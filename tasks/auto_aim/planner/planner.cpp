#include "planner.hpp"

#include <vector>

#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

namespace auto_aim
{
Planner::Planner(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  yaw_offset_ = tools::read<double>(yaml, "yaw_offset") / 57.3;
  pitch_offset_ = tools::read<double>(yaml, "pitch_offset") / 57.3;
  fire_thresh_ = tools::read<double>(yaml, "fire_thresh") / 57.3;
  decision_speed_ = tools::read<double>(yaml, "decision_speed");
  high_speed_delay_time_ = tools::read<double>(yaml, "high_speed_delay_time");
  low_speed_delay_time_ = tools::read<double>(yaml, "low_speed_delay_time");
  resistance_k_ = tools::read<double>(yaml,"resistance_k");
  traj_g_ = tools::read<double>(yaml,"traj_g");

  // 不同距离时调整offset
  is_multiple_offset_ = tools::read<bool>(yaml, "is_multiple_offset");
  planner_offset_distance_ = tools::read<std::vector<double>>(yaml, "planner_offset_distance");
  planner_offset_yaw_ = tools::read<std::vector<double>>(yaml, "planner_offset_yaw");
  planner_offset_pitch_ = tools::read<std::vector<double>>(yaml, "planner_offset_pitch");
  if (planner_offset_distance_.size() != planner_offset_yaw_.size() ||
      planner_offset_distance_.size() != planner_offset_pitch_.size()) {
    tools::logger()->warn(
      "[Planner] offset array size mismatch, using default offset");
    is_multiple_offset_ = false;
  }

  // 距离判断，在远距离时使用更严格的容差
  planner_judge_distance_ = tools::read<std::vector<double>>(yaml, "planner_judge_distance");
  planner_fire_thresh_ = tools::read<std::vector<double>>(yaml, "planner_fire_thresh");
  is_multiple_thresh_ = tools::read<bool>(yaml, "is_multiple_thresh");
  if (planner_fire_thresh_.size() != planner_judge_distance_.size()) {
    tools::logger()->warn(
      "[Planner]fire_thresh array size mismatch, using default fire_thresh");
    is_multiple_thresh_ = false;
  }

  //高速旋转时强制开火
  high_spin_force_enabled_planner_ = tools::read<bool>(yaml, "high_spin_force_enabled_planner");
  high_spin_force_enter_speed_planner_ = tools::read<double>(yaml, "high_spin_force_enter_speed_planner"); 
  high_spin_force_exit_speed_planner_ = tools::read<double>(yaml, "high_spin_force_exit_speed_planner");
  force_plus_high_ = tools::read<double>(yaml, "force_plus_high");
  force_plus_low_ = tools::read<double>(yaml, "force_plus_low");

  setup_yaw_solver(config_path);
  setup_pitch_solver(config_path);
}

Plan Planner::plan(Target target, double bullet_speed)
{
  // 0. Check bullet speed
  if (bullet_speed < 10 || bullet_speed > 25) {
    bullet_speed = 23;
  }

  // 1. Predict fly_time
  Eigen::Vector3d xyz;
  auto min_dist = 1e10;
  for (auto & xyza : target.armor_xyza_list()) {
    auto dist = xyza.head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      xyz = xyza.head<3>();
    }
  }
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z(), resistance_k_, traj_g_);
  target.predict(bullet_traj.fly_time);

  // 2. Get trajectory
  double yaw0;
  Trajectory traj;
  try {
    yaw0 = aim(target, bullet_speed)(0);
    traj = get_trajectory(target, yaw0, bullet_speed);
  } catch (const std::exception & e) {
    tools::logger()->warn("Unsolvable target {:.2f}, e = {}", bullet_speed, e.what());
    return {false};
  }

  // 3. Solve yaw
  Eigen::VectorXd x0(2);
  x0 << traj(0, 0), traj(1, 0);
  tiny_set_x0(yaw_solver_, x0);

  yaw_solver_->work->Xref = traj.block(0, 0, 2, HORIZON);
  tiny_solve(yaw_solver_);

  // 4. Solve pitch
  x0 << traj(2, 0), traj(3, 0);
  tiny_set_x0(pitch_solver_, x0);

  pitch_solver_->work->Xref = traj.block(2, 0, 2, HORIZON);
  tiny_solve(pitch_solver_);

  Plan plan;
  plan.control = true;

  plan.target_yaw = tools::limit_rad(traj(0, HALF_HORIZON) + yaw0);
  plan.target_pitch = traj(2, HALF_HORIZON);

  plan.yaw = tools::limit_rad(yaw_solver_->work->x(0, HALF_HORIZON) + yaw0);
  plan.yaw_vel = yaw_solver_->work->x(1, HALF_HORIZON);
  plan.yaw_acc = yaw_solver_->work->u(0, HALF_HORIZON);

  plan.pitch = pitch_solver_->work->x(0, HALF_HORIZON);
  plan.pitch_vel = pitch_solver_->work->x(1, HALF_HORIZON);
  plan.pitch_acc = pitch_solver_->work->u(0, HALF_HORIZON);
  
  // 4. 判断是否开火
  double final_fire_thresh_ = get_thresh(min_dist, std::abs(target.ekf_x()[7]));
  
  auto shoot_offset_ = 2;
  plan.fire =
    std::hypot(
      traj(0, HALF_HORIZON + shoot_offset_) - yaw_solver_->work->x(0, HALF_HORIZON + shoot_offset_),
      traj(2, HALF_HORIZON + shoot_offset_) -
        pitch_solver_->work->x(0, HALF_HORIZON + shoot_offset_)) < final_fire_thresh_;

  if(std::abs(last_yaw_ - plan.yaw) > final_fire_thresh_*2) plan.fire = false;
  last_yaw_ = plan.yaw;

  return plan;
}

Plan Planner::plan(std::optional<Target> target, double bullet_speed)
{
  if (!target.has_value()) return {false};

  double delay_time =
    std::abs(target->ekf_x()[7]) > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;

  auto future = std::chrono::steady_clock::now() + std::chrono::microseconds(int(delay_time * 1e6));

  target->predict(future);

  return plan(*target, bullet_speed);
}

void Planner::setup_yaw_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_yaw_acc = tools::read<double>(yaml, "max_yaw_acc");
  auto Q_yaw = tools::read<std::vector<double>>(yaml, "Q_yaw");
  auto R_yaw = tools::read<std::vector<double>>(yaml, "R_yaw");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_yaw.data());
  Eigen::Matrix<double, 1, 1> R(R_yaw.data());
  tiny_setup(&yaw_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -180);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 180);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_yaw_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_yaw_acc);
  tiny_set_bound_constraints(yaw_solver_, x_min, x_max, u_min, u_max);

  yaw_solver_->settings->max_iter = 10;
}

void Planner::setup_pitch_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_pitch_acc = tools::read<double>(yaml, "max_pitch_acc");
  auto Q_pitch = tools::read<std::vector<double>>(yaml, "Q_pitch");
  auto R_pitch = tools::read<std::vector<double>>(yaml, "R_pitch");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_pitch.data());
  Eigen::Matrix<double, 1, 1> R(R_pitch.data());
  tiny_setup(&pitch_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_pitch_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_pitch_acc);
  tiny_set_bound_constraints(pitch_solver_, x_min, x_max, u_min, u_max);

  pitch_solver_->settings->max_iter = 10;
}

Eigen::Matrix<double, 2, 1> Planner::aim(const Target & target, double bullet_speed)
{
  Eigen::Vector3d xyz;
  double yaw;
  auto min_dist = 1e10;

  for (auto & xyza : target.armor_xyza_list()) {
    auto dist = xyza.head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      xyz = xyza.head<3>();
      yaw = xyza[3];
    }
  }
  debug_xyza = Eigen::Vector4d(xyz.x(), xyz.y(), xyz.z(), yaw);

  auto azim = std::atan2(xyz.y(), xyz.x());
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z(), resistance_k_, traj_g_);
  if (bullet_traj.unsolvable) throw std::runtime_error("Unsolvable bullet trajectory!");

  // 不同距离时调整offset
  double final_yaw_offset = yaw_offset_;
  double final_pitch_offset = pitch_offset_;
  if (is_multiple_offset_) {
    for (int i = 0; i < planner_offset_distance_.size(); i++) {
      if (min_dist < planner_offset_distance_.at(i)) {
        final_yaw_offset = planner_offset_yaw_.at(i) / 57.3;
        final_pitch_offset = planner_offset_pitch_.at(i) / 57.3;
        break;
      }
    }
  }

  return {tools::limit_rad(azim + final_yaw_offset), -bullet_traj.pitch - final_pitch_offset};
}

Trajectory Planner::get_trajectory(Target & target, double yaw0, double bullet_speed)
{
  Trajectory traj;

  target.predict(-DT * (HALF_HORIZON + 1));
  auto yaw_pitch_last = aim(target, bullet_speed);

  target.predict(DT);  // [0] = -HALF_HORIZON * DT -> [HHALF_HORIZON] = 0
  auto yaw_pitch = aim(target, bullet_speed);

  for (int i = 0; i < HORIZON; i++) {
    target.predict(DT);
    auto yaw_pitch_next = aim(target, bullet_speed);

    auto yaw_vel = tools::limit_rad(yaw_pitch_next(0) - yaw_pitch_last(0)) / (2 * DT);
    auto pitch_vel = (yaw_pitch_next(1) - yaw_pitch_last(1)) / (2 * DT);

    traj.col(i) << tools::limit_rad(yaw_pitch(0) - yaw0), yaw_vel, yaw_pitch(1), pitch_vel;

    yaw_pitch_last = yaw_pitch;
    yaw_pitch = yaw_pitch_next;
  }

  return traj;
}

double Planner::get_thresh(double distance, double angular_speed)
{//计算开火阈值，默认是一个常数，也可以使用多个阈值分段
  
  double final_fire_thresh = fire_thresh_ / 57.3; // 默认使用fire_thresh_
  //===根据距离调整开火阈值===

  if (is_multiple_thresh_) { // 使用多个阈值分段的方式
    final_fire_thresh = planner_fire_thresh_.back(); // 默认使用最后一个阈值
    for(int i = 0; i < planner_judge_distance_.size(); i ++) {
      if(distance < planner_judge_distance_.at(i)){
        final_fire_thresh = planner_fire_thresh_.at(i) / 57.3;
        break;
      }
    }
  }

  //===判断目标旋转速度，过快时放宽开火限制===
  if (!high_spin_force_enabled_planner_) { // 参考shooter的高速旋转强制开火逻辑，当目标旋转过快时放宽开火限制
    high_spin_force_active_planner_ = false;
  } else if (high_spin_force_active_planner_) {
    if (angular_speed < high_spin_force_exit_speed_planner_) high_spin_force_active_planner_ = false;
  } else if (angular_speed > high_spin_force_enter_speed_planner_) {
    high_spin_force_active_planner_ = true;
  }

  if (high_spin_force_active_planner_) {
    double speed_range = 17 - high_spin_force_exit_speed_planner_;
    if(speed_range <= 0) { // 通常不太会小于等于0，但是以防万一加一下
      double plus_k = (force_plus_high_ - force_plus_low_) / speed_range;
      final_fire_thresh *= (angular_speed - high_spin_force_exit_speed_planner_) * plus_k + force_plus_low_;
    }
  }

  return final_fire_thresh;
}

}  // namespace auto_aim