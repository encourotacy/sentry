#include "buff_aimer.hpp"

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"

namespace auto_buff
{

Aimer::Aimer(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);

  yaw_offset_ = yaml["yaw_offset_buff"].as<double>() / 57.3;
  pitch_offset_ = yaml["pitch_offset_buff"].as<double>() / 57.3;
  fire_gap_time_ = yaml["fire_gap_time"].as<double>();
  predict_time_ = yaml["predict_time"].as<double>();

  last_fire_t_ = std::chrono::steady_clock::now();
}

Eigen::Vector3d Aimer::getCurrentArmCenter(Target & target) const
{
  return target.point_buff2world(Eigen::Vector3d(0, 0, 0.7));
}

bool Aimer::getSendAngle(
  Target & target, const double predict_time, const double bullet_speed, double & yaw, double & pitch)
{
  const auto snapshot = target.snapshot();
  auto restore_state = [&]() { target.restore(snapshot); };

  target.predict(predict_time);
  const Eigen::VectorXd predicted_state = target.ekf_x();
  if (predicted_state.size() == 0 || !predicted_state.allFinite()) {
    tools::logger()->debug("[Aimer] 跳过本帧：预测后的目标状态非法");
    restore_state();
    return false;
  }
  angle_ = target.ekf_x()[5];

  Eigen::Vector3d aim_in_world = getCurrentArmCenter(target);
  if (!aim_in_world.allFinite()) {
    tools::logger()->debug("[Aimer] 跳过本帧：目标点坐标非法");
    restore_state();
    return false;
  }

  double d = std::sqrt(aim_in_world[0] * aim_in_world[0] + aim_in_world[1] * aim_in_world[1]);
  double h = aim_in_world[2];
  if (!std::isfinite(d) || !std::isfinite(h)) {
    tools::logger()->debug("[Aimer] 跳过本帧：弹道输入非法 d={} h={}", d, h);
    restore_state();
    return false;
  }

  tools::Trajectory trajectory0(bullet_speed, d, h);
  if (trajectory0.unsolvable) {
    tools::logger()->debug(
      "[Aimer] Unsolvable trajectory0: {:.2f} {:.2f} {:.2f}", bullet_speed, d, h);
    restore_state();
    return false;
  }

  target.predict(trajectory0.fly_time);
  if (!target.ekf_x().allFinite()) {
    tools::logger()->debug("[Aimer] 跳过本帧：飞行时间预测后的目标状态非法");
    restore_state();
    return false;
  }
  aim_in_world = getCurrentArmCenter(target);

  if (!aim_in_world.allFinite()) {
    tools::logger()->debug("[Aimer] 跳过本帧：二次预测后的目标点坐标非法");
    restore_state();
    return false;
  }

  d = std::sqrt(aim_in_world[0] * aim_in_world[0] + aim_in_world[1] * aim_in_world[1]);
  h = aim_in_world[2];
  if (!std::isfinite(d) || !std::isfinite(h)) {
    tools::logger()->debug("[Aimer] 跳过本帧：二次弹道输入非法 d={} h={}", d, h);
    restore_state();
    return false;
  }

  tools::Trajectory trajectory1(bullet_speed, d, h);
  if (trajectory1.unsolvable) {
    tools::logger()->debug(
      "[Aimer] Unsolvable trajectory1: {:.2f} {:.2f} {:.2f}", bullet_speed, d, h);
    restore_state();
    return false;
  }

  auto time_error = trajectory1.fly_time - trajectory0.fly_time;
  if (std::abs(time_error) > 0.025) {
    tools::logger()->debug("[Aimer] Large time error: {:.3f}", time_error);
    restore_state();
    return false;
  }

  yaw = std::atan2(aim_in_world[1], aim_in_world[0]) + yaw_offset_;
  pitch = trajectory1.pitch + pitch_offset_;
  if (!std::isfinite(yaw) || !std::isfinite(pitch)) {
    tools::logger()->debug("[Aimer] 跳过本帧：输出角度非法 yaw={} pitch={}", yaw, pitch);
    restore_state();
    return false;
  }

  last_hit_ring_ = target.getLastHitRing();

  restore_state();
  return true;
}

io::Command Aimer::aim(
  Target & target, std::chrono::steady_clock::time_point & timestamp, double bullet_speed,
  bool to_now)
{
  io::Command command = {false, false, 0, 0};

  if (target.is_unsolve()) return command;

  if (bullet_speed < 10) bullet_speed = 23;

  auto now = std::chrono::steady_clock::now();
  auto detect_now_gap = tools::delta_time(now, timestamp);
  double future = to_now ? (detect_now_gap + predict_time_) : 0.1 + predict_time_;

  double yaw, pitch;

  if (getSendAngle(target, future, bullet_speed, yaw, pitch)) {
    command.yaw = yaw;
    command.pitch = -pitch;

    bool angle_changed =
      std::abs(last_yaw_ - yaw) > 5 / 57.3 || std::abs(last_pitch_ - pitch) > 5 / 57.3;

    if (mistake_count_ > 3) {
      switch_fanblade_ = true;
      mistake_count_ = 0;
      command.control = true;
    } else if (angle_changed) {
      switch_fanblade_ = true;
      mistake_count_++;
      command.control = false;
    } else {
      switch_fanblade_ = false;
      mistake_count_ = 0;
      command.control = true;
    }

    last_yaw_ = yaw;
    last_pitch_ = pitch;
  }

  if (switch_fanblade_) {
    command.shoot = false;
    last_fire_t_ = now;
  } else if (!switch_fanblade_ && tools::delta_time(now, last_fire_t_) > fire_gap_time_) {
    command.shoot = true;
    last_fire_t_ = now;
  }

  return command;
}

auto_aim::Plan Aimer::mpc_aim(
  Target & target, std::chrono::steady_clock::time_point & timestamp, io::GimbalState gs,
  bool to_now)
{
  auto_aim::Plan plan = {false, false, 0, 0, 0, 0, 0, 0, 0, 0};

  if (target.is_unsolve()) return plan;

  double bullet_speed = (gs.bullet_speed < 10) ? 23 : gs.bullet_speed;

  auto now = std::chrono::steady_clock::now();
  auto detect_now_gap = tools::delta_time(now, timestamp);
  double future = to_now ? (detect_now_gap + predict_time_) : 0.1 + predict_time_;

  double yaw, pitch;

  if (getSendAngle(target, future, bullet_speed, yaw, pitch)) {
    plan.yaw = yaw;
    plan.pitch = -pitch;

    bool angle_changed =
      std::abs(last_yaw_ - yaw) > 5 / 57.3 || std::abs(last_pitch_ - pitch) > 5 / 57.3;

    if (mistake_count_ > 3) {
      switch_fanblade_ = true;
      mistake_count_ = 0;
      plan.control = true;
      first_in_aimer_ = true;
    } else if (angle_changed) {
      switch_fanblade_ = true;
      mistake_count_++;
      plan.control = false;
      first_in_aimer_ = true;
    } else {
      switch_fanblade_ = false;
      mistake_count_ = 0;
      plan.control = true;
    }

    last_yaw_ = yaw;
    last_pitch_ = pitch;

    if (plan.control) {
      if (first_in_aimer_) {
        plan.yaw_vel = 0;
        plan.yaw_acc = 0;
        plan.pitch_vel = 0;
        plan.pitch_acc = 0;
        first_in_aimer_ = false;
      } else {
        auto dt = predict_time_;
        double last_yaw_mpc, last_pitch_mpc;
        getSendAngle(target, predict_time_ * -1, bullet_speed, last_yaw_mpc, last_pitch_mpc);

        plan.yaw_vel = tools::limit_rad(yaw - last_yaw_mpc) / (2 * dt);
        plan.yaw_acc =
          (tools::limit_rad(yaw - gs.yaw) - tools::limit_rad(gs.yaw - last_yaw_mpc)) /
          std::pow(dt, 2);

        plan.pitch_vel = tools::limit_rad(-pitch + last_pitch_mpc) / (2 * dt);
        plan.pitch_acc = (-pitch - gs.pitch - (gs.pitch + last_pitch_mpc)) / std::pow(dt, 2);
      }
    }
  }

  if (switch_fanblade_) {
    plan.fire = false;
    last_fire_t_ = now;
  } else if (!switch_fanblade_ && tools::delta_time(now, last_fire_t_) > fire_gap_time_) {
    plan.fire = true;
    last_fire_t_ = now;
  }

  return plan;
}

}  // namespace auto_buff
