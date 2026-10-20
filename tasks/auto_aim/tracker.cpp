#include "tracker.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <tuple>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
Tracker::Tracker(const std::string & config_path, Solver & solver)
: solver_{solver},
  detect_count_(0),
  temp_lost_count_(0),
  state_{"lost"},
  pre_state_{"lost"},
  last_timestamp_(std::chrono::steady_clock::now()),
  omni_target_priority_{ArmorPriority::fifth}
{
  auto yaml = YAML::LoadFile(config_path);
  enemy_color_ = (yaml["enemy_color"].as<std::string>() == "red") ? Color::red : Color::blue;
  min_detect_count_ = yaml["min_detect_count"].as<int>();
  max_temp_lost_count_ = yaml["max_temp_lost_count"].as<int>();
  outpost_max_temp_lost_count_ = yaml["outpost_max_temp_lost_count"].as<int>();
  normal_temp_lost_count_ = max_temp_lost_count_;
}

std::string Tracker::state() const { return state_; }

std::list<Target> Tracker::track(
  std::list<Armor> & armors, std::chrono::steady_clock::time_point t, bool use_enemy_color)
{
  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  handle_large_dt(dt);
  armors.remove_if([&](const auto_aim::Armor & a) { return a.color != enemy_color_; });

  armors.sort([](const Armor & a, const Armor & b) {
    cv::Point2f img_center(1440 / 2, 1080 / 2);
    auto distance_1 = cv::norm(a.center - img_center);
    auto distance_2 = cv::norm(b.center - img_center);
    return distance_1 < distance_2;
  });

  armors.sort(
    [](const auto_aim::Armor & a, const auto_aim::Armor & b) { return a.priority < b.priority; });

  bool found;
  if (state_ == "lost") {
    found = set_target(armors, t);
  } else {
    found = update_target(armors, t);
  }

  state_machine(found);

  if (state_ != "lost" && target_.diverged()) {
    tools::logger()->debug("[Tracker] Target diverged!");
    state_ = "lost";
    return {};
  }

  if (
    std::accumulate(
      target_.ekf().recent_nis_failures.begin(), target_.ekf().recent_nis_failures.end(), 0) >=
    (0.4 * target_.ekf().window_size)) {
    tools::logger()->debug("[Target] Bad Converge Found!");
    state_ = "lost";
    return {};
  }

  if (state_ == "lost") return {};

  std::list<Target> targets = {target_};
  (void)use_enemy_color;
  return targets;
}

std::tuple<omniperception::DetectionResult, std::list<Target>> Tracker::track(
  const std::vector<omniperception::DetectionResult> & detection_queue, std::list<Armor> & armors,
  std::chrono::steady_clock::time_point t, bool use_enemy_color)
{
  omniperception::DetectionResult switch_target{std::list<Armor>(), t, 0, 0};
  omniperception::DetectionResult temp_target{std::list<Armor>(), t, 0, 0};
  if (!detection_queue.empty()) {
    temp_target = detection_queue.front();
  }

  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  handle_large_dt(dt);
  
  armors.sort([](const Armor & a, const Armor & b) {
    cv::Point2f img_center(1440 / 2, 1080 / 2);
    auto distance_1 = cv::norm(a.center - img_center);
    auto distance_2 = cv::norm(b.center - img_center);
    return distance_1 < distance_2;
  });

  armors.sort([](const Armor & a, const Armor & b) { return a.priority < b.priority; });

  bool found;
  if (state_ == "lost") {
    found = set_target(armors, t);
  } else if (state_ == "tracking" && !armors.empty() && armors.front().priority < target_.priority) {
    found = set_target(armors, t);
    tools::logger()->debug("auto_aim switch target to {}", ARMOR_NAMES[armors.front().name]);
  } else if (
    state_ == "tracking" && !temp_target.armors.empty() &&
    temp_target.armors.front().priority < target_.priority && target_.convergened()) {
    state_ = "switching";
    switch_target = omniperception::DetectionResult{
      temp_target.armors, t, temp_target.delta_yaw, temp_target.delta_pitch};
    omni_target_priority_ = temp_target.armors.front().priority;
    found = false;
    tools::logger()->debug("omniperception find higher priority target");
  } else if (state_ == "switching") {
    found = !armors.empty() && armors.front().priority == omni_target_priority_;
  } else if (state_ == "detecting" && pre_state_ == "switching") {
    found = set_target(armors, t);
  } else {
    found = update_target(armors, t);
  }

  pre_state_ = state_;
  state_machine(found);

  if (state_ != "lost" && target_.diverged()) {
    tools::logger()->debug("[Tracker] Target diverged!");
    state_ = "lost";
    return {switch_target, {}};
  }

  if (state_ == "lost") return {switch_target, {}};

  std::list<Target> targets = {target_};
  (void)use_enemy_color;
  return {switch_target, targets};
}

void Tracker::state_machine(bool found)
{
  if (state_ == "lost") {
    if (!found) return;

    state_ = "detecting";
    detect_count_ = 1;
  } else if (state_ == "detecting") {
    if (found) {
      detect_count_++;
      if (detect_count_ >= min_detect_count_) state_ = "tracking";
    } else {
      detect_count_ = 0;
      state_ = "lost";
    }
  } else if (state_ == "tracking") {
    if (found) return;

    temp_lost_count_ = 1;
    state_ = "temp_lost";
  } else if (state_ == "switching") {
    if (found) {
      state_ = "detecting";
    } else {
      temp_lost_count_++;
      if (temp_lost_count_ > 200) state_ = "lost";
    }
  } else if (state_ == "temp_lost") {
    if (found) {
      state_ = "tracking";
    } else {
      temp_lost_count_++;
      if (target_.name == ArmorName::outpost)
        max_temp_lost_count_ = outpost_max_temp_lost_count_;
      else
        max_temp_lost_count_ = normal_temp_lost_count_;

      if (temp_lost_count_ > max_temp_lost_count_) state_ = "lost";
    }
  }
}

void Tracker::handle_large_dt(double dt)
{
  // 时间间隔过长，说明可能发生了相机离线。前哨站模型已经带有更长的 temp_lost
  // 保持窗口，单次时间戳跳变不应清空层级语义；后续由关联和状态机决定是否丢失。
  if (state_ == "lost" || dt <= 0.1) return;

  tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
  if (target_.name == ArmorName::outpost) return;

  state_ = "lost";
}

bool Tracker::set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  if (armors.empty()) return false;

  auto & armor = armors.front();
  solver_.solve(armor);

  auto is_balance = (armor.type == ArmorType::big) &&
                    (armor.name == ArmorName::three || armor.name == ArmorName::four ||
                     armor.name == ArmorName::five);

  if (is_balance) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
    target_ = Target(armor, t, 0.2, 2, P0_dig);
  } else if (armor.name == ArmorName::outpost) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 25, 81, 0.4, 100, 1e-4, 0, 0}};
    target_ = Target(armor, t, 0.2765, 3, P0_dig);
  } else if (armor.name == ArmorName::base) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0}};
    target_ = Target(armor, t, 0.3205, 3, P0_dig);
  } else {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
    target_ = Target(armor, t, 0.2, 4, P0_dig);
  }

  return true;
}

bool Tracker::update_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  target_.predict(t);

  if (target_.name != ArmorName::outpost) {
    int found_count = 0;
    for (auto & armor : armors) {
      if (armor.name != target_.name || armor.type != target_.armor_type) continue;
      found_count++;
      solver_.solve(armor);
      target_.update(armor);
    }
    return found_count > 0;
  }

  std::vector<Armor *> candidates;
  candidates.reserve(armors.size());
  for (auto & armor : armors) {
    if (armor.name != target_.name || armor.type != target_.armor_type) continue;
    candidates.push_back(&armor);
  }
  if (candidates.empty()) return false;

  const std::vector<Eigen::Vector4d> predicted_armors = target_.armor_xyza_list();
  const int predicted_count = static_cast<int>(predicted_armors.size());
  auto normalized_square = [](double value, double gate) {
    const double safe_gate = gate <= 1e-6 ? 1e-6 : gate;
    const double normalized = value / safe_gate;
    return normalized * normalized;
  };

  Armor * best_armor = nullptr;
  double best_score = std::numeric_limits<double>::infinity();
  int best_id = -1;
  std::array<double, 3> best_scores{
    std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::infinity()};

  for (auto * armor_ptr : candidates) {
    auto & armor = *armor_ptr;
    solver_.solve(armor);

    for (int id = 0; id < std::max(1, predicted_count); ++id) {
      double score = 0.0;
      if (predicted_count > 0) {
        const Eigen::Vector4d & xyza = predicted_armors[id];
        const Eigen::Vector3d predicted_ypd = tools::xyz2ypd(xyza.head(3));
        const double yaw_err = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3]));
        const double bearing_err =
          std::abs(tools::limit_rad(armor.ypd_in_world[0] - predicted_ypd[0]));
        const double pitch_err =
          std::abs(tools::limit_rad(armor.ypd_in_world[1] - predicted_ypd[1]));
        const double dist_err = std::abs(armor.ypd_in_world[2] - predicted_ypd[2]);
        const double z_err = std::abs(armor.xyz_in_world[2] - xyza[2]);

        score =
          1.0 * normalized_square(yaw_err, 0.45) +
          0.7 * normalized_square(bearing_err, 0.40) +
          0.4 * normalized_square(pitch_err, 0.30) +
          0.25 * normalized_square(dist_err, 0.60) +
          0.8 * normalized_square(z_err, 0.08);

        if (id == target_.last_id) score *= 0.95;
      }

      if (score < best_score) {
        best_score = score;
        best_armor = &armor;
        best_id = id;
      }
      if (id >= 0 && id < static_cast<int>(best_scores.size())) {
        best_scores[id] = std::min(best_scores[id], score);
      }
    }
  }

  if (best_armor == nullptr) return false;

  const double reject_score = target_.outpost_layer_locked() ? 28.0 : 80.0;
  if (best_score > reject_score) {
    target_.set_outpost_association_debug(best_id, best_scores, best_score, "score_gate");
    return false;
  }

  target_.set_outpost_association_debug(best_id, best_scores, best_score, "");
  target_.update(*best_armor, best_id);
  return true;
}


}  // namespace auto_aim
