#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <opencv2/opencv.hpp>
#include <string>

#include "io/camera.hpp"
#include "io/ros2/ros2_gimbal.hpp"
#include "io/ros2/subscribe.hpp"

#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"

#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tasks/auto_buff/buff_type.hpp"

#include "tasks/omniperception/ovsentry_omni_logic.hpp"

#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/recorder.hpp"

namespace
{
const std::string keys =
  "{help h usage ? |                           | 输出命令行参数说明}"
  "{@config-path   | configs/sentry.yaml    | 位置参数，yaml配置文件路径 }";

const char * energy_name(auto_buff::EnergyType energy_type)
{
  return energy_type == auto_buff::EnergyType::SMALL ? "small" : "big";
}

const char * mode_name(io::ROS2Mode mode)
{
  switch (mode) {
    case io::ROS2Mode::small_buff:
      return "small_buff";
    case io::ROS2Mode::big_buff:
      return "big_buff";
    case io::ROS2Mode::auto_aim:
    default:
      return "auto_aim";
  }
}

double unwrap_yaw(double yaw, double reference)
{
  return reference + tools::limit_rad(yaw - reference);
}

double target_big_yaw(const auto_aim::Target & target, double current_big_yaw)
{
  const auto & ekf_x = target.ekf_x();
  return unwrap_yaw(std::atan2(ekf_x[2], ekf_x[0]), current_big_yaw);
}

bool unlocked_outpost(const auto_aim::Target & target)
{
  return target.name == auto_aim::ArmorName::outpost && !target.outpost_layer_locked();
}

uint8_t armor_name_to_nav_id(auto_aim::ArmorName name)
{
  switch (name) {
    case auto_aim::ArmorName::one:
      return 1;
    case auto_aim::ArmorName::two:
      return 2;
    case auto_aim::ArmorName::three:
      return 3;
    case auto_aim::ArmorName::four:
      return 4;
    case auto_aim::ArmorName::five:
      return 5;
    case auto_aim::ArmorName::sentry:
      return 6;
    case auto_aim::ArmorName::outpost:
      return 7;
    case auto_aim::ArmorName::base:
      return 8;
    default:
      return 0;
  }
}
}  // namespace

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  tools::logger()->info("[OVSentry MPC] auto aim = mpc, buff aim = arm-only");

  tools::Exiter exiter;
  tools::Recorder recorder;

  io::ROS2Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::YOLO yolo(config_path);
  auto_aim::Solver aim_solver(config_path);
  auto_aim::Tracker tracker(config_path, aim_solver);
  auto_aim::Planner planner(config_path);

  const auto nav_config = io::load_nav_subscribe_config(config_path);
  io::ArmorIgnoreSubscriber armor_ignore(
    nav_config.auto_aim_ignore_topic, nav_config.auto_aim_ignore_msg_type);
  io::BuffRequestSubscriber buff_request(nav_config.request_buff_topic);

  auto_buff::Buff_Detector small_buff_detector(config_path, "small_buff_model");
  auto_buff::Buff_Detector big_buff_detector(config_path, "big_buff_model");
  auto_buff::Solver buff_solver(config_path);
  auto small_target = std::make_shared<auto_buff::SmallTarget>();
  auto big_target = std::make_shared<auto_buff::BigTarget>();
  auto_buff::Aimer buff_aimer(config_path);
  auto_buff::DoubleBoardController double_board_controller;

  // 将检测器和跟踪器配对
  small_buff_detector.setEnergyType(auto_buff::EnergyType::SMALL);
  small_buff_detector.setTarget(std::static_pointer_cast<auto_buff::Target>(small_target));
  big_buff_detector.setEnergyType(auto_buff::EnergyType::BIG);
  big_buff_detector.setTarget(std::static_pointer_cast<auto_buff::Target>(big_target));

  std::optional<auto_buff::EnergyType> active_buff_energy;
  auto configure_buff = [&](auto_buff::EnergyType energy_type) {
    if (active_buff_energy.has_value() && active_buff_energy.value() == energy_type) return;
    active_buff_energy = energy_type;
    buff_solver.setEnergyType(energy_type);
    tools::logger()->info("[OVSentry MPC] switch buff energy={}", energy_name(energy_type));
  };

  omniperception::OmniSearch omni(config_path);
  bool prev_lost = false;

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  int frame_count = 0;
  auto last_mode = io::ROS2Mode::auto_aim;

  tools::logger()->info("[OVSentry MPC] started, timing offset={:.2f}ms", gimbal.offset_ms());

  while (!exiter.exit()) {
    try {
      camera.read(img, timestamp);
      if (img.empty()) {
        tools::logger()->warn("读取到空图像，跳过此帧");
        continue;
      }
    } catch (const std::exception & e) {
      tools::logger()->error("读取摄像机图像失败: {}", e.what());
      continue;
    }

    frame_count++;
    omni.read_cameras();
    const auto mode = buff_request.mode();
    // const auto mode = io::ROS2Mode::auto_aim;
    // const auto mode = io::ROS2Mode::small_buff;
    // const auto mode = io::ROS2Mode::big_buff;
    if (mode != last_mode) {
      tools::logger()->info("[OVSentry MPC] switch mode={}", mode_name(mode));
      last_mode = mode;
    }

    const auto q = gimbal.imu_at_image(timestamp);
    aim_solver.set_R_gimbal2world(q);
    buff_solver.set_R_gimbal2world(q);
    recorder.record(img, q, timestamp);

    const bool buff_mode = mode == io::ROS2Mode::small_buff || mode == io::ROS2Mode::big_buff;
    if (!buff_mode) {
      auto armors = yolo.detect(img, frame_count);

      const auto ignore = armor_ignore.ignored();
      armors.remove_if([&](const auto_aim::Armor & armor) {
        return ignore.contains(armor_name_to_nav_id(armor.name));
      });

      auto targets = tracker.track(armors, timestamp, true);

      // 全向感知初始化与重置
      const auto tracker_state = tracker.state();
      const auto gimbal_state = gimbal.state();
      const bool lost = tracker_state == "lost";
      if (lost != prev_lost) omni.reset();
      prev_lost = lost;

      if (lost) {
        const auto steer = omni.update(
          gimbal_state.big_yaw,
          [&](std::chrono::steady_clock::time_point t) { return gimbal.big_yaw_at_image(t); },
          [&](uint8_t id) { return ignore.contains(id); }, frame_count);
        gimbal.send_mpc(
          steer.control, false, steer.big_yaw, steer.small_yaw, steer.pitch, 0, 0, 0, 0,
          steer.armor_id, 0, 0, 0);
      } else if (tracker_state == "detecting") {
        gimbal.send_mpc(
          true, false, gimbal_state.big_yaw, gimbal_state.yaw, -gimbal_state.pitch, 0, 0, 0, 0);
      } else {
        //=== planner & aimer start ===
        std::optional<auto_aim::Target> target_for_plan =
          targets.empty() ? std::nullopt : std::make_optional(targets.front());

        auto plan = planner.plan(target_for_plan, gimbal.bullet_speed());

        uint8_t armor_id = 0;
        double target_vx = 0.0;
        double target_vy = 0.0;
        double horizon_distance = 0.0;
        double big_yaw = plan.yaw;
        double small_yaw = plan.yaw;
        if (plan.control && !targets.empty()) {
          const auto ekf_x = targets.front().ekf_x();
          armor_id = armor_name_to_nav_id(targets.front().name);
          target_vx = ekf_x[1];
          target_vy = ekf_x[3];
          horizon_distance = std::hypot(ekf_x[0], ekf_x[2]);
          small_yaw = plan.yaw;
          big_yaw = unlocked_outpost(targets.front())
                      ? gimbal_state.big_yaw
                      : target_big_yaw(targets.front(), gimbal_state.big_yaw);
        }

        gimbal.send_mpc(
          plan.control, plan.fire, big_yaw, small_yaw, plan.pitch, plan.yaw_vel, plan.pitch_vel,
          plan.yaw_acc, plan.pitch_acc, armor_id, target_vx, target_vy, horizon_distance);
      }
    } else {
      if (prev_lost) omni.reset();
      prev_lost = false;
      const auto energy_type = mode == io::ROS2Mode::small_buff ? auto_buff::EnergyType::SMALL
                                                                : auto_buff::EnergyType::BIG;
      configure_buff(energy_type);
      auto & active_buff_detector =
        energy_type == auto_buff::EnergyType::SMALL ? small_buff_detector : big_buff_detector;

      std::optional<auto_buff::PowerRune> power_runes = active_buff_detector.detect(img);

      if (energy_type == auto_buff::EnergyType::BIG && power_runes.has_value()) {
        double_board_controller.updateSelection(power_runes.value(), timestamp);
      }

      const bool should_solve =
        power_runes.has_value() &&
        (energy_type == auto_buff::EnergyType::SMALL ||
         (power_runes->selectedBoardIndex() >= 0 &&
          power_runes->isBoardLit(power_runes->selectedBoardIndex())));

      if (should_solve) {
        buff_solver.solve(power_runes);
      }

      io::Command command{false, false, 0.0, 0.0};
      if (energy_type == auto_buff::EnergyType::SMALL) {
        small_target->get_target(power_runes, timestamp, energy_type, nullptr);
        if (!small_target->is_unsolve()) {
          command = buff_aimer.aim(*small_target, timestamp, gimbal.bullet_speed(), true);
          command.shoot = false;
        }
      } else {
        std::optional<auto_buff::PowerRune> tracking_power_runes = std::nullopt;
        if (
          power_runes.has_value() && power_runes->selectedBoardIndex() >= 0 &&
          power_runes->isBoardLit(power_runes->selectedBoardIndex())) {
          tracking_power_runes = power_runes;
        }
        auto sin_param = buff_solver.getSinusoidalParam();
        big_target->get_target(tracking_power_runes, timestamp, energy_type, &sin_param);
        if (!big_target->is_unsolve()) {
          command = buff_aimer.aim(*big_target, timestamp, gimbal.bullet_speed(), true);
          command.shoot = false;
        }
      }

      gimbal.send(command);
    }
  }

  gimbal.send_mpc(false, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  return 0;
}
