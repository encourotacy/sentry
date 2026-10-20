#include <chrono>
#include <memory>
#include <optional>
#include <opencv2/opencv.hpp>
#include <string>

#include "io/camera.hpp"
#include "io/ovgimbal.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tasks/auto_buff/buff_type.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/recorder.hpp"

namespace
{
const std::string keys =
  "{help h usage ? |                           | 输出命令行参数说明}"
  "{@config-path   | configs/standard4.yaml   | 位置参数，yaml配置文件路径 }";

const char * energy_name(auto_buff::EnergyType energy_type)
{
  return energy_type == auto_buff::EnergyType::SMALL ? "small" : "big";
}
}  // namespace

const std::unordered_map<int, auto_aim::ArmorName> num2armor_name = {
  {1, auto_aim::ArmorName::one},
  {2, auto_aim::ArmorName::two},
  {3, auto_aim::ArmorName::three},
  {4, auto_aim::ArmorName::four},
  {5, auto_aim::ArmorName::sentry},
  {6, auto_aim::ArmorName::outpost},
  {7, auto_aim::ArmorName::base},
  {0, auto_aim::ArmorName::not_armor}};

const std::unordered_map<auto_aim::ArmorName, uint8_t> armor_name2num = {
  {auto_aim::ArmorName::one, 1},
  {auto_aim::ArmorName::two, 2},
  {auto_aim::ArmorName::three, 3},
  {auto_aim::ArmorName::four, 4},
  {auto_aim::ArmorName::five, 0},
  {auto_aim::ArmorName::sentry, 5},
  {auto_aim::ArmorName::outpost, 6},
  {auto_aim::ArmorName::base, 7},
  {auto_aim::ArmorName::not_armor, 0}};

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  tools::logger()->info("[OVGimbal Standard MPC] auto aim=mpc, buff aim=arm-only");
  tools::Exiter exiter;
  tools::Recorder recorder;

  io::OVGimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::YOLO yolo(config_path);
  auto_aim::Solver aim_solver(config_path);
  auto_aim::Tracker tracker(config_path, aim_solver);
  // auto_aim::Aimer aimer(config_path);
  // auto_aim::Shooter shooter(config_path);
  auto_aim::Planner planner(config_path);

  auto_buff::Buff_Detector small_buff_detector(config_path, "small_buff_model");
  auto_buff::Buff_Detector big_buff_detector(config_path, "big_buff_model");
  auto_buff::Solver buff_solver(config_path);
  auto small_target = std::make_shared<auto_buff::SmallTarget>();
  auto big_target = std::make_shared<auto_buff::BigTarget>();
  auto_buff::Aimer buff_aimer(config_path);
  auto_buff::DoubleBoardController double_board_controller;

  small_buff_detector.setEnergyType(auto_buff::EnergyType::SMALL);
  small_buff_detector.setTarget(std::static_pointer_cast<auto_buff::Target>(small_target));
  big_buff_detector.setEnergyType(auto_buff::EnergyType::BIG);
  big_buff_detector.setTarget(std::static_pointer_cast<auto_buff::Target>(big_target));

  std::optional<auto_buff::EnergyType> active_buff_energy;
  auto configure_buff = [&](auto_buff::EnergyType energy_type) {
    if (active_buff_energy.has_value() && active_buff_energy.value() == energy_type) return;
    active_buff_energy = energy_type;
    buff_solver.setEnergyType(energy_type);
    tools::logger()->info("[OVGimbal Standard MPC] switch buff energy={}", energy_name(energy_type));
  };

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  int frame_count = 0;
  auto last_mode = io::AutoAimMode::unknown;

  tools::logger()->info(
    "[OVGimbal Standard MPC] started, timing offset={:.2f}ms", gimbal.offset_ms());

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
    const auto mode = gimbal.autoaim_mode();
    // const auto mode = io::AutoAimMode::big_energy;

    if (mode != last_mode) {
      tools::logger()->info(
        "[OVGimbal Standard MPC] switch mode={} raw=0x{:02X}", gimbal.autoaim_mode_name(),
        gimbal.raw_mode());
      last_mode = mode;
    }

    const auto q = gimbal.imu_at_image(timestamp);
    aim_solver.set_R_gimbal2world(q);
    buff_solver.set_R_gimbal2world(q);
    
    recorder.record(img, q ,timestamp);

    io::Command command{false, false, 0.0, 0.0};
    io::Command last_command{false, false, 0.0, 0.0};

    if (mode == io::AutoAimMode::normal || mode == io::AutoAimMode::anti_top
         || mode == io::AutoAimMode::outpost) {
      auto armors = yolo.detect(img, frame_count);

      if (mode == io::AutoAimMode::outpost) // 前哨模式下只打前哨
        armors.remove_if([&](const auto_aim::Armor & a) { return a.name != auto_aim::ArmorName::outpost; });
      auto invincibility_num = gimbal.invincibility();
      
      for (int i = 0; i < 5; i ++) { // 过滤无敌单位的装甲板
        if (invincibility_num % 2){
          auto filterd_armor = num2armor_name.at(i+1);
          armors.remove_if([&](const auto_aim::Armor & a) { return a.name == filterd_armor; });
        }
        invincibility_num = invincibility_num >> 1;
      }

      auto targets = tracker.track(armors, timestamp, true);
      std::optional<auto_aim::Target> target_for_plan =
        targets.empty() ? std::nullopt : std::make_optional(targets.front());
      auto plan = planner.plan(target_for_plan, gimbal.bullet_speed());
      plan.yaw *= 57.3;
      plan.pitch *= 57.3;
      plan.yaw_vel *= 57.3;
      plan.yaw_acc *= 57.3;
      plan.pitch_vel *= 57.3;
      plan.pitch_acc *= 57.3;

      uint8_t armor_id = 0;
      if (!targets.empty()) armor_id = armor_name2num.at(targets.front().name);

      gimbal.send(
        plan.control, plan.fire, plan.yaw, plan.pitch, plan.yaw_vel, plan.yaw_acc, plan.pitch_vel,
        plan.pitch_acc, armor_id);
    } else if (mode == io::AutoAimMode::small_energy || mode == io::AutoAimMode::big_energy) {
      const auto energy_type = mode == io::AutoAimMode::small_energy ? auto_buff::EnergyType::SMALL
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

      gimbal.send(
        command.control, command.shoot, static_cast<float>(command.yaw*57.3),
        static_cast<float>(command.pitch*57.3), 0.0f, 0.0f, 0.0f, 0.0f);
      last_command = command;
      } else {
      gimbal.send(
        false, false, static_cast<float>(last_command.yaw*57.3),
        static_cast<float>(last_command.pitch*57.3), 0.0f, 0.0f, 0.0f, 0.0f);
      // gimbal.send(false, false, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    }
  }

  gimbal.send(false, false, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
  return 0;
}
