#include <chrono>
#include <memory>
#include <optional>
#include <opencv2/opencv.hpp>
#include <string>

#include "io/camera.hpp"
#include "io/ovgimbal.hpp"
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
#include "tools/math_tools.hpp"
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

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  tools::logger()->info("[OVGimbal Standard] buff aim=arm-only");

  tools::Exiter exiter;
  tools::Recorder recorder;

  io::OVGimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::YOLO yolo(config_path);
  auto_aim::Solver aim_solver(config_path);
  auto_aim::Tracker tracker(config_path, aim_solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);
  constexpr bool aimer_to_now = true;

  auto_buff::Buff_Detector buff_detector(config_path);
  auto_buff::Solver buff_solver(config_path);
  auto small_target = std::make_shared<auto_buff::SmallTarget>();
  auto big_target = std::make_shared<auto_buff::BigTarget>();
  auto_buff::Aimer buff_aimer(config_path);
  auto_buff::DoubleBoardController double_board_controller;

  std::optional<auto_buff::EnergyType> active_buff_energy;
  auto configure_buff = [&](auto_buff::EnergyType energy_type) {
    if (active_buff_energy.has_value() && active_buff_energy.value() == energy_type) return;
    active_buff_energy = energy_type;
    buff_detector.setEnergyType(energy_type);
    buff_solver.setEnergyType(energy_type);
    buff_detector.setTarget(
      energy_type == auto_buff::EnergyType::SMALL
        ? std::static_pointer_cast<auto_buff::Target>(small_target)
        : std::static_pointer_cast<auto_buff::Target>(big_target));
    tools::logger()->info("[OVGimbal Standard] switch buff energy={}", energy_name(energy_type));
  };

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  int frame_count = 0;
  auto last_mode = io::AutoAimMode::unknown;

  tools::logger()->info("[OVGimbal Standard] started, timing offset={:.2f}ms", gimbal.offset_ms());

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
    if (mode != last_mode) {
      tools::logger()->info(
        "[OVGimbal Standard] switch mode={} raw=0x{:02X}", gimbal.autoaim_mode_name(),
        gimbal.raw_mode());
      last_mode = mode;
    }

    const auto q = gimbal.imu_at_image(timestamp);
    aim_solver.set_R_gimbal2world(q);
    buff_solver.set_R_gimbal2world(q);
    //recorder.record(img, q, timestamp);

    const Eigen::Vector3d ypr = tools::eulers(aim_solver.R_gimbal2world(), 2, 1, 0);
    io::Command command{false, false, 0.0, 0.0};

    if (mode == io::AutoAimMode::normal || mode == io::AutoAimMode::anti_top) {
      auto armors = yolo.detect(img, frame_count);
      auto targets = tracker.track(armors, timestamp);
      command = aimer.aim(targets, timestamp, gimbal.bullet_speed(), aimer_to_now);
      command.shoot = shooter.shoot(command, aimer, targets, ypr, tracker.state() == "tracking");
      gimbal.send(command);
    } 
    else if (mode == io::AutoAimMode::small_energy || mode == io::AutoAimMode::big_energy) {
      const auto energy_type = mode == io::AutoAimMode::small_energy ? auto_buff::EnergyType::SMALL
                                                                     : auto_buff::EnergyType::BIG;
      configure_buff(energy_type);

      std::optional<auto_buff::PowerRune> power_runes = buff_detector.detect(img);

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
        }
      }
      gimbal.send(command);
    } 
    else {
      gimbal.send(command);
    }
  }

  gimbal.send(io::Command{false, false, 0.0, 0.0});
  return 0;
}
