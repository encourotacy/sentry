#include <fmt/core.h>

#include <chrono>
#include <memory>
#include <nlohmann/json.hpp>
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
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

namespace
{
const std::string keys =
  "{help h usage ? |                           | 输出命令行参数说明}"
  "{@config-path   | configs/standard4.yaml   | 位置参数，yaml配置文件路径 }";

double rad2deg(double rad) { return rad * 57.3; }

const char * energy_name(auto_buff::EnergyType energy_type)
{
  return energy_type == auto_buff::EnergyType::SMALL ? "small" : "big";
}

std::string board_index_string(int index)
{
  return index >= 0 ? fmt::format("#{}", index + 1) : "none";
}

const char * chain_name(io::AutoAimMode mode)
{
  switch (mode) {
    case io::AutoAimMode::normal:
    case io::AutoAimMode::anti_top:
      return "auto_aim";
    case io::AutoAimMode::small_energy:
    case io::AutoAimMode::big_energy:
      return "auto_buff";
    case io::AutoAimMode::unknown:
    default:
      return "unknown";
  }
}

void put_common_plot_data(
  nlohmann::json & data, const io::OVGimbal & gimbal, const io::Command & command,
  io::AutoAimMode mode)
{
  const auto gs = gimbal.state();
  data["mode_raw"] = gimbal.raw_mode();
  data["mode_name"] = gimbal.autoaim_mode_name();
  data["active_chain"] = chain_name(mode);
  data["gimbal_yaw"] = rad2deg(gs.yaw);
  data["gimbal_pitch"] = rad2deg(gs.pitch);
  data["gimbal_yaw_vel"] = rad2deg(gs.yaw_vel);
  data["gimbal_pitch_vel"] = rad2deg(gs.pitch_vel);
  data["bullet_speed"] = gs.bullet_speed;
  data["cmd_control"] = command.control ? 1 : 0;
  data["cmd_shoot"] = command.shoot ? 1 : 0;
  if (command.control) {
    data["cmd_yaw"] = rad2deg(command.yaw);
    data["cmd_pitch"] = rad2deg(command.pitch);
  }
}

void draw_common_overlay(
  cv::Mat & img, const io::OVGimbal & gimbal, const io::Command & command,
  io::AutoAimMode mode)
{
  const auto gs = gimbal.state();
  tools::draw_text(
    img,
    fmt::format(
      "Mode: {}(0x{:02X}) | Chain: {}", gimbal.autoaim_mode_name(), gimbal.raw_mode(),
      chain_name(mode)),
    {10, 30}, {255, 255, 255});
  tools::draw_text(
    img,
    fmt::format(
      "Gimbal: yaw={:.1f}, pitch={:.1f}, bullet={:.1f}", rad2deg(gs.yaw), rad2deg(gs.pitch),
      gs.bullet_speed),
    {10, 60}, {255, 255, 255});
  tools::draw_text(
    img,
    fmt::format(
      "Cmd: control={}, yaw={:.2f}, pitch={:.2f}, shoot={}", command.control,
      rad2deg(command.yaw), rad2deg(command.pitch), command.shoot),
    {10, 90}, command.control ? cv::Scalar(154, 50, 205) : cv::Scalar(120, 120, 120));
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
  tools::Plotter plotter;
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
    const auto frame_start = std::chrono::steady_clock::now();
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
    recorder.record(img, q, timestamp);

    const Eigen::Vector3d ypr = tools::eulers(aim_solver.R_gimbal2world(), 2, 1, 0);
    io::Command command{false, false, 0.0, 0.0};
    nlohmann::json data;
    data["frame_count"] = frame_count;

    if (mode == io::AutoAimMode::normal || mode == io::AutoAimMode::anti_top) {
      const auto detect_start = std::chrono::steady_clock::now();
      auto armors = yolo.detect(img, frame_count);
      const auto detect_end = std::chrono::steady_clock::now();

      const auto track_start = std::chrono::steady_clock::now();
      auto targets = tracker.track(armors, timestamp);
      const auto track_end = std::chrono::steady_clock::now();

      const auto aim_start = std::chrono::steady_clock::now();
      command = aimer.aim(targets, timestamp, gimbal.bullet_speed(), aimer_to_now);
      const auto aim_end = std::chrono::steady_clock::now();
      command.shoot = shooter.shoot(command, aimer, targets, ypr, tracker.state() == "tracking");
      gimbal.send(command);

      data["armor_num"] = armors.size();
      data["tracker_state"] = tracker.state();
      data["auto_aim_detect_time"] = tools::delta_time(detect_end, detect_start) * 1e3;
      data["auto_aim_track_time"] = tools::delta_time(track_end, track_start) * 1e3;
      data["auto_aim_aim_time"] = tools::delta_time(aim_end, aim_start) * 1e3;

      if (!armors.empty()) {
        auto & armor = armors.front();
        aim_solver.solve(armor);
        data["armor_x"] = armor.xyz_in_world[0];
        data["armor_y"] = armor.xyz_in_world[1];
        data["armor_yaw"] = rad2deg(armor.ypr_in_world[0]);
        data["armor_yaw_raw"] = rad2deg(armor.yaw_raw);
      }

      if (!targets.empty()) {
        auto target = targets.front();
        for (const Eigen::Vector4d & xyza : target.armor_xyza_list()) {
          auto image_points =
            aim_solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
          tools::draw_points(img, image_points, {0, 255, 0});
        }

        auto aim_point = aimer.debug_aim_point;
        data["aim_point_valid"] = aim_point.valid ? 1 : 0;
        data["aim_point_source"] = aim_point.source;
        data["aim_point_armor_id"] = aim_point.armor_id;

        Eigen::Vector4d aim_xyza = aim_point.xyza;
        auto image_points =
          aim_solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, aim_point.valid ? cv::Scalar(0, 0, 255) :
                                                                cv::Scalar(255, 0, 0));

        Eigen::VectorXd x = target.ekf_x();
        data["x"] = x[0];
        data["vx"] = x[1];
        data["y"] = x[2];
        data["vy"] = x[3];
        data["z"] = x[4];
        data["vz"] = x[5];
        data["a"] = rad2deg(x[6]);
        data["w"] = x[7];
        data["r"] = x[8];
        data["l"] = x[9];
        data["h"] = x[10];
        data["last_id"] = target.last_id;
      }

      tools::draw_text(
        img, fmt::format("[{}]", tracker.state()), {10, 120}, {0, 255, 0});
    } else if (mode == io::AutoAimMode::small_energy || mode == io::AutoAimMode::big_energy) {
      const auto energy_type = mode == io::AutoAimMode::small_energy ? auto_buff::EnergyType::SMALL
                                                                     : auto_buff::EnergyType::BIG;
      configure_buff(energy_type);

      const auto detect_start = std::chrono::steady_clock::now();
      std::optional<auto_buff::PowerRune> power_runes = buff_detector.detect(img);
      const auto detect_end = std::chrono::steady_clock::now();

      if (energy_type == auto_buff::EnergyType::BIG && power_runes.has_value()) {
        double_board_controller.updateSelection(power_runes.value(), timestamp);
      }

      const bool should_solve =
        power_runes.has_value() &&
        (energy_type == auto_buff::EnergyType::SMALL ||
         (power_runes->selectedBoardIndex() >= 0 &&
          power_runes->isBoardLit(power_runes->selectedBoardIndex())));

      const auto solve_start = std::chrono::steady_clock::now();
      if (should_solve) {
        buff_solver.solve(power_runes);
      }
      const auto solve_end = std::chrono::steady_clock::now();

      const auto aim_start = std::chrono::steady_clock::now();
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
      const auto aim_end = std::chrono::steady_clock::now();
      gimbal.send(command);

      data["energy_type"] = energy_name(energy_type);
      data["buff_detect_time"] = tools::delta_time(detect_end, detect_start) * 1e3;
      data["buff_solve_time"] = tools::delta_time(solve_end, solve_start) * 1e3;
      data["buff_aim_time"] = tools::delta_time(aim_end, aim_start) * 1e3;

      if (power_runes.has_value()) {
        const auto & p = power_runes.value();
        const bool is_double_board_mode = energy_type == auto_buff::EnergyType::BIG;
        const bool has_selected_board =
          p.selectedBoardIndex() >= 0 && p.isBoardLit(p.selectedBoardIndex());

        if (!is_double_board_mode || has_selected_board) {
          data["buff_yaw"] = rad2deg(p.ypr_in_world[0]);
          data["buff_roll"] = rad2deg(p.ypr_in_world[2]);
        }
        data["hit_ring"] = p.hit_ring;
        data["activated_arms"] = p.activated_arms;
        data["selected_board"] = p.selectedBoardIndex();
        data["candidate_board"] = p.candidateBoardIndex();
        data["double_board_mode"] = is_double_board_mode ? 1 : 0;

        tools::draw_text(
          img, fmt::format("Energy: {}", energy_name(energy_type)), {10, 120}, {180, 255, 180});
        tools::draw_text(
          img, fmt::format("Hit Ring: {}", p.hit_ring), {10, 150}, {0, 255, 255});
        tools::draw_text(
          img, fmt::format("Arms: {}/5", p.activated_arms), {10, 180}, {0, 255, 255});
        if (is_double_board_mode) {
          tools::draw_text(
            img, fmt::format("Selected: {}", board_index_string(p.selectedBoardIndex())),
            {10, 210}, {0, 255, 0});
          tools::draw_text(
            img, fmt::format("Candidate: {}", board_index_string(p.candidateBoardIndex())),
            {10, 240}, {0, 215, 255});
        }
        if (
          is_double_board_mode && p.selectedBoardIndex() >= 0 &&
          p.isBoardLit(p.selectedBoardIndex())) {
          cv::circle(img, p.target().center, 24, cv::Scalar(0, 255, 0), 3);
        }
      } else {
        tools::draw_text(
          img, fmt::format("Energy: {} | no target", energy_name(energy_type)), {10, 120},
          {180, 255, 180});
      }
    } else {
      gimbal.send(command);
      data["unknown_mode"] = 1;
      tools::draw_text(
        img, fmt::format("Unknown mode 0x{:02X}: no control", gimbal.raw_mode()), {10, 120},
        {0, 0, 255});
    }

    data["gimbal_roll"] = rad2deg(ypr[2]);
    data["attitude_pitch"] = rad2deg(ypr[1]);
    data["attitude_yaw"] = rad2deg(ypr[0]);
    data["total_time"] = tools::delta_time(std::chrono::steady_clock::now(), frame_start) * 1e3;
    put_common_plot_data(data, gimbal, command, mode);
    plotter.plot(data);

    draw_common_overlay(img, gimbal, command, mode);
    cv::resize(img, img, {}, 0.5, 0.5);
    cv::imshow("OVGimbal Standard", img);
    auto key = cv::waitKey(1);
    if (key == 'q') break;
  }

  gimbal.send(io::Command{false, false, 0.0, 0.0});
  return 0;
}
