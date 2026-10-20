#include <fmt/core.h>

#include <chrono>
#include <cmath>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <opencv2/opencv.hpp>
#include <string>

#include "io/camera.hpp"
#include "io/usbcamera/usbcamera.hpp"
#include "io/ros2/ros2_gimbal.hpp"

#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"

#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tasks/auto_buff/buff_type.hpp"

#include "tasks/omniperception/decider.hpp"
#include "tasks/omniperception/ovsentry_omni_logic.hpp"

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
  "{@config-path   | configs/sentry.yaml    | 位置参数，yaml配置文件路径 }"
  "{left           | __yaml__                | 左前相机设备名(相对/dev)，默认读yaml.omni_left_path }"
  "{right          | __yaml__                | 右前相机设备名(相对/dev)，默认读yaml.omni_right_path }"
  "{back           | __yaml__                | 正后相机设备名(相对/dev)，默认读yaml.omni_back_path }"
  "{left_yaw       | 60                      | 左前相机中心yaw角(deg) }"
  "{right_yaw      | -60                     | 右前相机中心yaw角(deg) }"
  "{back_yaw       | 180                     | 正后相机中心yaw角(deg) }"
  "{fov_h          | 120                     | USB相机水平视场角(deg) }"
  "{fov_v          | 67                      | USB相机垂直视场角(deg) }"
  "{no-display     |                         | 关闭画面显示 }";

double rad2deg(double rad) { return rad * 57.3; }

const char * energy_name(auto_buff::EnergyType energy_type)
{
  return energy_type == auto_buff::EnergyType::SMALL ? "small" : "big";
}

const char * mode_name(io::Mode mode)
{
  switch (mode) {
    case io::Mode::small_buff:
      return "small_buff";
    case io::Mode::big_buff:
      return "big_buff";
    case io::Mode::auto_aim:
    default:
      return "auto_aim";
  }
}

std::string board_index_string(int index)
{
  return index >= 0 ? fmt::format("#{}", index + 1) : "none";
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

void put_common_plot_data(
  nlohmann::json & data, const io::ROS2Gimbal & gimbal, const io::Command & command, io::Mode mode)
{
  const auto gs = gimbal.state();
  data["gimbal_yaw"] = rad2deg(gs.yaw);
  data["gimbal_pitch"] = rad2deg(gs.pitch);
  data["gimbal_yaw_vel"] = rad2deg(gs.yaw_vel);
  data["gimbal_pitch_vel"] = rad2deg(gs.pitch_vel);
  if (command.control) {
    data["cmd_yaw"] = rad2deg(command.yaw);
    data["cmd_pitch"] = rad2deg(command.pitch);
  }
}

void draw_common_overlay(
  cv::Mat & img, const io::ROS2Gimbal & gimbal, const io::Command & command, io::Mode mode)
{
  const auto gs = gimbal.state();
  tools::draw_text(
    img, fmt::format("Mode: {}", mode_name(mode)), {10, 30}, {255, 255, 255});
  tools::draw_text(
    img,
    fmt::format(
      "Gimbal: yaw={:.1f}, big_yaw={:.1f}, pitch={:.1f}, bullet={:.1f}", rad2deg(gs.yaw),
      rad2deg(gs.big_yaw), rad2deg(gs.pitch), gs.bullet_speed),
    {10, 60}, {255, 255, 255});
  tools::draw_text(
    img,
    fmt::format(
      "Cmd: control={}, yaw={:.2f}, big_yaw={:.2f}, pitch={:.2f}, shoot={}", command.control,
      rad2deg(command.yaw), rad2deg(command.big_yaw), rad2deg(command.pitch), command.shoot),
    {10, 90}, command.control ? cv::Scalar(154, 50, 205) : cv::Scalar(120, 120, 120));
}
}  // namespace

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys); 
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage(); // 打印命令行参数说明
    return 0;
  }

  tools::logger()->info("[OVSentry MPC Debug] auto aim=mpc, buff aim=arm-only");

  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;

  io::ROS2Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::YOLO yolo(config_path);
  auto_aim::Solver aim_solver(config_path); 
  auto_aim::Tracker tracker(config_path, aim_solver); 
  auto_aim::Planner planner(config_path); 

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
    tools::logger()->info(
      "[OVSentry MPC Debug] switch buff energy={}", energy_name(energy_type)); 
  };

  cv::Mat img; 
  std::chrono::steady_clock::time_point timestamp; 
  int frame_count = 0;
  auto last_mode = io::Mode::auto_aim; 

  tools::logger()->info(
    "[OVSentry MPC Debug] started, timing offset={:.2f}ms", gimbal.offset_ms());

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
    const auto mode = gimbal.mode();
    // const auto mode = io::Mode::auto_aim;
    // const auto mode = io::Mode::small_buff;
    // const auto mode = io::Mode::big_buff;
    if (mode != last_mode) {
      tools::logger()->info(
        "[OVSentry MPC Debug] switch mode={}", io::MODES[static_cast<int>(mode)]);
      last_mode = mode;
    }

    const auto q = gimbal.imu_at_image(timestamp);
    aim_solver.set_R_gimbal2world(q);
    buff_solver.set_R_gimbal2world(q);
    recorder.record(img, q, timestamp);

    const Eigen::Vector3d ypr = tools::eulers(aim_solver.R_gimbal2world(), 2, 1, 0);
    io::Command command{false, false, 0.0, 0.0};
    io::Command command_aimer{false, false, 0.0, 0.0};
    nlohmann::json data;
    data["frame_count"] = frame_count;

    const bool buff_mode = mode == io::small_buff || mode == io::big_buff;
    if (!buff_mode) {
      const auto detect_start = std::chrono::steady_clock::now();
      auto armors = yolo.detect(img, frame_count);
      const auto detect_end = std::chrono::steady_clock::now();

      const auto track_start = std::chrono::steady_clock::now();
      auto targets = tracker.track(armors, timestamp, true);
      const auto track_end = std::chrono::steady_clock::now();
      //=== planner & aimer start ===
      const auto aim_start = std::chrono::steady_clock::now();
      std::optional<auto_aim::Target> target_for_plan =
        targets.empty() ? std::nullopt : std::make_optional(targets.front());

      auto plan = planner.plan(target_for_plan, gimbal.bullet_speed());
      const auto aim_end = std::chrono::steady_clock::now();

      uint8_t armor_id = 0;
      const auto gimbal_state = gimbal.state();
      double big_yaw = plan.yaw;
      double small_yaw = plan.yaw;
      if (plan.control && !targets.empty()) {
        armor_id = armor_name_to_nav_id(targets.front().name);
        small_yaw = plan.yaw;
        big_yaw = unlocked_outpost(targets.front())
                    ? gimbal_state.big_yaw
                    : target_big_yaw(targets.front(), gimbal_state.big_yaw);
      }

      command.control = plan.control;
      command.shoot = plan.fire;
      command.yaw = small_yaw;
      command.pitch = plan.pitch;
      command.big_yaw = big_yaw;
      command.small_yaw = small_yaw;
      command.has_target_yaw = plan.control;
      gimbal.send_mpc(
        plan.control, plan.fire, big_yaw, small_yaw, plan.pitch, plan.yaw_vel, plan.pitch_vel,
        plan.yaw_acc, plan.pitch_acc, armor_id);

      data["plan_target_yaw"] = plan.target_yaw;
      data["plan_target_pitch"] = plan.target_pitch;
      data["plan_yaw"] = rad2deg(plan.yaw);
      data["plan_pitch"] = rad2deg(plan.pitch);
      data["plan_yaw_vel"] = rad2deg(plan.yaw_vel);
      data["plan_pitch_vel"] = rad2deg(plan.pitch_vel);
      data["plan_yaw_acc"] = plan.yaw_acc;
      data["plan_pitch_acc"] = plan.pitch_acc;
      data["cmd_big_yaw"] = rad2deg(big_yaw);
      data["cmd_small_yaw"] = rad2deg(small_yaw);
      data["plan_fire"] = plan.fire ? 1 : 0;

      if (!armors.empty()) {
        auto & armor = armors.front();
        aim_solver.solve(armor);
        data["armor_x"] = armor.xyz_in_world[0];
        data["armor_y"] = armor.xyz_in_world[1];
        data["armor_z"] = armor.xyz_in_world[2] * 1000;
        data["armor_yaw"] = rad2deg(armor.ypr_in_world[0]);
        data["armor_pitch"] = rad2deg(armor.ypr_in_world[1]);
        data["armor_yaw_raw"] = rad2deg(armor.yaw_raw);
      }

      if (!targets.empty()) {
        auto target = targets.front();
        for (const Eigen::Vector4d & xyza : target.armor_xyza_list()) {
          auto image_points =
            aim_solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
          tools::draw_points(img, image_points, {0, 255, 0});
        };


        if (plan.control) {
          Eigen::Vector4d aim_xyza = planner.debug_xyza;
          auto image_points = aim_solver.reproject_armor(
            aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
          tools::draw_points(img, image_points, {0, 255, 255});
        }

        Eigen::VectorXd x = target.ekf_x();
        data["z"] = x[4];
        data["w"] = x[7];
        data["r"] = x[8];

        // tools::logger()->info("[OVGimbal Standard MPC Debug]rotation_vel={} , distance={}"
        //   , x[7], std::hypot(x[0], x[2]));
      }

      tools::draw_text(
        img, fmt::format("[{}]", tracker.state()), {10, 120}, {0, 255, 0});
    } else {
      const auto energy_type = mode == io::small_buff ? auto_buff::EnergyType::SMALL
                                                      : auto_buff::EnergyType::BIG;
      configure_buff(energy_type);
      auto & active_buff_detector =
        energy_type == auto_buff::EnergyType::SMALL ? small_buff_detector : big_buff_detector;

      const auto detect_start = std::chrono::steady_clock::now();
      std::optional<auto_buff::PowerRune> power_runes = active_buff_detector.detect(img);
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
      const auto aim_end = std::chrono::steady_clock::now();
      gimbal.send(command);
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
    }

    put_common_plot_data(data, gimbal, command, mode);
    plotter.plot(data);

    draw_common_overlay(img, gimbal, command, mode);
    cv::resize(img, img, {}, 0.5, 0.5);
    cv::imshow("OVGimbal Standard MPC Debug", img);
    auto key = cv::waitKey(1);
    if (key == 'q' || key == 'Q') break;
  }

  gimbal.send_mpc(false, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  return 0;
}
