#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>

#include "io/camera.hpp"
#include "io/ovgimbal.hpp"
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
struct CliOptions
{
  std::string config_path = "configs/standard4.yaml";
  std::string energy_type = "small";
  bool show_help = false;
};

std::string normalize_arg(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
}

void print_usage(const char * program)
{
  fmt::print(
    "Usage: {} [params]\n\n"
    "  -?, -h, --help, --usage\n"
    "      输出命令行参数说明\n"
    "  -c, --config-path <path>\n"
    "      yaml配置文件的路径，默认 configs/standard4.yaml\n"
    "  --energy-type, --etype <small|big>\n"
    "      手动指定能量机关类型，默认 small\n"
    "\n",
    program);
}

const char * energy_name(auto_buff::EnergyType energy_type)
{
  return energy_type == auto_buff::EnergyType::SMALL ? "small" : "big";
}

std::string board_index_string(int index)
{
  return index >= 0 ? fmt::format("#{}", index + 1) : "none";
}

bool matches_option(const std::string & arg, const std::string & long_opt, const std::string & short_opt)
{
  return arg == long_opt || arg == short_opt || arg.rfind(long_opt + "=", 0) == 0 ||
         arg.rfind(short_opt + "=", 0) == 0;
}

std::optional<std::string> extract_option_value(
  int & index, int argc, char * argv[], const std::string & arg, const std::string & long_opt,
  const std::string & short_opt, std::string & error)
{
  if (arg == long_opt || arg == short_opt) {
    if (index + 1 >= argc) {
      error = fmt::format("{} 缺少参数值", long_opt);
      return std::nullopt;
    }
    ++index;
    return std::string(argv[index]);
  }

  if (arg.rfind(long_opt + "=", 0) == 0) {
    auto value = arg.substr(long_opt.size() + 1);
    if (value.empty()) {
      error = fmt::format("{} 不能为空", long_opt);
      return std::nullopt;
    }
    return value;
  }

  if (arg.rfind(short_opt + "=", 0) == 0) {
    auto value = arg.substr(short_opt.size() + 1);
    if (value.empty()) {
      error = fmt::format("{} 不能为空", long_opt);
      return std::nullopt;
    }
    return value;
  }

  error = fmt::format("未能解析参数 {}", arg);
  return std::nullopt;
}

bool parse_args(int argc, char * argv[], CliOptions & options, std::string & error)
{
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help" || arg == "--usage" || arg == "-?") {
      options.show_help = true;
      continue;
    }

    if (matches_option(arg, "--config-path", "-c")) {
      auto value = extract_option_value(i, argc, argv, arg, "--config-path", "-c", error);
      if (!value.has_value()) return false;
      options.config_path = *value;
      continue;
    }

    if (matches_option(arg, "--energy-type", "--etype")) {
      auto value = extract_option_value(i, argc, argv, arg, "--energy-type", "--etype", error);
      if (!value.has_value()) return false;
      options.energy_type = normalize_arg(*value);
      continue;
    }

    if (!arg.empty() && arg[0] != '-') {
      options.config_path = arg;
      continue;
    }

    error = fmt::format("未知参数 {}", arg);
    return false;
  }

  return true;
}
}  // namespace

int main(int argc, char * argv[])
{
  CliOptions options;
  std::string parse_error;
  if (!parse_args(argc, argv, options, parse_error)) {
    tools::logger()->error(parse_error);
    print_usage(argv[0]);
    return 1;
  }

  if (options.show_help) {
    print_usage(argv[0]);
    return 0;
  }

  auto config_path = options.config_path;
  std::string energy_type_str = options.energy_type;

  if (config_path.empty()) {
    tools::logger()->error("config-path 不能为空");
    print_usage(argv[0]);
    return 1;
  }

  if (energy_type_str != "small" && energy_type_str != "big") {
    tools::logger()->error("非法 energy-type: {}，仅支持 small 或 big", energy_type_str);
    print_usage(argv[0]);
    return 1;
  }

  auto_buff::EnergyType active_energy_type =
    energy_type_str == "big" ? auto_buff::EnergyType::BIG : auto_buff::EnergyType::SMALL;

  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;

  io::OVGimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_buff::Buff_Detector detector(config_path);
  auto_buff::Solver solver(config_path);
  auto small_target = std::make_shared<auto_buff::SmallTarget>();
  auto big_target = std::make_shared<auto_buff::BigTarget>();
  auto_buff::Aimer aimer(config_path);
  auto_buff::DoubleBoardController double_board_controller;

  if (active_energy_type == auto_buff::EnergyType::SMALL) {
    detector.setTarget(small_target);
    detector.setEnergyType(auto_buff::EnergyType::SMALL);
    solver.setEnergyType(auto_buff::EnergyType::SMALL);
  } else {
    detector.setTarget(big_target);
    detector.setEnergyType(auto_buff::EnergyType::BIG);
    solver.setEnergyType(auto_buff::EnergyType::BIG);
  }

  cv::Mat img;
  std::chrono::steady_clock::time_point t;

  tools::logger()->info(
    "[OVGimbal Buff] aim=arm-only, energy={}", energy_name(active_energy_type));

  while (!exiter.exit()) {
    camera.read(img, t);
    auto q = gimbal.imu_at_image(t);
    solver.set_R_gimbal2world(q);
    //recorder.record(img, q, t);

    io::Command command{false, false, 0.0, 0.0};
    std::optional<auto_buff::PowerRune> power_runes;
    power_runes = detector.detect(img);

    if (active_energy_type == auto_buff::EnergyType::BIG) {
      if (power_runes.has_value()) {
        double_board_controller.updateSelection(power_runes.value(), t);
      }
    }

    const bool should_solve = power_runes.has_value() &&
                              (active_energy_type == auto_buff::EnergyType::SMALL ||
                               (power_runes->selectedBoardIndex() >= 0 &&
                                power_runes->isBoardLit(power_runes->selectedBoardIndex())));
    if (should_solve) {
      solver.solve(power_runes);
    }

    if (active_energy_type == auto_buff::EnergyType::SMALL) {
      small_target->get_target(power_runes, t, active_energy_type, nullptr);
      if (!small_target->is_unsolve()) {
        command = aimer.aim(*small_target, t, gimbal.bullet_speed(), true);
      }
    } else {
      std::optional<auto_buff::PowerRune> tracking_power_runes = std::nullopt;
      if (power_runes.has_value() && power_runes->selectedBoardIndex() >= 0 &&
          power_runes->isBoardLit(power_runes->selectedBoardIndex()))
      {
        tracking_power_runes = power_runes;
      }
      auto sin_param = solver.getSinusoidalParam();
      big_target->get_target(tracking_power_runes, t, active_energy_type, &sin_param);
      if (!big_target->is_unsolve()) {
        command = aimer.aim(*big_target, t, gimbal.bullet_speed(), true);
      }
    }

    gimbal.send(command);

    nlohmann::json data;
    auto gs = gimbal.state();
    data["gimbal_yaw"] = gs.yaw * 57.3;
    data["gimbal_pitch"] = gs.pitch * 57.3;
    data["bullet_speed"] = gs.bullet_speed;
    data["energy_type"] = energy_name(active_energy_type);
    if (power_runes.has_value()) {
      const auto & p = power_runes.value();
      const bool is_double_board_mode = active_energy_type == auto_buff::EnergyType::BIG;
      const bool has_selected_board =
        p.selectedBoardIndex() >= 0 && p.isBoardLit(p.selectedBoardIndex());
      if (!is_double_board_mode || has_selected_board) {
        data["buff_yaw"] = p.ypr_in_world[0] * 57.3;
        data["buff_roll"] = p.ypr_in_world[2] * 57.3;
      }
      data["hit_ring"] = p.hit_ring;
      data["activated_arms"] = p.activated_arms;
      data["selected_board"] = p.selectedBoardIndex();
      data["candidate_board"] = p.candidateBoardIndex();
      data["double_board_mode"] = is_double_board_mode ? 1 : 0;
    }
    if (command.control) {
      data["cmd_yaw"] = command.yaw * 57.3;
      data["cmd_pitch"] = command.pitch * 57.3;
      data["shoot"] = command.shoot ? 1 : 0;
    }
    plotter.plot(data);

    if (power_runes.has_value()) {
      const auto & p = power_runes.value();
      const bool is_double_board_mode = active_energy_type == auto_buff::EnergyType::BIG;
      tools::draw_text(
        img, fmt::format("Hit Ring: {}", p.hit_ring), {10, 30}, {0, 255, 255});
      tools::draw_text(
        img, fmt::format("Arms: {}/5", p.activated_arms), {10, 60}, {0, 255, 255});
      if (is_double_board_mode) {
        tools::draw_text(
          img, fmt::format("Selected: {}", board_index_string(p.selectedBoardIndex())), {10, 180},
          {0, 255, 0});
        tools::draw_text(
          img, fmt::format("Candidate: {}", board_index_string(p.candidateBoardIndex())),
          {10, 210}, {0, 215, 255});
      }
      if (is_double_board_mode && p.selectedBoardIndex() >= 0 && p.isBoardLit(p.selectedBoardIndex())) {
        cv::circle(img, p.target().center, 24, cv::Scalar(0, 255, 0), 3);
      }
    }
    tools::draw_text(
      img, fmt::format("Energy: {}", energy_name(active_energy_type)), {10, 120}, {180, 255, 180});
    if (command.control) {
      tools::draw_text(
        img,
        fmt::format(
          "Cmd: yaw={:.2f}, pitch={:.2f}, shoot={}", command.yaw * 57.3, command.pitch * 57.3,
          command.shoot),
        {10, 90}, {255, 200, 0});
    }

    cv::resize(img, img, {}, 0.5, 0.5);
    cv::imshow("OVGimbal Buff Test", img);
    auto key = cv::waitKey(1);
    if (key == 'q') break;
  }

  return 0;
}
