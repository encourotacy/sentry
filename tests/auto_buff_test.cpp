#include <fmt/core.h>

#include <chrono>
#include <cctype>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

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

namespace
{
struct CliOptions
{
  std::string input_path;
  std::string config_path = "configs/standard3.yaml";
  int start_index = 0;
  int end_index = 0;
  int skip_frames = 0;
  std::string energy_type = "big";
  bool show_help = false;
};

void print_usage(const char * program)
{
  fmt::print(
    "Usage: {} <input-path> [params]\n\n"
    "  <input-path>\n"
    "      不带后缀的录播路径，程序会读取 <input-path>.avi 和 <input-path>.txt\n"
    "  -?, -h, --help, --usage\n"
    "      输出命令行参数说明\n"
    "  -c, --config-path <path>\n"
    "      yaml配置文件路径，默认 configs/standard3.yaml\n"
    "  -s, --start-index <n>\n"
    "      视频起始帧下标，默认 0\n"
    "  -e, --end-index <n>\n"
    "      视频结束帧下标，默认 0，表示跑到结尾\n"
    "  -k, --skip-frames <n>\n"
    "      跳帧数，默认 0\n"
    "  --energy-type, --etype <small|big>\n"
    "      能量机关类型，默认 big\n",
    program);
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

bool parse_int_arg(const std::string & value, const std::string & name, int & output, std::string & error)
{
  try {
    output = std::stoi(value);
    return true;
  } catch (const std::exception &) {
    error = fmt::format("{} 需要整数，当前为 {}", name, value);
    return false;
  }
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

    if (matches_option(arg, "--start-index", "-s")) {
      auto value = extract_option_value(i, argc, argv, arg, "--start-index", "-s", error);
      if (!value.has_value()) return false;
      if (!parse_int_arg(*value, "--start-index", options.start_index, error)) return false;
      continue;
    }

    if (matches_option(arg, "--end-index", "-e")) {
      auto value = extract_option_value(i, argc, argv, arg, "--end-index", "-e", error);
      if (!value.has_value()) return false;
      if (!parse_int_arg(*value, "--end-index", options.end_index, error)) return false;
      continue;
    }

    if (matches_option(arg, "--skip-frames", "-k")) {
      auto value = extract_option_value(i, argc, argv, arg, "--skip-frames", "-k", error);
      if (!value.has_value()) return false;
      if (!parse_int_arg(*value, "--skip-frames", options.skip_frames, error)) return false;
      continue;
    }

    if (matches_option(arg, "--energy-type", "--etype")) {
      auto value = extract_option_value(i, argc, argv, arg, "--energy-type", "--etype", error);
      if (!value.has_value()) return false;
      options.energy_type = *value;
      continue;
    }

    if (!arg.empty() && arg[0] != '-') {
      if (options.input_path.empty()) {
        options.input_path = arg;
      } else {
        error = fmt::format("多余的位置参数 {}", arg);
        return false;
      }
      continue;
    }

    error = fmt::format("未知参数 {}", arg);
    return false;
  }

  return true;
}

std::string normalize_arg(std::string value)
{
  for (char & ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return value;
}

std::string energy_type_string(auto_buff::EnergyType energy_type)
{
  return energy_type == auto_buff::EnergyType::SMALL ? "SMALL (pi/3)" : "BIG (sin)";
}

std::string board_index_string(int index)
{
  return index >= 0 ? fmt::format("#{}", index + 1) : "none";
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

  if (options.input_path.empty()) {
    tools::logger()->error("缺少 input-path");
    print_usage(argv[0]);
    return 1;
  }

  if (options.skip_frames < 0) {
    tools::logger()->error("非法 skip-frames: {}，必须大于等于 0", options.skip_frames);
    return 1;
  }

  options.energy_type = normalize_arg(options.energy_type);

  auto_buff::EnergyType fixed_energy_type = options.energy_type == "small" ?
                                              auto_buff::EnergyType::SMALL :
                                              auto_buff::EnergyType::BIG;
  if (options.energy_type != "small" && options.energy_type != "big") {
    tools::logger()->error("非法 energy-type: {}，仅支持 small 或 big", options.energy_type);
    return 1;
  }

  tools::logger()->info(">>> 使用 {} 模式", fixed_energy_type == auto_buff::EnergyType::SMALL ? "小符" : "大符");
  tools::logger()->info(">>> 瞄准模式: ARM");

  const auto video_path = fmt::format("{}.avi", options.input_path);
  const auto text_path = fmt::format("{}.txt", options.input_path);
  if (!std::filesystem::exists(video_path)) {
    tools::logger()->error("无法找到视频文件: {}", video_path);
    return 1;
  }
  if (!std::filesystem::exists(text_path)) {
    tools::logger()->error("无法找到姿态文件: {}", text_path);
    return 1;
  }

  cv::VideoCapture video(video_path);
  if (!video.isOpened()) {
    tools::logger()->error("无法打开视频文件: {}", video_path);
    return 1;
  }

  std::ifstream text(text_path);
  if (!text.is_open()) {
    tools::logger()->error("无法打开文本文件: {}", text_path);
    return 1;
  }

  tools::Plotter plotter;
  tools::Exiter exiter;

  auto_buff::Buff_Detector detector(options.config_path);
  auto_buff::Solver solver(options.config_path);
  auto_buff::BigTarget big_target;
  auto_buff::SmallTarget small_target;
  auto_buff::Aimer aimer(options.config_path);
  auto_buff::DoubleBoardController double_board_controller;

  detector.setEnergyType(fixed_energy_type);
  solver.setEnergyType(fixed_energy_type);

  if (fixed_energy_type == auto_buff::EnergyType::SMALL) {
    detector.setTarget(std::shared_ptr<auto_buff::Target>(&small_target, [](auto *) {}));
  } else {
    detector.setTarget(std::shared_ptr<auto_buff::Target>(&big_target, [](auto *) {}));
  }

  const double video_fps = video.get(cv::CAP_PROP_FPS);
  const int total_frames = static_cast<int>(video.get(cv::CAP_PROP_FRAME_COUNT));
  tools::logger()->info(
    "视频信息: {:.2f} FPS, 总帧数: {}, 起始帧: {}, 结束帧: {}, 跳帧: {}",
    video_fps, total_frames, options.start_index, options.end_index, options.skip_frames);

  if (fixed_energy_type == auto_buff::EnergyType::BIG) {
    auto param = solver.getSinusoidalParam();
    tools::logger()->info(">>> 大符正弦参数: a={:.3f}, w={:.3f}, b={:.3f}", param.a, param.omega, param.b);
  }

  if (options.start_index > 0) {
    video.set(cv::CAP_PROP_POS_FRAMES, options.start_index);
  }
  for (int i = 0; i < options.start_index; ++i) {
    double t, w, x, y, z;
    text >> t >> w >> x >> y >> z;
  }

  cv::namedWindow("Auto Buff Test", cv::WINDOW_NORMAL);
  cv::resizeWindow("Auto Buff Test", 960, 540);
  cv::namedWindow("Debug Detection", cv::WINDOW_NORMAL);
  cv::resizeWindow("Debug Detection", 960, 540);

  cv::Mat img;
  const auto t0 = std::chrono::steady_clock::now();

  int processed_frames = 0;
  int frame_counter = 0;
  int consecutive_lost_frames = 0;
  constexpr int MAX_CONSECUTIVE_LOST = 15;
  int total_frames_processed = 0;
  int detected_frames = 0;
  int lost_frames = 0;
  double total_process_time_ms = 0.0;
  std::deque<double> recent_detection_rate;

  for (int frame_count = options.start_index; !exiter.exit(); ++frame_count) {
    if (options.end_index > 0 && frame_count > options.end_index) break;

    if (options.skip_frames > 0 && frame_counter++ % (options.skip_frames + 1) != 0) {
      if (!video.grab()) break;
      double t, w, x, y, z;
      text >> t >> w >> x >> y >> z;
      continue;
    }

    const auto frame_start = std::chrono::steady_clock::now();

    video.read(img);
    if (img.empty()) {
      tools::logger()->info("视频播放结束");
      break;
    }
    cv::Mat raw_img = img.clone();

    double t, w, x, y, z;
    text >> t >> w >> x >> y >> z;
    auto timestamp = t0 + std::chrono::microseconds(static_cast<int>(t * 1e6));

    solver.set_R_gimbal2world({w, x, y, z});

    auto power_runes = detector.detect(img);
    if (fixed_energy_type == auto_buff::EnergyType::BIG) {
      if (power_runes.has_value()) {
        double_board_controller.updateSelection(power_runes.value(), timestamp);
      }
    }

    total_frames_processed++;
    processed_frames++;
    if (power_runes.has_value()) {
      detected_frames++;
    } else {
      lost_frames++;
    }
    recent_detection_rate.push_back(power_runes.has_value() ? 1.0 : 0.0);
    if (recent_detection_rate.size() > 100) {
      recent_detection_rate.pop_front();
    }
    const double recent_rate =
      recent_detection_rate.empty() ? 0.0 :
                                      std::accumulate(
                                        recent_detection_rate.begin(), recent_detection_rate.end(), 0.0) /
                                        recent_detection_rate.size();

    std::vector<std::pair<std::string, cv::Scalar>> overlay_lines;
    overlay_lines.emplace_back(
      fmt::format("Frame: {}/{}", frame_count, total_frames), cv::Scalar(255, 255, 255));
    overlay_lines.emplace_back(
      fmt::format("Energy: {}", energy_type_string(fixed_energy_type)), cv::Scalar(255, 200, 0));
    overlay_lines.emplace_back("Aim: ARM", cv::Scalar(255, 200, 0));
    overlay_lines.emplace_back(
      fmt::format(
        "Detect: {:.1f}% total / {:.1f}% last100",
        total_frames_processed > 0 ? detected_frames * 100.0 / total_frames_processed : 0.0,
        recent_rate * 100.0),
      cv::Scalar(0, 255, 255));
    overlay_lines.emplace_back(
      fmt::format("Detected: {} Lost: {}", detected_frames, lost_frames), cv::Scalar(0, 255, 255));

    io::Command command{false, false, 0.0, 0.0};
    nlohmann::json data;
    data["energy_type"] = fixed_energy_type == auto_buff::EnergyType::SMALL ? 0 : 1;
    std::optional<auto_buff::PowerRune> tracking_power_rune = std::nullopt;

    if (power_runes.has_value()) {
      consecutive_lost_frames = 0;
      auto & p = power_runes.value();

      const bool is_double_board_mode = fixed_energy_type == auto_buff::EnergyType::BIG;
      const bool has_selected_board =
        p.selectedBoardIndex() >= 0 && p.isBoardLit(p.selectedBoardIndex());
      const bool has_candidate_board =
        p.candidateBoardIndex() >= 0 && p.isBoardLit(p.candidateBoardIndex());
      int best_target_id = p.selectedBoardIndex();
      if (is_double_board_mode) {
        overlay_lines.emplace_back("Double-Board Mode", cv::Scalar(0, 200, 255));
        overlay_lines.emplace_back(
          fmt::format("Selected Board: {}", board_index_string(best_target_id)), cv::Scalar(0, 255, 0));
        overlay_lines.emplace_back(
          fmt::format("Candidate Board: {}", board_index_string(p.candidateBoardIndex())),
          cv::Scalar(0, 215, 255));
        if (!has_selected_board) {
          overlay_lines.emplace_back("Locked board temporarily missing", cv::Scalar(0, 165, 255));
        }
      } else if (p.fanblades.size() > 1) {
        overlay_lines.emplace_back(
          "Multiple detections outside double-board mode", cv::Scalar(0, 165, 255));
        overlay_lines.emplace_back(
          fmt::format("Use default board: #{}", best_target_id), cv::Scalar(0, 255, 255));
      } else {
        overlay_lines.emplace_back("Single Target Mode", cv::Scalar(255, 255, 0));
      }

      overlay_lines.emplace_back("BUFF DETECTED", cv::Scalar(0, 255, 0));
      overlay_lines.emplace_back(fmt::format("Hit Ring: {}", p.hit_ring), cv::Scalar(0, 255, 255));
      overlay_lines.emplace_back(
        fmt::format("Arms: {}/5", p.activated_arms), cv::Scalar(0, 255, 255));

      int display_idx = 0;
      if (is_double_board_mode) {
        if (has_selected_board) {
          display_idx = p.selectedBoardIndex();
        } else if (has_candidate_board) {
          display_idx = p.candidateBoardIndex();
        } else {
          display_idx = -1;
        }
      } else if (p.fanblades.size() > 1 && best_target_id < static_cast<int>(p.fanblades.size())) {
        display_idx = best_target_id;
      }
      if (!p.fanblades.empty() && display_idx >= 0 &&
          display_idx < static_cast<int>(p.fanblades.size()))
      {
        const auto & blade = p.fanblades[display_idx];
        for (int i = 0; i < 4 && i < static_cast<int>(blade.points.size()); ++i) {
          cv::circle(img, blade.points[i], 4, cv::Scalar(0, 255, 255), 2);
        }
        for (int i = 0; i < static_cast<int>(blade.points.size()) && i < 4; ++i) {
          cv::line(
            img, blade.points[i], blade.points[(i + 1) % blade.points.size()],
            cv::Scalar(255, 255, 0), 1);
        }
      }
      cv::circle(img, p.r_center, 6, cv::Scalar(255, 0, 0), 2);

      data["hit_ring"] = p.hit_ring;
      data["activated_arms"] = p.activated_arms;

      if (fixed_energy_type == auto_buff::EnergyType::SMALL) {
        solver.solve(power_runes);
        data["buff_R_yaw"] = p.ypd_in_world[0];
        data["buff_R_pitch"] = p.ypd_in_world[1];
        data["buff_R_dis"] = p.ypd_in_world[2];
        data["buff_yaw"] = p.ypr_in_world[0] * 57.3;
        data["buff_pitch"] = p.ypr_in_world[1] * 57.3;
        data["buff_roll"] = p.ypr_in_world[2] * 57.3;
      } else {
        if (has_selected_board) {
          solver.solve(power_runes);
          data["buff_R_yaw"] = p.ypd_in_world[0];
          data["buff_R_pitch"] = p.ypd_in_world[1];
          data["buff_R_dis"] = p.ypd_in_world[2];
          data["buff_yaw"] = p.ypr_in_world[0] * 57.3;
          data["buff_pitch"] = p.ypr_in_world[1] * 57.3;
          data["buff_roll"] = p.ypr_in_world[2] * 57.3;
          tracking_power_rune = power_runes;
        }
      }

      if (fixed_energy_type == auto_buff::EnergyType::BIG && has_selected_board) {
        cv::circle(img, p.target().center, 24, cv::Scalar(0, 255, 0), 3);
      }
    } else {
      consecutive_lost_frames++;
      overlay_lines.emplace_back("NO BUFF DETECTED", cv::Scalar(0, 0, 255));
      overlay_lines.emplace_back(
        fmt::format("Lost frames: {}", consecutive_lost_frames), cv::Scalar(0, 0, 255));
      if (consecutive_lost_frames > MAX_CONSECUTIVE_LOST) {
        overlay_lines.emplace_back("TARGET RESET", cv::Scalar(255, 0, 0));
      }
    }

    if (fixed_energy_type == auto_buff::EnergyType::SMALL) {
      small_target.get_target(power_runes, timestamp, fixed_energy_type, nullptr);
      if (!small_target.is_unsolve()) {
        command = aimer.aim(small_target, timestamp, 22.0, false);

        Eigen::VectorXd x = small_target.ekf_x();
        if (x.size() > 0) data["R_yaw"] = x[0];
        if (x.size() > 1) data["R_V_yaw"] = x[1];
        if (x.size() > 2) data["R_pitch"] = x[2];
        if (x.size() > 3) data["R_dis"] = x[3];
        if (x.size() > 4) data["yaw"] = x[4] * 57.3;
        if (x.size() > 5) {
          data["angle"] = x[5] * 57.3;
          overlay_lines.emplace_back(
            fmt::format("Angle: {:.1f} deg", x[5] * 57.3), cv::Scalar(0, 255, 255));
        }
        overlay_lines.emplace_back(
          fmt::format("Speed: {:.1f} deg/s", small_target.spd * 57.3), cv::Scalar(0, 255, 255));
      } else if (power_runes.has_value()) {
        overlay_lines.emplace_back("TARGET UNSOLVABLE", cv::Scalar(0, 165, 255));
      }
    } else {
      auto sin_param = solver.getSinusoidalParam();
      big_target.get_target(tracking_power_rune, timestamp, fixed_energy_type, &sin_param);
      if (!big_target.is_unsolve()) {
        command = aimer.aim(big_target, timestamp, 22.0, false);

        Eigen::VectorXd x = big_target.ekf_x();
        if (x.size() > 0) data["R_yaw"] = x[0];
        if (x.size() > 1) data["R_V_yaw"] = x[1];
        if (x.size() > 2) data["R_pitch"] = x[2];
        if (x.size() > 3) data["R_dis"] = x[3];
        if (x.size() > 4) data["yaw"] = x[4] * 57.3;
        if (x.size() > 5) {
          data["angle"] = x[5] * 57.3;
          overlay_lines.emplace_back(
            fmt::format("Angle: {:.1f} deg", x[5] * 57.3), cv::Scalar(0, 255, 255));
        }
        if (x.size() > 6) {
          data["spd"] = x[6];
          overlay_lines.emplace_back(
            fmt::format("Speed: {:.1f} deg/s", x[6] * 57.3), cv::Scalar(0, 255, 255));
        }
        if (x.size() >= 10) {
          data["a"] = x[7];
          data["w"] = x[8];
          data["fi"] = x[9];
          data["spd0"] = big_target.spd;
          overlay_lines.emplace_back(
            fmt::format("a={:.3f} w={:.3f}", x[7], x[8]), cv::Scalar(200, 200, 255));
        }
      } else if (power_runes.has_value()) {
        overlay_lines.emplace_back("TARGET UNSOLVABLE", cv::Scalar(0, 165, 255));
      }
    }

    const Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
    data["gimbal_yaw"] = ypr[0] * 57.3;
    data["gimbal_pitch"] = -ypr[1] * 57.3;
    if (command.control) {
      data["cmd_yaw"] = command.yaw * 57.3;
      data["cmd_pitch"] = command.pitch * 57.3;
      overlay_lines.emplace_back(
        fmt::format("Cmd: yaw={:.1f} deg, pitch={:.1f} deg", command.yaw * 57.3, command.pitch * 57.3),
        cv::Scalar(255, 200, 0));
    }
    plotter.plot(data);

    const auto frame_end = std::chrono::steady_clock::now();
    const double frame_time_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
    total_process_time_ms += frame_time_ms;
    const double avg_process_time = total_process_time_ms / std::max(processed_frames, 1);
    const double current_fps = frame_time_ms > 1e-6 ? 1000.0 / frame_time_ms : 0.0;

    cv::Mat debug_img = raw_img.clone();
    std::vector<std::pair<std::string, cv::Scalar>> debug_lines;
    debug_lines.emplace_back("Debug Detection", cv::Scalar(255, 255, 255));
    if (power_runes.has_value()) {
      const auto & p = power_runes.value();
      const bool has_selected_board =
        p.selectedBoardIndex() >= 0 && p.isBoardLit(p.selectedBoardIndex());
      const bool has_candidate_board =
        p.candidateBoardIndex() >= 0 && p.isBoardLit(p.candidateBoardIndex());
      debug_lines.emplace_back(
        fmt::format("Detected boards: {}", p.getLightArmCount()), cv::Scalar(0, 255, 255));
      debug_lines.emplace_back(
        fmt::format("Selected: {}", board_index_string(p.selectedBoardIndex())), cv::Scalar(0, 255, 0));
      debug_lines.emplace_back(
        fmt::format("Candidate: {}", board_index_string(p.candidateBoardIndex())), cv::Scalar(0, 215, 255));

      std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 255, 0)};
      for (size_t i = 0; i < p.fanblades.size(); ++i) {
        const cv::Scalar color = colors[i % colors.size()];
        const bool is_selected = has_selected_board && static_cast<int>(i) == p.selectedBoardIndex();
        const bool is_candidate = has_candidate_board && static_cast<int>(i) == p.candidateBoardIndex();
        for (const auto & pt : p.fanblades[i].points) {
          cv::circle(debug_img, pt, 5, color, 2);
        }
        cv::circle(debug_img, p.fanblades[i].center, 8, color, 3);

        std::string board_tag = fmt::format("#{}", i);
        if (is_selected) {
          board_tag += " SELECTED";
        } else if (is_candidate) {
          board_tag += " CANDIDATE";
        }
        cv::Scalar label_color = color;
        if (is_selected) {
          label_color = cv::Scalar(0, 255, 0);
        } else if (is_candidate) {
          label_color = cv::Scalar(0, 215, 255);
        }
        cv::putText(
          debug_img, board_tag,
          cv::Point(static_cast<int>(p.fanblades[i].center.x - 18), static_cast<int>(p.fanblades[i].center.y - 14)),
          cv::FONT_HERSHEY_SIMPLEX, 0.65, label_color, 2, cv::LINE_AA);

        if (p.fanblades[i].points.size() >= 4) {
          cv::rectangle(debug_img, cv::boundingRect(p.fanblades[i].points), color, 2);
        }
        if (is_selected) {
          cv::circle(debug_img, p.fanblades[i].center, 24, cv::Scalar(0, 255, 0), 3);
        } else if (is_candidate) {
          cv::circle(debug_img, p.fanblades[i].center, 20, cv::Scalar(0, 215, 255), 2);
        }
      }
      cv::circle(debug_img, p.r_center, 10, cv::Scalar(0, 0, 255), 3);
    } else {
      debug_lines.emplace_back("Detected boards: 0", cv::Scalar(0, 0, 255));
      debug_lines.emplace_back("Selected: none", cv::Scalar(0, 0, 255));
      debug_lines.emplace_back("Candidate: none", cv::Scalar(0, 215, 255));
    }

    cv::Mat display_img;
    cv::resize(img, display_img, cv::Size(960, 540));

    int line_y = 28;
    for (const auto & line : overlay_lines) {
      cv::putText(
        display_img, line.first, cv::Point(12, line_y), cv::FONT_HERSHEY_SIMPLEX, 0.62,
        line.second, 2, cv::LINE_AA);
      line_y += 22;
      if (line_y > 300) break;
    }
    cv::putText(
      display_img, fmt::format("Proc: {:.1f}ms (avg: {:.1f}ms)", frame_time_ms, avg_process_time),
      cv::Point(708, 28), cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(255, 255, 255), 2,
      cv::LINE_AA);
    cv::putText(
      display_img, fmt::format("FPS: {:.1f}", current_fps), cv::Point(708, 54),
      cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::putText(
      display_img, "Controls: [Q]uit, [Space]Pause/Resume", cv::Point(12, 531),
      cv::FONT_HERSHEY_SIMPLEX, 0.56, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);

    cv::Mat debug_display;
    cv::resize(debug_img, debug_display, cv::Size(960, 540));
    int debug_line_y = 28;
    for (const auto & line : debug_lines) {
      cv::putText(
        debug_display, line.first, cv::Point(12, debug_line_y), cv::FONT_HERSHEY_SIMPLEX,
        0.62, line.second, 2, cv::LINE_AA);
      debug_line_y += 24;
    }
    cv::putText(
      debug_display, "Colored boxes = all detections", cv::Point(12, 530),
      cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);

    cv::imshow("Auto Buff Test", display_img);
    cv::imshow("Debug Detection", debug_display);

    int key_delay_ms = 1;
    if (video_fps > 1.0) {
      const double target_frame_time_ms = 1000.0 / video_fps;
      const double remaining_time_ms = target_frame_time_ms - frame_time_ms;
      if (remaining_time_ms > 1.0) {
        key_delay_ms = static_cast<int>(remaining_time_ms);
      }
    }

    int key = cv::waitKey(key_delay_ms);
    if (key == 'q' || key == 'Q') break;
    while (key == ' ') {
      int pause_key = cv::waitKey(30);
      if (pause_key == 'q' || pause_key == 'Q') {
        key = pause_key;
        break;
      }
      if (pause_key == ' ') break;
    }
    if (key == 'q' || key == 'Q') break;
  }

  cv::destroyAllWindows();
  text.close();
  return 0;
}
