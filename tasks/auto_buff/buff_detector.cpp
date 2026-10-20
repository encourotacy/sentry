#include "tasks/auto_buff/buff_detector.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>

#include "tasks/auto_buff/buff_target.hpp"
#include "tools/logger.hpp"

namespace auto_buff
{
const char * Buff_Detector::colorName(BuffColor color)
{
  return color == BuffColor::RED ? "red" : "blue";
}

BuffColor Buff_Detector::oppositeColor(BuffColor color)
{
  return color == BuffColor::RED ? BuffColor::BLUE : BuffColor::RED;
}

BuffColor Buff_Detector::classifyColor(const cv::Mat & bgr_img, const YOLO11_BUFF::Object & result)
{
  if (bgr_img.empty()) {
    return BuffColor::BLUE;
  }

  std::vector<cv::Point> polygon;
  polygon.reserve(result.kpt.size());
  for (const auto & point : result.kpt) {
    const int x = std::clamp(static_cast<int>(std::lround(point.x)), 0, bgr_img.cols - 1);
    const int y = std::clamp(static_cast<int>(std::lround(point.y)), 0, bgr_img.rows - 1);
    polygon.emplace_back(x, y);
  }

  const cv::Rect image_rect(0, 0, bgr_img.cols, bgr_img.rows);
  const cv::Rect result_rect(
    static_cast<int>(std::lround(result.rect.x)),
    static_cast<int>(std::lround(result.rect.y)),
    static_cast<int>(std::lround(result.rect.width)),
    static_cast<int>(std::lround(result.rect.height)));
  cv::Rect roi = polygon.empty() ? (result_rect & image_rect) : (cv::boundingRect(polygon) & image_rect);
  if (roi.width <= 0 || roi.height <= 0) {
    roi = result_rect & image_rect;
  }
  if (roi.width <= 0 || roi.height <= 0) {
    return BuffColor::BLUE;
  }

  cv::Mat mask = cv::Mat::zeros(roi.size(), CV_8UC1);
  if (!polygon.empty()) {
    std::vector<cv::Point> shifted_polygon;
    shifted_polygon.reserve(polygon.size());
    for (const auto & point : polygon) {
      shifted_polygon.emplace_back(point.x - roi.x, point.y - roi.y);
    }
    cv::fillConvexPoly(mask, shifted_polygon, cv::Scalar(255));
  } else {
    mask.setTo(cv::Scalar(255));
  }

  long long red_sum = 0;
  long long blue_sum = 0;
  int valid_pixels = 0;
  const cv::Mat roi_img = bgr_img(roi);
  for (int y = 0; y < roi_img.rows; ++y) {
    for (int x = 0; x < roi_img.cols; ++x) {
      if (mask.at<uint8_t>(y, x) == 0) continue;
      const auto & pixel = roi_img.at<cv::Vec3b>(y, x);
      if (std::max(pixel[0], pixel[2]) < 80) continue;
      blue_sum += pixel[0];
      red_sum += pixel[2];
      ++valid_pixels;
    }
  }

  if (valid_pixels == 0) {
    return BuffColor::BLUE;
  }
  return blue_sum > red_sum ? BuffColor::BLUE : BuffColor::RED;
}

std::vector<YOLO11_BUFF::Object> Buff_Detector::filterByConfiguredColor(
  const cv::Mat & bgr_img, const std::vector<YOLO11_BUFF::Object> & results) const
{
  if (!energy_color_configured_ || results.empty()) {
    return results;
  }

  std::vector<YOLO11_BUFF::Object> filtered;
  filtered.reserve(results.size());
  for (const auto & result : results) {
    if (classifyColor(bgr_img, result) == target_energy_color_) {
      filtered.push_back(result);
    }
  }

  if (filtered.empty()) {
    tools::logger()->debug(
      "[Buff_Detector] 未筛到目标颜色 {}，回退使用全部候选 {} 个",
      colorName(target_energy_color_), results.size());
    return results;
    // return filtered;
  }
  return filtered;
}

RingDetector::RingDetector() : max_radius_mm_(150.0), outer_ring_(1), inner_ring_(10), focal_length_(800.0) {}

void RingDetector::setParams(double max_radius_mm, int outer_ring, int inner_ring)
{
  max_radius_mm_ = max_radius_mm;
  outer_ring_ = outer_ring;
  inner_ring_ = inner_ring;
}

void RingDetector::setFocalLength(double focal_length)
{
  focal_length_ = focal_length;
}

double RingDetector::pixelToMm(double pixel_dist, double distance_m) const
{
  return pixel_dist * distance_m * 1000.0 / focal_length_;
}

int RingDetector::detectRing(
  const cv::Point2f & hit_point, const cv::Point2f & center, double distance_to_target) const
{
  double pixel_dist = cv::norm(hit_point - center);
  double radius_mm = pixelToMm(pixel_dist, distance_to_target);
  radius_mm = std::clamp(radius_mm, 0.0, max_radius_mm_);

  double t = radius_mm / max_radius_mm_;
  double ring_float;

  if (inner_ring_ > outer_ring_) {
    ring_float = inner_ring_ - t * (inner_ring_ - outer_ring_);
  } else {
    ring_float = outer_ring_ + t * (inner_ring_ - outer_ring_);
  }

  int ring = static_cast<int>(ring_float + 0.5);
  return std::clamp(ring, 1, 10);
}

ArmStateDetector::ArmStateDetector() : brightness_threshold_(100), use_color_detection_(false) {}

void ArmStateDetector::setParams(int brightness_threshold, bool use_color_detection)
{
  brightness_threshold_ = brightness_threshold;
  use_color_detection_ = use_color_detection;
}

int ArmStateDetector::detectActivatedArms(const std::vector<FanBlade> & fanblades) const
{
  int count = 0;
  for (const auto & blade : fanblades) {
    if (blade.type == _light) {
      count++;
    }
  }
  return count;
}

int ArmStateDetector::getCurrentTargetArmIndex(const std::vector<FanBlade> & fanblades) const
{
  for (size_t i = 0; i < fanblades.size(); i++) {
    if (fanblades[i].type == _light || fanblades[i].type == _target) {
      return static_cast<int>(i);
    }
  }
  return 0;
}

Buff_Detector::Buff_Detector(const std::string & config) : Buff_Detector(config, "model") {}

Buff_Detector::Buff_Detector(const std::string & config, const std::string & model_key)
  : status_(TrackStatus::LOSE),
    lose_(0),
    MODE_(config, model_key),
    energy_type_(EnergyType::SMALL),
    fx_(800.0),
    fy_(800.0),
    cx_(320.0),
    cy_(240.0),
    max_radius_mm_(150.0),
    outer_ring_(1),
    inner_ring_(10),
    lastlen_(0.0),
    configured_energy_color_(BuffColor::RED),
    target_energy_color_(BuffColor::BLUE),
    energy_color_configured_(false),
    target_(nullptr),
    prediction_valid_(false)
{
  try {
    auto yaml = YAML::LoadFile(config);
    std::string configured_color;
    if (yaml["enemy_color"]) {
      configured_color = yaml["enemy_color"].as<std::string>();
      energy_color_configured_ = true;
    } else if (yaml["energy_color"]) {
      configured_color = yaml["energy_color"].as<std::string>();
      energy_color_configured_ = true;
      tools::logger()->warn(
        "[Buff_Detector] energy_color 已过时，建议改用 enemy_color，buff 会自动使用反色");
    }

    if (energy_color_configured_) {
      std::transform(configured_color.begin(), configured_color.end(), configured_color.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
      });
      configured_energy_color_ = configured_color == "blue" ? BuffColor::BLUE : BuffColor::RED;
      target_energy_color_ = oppositeColor(configured_energy_color_);
      tools::logger()->info(
        "[Buff_Detector] enemy_color={}, buff_target_color={}",
        colorName(configured_energy_color_), colorName(target_energy_color_));
    }

    if (yaml["ring_mapping"]) {
      max_radius_mm_ = yaml["ring_mapping"]["max_radius_mm"].as<double>(150.0);
      outer_ring_ = yaml["ring_mapping"]["outer_ring"].as<int>(1);
      inner_ring_ = yaml["ring_mapping"]["inner_ring"].as<int>(10);
    }

    if (yaml["camera_matrix"]) {
      auto camera_data = yaml["camera_matrix"].as<std::vector<double>>();
      if (camera_data.size() >= 4) {
        fx_ = camera_data[0];
        fy_ = camera_data[4];
        cx_ = camera_data[2];
        cy_ = camera_data[5];
      }
    }

    if (yaml["arm_detection"]) {
      int brightness = yaml["arm_detection"]["brightness_threshold"].as<int>(100);
      bool use_color = yaml["arm_detection"]["use_color_detection"].as<bool>(false);
      arm_detector_.setParams(brightness, use_color);
    }
  } catch (const std::exception & e) {
    tools::logger()->warn("[Buff_Detector] 加载配置文件失败: {}", e.what());
  }

  ring_detector_.setParams(max_radius_mm_, outer_ring_, inner_ring_);
  ring_detector_.setFocalLength((fx_ + fy_) / 2.0);
}

void Buff_Detector::setCameraParams(double fx, double fy, double cx, double cy)
{
  fx_ = fx;
  fy_ = fy;
  cx_ = cx;
  cy_ = cy;
  ring_detector_.setFocalLength((fx + fy) / 2.0);
}

void Buff_Detector::setEnergyType(EnergyType type)
{
  energy_type_ = type;
}

void Buff_Detector::handle_img(const cv::Mat & bgr_img, cv::Mat & dilated_img)
{
  cv::Mat gray_img;
  cv::cvtColor(bgr_img, gray_img, cv::COLOR_BGR2GRAY);
  cv::Mat binary_img;
  cv::threshold(gray_img, binary_img, 100, 255, cv::THRESH_BINARY);
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
  cv::dilate(binary_img, dilated_img, kernel, cv::Point(-1, -1), 1);
}

cv::Point2f Buff_Detector::get_r_center(std::vector<FanBlade> & fanblades, cv::Mat & bgr_img)
{
  if (fanblades.empty()) {
    tools::logger()->debug("[Buff_Detector] 无法计算r_center!");
    return {0, 0};
  }

  cv::Point2f r_center_t = {0, 0};
  for (auto & fanblade : fanblades) {
    auto point5 = fanblade.points[4];
    auto point6 = fanblade.points[5];
    r_center_t += (point6 - point5) * 1.4 + point5;
  }
  r_center_t /= float(fanblades.size());

  cv::Mat dilated_img;
  handle_img(bgr_img, dilated_img);

  double radius = cv::norm(fanblades[0].points[2] - fanblades[0].center) * 0.8;
  cv::Mat mask = cv::Mat::zeros(dilated_img.size(), CV_8U);
  cv::circle(mask, r_center_t, radius, cv::Scalar(255), -1);
  cv::bitwise_and(dilated_img, mask, dilated_img);

  std::vector<std::vector<cv::Point>> contours;
  auto r_center = r_center_t;
  cv::findContours(dilated_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  double best_score = INFINITY;
  for (auto & contour : contours) {
    auto rotated_rect = cv::minAreaRect(contour);
    double ratio = rotated_rect.size.height > rotated_rect.size.width ?
      rotated_rect.size.height / rotated_rect.size.width :
      rotated_rect.size.width / rotated_rect.size.height;

    double score = ratio + cv::norm(rotated_rect.center - r_center_t) / (radius / 3);

    if (score < best_score) {
      best_score = score;
      r_center = rotated_rect.center;
    }
  }

  return r_center;
}

int Buff_Detector::detectHitRing(const PowerRune & powerrune, double distance_to_target) const
{
  if (powerrune.fanblades.empty()) return 0;
  const auto & blade = powerrune.target();
  return ring_detector_.detectRing(blade.center, powerrune.r_center, distance_to_target);
}

int Buff_Detector::detectActivatedArms(const std::vector<FanBlade> & fanblades) const
{
  return arm_detector_.detectActivatedArms(fanblades);
}

PowerRune Buff_Detector::constructPredictedPowerRune()
{
  PowerRune predicted;
  if (!last_powerrune_.has_value()) {
    prediction_valid_ = false;
    return predicted;
  }
  predicted = last_powerrune_.value();
  if (target_ != nullptr && !target_->is_unsolve()) {
    Eigen::Vector3d predicted_xyz = target_->point_buff2world(Eigen::Vector3d(0, 0, 0));
    predicted.xyz_in_world = predicted_xyz;
    predicted.ypd_in_world = tools::xyz2ypd(predicted_xyz);
    Eigen::Vector3d predicted_blade_xyz = target_->point_buff2world(Eigen::Vector3d(0, 0, 0.7));
    predicted.blade_xyz_in_world = predicted_blade_xyz;
    predicted.blade_ypd_in_world = tools::xyz2ypd(predicted_blade_xyz);
    Eigen::VectorXd x = target_->ekf_x();
    if (x.size() >= 6) {
      predicted.ypr_in_world = Eigen::Vector3d(x[4], 0.0, x[5]);
    }
    prediction_valid_ = true;
  } else {
    prediction_valid_ = false;
  }
  predicted.timestamp = cv::getTickCount() / cv::getTickFrequency();
  return predicted;
}

std::optional<PowerRune> Buff_Detector::getPredictedPowerRune() const
{
  if (prediction_valid_) return std::optional<PowerRune>(predicted_powerrune_);
  return std::nullopt;
}

void Buff_Detector::handle_lose()
{
  lose_++;
  if (lose_ >= LOSE_MAX) {
    status_ = TrackStatus::LOSE;
    last_powerrune_ = std::nullopt;
    prediction_valid_ = false;
    if (lose_ == LOSE_MAX) tools::logger()->debug("[Detector] 目标彻底丢失(丢失{}帧以上)，已重置", LOSE_MAX);
  } else if (
    lose_ < 20 && last_powerrune_.has_value() && target_ != nullptr && !target_->is_unsolve())
  {
    status_ = TrackStatus::PREDICT_TRACK;
    auto now = std::chrono::steady_clock::now();
    double dt = tools::delta_time(now, last_timestamp_);
    dt = std::min(dt, 0.15);
    target_->predict(dt);
    predicted_powerrune_ = constructPredictedPowerRune();
    last_timestamp_ = now;
    if (lose_ < 10 || lose_ > 18) tools::logger()->debug("[Detector] 预测跟踪模式，丢失{}帧, dt={:.3f}s", lose_, dt);
  } else {
    status_ = TrackStatus::TEM_LOSE;
    prediction_valid_ = false;
    if (lose_ == 20) tools::logger()->debug("[Detector] 临时丢失!");
  }
}

std::optional<PowerRune> Buff_Detector::detect_multi(cv::Mat & bgr_img)
{
  auto now = std::chrono::steady_clock::now();
  const cv::Mat color_reference = bgr_img.clone();
  std::vector<YOLO11_BUFF::Object> results = MODE_.get_multicandidateboxes(bgr_img);
  results = filterByConfiguredColor(color_reference, results);
  std::cout << "[detect_multi] get_multicandidateboxes 返回 " << results.size() << " 个结果"
            << std::endl;
  if (results.empty()) return std::nullopt;
  lose_ = 0;
  status_ = TrackStatus::TRACK;
  last_timestamp_ = now;
  prediction_valid_ = false;
  std::vector<FanBlade> fanblades;
  for (const auto & result : results) {
    fanblades.emplace_back(FanBlade(result.kpt, result.kpt[4], _light));
  }
  auto r_center = get_r_center(fanblades, bgr_img);
  PowerRune powerrune(fanblades, r_center, last_powerrune_);
  if (powerrune.is_unsolve()) return std::nullopt;
  double distance_to_target = 5.0;
  if (last_powerrune_.has_value()) distance_to_target = last_powerrune_.value().ypd_in_world[2];
  for (int index : powerrune.getLitBoardIndices()) {
    powerrune.setBoardHitRing(index, ring_detector_.detectRing(
                                     powerrune.fanblades[index].center, powerrune.r_center,
                                     distance_to_target));
  }
  fillBoardMetadata(powerrune, results);
  powerrune.hit_ring = powerrune.getHitRingForBoard(powerrune.selectedBoardIndex());
  powerrune.activated_arms = detectActivatedArms(fanblades);
  powerrune.energy_type = energy_type_;
  powerrune.timestamp = cv::getTickCount() / cv::getTickFrequency();
  std::optional<PowerRune> P;
  P.emplace(powerrune);
  last_powerrune_ = P;
  return P;
}

std::optional<PowerRune> Buff_Detector::detect(cv::Mat & bgr_img)
{
  auto now = std::chrono::steady_clock::now();
  const cv::Mat color_reference = bgr_img.clone();

  std::vector<YOLO11_BUFF::Object> results;
  // 由于蓝色靶子识别不是很好，使用红色模型识别反色后的靶子
  // if (target_energy_color_ == BuffColor::BLUE) {
  //   cv::Mat to_red;
  //   cv::cvtColor(bgr_img, to_red, cv::COLOR_BGR2RGB);
  //   results = MODE_.get_multicandidateboxes(to_red);
  // }
  // else {
  //   results = MODE_.get_multicandidateboxes(bgr_img);
  // }

  results = MODE_.get_multicandidateboxes(bgr_img);
  // results = filterByConfiguredColor(color_reference, results);
  updateDetectionInfo(results);
  if (results.empty()) {
    handle_lose();
    if (status_ == TrackStatus::PREDICT_TRACK && prediction_valid_) {
      return std::optional<PowerRune>(predicted_powerrune_);
    }
    return std::nullopt;
  }
  lose_ = 0;
  status_ = TrackStatus::TRACK;
  last_timestamp_ = now;
  prediction_valid_ = false;
  std::vector<FanBlade> fanblades;
  for (const auto & result : results) {
    fanblades.emplace_back(FanBlade(result.kpt, result.kpt[4], _light));
  }
  auto r_center = get_r_center(fanblades, bgr_img);
  PowerRune powerrune(fanblades, r_center, last_powerrune_);
  if (powerrune.is_unsolve()) {
    handle_lose();
    if (status_ == TrackStatus::PREDICT_TRACK && prediction_valid_) {
      return std::optional<PowerRune>(predicted_powerrune_);
    }
    return std::nullopt;
  }
  double distance_to_target = 5.0;
  if (last_powerrune_.has_value()) distance_to_target = last_powerrune_.value().ypd_in_world[2];
  for (int index : powerrune.getLitBoardIndices()) {
    powerrune.setBoardHitRing(index, ring_detector_.detectRing(
                                     powerrune.fanblades[index].center, powerrune.r_center,
                                     distance_to_target));
  }
  fillBoardMetadata(powerrune, results);
  powerrune.hit_ring = powerrune.getHitRingForBoard(powerrune.selectedBoardIndex());
  powerrune.activated_arms = detectActivatedArms(fanblades);
  powerrune.energy_type = energy_type_;
  powerrune.timestamp = cv::getTickCount() / cv::getTickFrequency();
  std::optional<PowerRune> P;
  P.emplace(powerrune);
  last_powerrune_ = P;
  return P;
}

std::optional<PowerRune> Buff_Detector::detect_24(cv::Mat & bgr_img)
{
  auto now = std::chrono::steady_clock::now();
  const cv::Mat color_reference = bgr_img.clone();
  std::vector<YOLO11_BUFF::Object> results = MODE_.get_multicandidateboxes(bgr_img);
  results = filterByConfiguredColor(color_reference, results);
  if (results.empty()) return std::nullopt;
  std::vector<FanBlade> fanblades;
  for (auto & result : results) {
    fanblades.emplace_back(FanBlade(result.kpt, result.kpt[4], _light));
  }
  auto r_center = get_r_center(fanblades, bgr_img);
  PowerRune powerrune(fanblades, r_center, last_powerrune_);
  if (powerrune.is_unsolve()) {
    handle_lose();
    if (status_ == TrackStatus::PREDICT_TRACK && prediction_valid_) {
      return std::optional<PowerRune>(predicted_powerrune_);
    }
    return std::nullopt;
  }
  lose_ = 0;
  status_ = TrackStatus::TRACK;
  last_timestamp_ = now;
  prediction_valid_ = false;
  double distance_to_target = 5.0;
  if (last_powerrune_.has_value()) distance_to_target = last_powerrune_.value().ypd_in_world[2];
  for (int index : powerrune.getLitBoardIndices()) {
    powerrune.setBoardHitRing(index, ring_detector_.detectRing(
                                     powerrune.fanblades[index].center, powerrune.r_center,
                                     distance_to_target));
  }
  fillBoardMetadata(powerrune, results);
  powerrune.hit_ring = powerrune.getHitRingForBoard(powerrune.selectedBoardIndex());
  powerrune.activated_arms = detectActivatedArms(fanblades);
  powerrune.energy_type = energy_type_;
  powerrune.timestamp = cv::getTickCount() / cv::getTickFrequency();
  std::optional<PowerRune> P;
  P.emplace(powerrune);
  last_powerrune_ = P;
  return P;
}

std::optional<PowerRune> Buff_Detector::detect_debug(cv::Mat & bgr_img, cv::Point2f v)
{
  const cv::Mat color_reference = bgr_img.clone();
  std::vector<YOLO11_BUFF::Object> results = MODE_.get_multicandidateboxes(bgr_img);
  results = filterByConfiguredColor(color_reference, results);
  if (results.empty()) return std::nullopt;
  std::vector<FanBlade> fanblades_t;
  for (auto & result : results) {
    fanblades_t.emplace_back(FanBlade(result.kpt, result.kpt[4], _light));
  }
  auto r_center = get_r_center(fanblades_t, bgr_img);
  std::vector<FanBlade> fanblades;
  for (auto & fanblade : fanblades_t) {
    if (cv::norm((fanblade.center - r_center) - v) < 10 || results.size() == 1) {
      fanblades.emplace_back(fanblade);
      break;
    }
  }
  if (fanblades.empty()) return std::nullopt;
  PowerRune powerrune(fanblades, r_center, std::nullopt);
  for (int index : powerrune.getLitBoardIndices()) {
    powerrune.setBoardHitRing(index, ring_detector_.detectRing(
                                     powerrune.fanblades[index].center, powerrune.r_center, 5.0));
  }
  fillBoardMetadata(powerrune, results);
  powerrune.hit_ring = powerrune.getHitRingForBoard(powerrune.selectedBoardIndex());
  powerrune.activated_arms = detectActivatedArms(fanblades);
  powerrune.energy_type = energy_type_;
  powerrune.timestamp = cv::getTickCount() / cv::getTickFrequency();
  std::optional<PowerRune> P;
  P.emplace(powerrune);
  return P;
}

void Buff_Detector::updateDetectionInfo(const std::vector<YOLO11_BUFF::Object> & results)
{
  std::vector<double> confidences;
  std::vector<cv::Rect> bboxes;
  std::vector<cv::Point2f> centers;
  std::vector<int> track_counts(results.size(), 1);
  std::vector<bool> prev_used(last_detection_centers_.size(), false);
  for (size_t i = 0; i < results.size(); ++i) {
    const auto & result = results[i];
    confidences.push_back(result.prob);
    bboxes.push_back(result.rect);
    centers.push_back(result.kpt[4]);
    float best_distance = std::numeric_limits<float>::max();
    int best_prev = -1;
    for (size_t j = 0; j < last_detection_centers_.size(); ++j) {
      if (prev_used[j]) continue;
      float distance = cv::norm(centers.back() - last_detection_centers_[j]);
      if (distance < best_distance) {
        best_distance = distance;
        best_prev = static_cast<int>(j);
      }
    }
    if (best_prev >= 0 && best_distance <= 80.0F &&
        best_prev < static_cast<int>(target_track_counts_.size()))
    {
      prev_used[best_prev] = true;
      track_counts[i] = std::min(MAX_TRACK_COUNT, target_track_counts_[best_prev] + 1);
    }
  }
  last_confidences_ = std::move(confidences);
  last_bboxes_ = std::move(bboxes);
  last_detection_centers_ = std::move(centers);
  target_track_counts_ = std::move(track_counts);
}

void Buff_Detector::fillBoardMetadata(
  PowerRune & powerrune, const std::vector<YOLO11_BUFF::Object> & results) const
{
  powerrune.board_confidences.assign(powerrune.fanblades.size(), 0.0);
  powerrune.board_tracked_frames.assign(powerrune.fanblades.size(), 1);
  powerrune.board_areas.assign(powerrune.fanblades.size(), 0.0);
  std::vector<bool> used(results.size(), false);
  for (size_t board_index = 0; board_index < powerrune.fanblades.size(); ++board_index) {
    float best_distance = std::numeric_limits<float>::max();
    int best_result = -1;
    for (size_t result_index = 0; result_index < results.size(); ++result_index) {
      if (used[result_index]) continue;
      float distance =
        cv::norm(powerrune.fanblades[board_index].center - results[result_index].kpt[4]);
      if (distance < best_distance) {
        best_distance = distance;
        best_result = static_cast<int>(result_index);
      }
    }
    if (best_result < 0) continue;
    used[best_result] = true;
    powerrune.board_confidences[board_index] = results[best_result].prob;
    if (best_result < static_cast<int>(target_track_counts_.size())) {
      powerrune.board_tracked_frames[board_index] = target_track_counts_[best_result];
    }
    const auto & rect = results[best_result].rect;
    powerrune.board_areas[board_index] = static_cast<double>(rect.width) * rect.height;
  }
}

size_t Buff_Detector::getTargetCount() const
{
  if (!last_powerrune_.has_value()) return 0;
  return last_powerrune_.value().fanblades.size();
}

cv::Point2f Buff_Detector::getTargetCenter(int index) const
{
  if (!last_powerrune_.has_value()) return cv::Point2f(0, 0);
  const auto & powerrune = last_powerrune_.value();
  if (index < 0 || index >= static_cast<int>(powerrune.fanblades.size())) {
    return cv::Point2f(0, 0);
  }
  return powerrune.fanblades[index].center;
}

std::vector<cv::Point2f> Buff_Detector::getAllTargetCenters() const
{
  std::vector<cv::Point2f> centers;
  if (!last_powerrune_.has_value()) return centers;
  const auto & powerrune = last_powerrune_.value();
  for (const auto & blade : powerrune.fanblades) {
    centers.push_back(blade.center);
  }
  return centers;
}

std::vector<double> Buff_Detector::getAllConfidences() const
{
  if (!last_powerrune_.has_value()) return {};
  return last_powerrune_->board_confidences;
}

std::vector<double> Buff_Detector::getAllDistances() const
{
  std::vector<double> distances;
  if (!last_powerrune_.has_value()) return distances;
  const auto & powerrune = last_powerrune_.value();
  double current_distance = powerrune.ypd_in_world[2];
  size_t target_count = powerrune.fanblades.size();
  distances.assign(target_count, current_distance);
  return distances;
}

std::vector<double> Buff_Detector::getAllAngularVelocities() const
{
  std::vector<double> velocities;
  size_t target_count = getTargetCount();
  if (target_ != nullptr && !target_->is_unsolve()) {
    Eigen::VectorXd x = target_->ekf_x();
    double speed_rad = 3.0;
    if (energy_type_ == EnergyType::SMALL) {
      if (x.size() > 6) {
        speed_rad = std::abs(x[6]);
      } else if (x.size() > 5) {
        speed_rad = std::abs(x[5]);
      }
    } else if (x.size() > 6) {
      speed_rad = std::abs(x[6]);
    }
    if (speed_rad < 0.5 || speed_rad > 12.0) {
      speed_rad = 3.0;
    }
    velocities.assign(target_count, speed_rad);
  } else {
    double default_speed = (energy_type_ == EnergyType::SMALL) ? 1.05 : 3.0;
    velocities.assign(target_count, default_speed);
  }
  return velocities;
}

std::vector<int> Buff_Detector::getAllTrackedFrames() const
{
  if (!last_powerrune_.has_value()) return {};
  return last_powerrune_->board_tracked_frames;
}

std::vector<double> Buff_Detector::getAllAreas() const
{
  if (!last_powerrune_.has_value()) return {};
  return last_powerrune_->board_areas;
}

}  // namespace auto_buff
