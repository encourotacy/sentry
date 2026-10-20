#include "target_prioritizer.hpp"

#include <fmt/core.h>
#include <iostream>

namespace auto_buff
{

std::string TargetScore::getScoreString() const
{
  return fmt::format(
    "ID:{} T:{:.2f} Y:{:.2f} P:{:.2f} D:{:.2f} C:{:.2f} S:{:.2f} Sp:{:.2f}", target_id,
    total_score, yaw_score, pitch_score, distance_score, confidence_score, stability_score,
    speed_score);
}

TargetPrioritizer::TargetPrioritizer() {}

TargetPrioritizer::TargetPrioritizer(const Config & config) : config_(config) {}

void TargetPrioritizer::setConfig(const Config & config)
{
  config_ = config;
}

TargetPrioritizer::Config TargetPrioritizer::getConfig() const
{
  return config_;
}

void TargetPrioritizer::updateGimbalPose(double yaw_deg, double pitch_deg)
{
  current_yaw_deg_ = yaw_deg;
  current_pitch_deg_ = pitch_deg;
}

double TargetPrioritizer::normalize(double value, double max_value, bool reverse)
{
  double normalized = std::min(1.0, std::max(0.0, value / max_value));
  return reverse ? (1.0 - normalized) : normalized;
}

double TargetPrioritizer::computeYawScore(double pixel_offset_x, double image_center_x)
{
  double offset = std::abs(pixel_offset_x);
  return normalize(offset, config_.max_yaw_offset_pixel, true);
}

double TargetPrioritizer::computePitchScore(double pixel_offset_y, double image_center_y)
{
  double offset = std::abs(pixel_offset_y);
  return normalize(offset, config_.max_pitch_offset_pixel, true);
}

double TargetPrioritizer::computeDistanceScore(double distance_m)
{
  return normalize(distance_m, config_.max_distance_m, true);
}

double TargetPrioritizer::computeConfidenceScore(double confidence)
{
  return std::min(1.0, std::max(0.0, confidence));
}

double TargetPrioritizer::computeStabilityScore(int tracked_frames)
{
  double score = std::min(1.0, tracked_frames / (double)config_.max_track_frames);
  return score;
}

double TargetPrioritizer::computeSpeedScore(double angular_velocity_rad)
{
  double speed = std::abs(angular_velocity_rad);
  if (speed <= config_.ideal_speed_rad) {
    return 1.0;
  }
  return normalize(speed, config_.max_speed_rad, true);
}

std::vector<TargetScore> TargetPrioritizer::prioritizeTargets(
  const std::vector<cv::Point2f> & target_centers, const std::vector<double> & confidences,
  const std::vector<double> & distances, const std::vector<double> & angular_velocities,
  const std::vector<int> & tracked_frames, const cv::Point2f & image_center)
{
  std::vector<TargetScore> scores;

  size_t n = std::min(
    {target_centers.size(), confidences.size(), distances.size(), angular_velocities.size(),
     tracked_frames.size()});

  for (size_t i = 0; i < n; ++i) {
    TargetScore score;
    score.target_id = static_cast<int>(i);
    double offset_x = target_centers[i].x - image_center.x;
    double offset_y = target_centers[i].y - image_center.y;
    score.yaw_score = computeYawScore(offset_x, image_center.x);
    score.pitch_score = computePitchScore(offset_y, image_center.y);
    score.distance_score = computeDistanceScore(distances[i]);
    score.confidence_score = computeConfidenceScore(confidences[i]);
    score.stability_score = computeStabilityScore(tracked_frames[i]);
    score.speed_score = computeSpeedScore(angular_velocities[i]);
    score.total_score = score.yaw_score * config_.weight_yaw +
                        score.pitch_score * config_.weight_pitch +
                        score.distance_score * config_.weight_distance +
                        score.confidence_score * config_.weight_confidence +
                        score.stability_score * config_.weight_stability +
                        score.speed_score * config_.weight_speed;
    scores.push_back(score);
  }

  std::sort(scores.begin(), scores.end(), [](const TargetScore & a, const TargetScore & b) {
    return a.total_score > b.total_score;
  });

  return scores;
}

int TargetPrioritizer::getBestTargetId(const std::vector<TargetScore> & scores)
{
  if (scores.empty()) return -1;
  return scores[0].target_id;
}

TargetScore TargetPrioritizer::getBestTargetScore(const std::vector<TargetScore> & scores)
{
  if (scores.empty()) {
    TargetScore empty;
    empty.target_id = -1;
    empty.total_score = 0;
    return empty;
  }
  return scores[0];
}

void TargetPrioritizer::visualizePriorities(
  cv::Mat & image, const std::vector<TargetScore> & scores,
  const std::vector<cv::Point2f> & target_centers)
{
  std::vector<cv::Scalar> colors = {
    cv::Scalar(0, 255, 0), cv::Scalar(255, 100, 0), cv::Scalar(0, 200, 255), cv::Scalar(0, 0, 255)};

  for (size_t rank = 0; rank < scores.size() && rank < target_centers.size(); ++rank) {
    const auto & score = scores[rank];
    if (score.target_id >= static_cast<int>(target_centers.size())) continue;
    const auto & center = target_centers[score.target_id];
    cv::Scalar color = colors[std::min(rank, colors.size() - 1)];
    std::string rank_text = rank == 0 ? "★ BEST" : fmt::format("#{}", rank + 1);
    cv::putText(
      image, rank_text, cv::Point(static_cast<int>(center.x - 25), static_cast<int>(center.y - 20)),
      cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
    cv::putText(
      image, fmt::format("{:.2f}", score.total_score),
      cv::Point(static_cast<int>(center.x - 20), static_cast<int>(center.y - 8)),
      cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
    cv::circle(image, center, 20, color, rank == 0 ? 3 : 2);
  }

  if (!scores.empty()) {
    const auto & best = scores[0];
    std::string decision =
      fmt::format("优先击打: 目标 #{} (得分: {:.2f})", best.target_id, best.total_score);
    cv::putText(
      image, decision, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7,
      cv::Scalar(0, 255, 0), 2);
    std::string details = fmt::format(
      "Y:{:.2f} P:{:.2f} D:{:.2f} C:{:.2f} Sp:{:.2f}", best.yaw_score, best.pitch_score,
      best.distance_score, best.confidence_score, best.speed_score);
    cv::putText(
      image, details, cv::Point(10, 85), cv::FONT_HERSHEY_SIMPLEX, 0.5,
      cv::Scalar(200, 200, 200), 1);
  }
}

}  // namespace auto_buff
