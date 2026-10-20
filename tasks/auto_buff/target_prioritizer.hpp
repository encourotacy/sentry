#ifndef TARGET_PRIORITIZER_HPP
#define TARGET_PRIORITIZER_HPP

#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace auto_buff
{

struct TargetScore {
  int target_id;
  double total_score;
  double yaw_score;
  double pitch_score;
  double distance_score;
  double confidence_score;
  double stability_score;
  double speed_score;

  std::string getScoreString() const;
};

class TargetPrioritizer
{
public:
  struct Config {
    double weight_yaw = 0.30;
    double weight_pitch = 0.10;
    double weight_distance = 0.25;
    double weight_confidence = 0.20;
    double weight_stability = 0.10;
    double weight_speed = 0.05;

    double max_yaw_offset_pixel = 320.0;
    double max_pitch_offset_pixel = 240.0;
    double max_distance_m = 15.0;
    double ideal_speed_rad = 2.0;
    double max_speed_rad = 8.0;
    int max_track_frames = 100;
  };

  TargetPrioritizer();
  explicit TargetPrioritizer(const Config & config);

  void setConfig(const Config & config);
  Config getConfig() const;
  void updateGimbalPose(double yaw_deg, double pitch_deg);

  std::vector<TargetScore> prioritizeTargets(
    const std::vector<cv::Point2f> & target_centers, const std::vector<double> & confidences,
    const std::vector<double> & distances, const std::vector<double> & angular_velocities,
    const std::vector<int> & tracked_frames,
    const cv::Point2f & image_center = cv::Point2f(320, 240));

  int getBestTargetId(const std::vector<TargetScore> & scores);
  TargetScore getBestTargetScore(const std::vector<TargetScore> & scores);
  void visualizePriorities(
    cv::Mat & image, const std::vector<TargetScore> & scores,
    const std::vector<cv::Point2f> & target_centers);

private:
  Config config_;
  double current_yaw_deg_ = 0.0;
  double current_pitch_deg_ = 0.0;

  double normalize(double value, double max_value, bool reverse = true);
  double computeYawScore(double pixel_offset_x, double image_center_x);
  double computePitchScore(double pixel_offset_y, double image_center_y);
  double computeDistanceScore(double distance_m);
  double computeConfidenceScore(double confidence);
  double computeStabilityScore(int tracked_frames);
  double computeSpeedScore(double angular_velocity_rad);
};

}  // namespace auto_buff

#endif
