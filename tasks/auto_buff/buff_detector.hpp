#ifndef AUTO_BUFF__DETECTOR_HPP
#define AUTO_BUFF__DETECTOR_HPP

#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cmath>
#include <deque>
#include <memory>
#include <optional>
#include <vector>

#include "buff_type.hpp"
#include "tools/img_tools.hpp"
#include "tools/math_tools.hpp"
#include "yolo11_buff.hpp"

namespace auto_buff
{
enum class BuffColor {
  RED,
  BLUE
};

class Target;
}

const int LOSE_MAX = 60;

namespace auto_buff
{

enum class TrackStatus {
  TRACK,
  TEM_LOSE,
  PREDICT_TRACK,
  LOSE
};

class RingDetector
{
public:
  RingDetector();
  void setParams(double max_radius_mm, int outer_ring, int inner_ring);
  void setFocalLength(double focal_length);
  int detectRing(
    const cv::Point2f & hit_point, const cv::Point2f & center, double distance_to_target) const;

private:
  double pixelToMm(double pixel_dist, double distance_m) const;
  double max_radius_mm_;
  int outer_ring_;
  int inner_ring_;
  double focal_length_;
};

class ArmStateDetector
{
public:
  ArmStateDetector();
  void setParams(int brightness_threshold, bool use_color_detection);
  int detectActivatedArms(const std::vector<FanBlade> & fanblades) const;
  int getCurrentTargetArmIndex(const std::vector<FanBlade> & fanblades) const;

private:
  int brightness_threshold_;
  bool use_color_detection_;
};

class Buff_Detector
{
public:
  explicit Buff_Detector(const std::string & config);
  Buff_Detector(const std::string & config, const std::string & model_key);

  std::optional<PowerRune> detect_multi(cv::Mat & bgr_img);
  std::optional<PowerRune> detect(cv::Mat & bgr_img);
  std::optional<PowerRune> detect_24(cv::Mat & bgr_img);
  std::optional<PowerRune> detect_debug(cv::Mat & bgr_img, cv::Point2f v);

  void setEnergyType(EnergyType type);
  void setCameraParams(double fx, double fy, double cx, double cy);
  void setTarget(std::shared_ptr<Target> target_ptr) { target_ = target_ptr; }

  EnergyType getEnergyType() const { return energy_type_; }
  TrackStatus getStatus() const { return status_; }
  int getLoseCount() const { return lose_; }
  std::optional<PowerRune> getPredictedPowerRune() const;

  std::vector<cv::Point2f> getAllTargetCenters() const;
  std::vector<double> getAllConfidences() const;
  std::vector<double> getAllDistances() const;
  std::vector<double> getAllAngularVelocities() const;
  std::vector<int> getAllTrackedFrames() const;
  std::vector<double> getAllAreas() const;
  size_t getTargetCount() const;
  cv::Point2f getTargetCenter(int index) const;
  const std::optional<PowerRune> & getLastPowerRune() const { return last_powerrune_; }

private:
  static const char * colorName(BuffColor color);
  static BuffColor oppositeColor(BuffColor color);
  static BuffColor classifyColor(const cv::Mat & bgr_img, const YOLO11_BUFF::Object & result);
  std::vector<YOLO11_BUFF::Object> filterByConfiguredColor(
    const cv::Mat & bgr_img, const std::vector<YOLO11_BUFF::Object> & results) const;

  void handle_img(const cv::Mat & bgr_img, cv::Mat & dilated_img);
  cv::Point2f get_r_center(std::vector<FanBlade> & fanblades, cv::Mat & bgr_img);
  void handle_lose();
  int detectHitRing(const PowerRune & powerrune, double distance_to_target) const;
  int detectActivatedArms(const std::vector<FanBlade> & fanblades) const;
  PowerRune constructPredictedPowerRune();
  void updateDetectionInfo(const std::vector<YOLO11_BUFF::Object> & results);
  void fillBoardMetadata(
    PowerRune & powerrune, const std::vector<YOLO11_BUFF::Object> & results) const;

  YOLO11_BUFF MODE_;
  TrackStatus status_;
  int lose_;
  double lastlen_;
  std::optional<PowerRune> last_powerrune_;
  BuffColor configured_energy_color_;
  BuffColor target_energy_color_;
  bool energy_color_configured_;

  std::shared_ptr<Target> target_;
  std::chrono::steady_clock::time_point last_timestamp_;
  PowerRune predicted_powerrune_;
  bool prediction_valid_;

  EnergyType energy_type_;
  RingDetector ring_detector_;
  ArmStateDetector arm_detector_;

  double fx_, fy_, cx_, cy_;
  double max_radius_mm_;
  int outer_ring_;
  int inner_ring_;

  std::vector<double> last_confidences_;
  std::vector<cv::Rect> last_bboxes_;
  std::vector<cv::Point2f> last_detection_centers_;
  std::vector<int> target_track_counts_;
  static constexpr int MAX_TRACK_COUNT = 100;
};

}  // namespace auto_buff

#endif  // AUTO_BUFF__DETECTOR_HPP
