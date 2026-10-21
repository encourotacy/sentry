#ifndef OMNIPERCEPTION__DETECTION_HPP
#define OMNIPERCEPTION__DETECTION_HPP

#include <chrono>
#include <list>
#include <string>

#include "tasks/auto_aim/armor.hpp"

namespace omniperception
{
enum class OmniCameraSlot
{
  unknown = 0,
  left,
  right,
  back,
  extra
};

struct CameraSpec
{
  OmniCameraSlot slot = OmniCameraSlot::unknown;
  std::string label;
  std::string dev_path;
  double center_yaw_deg = 0.0;
  double fov_h_deg = 0.0;
  double fov_v_deg = 0.0;
};

//一个识别结果可能包含多个armor,需要排序和过滤。armors, timestamp, delta_yaw, delta_pitch
struct DetectionResult
{
  std::list<auto_aim::Armor> armors;
  std::chrono::steady_clock::time_point timestamp;
  double delta_yaw;    //rad
  double delta_pitch;  //rad

  // Assignment operator
  DetectionResult & operator=(const DetectionResult & other)
  {
    if (this != &other) {
      armors = other.armors;
      timestamp = other.timestamp;
      delta_yaw = other.delta_yaw;
      delta_pitch = other.delta_pitch;
    }
    return *this;
  }
};
}  // namespace omniperception

#endif