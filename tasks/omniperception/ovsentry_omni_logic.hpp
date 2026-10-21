#ifndef OMNIPERCEPTION__OVSENTRY_OMNI_LOGIC_HPP
#define OMNIPERCEPTION__OVSENTRY_OMNI_LOGIC_HPP

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "detection.hpp"
#include "io/usbcamera/usbcamera.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/yolo.hpp"

namespace omniperception
{
class Decider;

struct OmniSteer
{
  bool control = false;
  double big_yaw = 0;
  double small_yaw = 0;
  double pitch = 0.26;
  uint8_t armor_id = 0;
};

// 只在主相机 lost 时调用。跟踪中不要用侧相机抢目标。
class OmniSearch
{
public:
  explicit OmniSearch(const std::string & config_path);
  ~OmniSearch();

  void reset();

  OmniSteer update(
    double current_big_yaw,
    const std::function<double(std::chrono::steady_clock::time_point)> & big_yaw_at_image,
    const std::function<bool(uint8_t)> & ignored, int frame_count);

  struct Lock
  {
    OmniCameraSlot slot = OmniCameraSlot::unknown;
    auto_aim::ArmorName name = auto_aim::ArmorName::not_armor;
    double abs_yaw = 0;
    uint8_t armor_id = 0;
  };

private:
  struct Cam
  {
    CameraSpec spec;
    std::unique_ptr<io::USBCamera> camera;
  };

  std::unique_ptr<Decider> decider_;
  auto_aim::YOLO yolo_;
  std::vector<Cam> cams_;
  double fov_h_deg_ = 120;
  double min_delta_deg_ = 20;
  double release_deg_ = 3;
  double timeout_floor_s_ = 0.5;
  double yaw_rate_deg_s_ = 90;
  double timeout_margin_s_ = 0.4;
  std::chrono::milliseconds read_timeout_{10};
  std::chrono::steady_clock::duration cooldown_{};
  std::optional<Lock> session_;
  std::optional<Lock> cooldown_anchor_;
  std::chrono::steady_clock::time_point cooldown_until_{};
  std::chrono::steady_clock::time_point timeout_until_{};
  bool timing_ = false;
};

}  // namespace omniperception

#endif
