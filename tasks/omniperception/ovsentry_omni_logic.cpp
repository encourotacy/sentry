#include "ovsentry_omni_logic.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include "decider.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace omniperception
{
namespace
{
constexpr double kPitchRad = 0.26;

double ang_deg(double lhs, double rhs)
{
  return std::abs(tools::limit_rad(lhs - rhs)) * 57.3;
}

uint8_t nav_id(auto_aim::ArmorName name)
{
  switch (name) {
    case auto_aim::ArmorName::one: return 1;
    case auto_aim::ArmorName::two: return 2;
    case auto_aim::ArmorName::three: return 3;
    case auto_aim::ArmorName::four: return 4;
    case auto_aim::ArmorName::five: return 5;
    case auto_aim::ArmorName::sentry: return 6;
    case auto_aim::ArmorName::outpost: return 7;
    case auto_aim::ArmorName::base: return 8;
    default: return 0;
  }
}

bool same_target(const OmniSearch::Lock & lhs, const OmniSearch::Lock & rhs, double gate_deg)
{
  return lhs.slot == rhs.slot && lhs.name == rhs.name && ang_deg(lhs.abs_yaw, rhs.abs_yaw) < gate_deg;
}

OmniSteer steer_to(const OmniSearch::Lock & lock)
{
  OmniSteer out;
  out.control = true;
  out.big_yaw = lock.abs_yaw;
  out.small_yaw = tools::limit_rad(lock.abs_yaw);
  out.pitch = kPitchRad;
  out.armor_id = lock.armor_id;
  return out;
}

std::string dev_name(const YAML::Node & yaml, const char * key, const char * fallback)
{
  auto value = yaml[key] ? yaml[key].as<std::string>() : std::string(fallback);
  if (value.rfind("/dev/", 0) == 0) value = value.substr(5);
  return value;
}

double yaml_double(const YAML::Node & yaml, const char * key, double fallback)
{
  return yaml[key] ? yaml[key].as<double>() : fallback;
}

struct Hit
{
  OmniSearch::Lock lock;
  auto_aim::ArmorPriority priority = auto_aim::ArmorPriority::fifth;
  double confidence = 0;
};

}  // namespace

OmniSearch::OmniSearch(const std::string & config_path)
: decider_(std::make_unique<Decider>(config_path)), yolo_(config_path, false, "omni_device")
{
  const auto yaml = tools::load(config_path);
  fov_h_deg_ = yaml_double(yaml, "omni_fov_h_deg", 120);
  min_delta_deg_ = yaml_double(yaml, "omni_min_delta_deg", 20);
  release_deg_ = yaml_double(yaml, "omni_release_deg", 3);
  timeout_floor_s_ = yaml_double(yaml, "omni_timeout_s", 0.5);
  yaw_rate_deg_s_ = std::max(1.0, yaml_double(yaml, "omni_yaw_rate_dps", 90));
  timeout_margin_s_ = yaml_double(yaml, "omni_margin_s", 0.4);
  read_timeout_ = std::chrono::milliseconds(
    std::max(1, yaml["omni_camera_read_timeout_ms"] ? yaml["omni_camera_read_timeout_ms"].as<int>() : 10));
  cooldown_ = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(yaml_double(yaml, "omni_cooldown_s", 2.5)));

  const double fov_v = yaml_double(yaml, "omni_fov_v_deg", 67);
  auto add = [&](OmniCameraSlot slot, const char * label, const char * path_key, const char * path_fallback,
                 const char * yaw_key, double yaw_fallback) {
    Cam cam;
    cam.spec = {slot, label, dev_name(yaml, path_key, path_fallback), yaml_double(yaml, yaw_key, yaw_fallback),
                fov_h_deg_, fov_v};
    cam.camera = std::make_unique<io::USBCamera>(cam.spec.dev_path, config_path);
    cam.camera->device_name = label;
    cams_.push_back(std::move(cam));
  };
  add(OmniCameraSlot::left, "left", "omni_left_path", "video0", "omni_left_yaw_deg", 60);
  add(OmniCameraSlot::right, "right", "omni_right_path", "video2", "omni_right_yaw_deg", -60);
  add(OmniCameraSlot::back, "back", "omni_back_path", "video4", "omni_back_yaw_deg", 180);
}

OmniSearch::~OmniSearch() = default;

void OmniSearch::read_cameras()
{
  for (auto & cam : cams_) {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
    if (!cam.camera->read_with_timeout(img, timestamp, read_timeout_) || img.empty()) continue;
    cam.image = img.clone();
    cam.image_ts = timestamp;
  }
}

std::vector<OmniCameraFrame> OmniSearch::frames() const
{
  std::vector<OmniCameraFrame> frames;
  frames.reserve(cams_.size());
  for (const auto & cam : cams_) frames.push_back({cam.spec.label, cam.image});
  return frames;
}

void OmniSearch::reset()
{
  session_.reset();
  cooldown_anchor_.reset();
  cooldown_until_ = {};
  timeout_until_ = {};
  timing_ = false;
}

OmniSteer OmniSearch::update(
  double current_big_yaw,
  const std::function<double(std::chrono::steady_clock::time_point)> & big_yaw_at_image,
  const std::function<bool(uint8_t)> & ignored, int frame_count)
{
  const auto now = std::chrono::steady_clock::now();
  if (cooldown_anchor_ && now >= cooldown_until_) cooldown_anchor_.reset();

  std::vector<Hit> hits;
  for (auto & cam : cams_) {
    if (cam.image.empty()) continue;

    auto armors = yolo_.detect(cam.image, frame_count);
    decider_->filter_enemy_color(armors);
    decider_->set_priority(armors);
    armors.remove_if([&](const auto_aim::Armor & armor) { return ignored(nav_id(armor.name)); });
    if (armors.empty()) continue;

    const auto best = std::min_element(
      armors.begin(), armors.end(), [](const auto_aim::Armor & a, const auto_aim::Armor & b) {
        if (a.priority != b.priority) return a.priority < b.priority;
        return a.confidence > b.confidence;
      });
    const double delta_yaw = cam.spec.center_yaw_deg + (0.5 - best->center_norm.x) * cam.spec.fov_h_deg;
    hits.push_back(Hit{
      Lock{cam.spec.slot, best->name, big_yaw_at_image(cam.image_ts) + delta_yaw / 57.3, nav_id(best->name)},
      best->priority, best->confidence});
  }

  const Lock * reference = session_ ? &*session_ : (cooldown_anchor_ ? &*cooldown_anchor_ : nullptr);
  std::sort(hits.begin(), hits.end(), [&](const Hit & a, const Hit & b) {
    if (a.priority != b.priority) return a.priority < b.priority;
    if (reference != nullptr) {
      const bool a_same = same_target(a.lock, *reference, min_delta_deg_);
      const bool b_same = same_target(b.lock, *reference, min_delta_deg_);
      if (a_same != b_same) return a_same;
    }
    const auto delta = [&](const Hit & hit) {
      return ang_deg(hit.lock.abs_yaw, reference != nullptr ? reference->abs_yaw : current_big_yaw);
    };
    if (std::abs(delta(a) - delta(b)) > 1e-6) return delta(a) < delta(b);
    return a.confidence > b.confidence;
  });

  OmniSteer out;
  if (!hits.empty()) {
    const auto & chosen = hits.front().lock;
    const double delta = reference != nullptr ? ang_deg(chosen.abs_yaw, reference->abs_yaw)
                                              : ang_deg(chosen.abs_yaw, current_big_yaw);
    const bool same = reference != nullptr && same_target(chosen, *reference, min_delta_deg_);
    const bool large = !same && delta >= min_delta_deg_;
    if (large && cooldown_anchor_ && now < cooldown_until_ && reference != nullptr) {
      out = steer_to(*reference);
    } else {
      out = steer_to(chosen);
      if (!timing_ || !session_ || !same_target(*session_, chosen, min_delta_deg_)) {
        const double slew_deg = ang_deg(chosen.abs_yaw, current_big_yaw);
        const double seconds = std::max(timeout_floor_s_, slew_deg / yaw_rate_deg_s_ + timeout_margin_s_);
        timeout_until_ = now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                 std::chrono::duration<double>(seconds));
        timing_ = true;
      }
      session_ = chosen;
      if (large) {
        cooldown_anchor_ = chosen;
        cooldown_until_ = now + cooldown_;
      }
    }
  } else if (session_ && ang_deg(session_->abs_yaw, current_big_yaw) > release_deg_) {
    out = steer_to(*session_);
  }

  if (!out.control) {
    timing_ = false;
    return out;
  }

  const double error_deg = ang_deg(out.big_yaw, current_big_yaw);
  if (error_deg <= release_deg_) {
    timing_ = false;
    return out;
  }
  if (timing_ && now > timeout_until_) {
    tools::logger()->warn(
      "[Omni] slew still {:.1f} deg when the allowed window elapsed", error_deg);
    reset();
    return {};
  }
  return out;
}

}  // namespace omniperception
