#ifndef AUTO_BUFF__AIMER_HPP
#define AUTO_BUFF__AIMER_HPP

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <memory>

#include "../auto_aim/planner/planner.hpp"
#include "buff_target.hpp"
#include "buff_type.hpp"
#include "io/command.hpp"
#include "io/gimbal/gimbal.hpp"

namespace auto_buff
{

class Aimer
{
public:
  Aimer(const std::string & config_path);

  io::Command aim(
    Target & target, std::chrono::steady_clock::time_point & timestamp, double bullet_speed,
    bool to_now = true);

  auto_aim::Plan mpc_aim(
    Target & target, std::chrono::steady_clock::time_point & timestamp, io::GimbalState gs,
    bool to_now = true);

  double getCurrentAngle() const { return angle_; }
  int getLastHitRing() const { return last_hit_ring_; }

private:
  bool getSendAngle(
    Target & target, const double predict_time, const double bullet_speed, double & yaw,
    double & pitch);

  Eigen::Vector3d getCurrentArmCenter(Target & target) const;

  double yaw_offset_;
  double pitch_offset_;
  double fire_gap_time_;
  double predict_time_;

  int mistake_count_ = 0;
  bool switch_fanblade_ = false;
  double last_yaw_ = 0;
  double last_pitch_ = 0;
  double angle_ = 0;
  int last_hit_ring_ = 0;

  bool first_in_aimer_ = true;
  std::chrono::steady_clock::time_point last_fire_t_;
};

}  // namespace auto_buff
#endif  // AUTO_AIM__AIMER_HPP
