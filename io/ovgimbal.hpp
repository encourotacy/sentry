#ifndef IO__OVGIMBAL_HPP
#define IO__OVGIMBAL_HPP

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <thread>

#include "io/cboard.hpp"
#include "io/command.hpp"
#include "serial/serial.h"
#include "tools/thread_safe_queue.hpp"
#include "tasks/auto_aim/armor.hpp"

namespace io
{
struct OVGimbalState
{
  double yaw;
  double yaw_vel;
  double pitch;
  double pitch_vel;
  double bullet_speed;
};

enum class AutoAimMode : uint8_t
{
  normal = 0x00,
  small_energy = 0x01,
  big_energy = 0x02,
  anti_top = 0x03,
  outpost = 0x04,
  unknown = 0xff
};

class OVGimbal
{
public:
  explicit OVGimbal(const std::string & config_path);
  ~OVGimbal();

  Eigen::Quaterniond imu_at(std::chrono::steady_clock::time_point timestamp);
  Eigen::Quaterniond imu_at_image(std::chrono::steady_clock::time_point image_timestamp);
  double offset_ms() const;
  bool has_imu() const;

  void send(const io::Command & command);
  void send(
    bool control, bool fire, float yaw, float pitch, float yaw_vel, float yaw_acc, float pitch_vel,
    float pitch_acc, uint8_t armor_name_num = 0);

  double bullet_speed() const;
  Mode mode() const;
  uint8_t raw_mode() const;
  AutoAimMode autoaim_mode() const;
  std::string autoaim_mode_name() const;
  OVGimbalState state() const;
  uint8_t invincibility() const;

private:
  struct IMUData
  {
    Eigen::Quaterniond q;
    std::chrono::steady_clock::time_point timestamp;
  };

  serial::Serial serial_;
  std::thread thread_;
  std::atomic<bool> quit_{false};
  mutable std::mutex mtx_;

  tools::ThreadSafeQueue<IMUData> queue_{1000};
  Mode mode_{Mode::idle};
  uint8_t raw_mode_{0};
  double yaw_ = 0.0;
  double pitch_ = 0.0;
  double yaw_vel_ = 0.0;
  double pitch_vel_ = 0.0;
  double bullet_speed_ = 23.0;
  IMUData data_prev_;
  IMUData data_ahead_;
  IMUData data_behind_;
  bool has_prev_{false};
  bool has_latest_data_{false};
  IMUData latest_data_;
  std::chrono::microseconds imu_query_offset_{-1000};
  
  uint8_t invincibility_ = 0;

  void read_thread();
  bool read_exact(uint8_t * buffer, size_t size);
  void reconnect();
};

}  // namespace io

#endif  // IO__OVGIMBAL_HPP
