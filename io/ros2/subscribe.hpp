#ifndef IO__SUBSCRIBE_HPP
#define IO__SUBSCRIBE_HPP

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace io
{
enum class ROS2Mode : int
{
  auto_aim = 0,
  small_buff = 1,
  big_buff = 2
};

struct ArmorIgnoreList
{
  bool enabled = false;
  std::vector<uint8_t> ignored_ids;

  bool contains(uint8_t id) const
  {
    if (!enabled || id == 0) return false;
    return std::find(ignored_ids.begin(), ignored_ids.end(), id) != ignored_ids.end();
  }
};

class ArmorIgnoreSubscriber
{
public:
  ArmorIgnoreSubscriber(
    const std::string & topic = "/request_auto_aim_ignore",
    const std::string & msg_type = "rm_interfaces/msg/RequestAutoAimIgnore");
  ~ArmorIgnoreSubscriber();

  ArmorIgnoreList ignored() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

struct NavSubscribeConfig
{
  std::string request_buff_topic = "/request_buff";
  std::string auto_aim_ignore_topic = "/request_auto_aim_ignore";
  std::string auto_aim_ignore_msg_type = "rm_interfaces/msg/RequestAutoAimIgnore";
};

NavSubscribeConfig load_nav_subscribe_config(const std::string & config_path);

class BuffRequestSubscriber
{
public:
  explicit BuffRequestSubscriber(const std::string & topic = "/request_buff");
  ~BuffRequestSubscriber();

  ROS2Mode mode() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace io

#endif  // IO__SUBSCRIBE_HPP
