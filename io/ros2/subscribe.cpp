#include "subscribe.hpp"

#include <fastcdr/Cdr.h>
#include <fastcdr/FastBuffer.h>

#include <atomic>
#include <mutex>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int32.hpp>

#include "tools/logger.hpp"
#include "tools/yaml.hpp"

namespace io
{
namespace
{
const char * ros2_mode_name(ROS2Mode mode)
{
  switch (mode) {
    case ROS2Mode::small_buff:
      return "small_buff";
    case ROS2Mode::big_buff:
      return "big_buff";
    case ROS2Mode::auto_aim:
    default:
      return "auto_aim";
  }
}

std::vector<uint8_t> deserialize_ignore_ids(const rclcpp::SerializedMessage & serialized_message)
{
  const auto & raw = serialized_message.get_rcl_serialized_message();
  eprosima::fastcdr::FastBuffer buffer(reinterpret_cast<char *>(raw.buffer), raw.buffer_length);
  eprosima::fastcdr::Cdr cdr(buffer);
  cdr.read_encapsulation();

  uint32_t count = 0;
  cdr >> count;

  std::vector<uint8_t> ids;
  ids.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    uint8_t id = 0;
    cdr >> id;
    ids.push_back(id);
  }

  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  return ids;
}

}  // namespace

class ArmorIgnoreSubscriber::Impl
{
public:
  Impl(const std::string & topic, const std::string & msg_type)
  {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
      self_initialized_ = true;
    }

    node_ = std::make_shared<rclcpp::Node>("auto_aim_ignore_subscriber");
    try {
      subscription_ = node_->create_generic_subscription(
        topic, msg_type, rclcpp::SensorDataQoS(),
        [this](const std::shared_ptr<rclcpp::SerializedMessage> message) {
          this->callback(message);
        });

      executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
      executor_->add_node(node_);
      spin_thread_ = std::thread([this]() { executor_->spin(); });
      tools::logger()->info("[AutoAimIgnore] Subscribed '{}' as '{}'.", topic, msg_type);
    } catch (const std::exception & e) {
      tools::logger()->warn("[AutoAimIgnore] Failed to subscribe '{}': {}", topic, e.what());
    }
  }

  ~Impl()
  {
    if (executor_) executor_->cancel();
    if (spin_thread_.joinable()) spin_thread_.join();
    if (executor_ && node_) executor_->remove_node(node_);
    if (self_initialized_ && rclcpp::ok()) rclcpp::shutdown();
  }

  ArmorIgnoreList ignored() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return list_;
  }

private:
  void callback(const std::shared_ptr<rclcpp::SerializedMessage> & message)
  {
    try {
      auto ids = deserialize_ignore_ids(*message);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        list_.enabled = !ids.empty();
        list_.ignored_ids = ids;
      }
      for (const auto id : ids) {
        tools::logger()->info("[AutoAimIgnore] ignore armor id: {}", static_cast<int>(id));
      }
    } catch (const std::exception & e) {
      tools::logger()->warn("[AutoAimIgnore] Failed to parse ignore ids: {}", e.what());
    }
  }

  mutable std::mutex mutex_;
  ArmorIgnoreList list_;
  bool self_initialized_ = false;
  std::shared_ptr<rclcpp::Node> node_;
  std::shared_ptr<rclcpp::GenericSubscription> subscription_;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  std::thread spin_thread_;
};

class BuffRequestSubscriber::Impl
{
public:
  explicit Impl(const std::string & topic)
  {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
      self_initialized_ = true;
    }

    node_ = std::make_shared<rclcpp::Node>("buff_request_subscriber");
    subscription_ = node_->create_subscription<std_msgs::msg::Int32>(
      topic, 10, [this](const std_msgs::msg::Int32::SharedPtr message) {
        this->callback(message);
      });

    executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
    executor_->add_node(node_);
    spin_thread_ = std::thread([this]() { executor_->spin(); });
    tools::logger()->info("[BuffRequest] Subscribed '{}' as Int32.", topic);
  }

  ~Impl()
  {
    if (executor_) executor_->cancel();
    if (spin_thread_.joinable()) spin_thread_.join();
    if (executor_ && node_) executor_->remove_node(node_);
    if (self_initialized_ && rclcpp::ok()) rclcpp::shutdown();
  }

  ROS2Mode mode() const { return static_cast<ROS2Mode>(mode_.load()); }

private:
  void callback(const std_msgs::msg::Int32::SharedPtr message)
  {
    ROS2Mode mode = ROS2Mode::auto_aim;
    switch (message->data) {
      case 0:
        mode = ROS2Mode::auto_aim;
        break;
      case 1:
        mode = ROS2Mode::small_buff;
        break;
      case 2:
        mode = ROS2Mode::big_buff;
        break;
      default:
        tools::logger()->warn("[BuffRequest] ignore request_buff={}", message->data);
        return;
    }

    const auto previous = static_cast<ROS2Mode>(mode_.exchange(static_cast<int>(mode)));
    if (previous == mode) return;
    tools::logger()->info(
      "[BuffRequest] request_buff={} mode={}", message->data, ros2_mode_name(mode));
  }

  std::atomic<int> mode_{static_cast<int>(ROS2Mode::auto_aim)};
  bool self_initialized_ = false;
  std::shared_ptr<rclcpp::Node> node_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr subscription_;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  std::thread spin_thread_;
};

NavSubscribeConfig load_nav_subscribe_config(const std::string & config_path)
{
  NavSubscribeConfig config;
  const auto yaml = tools::load(config_path);
  const auto bridge = yaml["ros2_gimbal"];
  if (!bridge) return config;

  const auto subscribe = bridge["subscribe"];
  const auto pick = [&](const char * key) {
    if (subscribe && subscribe[key]) return subscribe[key];
    return bridge[key];
  };

  if (const auto topic = pick("request_buff_topic")) {
    config.request_buff_topic = topic.as<std::string>();
  }
  if (const auto topic = pick("auto_aim_ignore_topic")) {
    config.auto_aim_ignore_topic = topic.as<std::string>();
  }
  if (const auto msg_type = pick("auto_aim_ignore_msg_type")) {
    config.auto_aim_ignore_msg_type = msg_type.as<std::string>();
  }
  return config;
}

ArmorIgnoreSubscriber::ArmorIgnoreSubscriber(const std::string & topic, const std::string & msg_type)
: impl_(std::make_unique<Impl>(topic, msg_type))
{
}

ArmorIgnoreSubscriber::~ArmorIgnoreSubscriber() = default;

ArmorIgnoreList ArmorIgnoreSubscriber::ignored() const { return impl_->ignored(); }

BuffRequestSubscriber::BuffRequestSubscriber(const std::string & topic)
: impl_(std::make_unique<Impl>(topic))
{
}

BuffRequestSubscriber::~BuffRequestSubscriber() = default;

ROS2Mode BuffRequestSubscriber::mode() const { return impl_->mode(); }

}  // namespace io
