#ifndef AUTO_BUFF__SOLVER_HPP
#define AUTO_BUFF__SOLVER_HPP

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <optional>
#include <random>

#include "buff_type.hpp"
#include "tools/math_tools.hpp"

namespace auto_buff
{
const double THETA = 2.0 * CV_PI / 5.0;

class Solver
{
public:
  explicit Solver(const std::string & config_path);

  Eigen::Matrix3d R_gimbal2world() const;
  void set_R_gimbal2world(const Eigen::Quaterniond & q);
  void solve(std::optional<PowerRune> & ps) const;

  cv::Point2f point_buff2pixel(cv::Point3f x);
  std::vector<cv::Point2f> reproject_buff(
    const Eigen::Vector3d & xyz_in_world, double yaw, double row) const;

  void setEnergyType(EnergyType type);
  const SinusoidalParam & getSinusoidalParam() const { return sinusoidal_param_; }
  double getAMin() const { return a_min_; }
  double getAMax() const { return a_max_; }
  double getOmegaMin() const { return omega_min_; }
  double getOmegaMax() const { return omega_max_; }
  double getBBase() const { return b_base_; }
  double getCurrentSpeed(double current_time) const;
  double predictAngle(double start_angle, double current_time) const;
  double getActivationTime() const { return activation_time_; }

private:
  cv::Mat camera_matrix_;
  cv::Mat distort_coeffs_;
  Eigen::Matrix3d R_gimbal2imubody_;
  Eigen::Matrix3d R_camera2gimbal_;
  Eigen::Vector3d t_camera2gimbal_;
  Eigen::Matrix3d R_gimbal2world_;

  cv::Vec3d rvec_, tvec_;

  const std::vector<cv::Point3f> OBJECT_POINTS = {
    cv::Point3f(0, 0, 825e-3), cv::Point3f(0, 125e-3, 700e-3), cv::Point3f(0, 0, 575e-3),
    cv::Point3f(0, -125e-3, 700e-3), cv::Point3f(0, 0, 700e-3), cv::Point3f(0, 0, 180e-3),
    cv::Point3f(0, 0, 0)};

  EnergyType energy_type_;
  SinusoidalParam sinusoidal_param_;
  bool param_fixed_;
  double activation_time_;

  double a_min_, a_max_;
  double omega_min_, omega_max_;
  double b_base_;

  std::random_device rd_;
  std::mt19937 gen_;

  void loadConfig(const YAML::Node & config);
  void generateRandomParam();
  cv::Matx33f rotation_matrix(double angle) const;
  void compute_rotated_points(std::vector<std::vector<cv::Point3f>> & object_points);
};
}  // namespace auto_buff
#endif  // AUTO_AIM__SOLVER_HPP
