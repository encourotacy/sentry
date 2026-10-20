#include "buff_solver.hpp"

#include <cmath>

namespace auto_buff
{
cv::Matx33f Solver::rotation_matrix(double angle) const
{
  return cv::Matx33f(
    1, 0, 0, 0, std::cos(angle), -std::sin(angle), 0, std::sin(angle), std::cos(angle));
}

void Solver::compute_rotated_points(std::vector<std::vector<cv::Point3f>> & object_points)
{
  const std::vector<cv::Point3f> & base_points = object_points[0];
  for (int i = 1; i < 5; ++i) {
    double angle = i * THETA;
    cv::Matx33f R = rotation_matrix(angle);
    std::vector<cv::Point3f> rotated_points;
    for (const auto & point : base_points) {
      cv::Vec3f vec(point.x, point.y, point.z);
      cv::Vec3f rotated_vec = R * vec;
      rotated_points.emplace_back(rotated_vec[0], rotated_vec[1], rotated_vec[2]);
    }
    object_points[i] = rotated_points;
  }
}

Solver::Solver(const std::string & config_path)
  : R_gimbal2world_(Eigen::Matrix3d::Identity()),
    energy_type_(EnergyType::SMALL),
    param_fixed_(false),
    activation_time_(0.0),
    gen_(rd_())
{
  auto yaml = YAML::LoadFile(config_path);

  auto R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  auto R_camera2gimbal_data = yaml["R_camera2gimbal"].as<std::vector<double>>();
  auto t_camera2gimbal_data = yaml["t_camera2gimbal"].as<std::vector<double>>();
  R_gimbal2imubody_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_gimbal2imubody_data.data());
  R_camera2gimbal_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_camera2gimbal_data.data());
  t_camera2gimbal_ = Eigen::Matrix<double, 3, 1>(t_camera2gimbal_data.data());

  auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(camera_matrix_data.data());
  Eigen::Matrix<double, 1, 5> distort_coeffs(distort_coeffs_data.data());
  cv::eigen2cv(camera_matrix, camera_matrix_);
  cv::eigen2cv(distort_coeffs, distort_coeffs_);

  loadConfig(yaml);
}

void Solver::loadConfig(const YAML::Node & config)
{
  if (config["big_energy"]) {
    a_min_ = config["big_energy"]["a_min"].as<double>(0.780);
    a_max_ = config["big_energy"]["a_max"].as<double>(1.045);
    omega_min_ = config["big_energy"]["omega_min"].as<double>(1.884);
    omega_max_ = config["big_energy"]["omega_max"].as<double>(2.000);
    b_base_ = config["big_energy"]["b_base"].as<double>(2.090);
  } else {
    a_min_ = 0.780;
    a_max_ = 1.045;
    omega_min_ = 1.884;
    omega_max_ = 2.000;
    b_base_ = 2.090;
  }
}

void Solver::generateRandomParam()
{
  std::uniform_real_distribution<> a_dist(a_min_, a_max_);
  std::uniform_real_distribution<> omega_dist(omega_min_, omega_max_);

  sinusoidal_param_.a = a_dist(gen_);
  sinusoidal_param_.omega = omega_dist(gen_);
  sinusoidal_param_.b = b_base_ - sinusoidal_param_.a;
  sinusoidal_param_.phi = 0.0;

  param_fixed_ = true;
}

void Solver::setEnergyType(EnergyType type)
{
  if (energy_type_ == type) return;

  energy_type_ = type;

  if (type == EnergyType::BIG && !param_fixed_) {
    generateRandomParam();
  } else if (type == EnergyType::SMALL) {
    param_fixed_ = false;
  }
}

double Solver::getCurrentSpeed(double current_time) const
{
  if (energy_type_ == EnergyType::SMALL) {
    return CV_PI / 3.0;
  } else {
    if (!param_fixed_) {
      return CV_PI / 3.0;
    }
    double t = current_time - activation_time_;
    if (t < 0) t = 0;
    return sinusoidal_param_.getSpeed(t);
  }
}

double Solver::predictAngle(double start_angle, double current_time) const
{
  if (energy_type_ == EnergyType::SMALL) {
    double dt = current_time - activation_time_;
    if (dt < 0) dt = 0;
    return start_angle + (CV_PI / 3.0) * dt;
  } else {
    if (!param_fixed_) {
      double dt = current_time - activation_time_;
      if (dt < 0) dt = 0;
      return start_angle + (CV_PI / 3.0) * dt;
    }
    double t = current_time - activation_time_;
    if (t < 0) t = 0;
    return sinusoidal_param_.getAngle(start_angle, t);
  }
}

Eigen::Matrix3d Solver::R_gimbal2world() const { return R_gimbal2world_; }

void Solver::set_R_gimbal2world(const Eigen::Quaterniond & q)
{
  Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
  R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
}

void Solver::solve(std::optional<PowerRune> & ps) const
{
  if (!ps.has_value()) return;
  PowerRune & p = ps.value();

  std::vector<cv::Point2f> image_points = p.target().points;
  image_points.emplace_back(p.r_center);

  std::vector<cv::Point2f> image_points_fourth(image_points.begin(), image_points.begin() + 4);
  std::vector<cv::Point3f> OBJECT_POINTS_FOURTH(OBJECT_POINTS.begin(), OBJECT_POINTS.begin() + 4);
  cv::solvePnP(
    OBJECT_POINTS_FOURTH, image_points_fourth, camera_matrix_, distort_coeffs_, rvec_, tvec_,
    false, cv::SOLVEPNP_IPPE);

  Eigen::Vector3d t_buff2camera;
  cv::cv2eigen(tvec_, t_buff2camera);
  cv::Mat rmat;
  cv::Rodrigues(rvec_, rmat);
  Eigen::Matrix3d R_buff2camera;
  cv::cv2eigen(rmat, R_buff2camera);

  Eigen::Vector3d blade_xyz_in_buff{{0, 0, 700e-3}};

  Eigen::Vector3d xyz_in_camera = t_buff2camera;
  Eigen::Vector3d blade_xyz_in_camera = R_buff2camera * blade_xyz_in_buff + t_buff2camera;

  Eigen::Matrix3d R_buff2gimbal = R_camera2gimbal_ * R_buff2camera;
  Eigen::Vector3d xyz_in_gimbal = R_camera2gimbal_ * xyz_in_camera + t_camera2gimbal_;
  Eigen::Vector3d blade_xyz_in_gimbal = R_camera2gimbal_ * blade_xyz_in_camera + t_camera2gimbal_;

  Eigen::Matrix3d R_buff2world = R_gimbal2world_ * R_buff2gimbal;

  p.xyz_in_world = R_gimbal2world_ * xyz_in_gimbal;
  p.ypd_in_world = tools::xyz2ypd(p.xyz_in_world);

  p.blade_xyz_in_world = R_gimbal2world_ * blade_xyz_in_gimbal;
  p.blade_ypd_in_world = tools::xyz2ypd(p.blade_xyz_in_world);

  p.ypr_in_world = tools::eulers(R_buff2world, 2, 1, 0);
  p.energy_type = energy_type_;
}

cv::Point2f Solver::point_buff2pixel(cv::Point3f x)
{
  std::vector<cv::Point3d> world_points;
  std::vector<cv::Point2d> image_points;
  world_points.push_back(x);
  cv::projectPoints(world_points, rvec_, tvec_, camera_matrix_, distort_coeffs_, image_points);
  return image_points.back();
}

std::vector<cv::Point2f> Solver::reproject_buff(
  const Eigen::Vector3d & xyz_in_world, double yaw, double row) const
{
  auto R_buff2world = tools::rotation_matrix(Eigen::Vector3d(yaw, 0.0, row));

  const Eigen::Vector3d & t_buff2world = xyz_in_world;
  Eigen::Matrix3d R_buff2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * R_buff2world;
  Eigen::Vector3d t_buff2camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * t_buff2world - t_camera2gimbal_);

  cv::Vec3d rvec;
  cv::Mat R_buff2camera_cv;
  cv::eigen2cv(R_buff2camera, R_buff2camera_cv);
  cv::Rodrigues(R_buff2camera_cv, rvec);
  cv::Vec3d tvec(t_buff2camera[0], t_buff2camera[1], t_buff2camera[2]);

  std::vector<cv::Point2f> image_points;
  cv::projectPoints(OBJECT_POINTS, rvec, tvec, camera_matrix_, distort_coeffs_, image_points);
  return image_points;
}

}  // namespace auto_buff
