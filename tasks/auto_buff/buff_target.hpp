#ifndef BUFF_TARGET_HPP
#define BUFF_TARGET_HPP

#include <algorithm>
#include <chrono>
#include <optional>
#include <random>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "tasks/auto_buff/buff_type.hpp"
#include "tools/extended_kalman_filter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_buff
{

class Voter
{
public:
  Voter();
  void vote(const double angle_last, const double angle_now);
  int clockwise() const;

private:
  int clockwise_;
};

class Target
{
public:
  struct FilterSnapshot
  {
    Eigen::VectorXd x;
    Eigen::MatrixXd P;
    double spd = 0.0;
  };

  Target();
  virtual ~Target() = default;

  virtual void get_target(
    const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp,
    EnergyType energy_type, const SinusoidalParam * sinusoidal_param = nullptr) = 0;

  virtual void predict(double dt) = 0;

  Eigen::Vector3d point_buff2world(const Eigen::Vector3d & point_in_buff) const;
  bool is_unsolve() const;
  virtual Eigen::VectorXd ekf_x() const = 0;
  virtual Eigen::MatrixXd ekf_P() const = 0;
  FilterSnapshot snapshot() const;
  void restore(const FilterSnapshot & snapshot);

  virtual int getLastHitRing() const { return last_hit_ring_; }
  virtual int getActivatedArms() const { return activated_arms_; }

  double spd;

protected:
  bool hasFiniteState() const;

  tools::ExtendedKalmanFilter ekf_;
  Voter voter;
  bool first_in_;
  bool unsolvable_;
  double lasttime_;

  int last_hit_ring_;
  int activated_arms_;

  Eigen::VectorXd x0_;
  Eigen::MatrixXd P0_;
  Eigen::MatrixXd A_;
  Eigen::MatrixXd Q_;
  Eigen::MatrixXd H_;
  Eigen::MatrixXd R_;

  std::random_device rd_;
  std::mt19937 gen_;
};

class SmallTarget : public Target
{
public:
  SmallTarget();

  void get_target(
    const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp,
    EnergyType energy_type, const SinusoidalParam * sinusoidal_param = nullptr) override;

  void predict(double dt) override;

  Eigen::VectorXd ekf_x() const override;
  Eigen::MatrixXd ekf_P() const override;

  static constexpr double SMALL_W = CV_PI / 3;

private:
  void init(double nowtime, const PowerRune & p);
  void update(double nowtime, const PowerRune & p);
  Eigen::MatrixXd h_jacobian() const;
};

class BigTarget : public Target
{
public:
  BigTarget();

  void get_target(
    const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp,
    EnergyType energy_type, const SinusoidalParam * sinusoidal_param = nullptr) override;

  void predict(double dt) override;

  Eigen::VectorXd ekf_x() const override { return ekf_.x; }
  Eigen::MatrixXd ekf_P() const override { return ekf_.P; }

  void setSinusoidalRange(
    double a_min, double a_max, double omega_min, double omega_max, double b_base);

private:
  double fit_spd_;
  int consecutive_diverge_count;

  SinusoidalParam param_;
  double activation_time_;

  double a_min_, a_max_;
  double omega_min_, omega_max_;
  double b_base_;

  bool param_fixed_;

  void init(double nowtime, const PowerRune & p);
  void update(double nowtime, const PowerRune & p);
  Eigen::MatrixXd h_jacobian() const;

  void generateRandomParam();
};

}  // namespace auto_buff

#endif  // BUFF_TARGET_HPP
