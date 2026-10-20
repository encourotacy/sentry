#include "tasks/auto_buff/buff_target.hpp"

namespace auto_buff
{
namespace
{
bool hasValidMeasurement(const PowerRune & p)
{
  return p.ypd_in_world.allFinite() && p.ypr_in_world.allFinite() && p.blade_ypd_in_world.allFinite();
}
}  // namespace

Voter::Voter() : clockwise_(0) {}

void Voter::vote(const double angle_last, const double angle_now)
{
  if (std::abs(clockwise_) > 50) return;

  double diff = angle_now - angle_last;
  if (diff > CV_PI) diff -= 2 * CV_PI;
  if (diff < -CV_PI) diff += 2 * CV_PI;

  if (diff > 0)
    clockwise_++;
  else if (diff < 0)
    clockwise_--;
}

int Voter::clockwise() const { return clockwise_ > 0 ? 1 : -1; }

Target::Target()
  : first_in_(true), unsolvable_(true), spd(0.0), last_hit_ring_(0), activated_arms_(0), gen_(rd_())
{}

bool Target::hasFiniteState() const
{
  return ekf_.x.size() > 0 && ekf_.P.size() > 0 && ekf_.x.allFinite() && ekf_.P.allFinite();
}

Eigen::Vector3d Target::point_buff2world(const Eigen::Vector3d & point_in_buff) const
{
  if (unsolvable_ || !hasFiniteState()) return Eigen::Vector3d(0, 0, 0);

  Eigen::Matrix3d R_buff2world =
    tools::rotation_matrix(Eigen::Vector3d(ekf_.x[4], 0.0, ekf_.x[5]));

  auto R_yaw = ekf_.x[0];
  auto R_pitch = ekf_.x[2];
  auto R_dis = ekf_.x[3];

  Eigen::Vector3d point_in_world = R_buff2world * point_in_buff +
                                   Eigen::Vector3d(
                                     R_dis * std::cos(R_pitch) * std::cos(R_yaw),
                                     R_dis * std::cos(R_pitch) * std::sin(R_yaw),
                                     R_dis * std::sin(R_pitch));

  return point_in_world;
}

bool Target::is_unsolve() const { return unsolvable_; }
Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }
Eigen::MatrixXd Target::ekf_P() const { return ekf_.P; }

Target::FilterSnapshot Target::snapshot() const
{
  return FilterSnapshot{ekf_.x, ekf_.P, spd};
}

void Target::restore(const FilterSnapshot & snapshot)
{
  ekf_.x = snapshot.x;
  ekf_.P = snapshot.P;
  spd = snapshot.spd;
}

SmallTarget::SmallTarget() : Target()
{
  spd = SMALL_W;
}

void SmallTarget::get_target(
  const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp,
  EnergyType energy_type, const SinusoidalParam * sinusoidal_param)
{
  (void)sinusoidal_param;
  if (energy_type == EnergyType::BIG) {
    unsolvable_ = true;
    return;
  }

  static int lost_cn = 0;
  if (!p.has_value()) {
    unsolvable_ = true;
    lost_cn++;
    return;
  }

  if (!hasValidMeasurement(p.value())) {
    unsolvable_ = true;
    tools::logger()->warn("[SmallTarget] 输入观测量非法，跳过本帧");
    return;
  }

  last_hit_ring_ = p.value().hit_ring;
  activated_arms_ = p.value().activated_arms;

  static std::chrono::steady_clock::time_point start_timestamp = timestamp;
  auto time_gap = tools::delta_time(timestamp, start_timestamp);

  if (first_in_) {
    unsolvable_ = true;
    init(time_gap, p.value());
    first_in_ = false;
  }

  if (lost_cn > 10) {
    unsolvable_ = true;
    tools::logger()->debug("[SmallTarget] 丢失小符");
    lost_cn = 0;
    first_in_ = true;
    return;
  }

  unsolvable_ = false;
  update(time_gap, p.value());

  if (std::abs(ekf_.x[6]) > SMALL_W * 3.0 || std::abs(ekf_.x[6]) < SMALL_W * 0.1) {
    unsolvable_ = true;
    tools::logger()->debug("[SmallTarget] 小符角度发散 spd: {:.2f}", ekf_.x[6] * 180 / CV_PI);
    first_in_ = true;
    return;
  }

  lost_cn = 0;
}

void SmallTarget::predict(double dt)
{
  A_.resize(7, 7);
  A_ << 1.0, dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;

  auto v1 = 0.005;
  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;

  Q_.resize(7, 7);
  Q_ << a * v1, b * v1, 0.0, 0.0, 0.0, 0.0, 0.0, b * v1, c * v1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1;

  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = A_ * x;
    x_prior[0] = tools::limit_rad(x_prior[0]);
    x_prior[2] = tools::limit_rad(x_prior[2]);
    x_prior[4] = tools::limit_rad(x_prior[4]);
    x_prior[5] = tools::limit_rad(x_prior[5]);
    return x_prior;
  };

  ekf_.predict(A_, Q_, f);
}

void SmallTarget::init(double nowtime, const PowerRune & p)
{
  lasttime_ = nowtime;

  x0_.resize(7);
  P0_.resize(7, 7);
  A_.resize(7, 7);
  Q_.resize(7, 7);
  H_.resize(7, 7);
  R_.resize(7, 7);

  x0_ << p.ypd_in_world[0], 0.0, p.ypd_in_world[1], p.ypd_in_world[2], p.ypr_in_world[0],
    p.ypr_in_world[2], SMALL_W * voter.clockwise();

  P0_ << 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5;

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[0] = tools::limit_rad(c[0]);
    c[2] = tools::limit_rad(c[2]);
    c[4] = tools::limit_rad(c[4]);
    c[5] = tools::limit_rad(c[5]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0_, P0_, x_add);
}

void SmallTarget::update(double nowtime, const PowerRune & p)
{
  // if (nowtime - lasttime_ < 0 || nowtime - lasttime_ > 0.15 )
  //   tools::logger()->warn("[SmallTarget] Camera timestamp error! dt = {:.2f}", nowtime - lasttime_);
  //调试：检查相机时间戳间隔
  const Eigen::VectorXd & R_ypd = p.ypd_in_world;
  const Eigen::VectorXd & ypr = p.ypr_in_world;
  const Eigen::VectorXd & B_ypd = p.blade_ypd_in_world;

  if (abs(ypr[2] - ekf_.x[5]) > CV_PI / 12) {
    for (int i = -5; i <= 5; i++) {
      double angle_c = ekf_.x[5] + i * 2 * CV_PI / 5;
      if (std::fabs(angle_c - ypr[2]) < CV_PI / 5) {
        ekf_.x[5] += i * 2 * CV_PI / 5;
        break;
      }
    }
  }

  voter.vote(ekf_.x[5], ypr[2]);
  if (voter.clockwise() * ekf_.x[6] < 0) ekf_.x[6] *= -1;

  predict(nowtime - lasttime_);
  if (!hasFiniteState()) {
    unsolvable_ = true;
    first_in_ = true;
    tools::logger()->warn("[SmallTarget] predict后状态非法，重置跟踪器");
    return;
  }

  Eigen::MatrixXd H1(4, 7);
  H1 << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;

  Eigen::MatrixXd R1(4, 4);
  R1 << 0.02, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.15;

  auto z_subtract1 = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  Eigen::VectorXd z1(4);
  z1 << R_ypd[0], R_ypd[1], R_ypd[2], ypr[2];
  ekf_.update(z1, H1, R1, z_subtract1);
  if (!hasFiniteState()) {
    unsolvable_ = true;
    first_in_ = true;
    tools::logger()->warn("[SmallTarget] 一次观测更新后状态非法，重置跟踪器");
    return;
  }

  Eigen::MatrixXd H2 = h_jacobian();

  Eigen::MatrixXd R2(3, 3);
  R2 << 0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.3;

  auto h2 = [&](const Eigen::VectorXd & x) -> Eigen::Vector3d {
    Eigen::VectorXd B_xyz = point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.7));
    Eigen::VectorXd B_ypd = tools::xyz2ypd(B_xyz);
    return B_ypd;
  };

  auto z_subtract2 = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    return c;
  };

  Eigen::VectorXd z2(3);
  z2 << B_ypd[0], B_ypd[1], B_ypd[2];
  ekf_.update(z2, H2, R2, h2, z_subtract2);
  if (!hasFiniteState()) {
    unsolvable_ = true;
    first_in_ = true;
    tools::logger()->warn("[SmallTarget] 二次观测更新后状态非法，重置跟踪器");
    return;
  }

  this->spd = ekf_.x[6];
  lasttime_ = nowtime;
}

Eigen::MatrixXd SmallTarget::h_jacobian() const
{
  Eigen::MatrixXd H0(5, 7);
  H0 << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 0.0;

  Eigen::VectorXd R_ypd(3);
  R_ypd << ekf_.x[0], ekf_.x[2], ekf_.x[3];
  Eigen::MatrixXd H_ypd2xyz = tools::ypd2xyz_jacobian(R_ypd);

  Eigen::MatrixXd H1(3, 5);
  H1 << H_ypd2xyz(0, 0), H_ypd2xyz(0, 1), H_ypd2xyz(0, 2), 0.0, 0.0, H_ypd2xyz(1, 0),
    H_ypd2xyz(1, 1), H_ypd2xyz(1, 2), 0.0, 0.0, H_ypd2xyz(2, 0), H_ypd2xyz(2, 1),
    H_ypd2xyz(2, 2), 0.0, 0.0;

  double yaw = ekf_.x[4];
  double roll = ekf_.x[5];
  double cos_yaw = cos(yaw);
  double sin_yaw = sin(yaw);
  double cos_roll = cos(roll);
  double sin_roll = sin(roll);

  Eigen::MatrixXd H2(3, 5);
  H2 << 1.0, 0.0, 0.0, 0.7 * cos_yaw * sin_roll, 0.7 * sin_yaw * cos_roll, 0.0, 1.0, 0.0,
    0.7 * sin_yaw * sin_roll, -0.7 * cos_yaw * cos_roll, 0.0, 0.0, 1.0, 0.0, -0.7 * sin_roll;

  Eigen::VectorXd B_xyz = point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.7));
  Eigen::MatrixXd H3 = tools::xyz2ypd_jacobian(B_xyz);

  return H3 * H2 * H1 * H0;
}

Eigen::VectorXd SmallTarget::ekf_x() const { return ekf_.x; }
Eigen::MatrixXd SmallTarget::ekf_P() const { return ekf_.P; }

BigTarget::BigTarget()
  : Target(), fit_spd_(0.0), consecutive_diverge_count(0), activation_time_(0.0), param_fixed_(false)
{
  a_min_ = 0.78;
  a_max_ = 1.045;
  omega_min_ = 1.884;
  omega_max_ = 2.0;
  b_base_ = 2.090;
}

void BigTarget::setSinusoidalRange(
  double a_min, double a_max, double omega_min, double omega_max, double b_base)
{
  a_min_ = a_min;
  a_max_ = a_max;
  omega_min_ = omega_min;
  omega_max_ = omega_max;
  b_base_ = b_base;
}

void BigTarget::generateRandomParam()
{
  std::uniform_real_distribution<> a_dist(a_min_, a_max_);
  std::uniform_real_distribution<> omega_dist(omega_min_, omega_max_);

  param_.a = a_dist(gen_);
  param_.omega = omega_dist(gen_);
  param_.b = b_base_ - param_.a;
  param_.phi = 0.0;

  param_fixed_ = true;
  tools::logger()->debug(
    "[BigTarget] 生成正弦参数 a={:.3f}, ω={:.3f}, b={:.3f}", param_.a, param_.omega, param_.b);
}

void BigTarget::get_target(
  const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp,
  EnergyType energy_type, const SinusoidalParam * sinusoidal_param)
{
  static int lost_cn = 0;

  if (energy_type == EnergyType::SMALL) {
    unsolvable_ = true;
    return;
  }

  if (!p.has_value()) {
    unsolvable_ = true;
    lost_cn++;
    return;
  }

  if (!hasValidMeasurement(p.value())) {
    unsolvable_ = true;
    tools::logger()->warn("[BigTarget] 输入观测量非法，跳过本帧");
    return;
  }

  last_hit_ring_ = p.value().hit_ring;
  activated_arms_ = p.value().activated_arms;

  static std::chrono::steady_clock::time_point start_timestamp = timestamp;
  auto time_gap = tools::delta_time(timestamp, start_timestamp);

  if (sinusoidal_param != nullptr && !param_fixed_) {
    param_ = *sinusoidal_param;
    param_fixed_ = true;
    activation_time_ = time_gap;
    tools::logger()->debug("[BigTarget] 使用Solver提供的正弦参数");
  }

  if (first_in_) {
    unsolvable_ = true;

    if (p.value().ypd_in_world[2] > 3.0 && p.value().ypd_in_world[2] < 8.0) {
      init(time_gap, p.value());
      first_in_ = false;
      consecutive_diverge_count = 0;

      if (!param_fixed_) {
        generateRandomParam();
        activation_time_ = time_gap;
      }

      tools::logger()->debug("[BigTarget] 初始化成功, R_dis: {:.2f}", p.value().ypd_in_world[2]);
    } else {
      tools::logger()->debug("[BigTarget] 初始化数据不合理, R_dis: {:.2f}", p.value().ypd_in_world[2]);
      return;
    }
  }

  if (lost_cn > 20) {
    unsolvable_ = true;
    tools::logger()->debug("[BigTarget] 丢失大符");
    lost_cn = 0;
    first_in_ = true;
    param_fixed_ = false;
    consecutive_diverge_count = 0;
    return;
  }

  unsolvable_ = false;
  update(time_gap, p.value());

  bool diverged = false;

  if (ekf_.x[7] > a_max_ + 0.2 || ekf_.x[7] < a_min_ - 0.2) {
    diverged = true;
    tools::logger()->debug("[BigTarget] a发散: {:.3f}", ekf_.x[7]);
  }
  if (ekf_.x[8] > omega_max_ + 0.2 || ekf_.x[8] < omega_min_ - 0.2) {
    diverged = true;
    tools::logger()->debug("[BigTarget] ω发散: {:.3f}", ekf_.x[8]);
  }

  if (diverged) {
    if (consecutive_diverge_count++ < 15) {
      ekf_.x[7] = std::clamp(ekf_.x[7], a_min_, a_max_);
      ekf_.x[8] = std::clamp(ekf_.x[8], omega_min_, omega_max_);
      ekf_.P(7, 7) *= 3.0;
      ekf_.P(8, 8) *= 3.0;
    } else {
      tools::logger()->debug("[BigTarget] 连续发散，重新初始化");
      first_in_ = true;
      param_fixed_ = false;
      consecutive_diverge_count = 0;
    }
    return;
  }

  consecutive_diverge_count = std::max(0, consecutive_diverge_count - 1);
  lost_cn = 0;
}

void BigTarget::predict(double dt)
{
  double a = ekf_.x[7];
  double w = ekf_.x[8];
  double fi = ekf_.x[9];
  double t = lasttime_ + dt - activation_time_;
  double t_last = lasttime_ - activation_time_;

  A_ = Eigen::MatrixXd::Identity(10, 10);
  A_(0, 1) = dt;
  A_(5, 6) = voter.clockwise() * dt;
  A_(5, 7) =
    voter.clockwise() *
    ((-1.0 / w) * (std::cos(w * t + fi) - std::cos(w * t_last + fi)) - (t - t_last));
  A_(5, 8) = voter.clockwise() *
             (a / (w * w) * (std::cos(w * t + fi) - std::cos(w * t_last + fi)) +
              (-a / w) * (-std::sin(w * t + fi) * t + std::sin(w * t_last + fi) * t_last));
  A_(5, 9) = voter.clockwise() * ((a / w) * (std::sin(w * t + fi) - std::sin(w * t_last + fi)));
  A_(6, 7) = std::sin(w * t + fi) - 1.0;
  A_(6, 8) = a * std::cos(w * t + fi) * t;
  A_(6, 9) = a * std::cos(w * t + fi);

  double adapt_factor = 1.0;
  if (ekf_.P(7, 7) > 100.0 || ekf_.P(8, 8) > 100.0) {
    adapt_factor = 2.0;
  }

  auto v1 = 3.0 * adapt_factor;
  auto a1 = dt * dt * dt * dt / 4;
  auto b1 = dt * dt * dt / 2;
  auto c1 = dt * dt;

  Q_ = Eigen::MatrixXd::Zero(10, 10);
  Q_(0, 0) = a1 * v1;
  Q_(0, 1) = b1 * v1;
  Q_(1, 0) = b1 * v1;
  Q_(1, 1) = c1 * v1;
  Q_(2, 2) = 0.02;
  Q_(3, 3) = 0.25 * adapt_factor;
  Q_(4, 4) = 0.05 * adapt_factor;
  Q_(5, 5) = 8.0 * adapt_factor;
  Q_(6, 6) = 1.0 * adapt_factor;
  Q_(7, 7) = 0.05 * adapt_factor;
  Q_(8, 8) = 0.02 * adapt_factor;
  Q_(9, 9) = 6.0 * adapt_factor;

  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = x;
    double t_now = lasttime_ + dt - activation_time_;
    double t_prev = lasttime_ - activation_time_;

    x_prior[0] = tools::limit_rad(x[0] + dt * x[1]);
    x_prior[1] = x[1];
    x_prior[2] = tools::limit_rad(x[2]);
    x_prior[3] = x[3];
    x_prior[4] = tools::limit_rad(x[4]);

    double a = x[7];
    double w = x[8];
    double fi = x[9];

    double angle_increment = voter.clockwise() *
                             ((-a / w) * (std::cos(w * t_now + fi) - std::cos(w * t_prev + fi)) +
                              (b_base_ - a) * (t_now - t_prev));

    x_prior[5] = tools::limit_rad(x[5] + angle_increment);
    x_prior[6] = a * std::sin(w * t_now + fi) + (b_base_ - a);
    x_prior[7] = a;
    x_prior[8] = w;
    x_prior[9] = tools::limit_rad(fi);

    return x_prior;
  };

  ekf_.predict(A_, Q_, f);

  ekf_.x[7] = std::clamp(ekf_.x[7], a_min_, a_max_);
  ekf_.x[8] = std::clamp(ekf_.x[8], omega_min_, omega_max_);
  ekf_.x[9] = tools::limit_rad(ekf_.x[9]);
}

void BigTarget::init(double nowtime, const PowerRune & p)
{
  lasttime_ = nowtime;
  unsolvable_ = true;

  x0_.resize(10);
  P0_.resize(10, 10);
  A_.resize(10, 10);
  Q_.resize(10, 10);
  H_.resize(7, 10);
  R_.resize(7, 7);

  double a_init = (a_min_ + a_max_) / 2.0;
  double omega_init = (omega_min_ + omega_max_) / 2.0;

  x0_ << p.ypd_in_world[0], 0.0, p.ypd_in_world[1], p.ypd_in_world[2], p.ypr_in_world[0],
    p.ypr_in_world[2], std::abs(p.ypr_in_world[2]) * 2.09, a_init, omega_init, 0.0;

  P0_ << 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 150.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 500.0;

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[0] = tools::limit_rad(c[0]);
    c[2] = tools::limit_rad(c[2]);
    c[4] = tools::limit_rad(c[4]);
    c[5] = tools::limit_rad(c[5]);
    c[9] = tools::limit_rad(c[9]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0_, P0_, x_add);
}

void BigTarget::update(double nowtime, const PowerRune & p)
{
  const Eigen::VectorXd & R_ypd = p.ypd_in_world;
  const Eigen::VectorXd & ypr = p.ypr_in_world;
  const Eigen::VectorXd & B_ypd = p.blade_ypd_in_world;

  if (abs(ypr[2] - ekf_.x[5]) > CV_PI / 12) {
    for (int i = -5; i <= 5; i++) {
      double angle_c = ekf_.x[5] + i * 2 * CV_PI / 5;
      if (std::fabs(angle_c - ypr[2]) < CV_PI / 5) {
        ekf_.x[5] += i * 2 * CV_PI / 5;
        break;
      }
    }
  }

  voter.vote(ekf_.x[5], ypr[2]);
  auto anglelast = ekf_.x[5];

  predict(nowtime - lasttime_);
  if (!hasFiniteState()) {
    unsolvable_ = true;
    first_in_ = true;
    param_fixed_ = false;
    consecutive_diverge_count = 0;
    tools::logger()->warn("[BigTarget] predict后状态非法，重置跟踪器");
    return;
  }

  Eigen::MatrixXd H1(4, 10);
  H1 << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0, 0.0;

  Eigen::MatrixXd R1(4, 4);
  R1 << 0.08, 0.0, 0.0, 0.0, 0.0, 0.08, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.3;

  auto z_subtract1 = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  Eigen::VectorXd z1(4);
  z1 << R_ypd[0], R_ypd[1], R_ypd[2], ypr[2];
  ekf_.update(z1, H1, R1, z_subtract1);
  if (!hasFiniteState()) {
    unsolvable_ = true;
    first_in_ = true;
    param_fixed_ = false;
    consecutive_diverge_count = 0;
    tools::logger()->warn("[BigTarget] 一次观测更新后状态非法，重置跟踪器");
    return;
  }

  Eigen::MatrixXd H2 = h_jacobian();

  Eigen::MatrixXd R2(3, 3);
  R2 << 0.08, 0.0, 0.0, 0.0, 0.08, 0.0, 0.0, 0.0, 1.5;

  auto h2 = [&](const Eigen::VectorXd & x) -> Eigen::Vector3d {
    Eigen::VectorXd B_xyz = point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.7));
    Eigen::VectorXd B_ypd = tools::xyz2ypd(B_xyz);
    return B_ypd;
  };

  auto z_subtract2 = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    return c;
  };

  Eigen::VectorXd z2(3);
  z2 << B_ypd[0], B_ypd[1], B_ypd[2];
  ekf_.update(z2, H2, R2, h2, z_subtract2);
  if (!hasFiniteState()) {
    unsolvable_ = true;
    first_in_ = true;
    param_fixed_ = false;
    consecutive_diverge_count = 0;
    tools::logger()->warn("[BigTarget] 二次观测更新后状态非法，重置跟踪器");
    return;
  }

  double angle_now = ekf_.x[5];
  double dt = nowtime - lasttime_;
  if (dt > 1e-6) {
    this->spd = voter.clockwise() * (angle_now - anglelast) / dt;
    if (std::abs(this->spd) > 8.0 || std::abs(this->spd) < 0.1) {
      this->spd = ekf_.x[6];
    }
  }

  last_hit_ring_ = p.hit_ring;
  activated_arms_ = p.activated_arms;
  lasttime_ = nowtime;
  unsolvable_ = false;
}

Eigen::MatrixXd BigTarget::h_jacobian() const
{
  Eigen::MatrixXd H0(5, 10);
  H0 << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;

  Eigen::VectorXd R_ypd(3);
  R_ypd << ekf_.x[0], ekf_.x[2], ekf_.x[3];
  Eigen::MatrixXd H_ypd2xyz = tools::ypd2xyz_jacobian(R_ypd);

  Eigen::MatrixXd H1(3, 5);
  H1 << H_ypd2xyz(0, 0), H_ypd2xyz(0, 1), H_ypd2xyz(0, 2), 0.0, 0.0, H_ypd2xyz(1, 0),
    H_ypd2xyz(1, 1), H_ypd2xyz(1, 2), 0.0, 0.0, H_ypd2xyz(2, 0), H_ypd2xyz(2, 1),
    H_ypd2xyz(2, 2), 0.0, 0.0;

  double yaw = ekf_.x[4];
  double roll = ekf_.x[5];
  double cos_yaw = cos(yaw);
  double sin_yaw = sin(yaw);
  double cos_roll = cos(roll);
  double sin_roll = sin(roll);

  Eigen::MatrixXd H2(3, 5);
  H2 << 1.0, 0.0, 0.0, 0.7 * cos_yaw * sin_roll, 0.7 * sin_yaw * cos_roll, 0.0, 1.0, 0.0,
    0.7 * sin_yaw * sin_roll, -0.7 * cos_yaw * cos_roll, 0.0, 0.0, 1.0, 0.0, -0.7 * sin_roll;

  Eigen::VectorXd B_xyz = point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.7));
  Eigen::MatrixXd H3 = tools::xyz2ypd_jacobian(B_xyz);

  return H3 * H2 * H1 * H0;
}

}  // namespace auto_buff
