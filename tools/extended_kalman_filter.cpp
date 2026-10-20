#include "extended_kalman_filter.hpp"

#include <numeric>

namespace tools
{
ExtendedKalmanFilter::ExtendedKalmanFilter(
  const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add)
: x(x0), P(P0), I(Eigen::MatrixXd::Identity(x0.rows(), x0.rows())), x_add(x_add)
{
  data["residual_yaw"] = 0.0;
  data["residual_pitch"] = 0.0;
  data["residual_distance"] = 0.0;
  data["residual_angle"] = 0.0;
  data["nis"] = 0.0;
  data["nees"] = 0.0;
  data["nis_fail"] = 0.0;
  data["nees_fail"] = 0.0;
  data["recent_nis_failures"] = 0.0;
}

Eigen::VectorXd ExtendedKalmanFilter::predict(const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q)
{
  return predict(F, Q, [&](const Eigen::VectorXd & x) { return F * x; });
}

Eigen::VectorXd ExtendedKalmanFilter::predict(
  const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f)
{
  P = F * P * F.transpose() + Q;
  x = f(x);
  return x;
}

Eigen::VectorXd ExtendedKalmanFilter::update(
  const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)
{
  return update(z, H, R, [&](const Eigen::VectorXd & x) { return H * x; }, z_subtract);
}

Eigen::VectorXd ExtendedKalmanFilter::update(
  const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> h,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)
{
  Eigen::VectorXd x_prior = x;
  Eigen::MatrixXd P_prior = P;
  Eigen::MatrixXd innovation_cov = H * P * H.transpose() + R;
  if (!innovation_cov.allFinite()) {
    return x;
  }

  Eigen::FullPivLU<Eigen::MatrixXd> innovation_lu(innovation_cov);
  if (!innovation_lu.isInvertible()) {
    return x;
  }

  Eigen::MatrixXd innovation_cov_inv = innovation_lu.inverse();
  if (!innovation_cov_inv.allFinite()) {
    return x;
  }

  Eigen::MatrixXd K = P * H.transpose() * innovation_cov_inv;
  if (!K.allFinite()) {
    return x;
  }

  // Stable Compution of the Posterior Covariance
  // https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb
  Eigen::MatrixXd P_posterior =
    (I - K * H) * P * (I - K * H).transpose() + K * R * K.transpose();
  if (!P_posterior.allFinite()) {
    return x;
  }

  Eigen::VectorXd innovation = z_subtract(z, h(x));
  Eigen::VectorXd x_posterior = x_add(x, K * innovation);
  if (!innovation.allFinite() || !x_posterior.allFinite()) {
    return x;
  }

  P = P_posterior;
  x = x_posterior;

  /// 卡方检验
  Eigen::VectorXd residual = z_subtract(z, h(x));
  if (!residual.allFinite()) {
    P = P_prior;
    x = x_prior;
    return x;
  }
  // 新增检验
  Eigen::MatrixXd S = H * P * H.transpose() + R;
  if (!S.allFinite()) {
    P = P_prior;
    x = x_prior;
    return x;
  }

  Eigen::FullPivLU<Eigen::MatrixXd> s_lu(S);
  Eigen::FullPivLU<Eigen::MatrixXd> p_lu(P);
  if (!s_lu.isInvertible() || !p_lu.isInvertible()) {
    return x;
  }

  Eigen::MatrixXd S_inv = s_lu.inverse();
  Eigen::MatrixXd P_inv = p_lu.inverse();
  if (!S_inv.allFinite() || !P_inv.allFinite()) {
    return x;
  }

  double nis = residual.transpose() * S_inv * residual;
  double nees = (x - x_prior).transpose() * P_inv * (x - x_prior);

  // 卡方检验阈值（自由度=4，取置信水平95%）
  constexpr double nis_threshold = 0.711;
  constexpr double nees_threshold = 0.711;

  if (nis > nis_threshold) nis_count_++, data["nis_fail"] = 1;
  if (nees > nees_threshold) nees_count_++, data["nees_fail"] = 1;
  total_count_++;
  last_nis = nis;

  recent_nis_failures.push_back(nis > nis_threshold ? 1 : 0);

  if (recent_nis_failures.size() > window_size) {
    recent_nis_failures.pop_front();
  }

  int recent_failures = std::accumulate(recent_nis_failures.begin(), recent_nis_failures.end(), 0);
  double recent_rate = static_cast<double>(recent_failures) / recent_nis_failures.size();

  data["residual_yaw"] = residual.size() > 0 ? residual[0] : 0.0;
  data["residual_pitch"] = residual.size() > 1 ? residual[1] : 0.0;
  data["residual_distance"] = residual.size() > 2 ? residual[2] : 0.0;
  data["residual_angle"] = residual.size() > 3 ? residual[3] : 0.0;
  data["nis"] = nis;
  data["nees"] = nees;
  data["recent_nis_failures"] = recent_rate;

  return x;
}

}  // namespace tools
