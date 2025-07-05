#include "ekf.hpp"
#include <cmath>

namespace ekf_standalone {

ExtendedKalmanFilter::ExtendedKalmanFilter(
  const Eigen::Vector4f& x0,
  const Eigen::Matrix4f& P0,
  const Eigen::Matrix4f& Q,
  const Eigen::Matrix2f& R
) : x_(x0), P_(P0), Q_(Q), R_(R) {}

Eigen::Vector4f ExtendedKalmanFilter::motionModel(const Eigen::Vector4f& x, 
                                                  const Eigen::Vector2f& u, float dt) {
  float px = x(0); 
  float py = x(1); 
  float theta = x(2);
  float v = u(0); 
  float omega = u(1);

  Eigen::Vector4f xp;
  xp(0) = px + v * std::cos(theta) * dt;  
  xp(1) = py + v * std::sin(theta) * dt;  
  xp(2) = theta + omega * dt;                
  xp(3) = v;                                 
  return xp;
}

Eigen::Matrix4f ExtendedKalmanFilter::jacobianF(const Eigen::Vector4f& x, 
                                                const Eigen::Vector2f& u, float dt) {
  float theta = x(2);
  float v = u(0);  
  Eigen::Matrix4f F = Eigen::Matrix4f::Identity();
  F(0,2) = -v * std::sin(theta) * dt;  
  F(0,3) = 0;                          
  F(1,2) =  v * std::cos(theta) * dt; 
  F(1,3) = 0;                          
  
  return F;
}

Eigen::Vector2f ExtendedKalmanFilter::observeModel(const Eigen::Vector4f& x) {
  return x.head<2>(); 
}

Eigen::Matrix<float,2,4> ExtendedKalmanFilter::jacobianH(const Eigen::Vector4f&) {
  Eigen::Matrix<float,2,4> H;
  H << 1,0,0,0,
       0,1,0,0;
  return H;
}

void ExtendedKalmanFilter::predict(const Eigen::Vector2f& u, float dt) {
  Eigen::Vector4f x_pred = motionModel(x_, u, dt);
  Eigen::Matrix4f F = jacobianF(x_, u, dt);
  Eigen::Matrix4f P_pred = F * P_ * F.transpose() + Q_;
  x_ = x_pred;  
  P_ = P_pred;
}

void ExtendedKalmanFilter::update(const Eigen::Vector2f& z) {
  Eigen::Vector2f z_pred = observeModel(x_);
  Eigen::Vector2f y      = z - z_pred;
  auto H = jacobianH(x_);
  Eigen::Matrix2f S = H * P_ * H.transpose() + R_;
  Eigen::Matrix<float,4,2> K = P_ * H.transpose() * S.inverse();
  x_ = x_ + K * y;
  P_ = (Eigen::Matrix4f::Identity() - K * H) * P_;
}

const Eigen::Vector4f& ExtendedKalmanFilter::state() const {
  return x_;
}

const Eigen::Matrix4f& ExtendedKalmanFilter::covariance() const {
  return P_;
}

} // namespace ekf_standalone
