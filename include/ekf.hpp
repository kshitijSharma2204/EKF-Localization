#pragma once
#include <Eigen/Eigen>

namespace ekf_standalone {

/**
 * @brief 4-state Extended Kalman Filter: [x, y, θ, v]
 */
class ExtendedKalmanFilter {
public:
  explicit ExtendedKalmanFilter(
    const Eigen::Vector4f& x0,
    const Eigen::Matrix4f& P0,
    const Eigen::Matrix4f& Q,
    const Eigen::Matrix2f& R
  );

  /// Predict step: propagate with control u over dt
  void predict(const Eigen::Vector2f& u, float dt);

  /// Update step: fuse in measurement z
  void update(const Eigen::Vector2f& z);

  const Eigen::Vector4f& state() const;
  const Eigen::Matrix4f& covariance() const;

private:
  Eigen::Vector4f motionModel   (const Eigen::Vector4f& x, const Eigen::Vector2f& u, float dt);
  Eigen::Matrix4f jacobianF     (const Eigen::Vector4f& x, const Eigen::Vector2f& u, float dt);
  Eigen::Vector2f observeModel  (const Eigen::Vector4f& x);
  Eigen::Matrix<float,2,4> jacobianH(const Eigen::Vector4f& x);

  Eigen::Vector4f x_;
  Eigen::Matrix4f P_;
  Eigen::Matrix4f Q_;
  Eigen::Matrix2f R_;
};

} // namespace ekf_standalone
