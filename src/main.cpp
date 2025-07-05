#include "ekf.hpp"
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>
#include <iostream>
#include <algorithm>
#include <limits>
#include <cmath>

using namespace ekf_standalone;

/**
 * @brief Simulates the unicycle motion model:
 *        x₊ = x + v · cos(θ) · dt
 *        y₊ = y + v · sin(θ) · dt
 *        θ₊ = θ + ω · dt
 *        v₊ = v
 *
 * @param x     Current state [x, y, θ, v].
 * @param u     Control input [v, ω].
 * @param dt    Time step.
 * @return      Next state.
 */
Eigen::Vector4f motionModel(
  const Eigen::Vector4f& x,
  const Eigen::Vector2f& u,
  float dt
) {
  float px    = x(0), py    = x(1);
  float theta = x(2);
  float v     = u(0),      // commanded forward speed
        omega = u(1);      // commanded yaw rate

  Eigen::Vector4f xp;
  xp(0) = px    + v * std::cos(theta) * dt;
  xp(1) = py    + v * std::sin(theta) * dt;
  xp(2) = theta + omega      * dt;
  xp(3) = v;  // maintain speed
  return xp;
}

/**
 * @brief Draws all trajectories (ground truth, odometry, observations, EKF estimate)
 *        up to the current frame, automatically scaling and centering them to fit.
 *
 * @param canvas     OpenCV image to draw on.
 * @param gt         Ground truth points.
 * @param dr         Dead‐reckoned (odometry) points.
 * @param obs        Noisy observation points.
 * @param ekf_pts    EKF estimated points.
 * @param outScale   Output scale (world→pixel).
 * @param outOffset  Output offset (world→pixel).
 * @param frame_idx  Number of points to draw.
 */
void drawTrajectories(
  cv::Mat &canvas,
  const std::vector<cv::Point2f>& gt,
  const std::vector<cv::Point2f>& dr,
  const std::vector<cv::Point2f>& obs,
  const std::vector<cv::Point2f>& ekf_pts,
  cv::Point2f &outScale,
  cv::Point2f &outOffset,
  int frame_idx
) {
  canvas.setTo(cv::Scalar(255,255,255));

  // 1) Compute axis-aligned bounding box of all data points so far
  float min_x =  std::numeric_limits<float>::infinity();
  float min_y =  std::numeric_limits<float>::infinity();
  float max_x = -std::numeric_limits<float>::infinity();
  float max_y = -std::numeric_limits<float>::infinity();

  auto updateBounds = [&](const cv::Point2f &p) {
    min_x = std::min(min_x, p.x);
    max_x = std::max(max_x, p.x);
    min_y = std::min(min_y, p.y);
    max_y = std::max(max_y, p.y);
  };

  auto scan = [&](const std::vector<cv::Point2f>& pts) {
    int N = std::min((int)pts.size(), frame_idx);
    for (int i = 0; i < N; ++i) updateBounds(pts[i]);
  };

  scan(gt);   scan(dr);   scan(obs);   scan(ekf_pts);

  // 2) Determine scale and offset to fit all points with a margin
  const float margin = 20.0f;
  float sx = (canvas.cols - 2 * margin) / (max_x - min_x);
  float sy = (canvas.rows - 2 * margin) / (max_y - min_y);
  float s  = std::min(sx, sy);

  cv::Point2f offset(
    -min_x * s + margin,
    -min_y * s + margin
  );
  outScale  = cv::Point2f(s, s);
  outOffset = offset;

  // 3) World→pixel mapping (flip Y so up is positive)
  auto toCv = [&](const cv::Point2f &p) {
    float px = p.x * s + offset.x;
    float py = canvas.rows - (p.y * s + offset.y);
    return cv::Point2f(px, py);
  };

  // 4) Draw lines for each trajectory
  auto drawPath = [&](const std::vector<cv::Point2f>& pts,
                      cv::Scalar col, int thickness) {
    int N = std::min((int)pts.size(), frame_idx);
    for (int i = 1; i < N; ++i) {
      cv::line(canvas,
               toCv(pts[i-1]),
               toCv(pts[i]),
               col, thickness);
    }
  };
  drawPath(gt,       {255,   0, 255}, 2);  // magenta = ground truth
  drawPath(dr,       {  0, 255,   0}, 2);  // green   = odometry
  drawPath(ekf_pts,  {  0, 165, 255}, 2);  // orange  = EKF estimate

  // 5) Draw observations as filled circles
  int M = std::min((int)obs.size(), frame_idx);
  for (int i = 0; i < M; ++i) {
    cv::circle(canvas, toCv(obs[i]), 2, {255,200,0}, -1);  // yellow
  }
}

/**
 * @brief Runs the EKF simulation, displays a live window, and optionally records MP4.
 *
 * @param recordVideo  If true, writes `ekf_demo.mp4` alongside live display.
 */
void runSimulation(bool recordVideo)
{
  // Simulation parameters
  const float dt       = 0.1f;
  const float sim_time = 30.0f;

  // Random‐noise generators
  std::mt19937 gen{42};
  std::normal_distribution<float> noise(0,1);
  float odom_std_v = 0.1f, odom_std_w = 0.05f;
  float meas_std   = 0.3f;

  // EKF initialization
  Eigen::Vector4f x_gt  = Eigen::Vector4f::Zero();
  Eigen::Vector4f x_dr  = Eigen::Vector4f::Zero();
  Eigen::Vector4f x_est = Eigen::Vector4f::Zero();
  Eigen::Matrix4f P0    = Eigen::Matrix4f::Identity() * 0.1f;
  Eigen::Matrix4f Q     = Eigen::Matrix4f::Identity() * 0.01f;
  Eigen::Matrix2f R     = Eigen::Matrix2f::Identity() * (meas_std*meas_std);
  ExtendedKalmanFilter ekf{x_est, P0, Q, R};

  // Data storage
  std::vector<cv::Point2f> pts_gt, pts_dr, pts_obs, pts_est;
  int total_frames = int(sim_time / dt);

  // Optional video writer setup
  cv::VideoWriter writer;
  if (recordVideo) {
    writer.open("ekf_demo.mp4",
                cv::VideoWriter::fourcc('m','p','4','v'),
                1.0/dt,
                {800,800},
                true);
    if (!writer.isOpened()) {
      std::cerr << "ERROR: could not open video writer, disabling recording\n";
      recordVideo = false;
    }
  }

  cv::Mat canvas(800,800,CV_8UC3);
  for (int frame = 0; frame < total_frames; ++frame) {
    float t = frame * dt;

    // --- Phase‐based control: sinusoid → straight down → semicircle ---
    Eigen::Vector2f u_phase;
    if (t < 5.0f) {
      // oscillating turn‐rate
      u_phase = {1.0f, std::sin(2.0f * M_PI * 0.5f * t)};
    } else if (t < 10.0f) {
      static bool turned = false;
      if (!turned) {
        // snap heading to -90°, then go straight
        float desired = -M_PI/2, curr = x_gt(2);
        u_phase = {1.0f, (desired - curr)/dt};
        turned = true;
      } else {
        u_phase = {1.0f, 0.0f};
      }
    } else {
      // half‐circle at constant turn‐rate
      u_phase = {1.0f, float(M_PI/10.0f)};
    }

    // 1) Simulate true motion
    x_gt = motionModel(x_gt, u_phase, dt);
    pts_gt.emplace_back(x_gt(0), x_gt(1));

    // 2) Simulate noisy odometry
    Eigen::Vector2f u_noisy {
      u_phase(0) + noise(gen)*odom_std_v,
      u_phase(1) + noise(gen)*odom_std_w
    };
    x_dr = motionModel(x_dr, u_noisy, dt);
    pts_dr.emplace_back(x_dr(0), x_dr(1));

    // 3) Simulate noisy measurement
    Eigen::Vector2f z {
      x_gt(0) + noise(gen)*meas_std,
      x_gt(1) + noise(gen)*meas_std
    };
    pts_obs.emplace_back(z(0), z(1));

    // 4) EKF predict & update
    ekf.predict(u_noisy, dt);
    ekf.update(z);
    auto xk = ekf.state();
    pts_est.emplace_back(xk(0), xk(1));

    // 5) Draw all trajectories with dynamic scaling
    cv::Point2f scale, offset;
    drawTrajectories(canvas,
                     pts_gt, pts_dr, pts_obs, pts_est,
                     scale, offset,
                     frame + 1);

    // 6) Draw covariance ellipse in red
    {
      auto P = ekf.covariance();
      float Pxx = P(0,0), Pyy = P(1,1), Pxy = P(0,1);
      float sx_lin = std::sqrt(Pxx), sy_lin = std::sqrt(Pyy);
      float ang = 0.5f * std::atan2(2*Pxy, Pxx-Pyy) * 180.0f / CV_PI;
      int ax = std::max(int(sx_lin*scale.x), 5),
          ay = std::max(int(sy_lin*scale.y), 5);
      cv::Point2f center(
        xk(0)*scale.x + offset.x,
        canvas.rows - (xk(1)*scale.y + offset.y)
      );
      cv::ellipse(canvas, center, cv::Size(ax, ay),
                  ang, 0, 360,
                  cv::Scalar(0, 0, 255), 2);
    }

    // 7) Display & record
    cv::imshow("EKF Demo", canvas);
    if (recordVideo) writer.write(canvas);
    if (cv::waitKey(int(dt * 1000)) == 27) break;
  }

  if (recordVideo) {
    writer.release();
    std::cout << "Saved ekf_demo.mp4\n";
  }
  cv::destroyAllWindows();
}

/**
 * @brief Entry point: pass `--record` to save MP4.
 */
int main(int argc, char** argv)
{
  bool record = (argc > 1 && std::string(argv[1]) == "--record");
  runSimulation(record);
  return 0;
}
