# Extended Kalman Filter Standalone Demo

This repository contains a pure C++ implementation of an Extended Kalman Filter (EKF)  
with a three-phase synthetic trajectory (sinusoid, straight, semicircle),  
live OpenCV visualization, and optional MP4 recording.

<p align="left">
  <img src="output/ekf_demo.gif" alt="EKF Demo" width="500"/>
</p>

## Getting Started

### Dependencies

- C++17 compiler  
- [Eigen3](https://eigen.tuxfamily.org/)  
- [OpenCV](https://opencv.org/)  

### Build

```bash
git clone [<your-repo-url>](https://github.com/kshitijSharma2204/EKF-Localization.git)
cd EKF_Localization
mkdir build && cd build
cmake ..
make
```

## Run the Demo

* Live Only - inside `build/`
  ```bash
  ./ekf_demo
  ```

* Live + Record (MP4) - inside `build/`
  ```bash
  ./ekf_demo --record
  ```
