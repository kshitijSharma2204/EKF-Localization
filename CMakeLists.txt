cmake_minimum_required(VERSION 3.5)
project(ekf_standalone)

find_package(OpenCV   REQUIRED)
find_package(Eigen3   REQUIRED NO_MODULE)

include_directories(include)

add_executable(ekf_demo
  src/main.cpp
  src/ekf.cpp
)
target_link_libraries(ekf_demo
  ${OpenCV_LIBS}
  Eigen3::Eigen
)
