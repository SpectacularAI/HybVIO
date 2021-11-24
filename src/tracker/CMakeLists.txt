cmake_minimum_required(VERSION 3.0)
project(tracker)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# The core library, tracker
add_library(tracker
  camera.cpp
  image.cpp
  ptsetreg.cpp
  five_point.cpp
  rot_ransac.cpp
  tracker.cpp
  util.cpp
  gpu_util.cpp
  image_pyramid.cpp
  feature_detector.cpp
  feature_detector_legacy.cpp
  optical_flow.cpp
  ransac_pipeline.cpp
  stereo_disparity.cpp
  stereo_rectifier.cpp
  stereo_upright_2p.cpp
  subpixel_adjuster.cpp
  undistorter.cpp)
target_link_libraries(tracker
  parameters
  mobile-cv-suite::core)
