cmake_minimum_required(VERSION 3.0)
project(hybvio VERSION 1.0.0 DESCRIPTION "HybVIO VIO/SLAM runner")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Wextra -O2")
# allow building a single shared library (adds -fPIC when compiling static libs)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Generate `compile_commands.json` for `clang-tidy`.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FIND_BLAS "Force -DEIGEN_USE_BLAS." OFF)
option(MARCH_NATIVE "Enable -march=native." OFF)
if (MARCH_NATIVE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

if (NOT CMAKE_BUILD_TYPE)
  # TODO: setting this to RelWithDebInfo should be used for optimization flags
  # However: that seems to result to crashes for unknown reasons
  set(CMAKE_BUILD_TYPE Debug)
endif()

set(BUILD_CLI_TOOLS ON CACHE STRING "Build `main` binaries and tests.")
set(LOGGING_ENABLED ON CACHE STRING "Enable logging")

set(BUILD_WITH_GPU ON CACHE STRING "Build with GPU (OpenGL) support")

set(BUILD_WITH_GLFW ON CACHE STRING "Support headless GPU processing on desktop with GLFW")
set(BUILD_VISUALIZATIONS OFF CACHE STRING "Build Pangolin-based visualizations")
set(LOGGING_LIBS "")
if(BUILD_CLI_TOOLS AND LOGGING_ENABLED)
  add_definitions(-DUSE_LOGURU)
endif()
if(LOGGING_ENABLED)
  add_definitions(-DLOGGING_ENABLED)
endif()


# Make sure no Eigen LGPL code is used. It's mostly stuff like sparse matrix
# operations that we would not want to use through Eigen in any case
add_definitions("-DEIGEN_MPL2_ONLY")

# Do not auto-parallelize matrix products (mainly affects the Odometry class)
# Note that the odometry module may be faster without this flag, if also using
# OpenMP, but then it may hog 100% CPU (on all cores)
# from time to time, which is not what we're looking for. This may not have
# an effect unless Eigen is configured to use OpenMP
add_definitions("-DEIGEN_DONT_PARALLELIZE")

# Note: try these too for debugging. For example, NaN initialization should
# not affect the results if everything is implemented correctly.
#add_definitions("-DEIGEN_INITIALIZE_MATRICES_BY_NAN")
#add_definitions("-DEIGEN_NO_AUTOMATIC_RESIZING")

if (FIND_BLAS)
  # Use Accelerate framework on Mac or system BLAS on Linux for Eigen.
  # This may or may not be faster than without BLAS. Similarly to the OpenMP
  # stuff in Eigen (disabled with EIGEN_DONT_PARALLELIZE), this can cause
  # CPU hogging on multiple-threads and the end result may not be optimal.
  find_package(BLAS REQUIRED)
  if(BLAS_FOUND)
    message(STATUS "using BLAS with Eigen")
    add_definitions("-DEIGEN_USE_BLAS")
    list(APPEND LIBS "${BLAS_LIBRARIES}") # appends "-framework Accelerate"
  endif()
endif()

option(USE_SLAM "Compile with SLAM support" OFF)
set(TARGET_ARCH "host" CACHE STRING "Target architecture")
find_package(mobile-cv-suite REQUIRED PATHS 3rdparty/mobile-cv-suite)

add_subdirectory("codegen")
foreach(X ${generated_sources})
  message(STATUS "marking ${X} as a generated source file")
  set_source_files_properties(${X} PROPERTIES GENERATED TRUE)
endforeach()

add_library(parameters
  codegen/output/parameters.cpp
  src/util/util.cpp
  src/util/parameter_parser.cpp
  codegen/output/cmd_parameters.cpp
  src/util/util.cpp
  src/util/timer.cpp
  src/util/parameter_parser.cpp)
add_dependencies(parameters generated_parameters cmd_generated_parameters)
target_link_libraries(parameters mobile-cv-suite::core)

if (BUILD_WITH_GPU)
  add_definitions(-DDAZZLING_GPU_ENABLED)
  if (BUILD_WITH_GLFW)
    add_definitions(-DDAZZLING_GLFW_ENABLED)
    find_package(glfw3 REQUIRED)
    list(APPEND LIBS glfw)
  endif()
  find_package(OpenGL REQUIRED) # can't find GL ES :(
  list(APPEND LIBS ${OPENGL_LIBRARIES})
endif()

# This causes the targets in feature-tracking/CMakeList.txt to be available
# here without including the individual source files
add_subdirectory("src/tracker")

if (USE_SLAM)
  # add SLAM
  add_subdirectory("src/slam")
else()
  add_library(slam src/util/slam_noop.cpp)
  target_link_libraries(slam mobile-cv-suite::core) # Required for Eigen includes
endif()

# Static library that main program and tests can link to.
add_library(odometry
  src/odometry/control.cpp
  src/odometry/ekf.cpp
  src/odometry/triangulation.cpp
  src/odometry/util.cpp
  src/odometry/sample_sync.cpp
  src/odometry/ekf_state_index.cpp
  src/odometry/backend.cpp
  src/odometry/output.cpp
  src/views/api_visualization_helpers.cpp
  src/views/visualization_pose.cpp
  src/views/visualization_internals.cpp
  src/api/visualizations.cpp
  src/api/api.cpp
  src/util/util.cpp
  src/api/type_convert.cpp
  )

target_link_libraries(odometry slam tracker ${LIBS} mobile-cv-suite::core)
target_include_directories(odometry PUBLIC ".")

if(BUILD_CLI_TOOLS)
  if (BUILD_VISUALIZATIONS)
    add_definitions(-DBUILD_VISUALIZATIONS)
    set(VISU_SRCS
      src/commandline/visual_update_viewer.cpp
      src/commandline/draw_gl.cpp)
    set(VISU_LIBS
      Threads::Threads
      parameters
      ${LOGGING_LIBS}
      mobile-cv-suite::visualization
      ${OPENGL_LIBRARIES})
  else()
    set(VISU_SRCS "")
    set(VISU_LIBS "")
  endif()
  if(USE_SLAM)
    add_definitions(-DUSE_DBOW2)
    add_definitions(-DUSE_SLAM)

    set(SLAM_SRCS "")
    set(SLAM_LIBS
      Threads::Threads
      parameters
      mobile-cv-suite::slam)

    # Select newer OpenGL, see `cmake --help-policy CMP0072`
    # cmake_policy(SET CMP0072 NEW) # TODO what CMake version does this require?

    find_package(Threads REQUIRED)
  else()
    set(SLAM_LIBS "")
    set(SLAM_SRCS "")
  endif()

  add_executable(main
    src/commandline/command_queue.cpp
    src/commandline/input.cpp
    src/commandline/input_csv.cpp
    src/commandline/input_jsonl.cpp
    src/commandline/main.cpp
    src/commandline/video_input.cpp
    src/commandline/videoutil.cpp
    ${VISU_SRCS}
    ${SLAM_SRCS})
  target_link_libraries(main PRIVATE odometry parameters ${LIBS} ${VISU_LIBS} ${SLAM_LIBS})

  # Put binaries from subdirectories (test) to the target directory.
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

  # Copy test data to the target directory so that we can simply run the test binary.
  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/test)

  # Lets `make test` and `ctest` commands run tests.
  enable_testing()

  list(APPEND LIBS mobile-cv-suite::core) # applies to run-tests
  add_subdirectory(test)
endif()
