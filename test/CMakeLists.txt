set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

set(TARGET_NAME run-tests)

add_executable(${TARGET_NAME}
  test_main.cpp
  camera.cpp
  ekf.cpp
  parameter_parser.cpp
  sample_sync.cpp
  triangulation.cpp
  util.cpp
  helpers.cpp
  allocator.cpp)

target_link_libraries(${TARGET_NAME} odometry ${LIBS})
add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
