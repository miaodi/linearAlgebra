cmake_minimum_required(VERSION 3.5...3.20)
project(tests)

include(FetchContent)

# Add Google Test
include(../cmake/GoogleTest.cmake)

add_executable(
  mkl_sparse_mat
  mkl_sparse_mat_test.cpp
)

target_link_libraries(
  mkl_sparse_mat PUBLIC GTest::gtest_main mkl_wrapper
)

add_executable(
  circular_buffer
  circularbuffer_test.cpp
)

target_link_libraries(
  circular_buffer PUBLIC GTest::gtest_main GTest::gmock_main mkl_wrapper
)

add_executable(
  bfs
  bfs_test.cpp
)

target_link_libraries(
  bfs PUBLIC GTest::gtest_main GTest::gmock_main reordering
)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/data/ex5.mtx ${CMAKE_CURRENT_BINARY_DIR}/data/ex5.mtx COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/data/rdist1.mtx ${CMAKE_CURRENT_BINARY_DIR}/data/rdist1.mtx COPYONLY)

include(GoogleTest)
gtest_discover_tests(mkl_sparse_mat)
gtest_discover_tests(circular_buffer)
gtest_discover_tests(bfs)