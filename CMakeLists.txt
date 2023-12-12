cmake_minimum_required(VERSION 3.15)
project(Linear_Algebra)

include(FetchContent)
FetchContent_Declare(
  fast_matrix_market
  GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market.git
  GIT_TAG main
  OVERRIDE_FIND_PACKAGE
)
# The following will automatically forward through to FetchContent_MakeAvailable()
# find_package(fast_matrix_market)
FetchContent_MakeAvailable(fast_matrix_market)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)


set(MKL_ARCH intel64)
set(MKL_THREADING gnu_thread)
set(MKL_INTERFACE ilp64)
find_package(MKL CONFIG REQUIRED)

add_executable(test main.cpp)
target_link_libraries(test PRIVATE fast_matrix_market::fast_matrix_market Eigen3::Eigen)