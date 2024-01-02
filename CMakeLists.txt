cmake_minimum_required(VERSION 3.5...3.20)
project(Linear_Algebra)

include(FetchContent)
FetchContent_Declare(
  fast_matrix_market
  GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market.git
  GIT_TAG main
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(fast_matrix_market)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

FetchContent_Declare(
  Spectra
  URL https://codeload.github.com/yixuan/spectra/tar.gz/refs/tags/v1.0.1
)
FetchContent_MakeAvailable(Spectra)

set(MKL_ARCH intel64)
set(MKL_THREADING gnu_thread)
set(MKL_INTERFACE ilp64)

find_package(MKL CONFIG REQUIRED)
find_package(arpackng REQUIRED)

add_subdirectory(utils)
add_subdirectory(mkl_wrapper)

add_subdirectory(execs)
add_subdirectory(tests)