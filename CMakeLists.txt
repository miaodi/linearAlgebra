cmake_minimum_required(VERSION 3.5...3.20)

project(Linear_Algebra CXX Fortran)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/morse_cmake/modules" )
# Include the init module
include(MorseInit)


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
  GIT_REPOSITORY https://github.com/yixuan/spectra.git
  GIT_TAG master
  GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(Spectra)

set(MKL_ARCH intel64)
set(MKL_THREADING gnu_thread)
set(MKL_INTERFACE lp64)
set(ARPACKNG_DIR)
find_package(MKL CONFIG REQUIRED)
find_package(TBB REQUIRED)
find_package(arpackng REQUIRED PATHS ${ARPACKNG_DIR})
find_package(OpenMP REQUIRED)

find_package(LAPACK REQUIRED)
set(MUMPS_DIR)
find_package(MUMPS REQUIRED COMPONENTS SEQ METIS OPENMP)

message("MUMPS_LIBRARIES: ${MUMPS_LIBRARIES}")

add_subdirectory(utils)
add_subdirectory(mkl_wrapper)
add_subdirectory(reordering)

add_subdirectory(execs)
enable_testing()
add_subdirectory(tests)

add_subdirectory(benchmarks)