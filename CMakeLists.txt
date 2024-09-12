cmake_minimum_required(VERSION 3.5...3.20)

project(Linear_Algebra CXX Fortran)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  # set(CMAKE_CXX_FLAGS "-g -O0 -Wall -Wextra -Wno-unused-variable -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer -DDEBUG")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "-Wno-unused-variable")
endif()
# set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_STANDARD 20)

option(USE_AMGCL_LIB "Use the amg library amgcl" OFF)

option(USE_MUMPS_LIB "Use the linear solver library mumps. Note that only non-mpi version is supported" OFF)

option(USE_ARPACK_LIB "Use arpack for generalize eigenvalue problem" OFF)

option(USE_METIS_LIB "Use metis for matrix reordering" OFF)

option(USE_BOOST_LIB "Use boost library" OFF)

option(USE_CUDA "Use cuda (cudss)" OFF)

option(BUILD_BENCHMARKS "Build benchmarks" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/morse_cmake/modules")
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
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

FetchContent_Declare(
  Spectra
  GIT_REPOSITORY https://github.com/yixuan/spectra.git
  GIT_TAG master
  GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(Spectra)
FetchContent_Declare(
  cereal
  GIT_REPOSITORY https://github.com/USCiLab/cereal.git
  GIT_TAG master
  GIT_SHALLOW TRUE
)

set(CMAKE_CXX_FLAGS_OLD "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-dangling-reference")
set(SKIP_PERFORMANCE_COMPARISON ON)
FetchContent_MakeAvailable(cereal)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_OLD}")
set(CMAKE_CXX_STANDARD 20)

FetchContent_Declare(
  AMGCL
  GIT_REPOSITORY https://github.com/ddemidov/amgcl.git
  GIT_TAG master
  GIT_SHALLOW TRUE
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # using GCC
  set(MKL_THREADING gnu_thread)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  # using Intel C++
  set(MKL_THREADING intel_thread)
endif()

set(MKL_ARCH intel64)
set(MKL_INTERFACE lp64)
set(MKL_LINK static)
find_package(MKL CONFIG REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)

find_package(LAPACK REQUIRED)

if(USE_AMGCL_LIB)
  FetchContent_MakeAvailable(AMGCL)
endif(USE_AMGCL_LIB)

if(USE_MUMPS_LIB)
  find_package(MUMPS REQUIRED COMPONENTS SEQ METIS OPENMP)
endif(USE_MUMPS_LIB)

if(USE_ARPACK_LIB)
  find_package(arpackng REQUIRED)
endif(USE_ARPACK_LIB)

if(USE_METIS_LIB)
  find_package(METIS REQUIRED)
endif(USE_METIS_LIB)

if(USE_BOOST_LIB)
  find_package(Boost REQUIRED)
endif(USE_BOOST_LIB)

if(USE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  find_package(cudss REQUIRED)
endif(USE_CUDA)

configure_file(
  "${CMAKE_SOURCE_DIR}/cmake/config.h.in"
  "${CMAKE_BINARY_DIR}/config.h")
set(CMAKE_INCLUDE_CURRENT_DIR ON)


add_subdirectory(utils)
add_subdirectory(sparse_mat_op)
add_subdirectory(mkl_wrapper)
add_subdirectory(reordering)

add_subdirectory(execs)
enable_testing()
add_subdirectory(tests)

if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif(BUILD_BENCHMARKS)
