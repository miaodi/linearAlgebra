cmake_minimum_required(VERSION 3.5...3.20)

# Set compilers before project() to ensure proper detection
# Note: These can be overridden via environment variables (CC, CXX) or cmake -D flags
if(DEFINED ENV{CC})
  set(CMAKE_C_COMPILER $ENV{CC})
endif()
if(DEFINED ENV{CXX})
  set(CMAKE_CXX_COMPILER $ENV{CXX})
endif()

project(Linear_Algebra CXX Fortran)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -fno-strict-aliasing -Wno-unused-variable")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "-Wno-unused-variable")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "-Wno-unused-variable")
  add_compile_definitions(NDEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set(CMAKE_CXX_FLAGS "-Wno-unused-variable -O2 -g -fno-omit-frame-pointer")
endif()

option(USE_AMGCL_LIB "Use the amg library amgcl" OFF)

option(USE_MUMPS_LIB "Use the linear solver library mumps. Note that only non-mpi version is supported" OFF)

option(USE_ARPACK_LIB "Use arpack for generalize eigenvalue problem" OFF)

option(USE_METIS_LIB "Use metis for matrix reordering" OFF)

option(USE_BOOST_LIB "Use boost library" OFF)

option(USE_CUDA "Use cuda" OFF)

option(USE_CUDSS "Use cudss library for direct solver. Note that USE_CUDA must be ON" OFF)

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
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG master
  GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(cxxopts)

# FetchContent_Declare(
#   cereal
#   GIT_REPOSITORY https://github.com/USCiLab/cereal.git
#   GIT_TAG master
#   GIT_SHALLOW TRUE
# )
# set(CMAKE_CXX_FLAGS_OLD "${CMAKE_CXX_FLAGS}")
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-dangling-reference")
# set(SKIP_PERFORMANCE_COMPARISON ON)
# FetchContent_MakeAvailable(cereal)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_OLD}")
# set(CMAKE_CXX_STANDARD 20)

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

find_package(Python3 REQUIRED COMPONENTS Interpreter)
message(STATUS "Using Python interpreter: ${Python3_EXECUTABLE}")

# Check if ssgetpy is available
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import ssgetpy"
    RESULT_VARIABLE SSGETPY_RESULT
    ERROR_QUIET OUTPUT_QUIET
)

if(NOT SSGETPY_RESULT EQUAL 0)
    message(FATAL_ERROR "Python package 'ssgetpy' not found. 
Please install it with: pip install ssgetpy")
endif()

if(USE_AMGCL_LIB)
  FetchContent_MakeAvailable(AMGCL)
endif()

if(USE_MUMPS_LIB)
  find_package(MUMPS REQUIRED COMPONENTS SEQ METIS OPENMP)
endif()

if(USE_ARPACK_LIB)
  find_package(arpackng REQUIRED)
endif()

if(USE_METIS_LIB)
  find_package(METIS REQUIRED)
endif()

if(USE_BOOST_LIB)
  find_package(Boost REQUIRED)
endif()

if(USE_CUDA)
  # Set CUDA compiler explicitly if not already set
  if(NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "nvcc")
  endif()
  enable_language(CUDA)
  
  # Set CUDA standard to match C++ standard
  set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  
  # Find CUDA toolkit (optional but recommended for better error messages)
  find_package(CUDAToolkit REQUIRED)
  
  # Set CUDA architectures if not already set
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86")  # Common architectures
  endif()
  
  # Set CUDA compiler flags for different build types
  # Note: For multi-config generators, use CMAKE_CUDA_FLAGS_<CONFIG> instead
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")  # Debug info for device code
  elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
  endif()
  
  # CUDSS support
  if(USE_CUDSS)
    add_compile_definitions(USE_CUDSS)
    find_package(cudss REQUIRED)
  endif()
  
  message(STATUS "CUDA enabled with architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
  message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
endif()

configure_file(
  "${CMAKE_SOURCE_DIR}/cmake/config.h.in"
  "${CMAKE_BINARY_DIR}/config.h")
set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories(${CMAKE_BINARY_DIR})  

add_subdirectory(utils)
add_subdirectory(sparse_mat_op)
add_subdirectory(mkl_wrapper)
add_subdirectory(reordering)
add_subdirectory(factorization)
add_subdirectory(data)

add_subdirectory(execs)
add_subdirectory(scratch)
enable_testing()
add_subdirectory(tests)

# file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/data)
# file(GLOB MATRIX_FILES ${CMAKE_CURRENT_SOURCE_DIR}/data/*)
# foreach(MATRIX_FILE ${MATRIX_FILES})
#    get_filename_component(FILENAME ${MATRIX_FILE} NAME)
#    file(CREATE_LINK ${MATRIX_FILE} ${CMAKE_CURRENT_BINARY_DIR}/data/${FILENAME} SYMBOLIC)
# endforeach()

if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
