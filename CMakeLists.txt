cmake_minimum_required(VERSION 3.18...3.20)

# Set compilers before project() to ensure proper detection
# Note: These can be overridden via environment variables (CC, CXX) or cmake -D flags
if(DEFINED ENV{CC})
  set(CMAKE_C_COMPILER $ENV{CC})
endif()
if(DEFINED ENV{CXX})
  set(CMAKE_CXX_COMPILER $ENV{CXX})
endif()

project(Linear_Algebra CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(LINEAR_ALGEBRA_LINK_COMPILE_COMMANDS_FOR_CLANGD
  "Create a source-tree compile_commands.json link for clangd" ON)

if(CMAKE_EXPORT_COMPILE_COMMANDS
   AND LINEAR_ALGEBRA_LINK_COMPILE_COMMANDS_FOR_CLANGD
   AND UNIX
   AND NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
      "${CMAKE_BINARY_DIR}/compile_commands.json"
      "${CMAKE_SOURCE_DIR}/compile_commands.json"
  )
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Keep FetchContent checkouts in a descriptive directory while preserving
# explicit user overrides and migrating existing build trees from CMake's default.
if(NOT DEFINED FETCHCONTENT_BASE_DIR
   OR FETCHCONTENT_BASE_DIR STREQUAL ""
   OR FETCHCONTENT_BASE_DIR STREQUAL "${CMAKE_BINARY_DIR}/_deps")
  set(FETCHCONTENT_BASE_DIR "${CMAKE_BINARY_DIR}/third-party"
    CACHE PATH "Directory used by FetchContent for third-party dependencies" FORCE)
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
# To force using METIS from GitHub instead of system METIS, set: -DFORCE_FETCH_METIS=ON

option(USE_BOOST_LIB "Use boost library" OFF)

option(USE_SPECTRA_LIB "Use Spectra library for eigenvalue problems" OFF)

option(USE_MKL "Use Intel MKL for optimized linear algebra operations" OFF)

option(USE_CUDA "Use cuda" OFF)

option(USE_CUDSS "Use cudss library for direct solver. Note that USE_CUDA must be ON" OFF)

option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(ENABLE_CUDA_PROFILER_RANGE "Enable cudaProfilerStart/Stop scopes in CUDA benchmarks" OFF)
option(ENABLE_TESTS "Enable building tests and downloading test data" ON)
option(ENABLE_DATA_DOWNLOADS "Download SuiteSparse matrices into the build tree" ON)
option(ENABLE_CSR5_REFERENCE_TESTS "Fetch CSR5 reference code and build CSR5 reference comparison tests/benchmarks" OFF)
option(BUILD_DATA "Build data subdirectory" OFF)

list(APPEND CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/cmake"
  "${CMAKE_SOURCE_DIR}/cmake/modules")

include(FetchContent)
include(cmake/SimdOptions.cmake)
include(cmake/DownloadCache.cmake)

FetchContent_Declare(
  fast_matrix_market
  GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market.git
  GIT_TAG main
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(fast_matrix_market)
find_package(Eigen3 REQUIRED NO_MODULE)

FetchContent_Declare(
  Spectra
  GIT_REPOSITORY https://github.com/yixuan/spectra.git
  GIT_TAG master
  GIT_SHALLOW TRUE
)

FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG master
  GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(cxxopts)
find_package(OpenMP REQUIRED)

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

# METIS declarations moved to cmake/SetupMETIS.cmake

if(USE_MKL)
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
  
  # add_compile_definitions(USE_MKL)
  message(STATUS "MKL enabled with interface: ${MKL_INTERFACE}, threading: ${MKL_THREADING}")
else()
  message(STATUS "MKL disabled - using standard LAPACK")
endif()

find_package(LAPACK REQUIRED)

if(ENABLE_TESTS OR ENABLE_DATA_DOWNLOADS OR BUILD_DATA)
  linear_algebra_prefer_user_python()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  message(STATUS "Using Python interpreter: ${Python3_EXECUTABLE}")

  # Check if ssgetpy is available for downloading test matrices
  execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import ssgetpy"
      RESULT_VARIABLE SSGETPY_RESULT
      ERROR_QUIET OUTPUT_QUIET
  )

  if(NOT SSGETPY_RESULT EQUAL 0)
      message(FATAL_ERROR
        "Python package 'ssgetpy' was not found in the selected Python interpreter:\n"
        "  Python3_EXECUTABLE='${Python3_EXECUTABLE}'\n"
        "\n"
        "SuiteSparse matrix downloads require a Python environment with ssgetpy installed.\n"
        "Choose one of the following fixes:\n"
        "  1. Install ssgetpy into this interpreter:\n"
        "       ${Python3_EXECUTABLE} -m pip install ssgetpy\n"
        "  2. Reconfigure with an interpreter that already has ssgetpy:\n"
        "       cmake -S . -B build -DPython3_EXECUTABLE=/path/to/python\n"
        "  3. Or set the helper environment variable before configuring:\n"
        "       export PYTHONWITHSSGETPY=/path/to/python\n"
        "       cmake -S . -B build\n"
        "\n"
        "If you do not need tests or matrix downloads for this configure, disable them with:\n"
        "  cmake -S . -B build -DENABLE_TESTS=OFF -DENABLE_DATA_DOWNLOADS=OFF")
  endif()
endif()

if(USE_AMGCL_LIB)
  FetchContent_MakeAvailable(AMGCL)
endif()

if(USE_SPECTRA_LIB)
  FetchContent_MakeAvailable(Spectra)
endif()

if(USE_MUMPS_LIB)
  find_package(MUMPS REQUIRED COMPONENTS SEQ METIS OPENMP)
endif()

if(USE_ARPACK_LIB)
  find_package(arpackng REQUIRED)
endif()

if(USE_METIS_LIB)
  # METIS setup is handled in cmake/SetupMETIS.cmake
  # Set FORCE_FETCH_METIS=ON to always fetch from GitHub instead of using system METIS
  include(cmake/SetupMETIS.cmake)
endif()

if(USE_BOOST_LIB)
  find_package(Boost REQUIRED)
endif()

if(USE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "75-real;80-real;86-real;89-real;90-real" CACHE STRING
        "CUDA architectures to build; use 90-real for H100 deployments that must avoid PTX JIT")
  endif()

  enable_language(CUDA)
  
  # Set CUDA standard to match C++ standard
  set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  
  # Find CUDA (handles both standard CUDA Toolkit and NVIDIA HPC SDK)
  find_package(CUDAHPC REQUIRED)
  
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # Enable device debug code.
  endif()
  
  # Enable extended device lambdas (required by cuco)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
  
  # CUDSS support
  if(USE_CUDSS)
    add_compile_definitions(USE_CUDSS)
    find_package(cudss REQUIRED)
  endif()
  
  message(STATUS "CUDA enabled with architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
  message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")

  # cuCollections (cuco) for GPU hash set support
  set(LINEAR_ALGEBRA_BUILD_BENCHMARKS ${BUILD_BENCHMARKS})
  set(BUILD_TESTS OFF)
  set(BUILD_BENCHMARKS OFF)
  set(BUILD_EXAMPLES OFF)
  FetchContent_Declare(
    cuco
    GIT_REPOSITORY https://github.com/NVIDIA/cuCollections.git
    GIT_TAG dev
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(cuco)
  set(BUILD_BENCHMARKS ${LINEAR_ALGEBRA_BUILD_BENCHMARKS})
endif()

configure_file(
  "${CMAKE_SOURCE_DIR}/cmake/config.h.in"
  "${CMAKE_BINARY_DIR}/config.h")
set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories(${CMAKE_BINARY_DIR})  

add_subdirectory(enums)
add_subdirectory(utils)
add_subdirectory(third-party/stb)
add_subdirectory(sparse_mat_op)
add_subdirectory(graph)
if(USE_MKL)
  add_subdirectory(mkl_wrapper)
endif()
add_subdirectory(reordering)
add_subdirectory(factorization)
add_subdirectory(solver)
if(ENABLE_DATA_DOWNLOADS OR BUILD_DATA)
  add_subdirectory(data)
endif()

add_subdirectory(execs)
add_subdirectory(scratch)
if(ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/data)
# file(GLOB MATRIX_FILES ${CMAKE_CURRENT_SOURCE_DIR}/data/*)
# foreach(MATRIX_FILE ${MATRIX_FILES})
#    get_filename_component(FILENAME ${MATRIX_FILE} NAME)
#    file(CREATE_LINK ${MATRIX_FILE} ${CMAKE_CURRENT_BINARY_DIR}/data/${FILENAME} SYMBOLIC)
# endforeach()

if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
