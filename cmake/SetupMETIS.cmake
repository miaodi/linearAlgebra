# SetupMETIS.cmake
# This module handles METIS setup, either by finding a system installation
# or by fetching and building from GitHub

# First, try to find METIS in the system
if(NOT FORCE_FETCH_METIS)
  find_package(METIS QUIET)
  
  if(METIS_FOUND)
    message(STATUS "Found system METIS: ${METIS_LIBRARIES}")
    
    # Create METIS::METIS target if it doesn't exist
    if(NOT TARGET METIS::METIS)
      add_library(METIS::METIS INTERFACE IMPORTED)
      set_target_properties(METIS::METIS PROPERTIES
        INTERFACE_LINK_LIBRARIES "${METIS_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${METIS_INCLUDE_DIRS}"
      )
    endif()
    
    return()
  else()
    message(STATUS "System METIS not found, will fetch from GitHub")
  endif()
endif()

# If we reach here, we need to fetch METIS from GitHub
message(STATUS "Fetching METIS and GKlib from GitHub...")

include(FetchContent)

# Set policy to allow FetchContent_Populate for patching purposes
# We need to use Populate to patch files before building
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

# Declare METIS and GKlib
FetchContent_Declare(
  METIS
  GIT_REPOSITORY https://github.com/KarypisLab/METIS.git
  GIT_TAG v5.2.1
  GIT_SHALLOW TRUE
)

FetchContent_Declare(
  GKlib
  GIT_REPOSITORY https://github.com/KarypisLab/GKlib.git
  GIT_TAG master
  GIT_SHALLOW TRUE
)

# Fetch and build GKlib first (dependency of METIS)
FetchContent_GetProperties(GKlib)
if(NOT gklib_POPULATED)
  # Patch GKlib CMakeLists.txt before making available
  FetchContent_GetProperties(GKlib SOURCE_DIR gklib_SOURCE_DIR)
  if(NOT gklib_SOURCE_DIR)
    # Need to populate to get source dir for patching
    FetchContent_Populate(GKlib)
    FetchContent_GetProperties(GKlib SOURCE_DIR gklib_SOURCE_DIR BINARY_DIR gklib_BINARY_DIR)
  endif()
  
  # Patch GKlib CMakeLists.txt to fix CMake version if needed
  file(READ "${gklib_SOURCE_DIR}/CMakeLists.txt" GKLIB_CMAKE_CONTENT)
  string(REPLACE "cmake_minimum_required(VERSION 2.8)" 
                 "cmake_minimum_required(VERSION 3.5)" 
                 GKLIB_CMAKE_CONTENT "${GKLIB_CMAKE_CONTENT}")
  file(WRITE "${gklib_SOURCE_DIR}/CMakeLists.txt" "${GKLIB_CMAKE_CONTENT}")
  
  # Build GKlib
  add_subdirectory(${gklib_SOURCE_DIR} ${gklib_BINARY_DIR})
  set(gklib_POPULATED TRUE)
endif()

# Now fetch and build METIS
FetchContent_GetProperties(METIS)
if(NOT metis_POPULATED)
  # Patch METIS before making available
  FetchContent_GetProperties(METIS SOURCE_DIR metis_SOURCE_DIR)
  if(NOT metis_SOURCE_DIR)
    # Need to populate to get source dir for patching
    FetchContent_Populate(METIS)
    FetchContent_GetProperties(METIS SOURCE_DIR metis_SOURCE_DIR BINARY_DIR metis_BINARY_DIR)
  endif()
  
  # Patch METIS CMakeLists.txt to fix CMake version requirement and remove build/xinclude subdirectory
  file(READ "${metis_SOURCE_DIR}/CMakeLists.txt" METIS_CMAKE_CONTENT)
  string(REPLACE "cmake_minimum_required(VERSION 2.8)" 
                 "cmake_minimum_required(VERSION 3.5)" 
                 METIS_CMAKE_CONTENT "${METIS_CMAKE_CONTENT}")
  string(REPLACE "add_subdirectory(\"build/xinclude\")"
                 "# add_subdirectory(\"build/xinclude\") # Commented out for FetchContent"
                 METIS_CMAKE_CONTENT "${METIS_CMAKE_CONTENT}")
  file(WRITE "${metis_SOURCE_DIR}/CMakeLists.txt" "${METIS_CMAKE_CONTENT}")
  
  # Patch metislib.h to include metis.h before GKlib.h
  file(READ "${metis_SOURCE_DIR}/libmetis/metislib.h" METISLIB_CONTENT)
  string(REPLACE "#include <GKlib.h>\n\n#if defined(ENABLE_OPENMP)\n  #include <omp.h>\n#endif\n\n\n#include <metis.h>"
                 "#include <metis.h>\n#include <GKlib.h>\n\n#if defined(ENABLE_OPENMP)\n  #include <omp.h>\n#endif"
                 METISLIB_CONTENT "${METISLIB_CONTENT}")
  file(WRITE "${metis_SOURCE_DIR}/libmetis/metislib.h" "${METISLIB_CONTENT}")
  
  # Create build/xinclude directory and copy required headers
  file(MAKE_DIRECTORY "${metis_SOURCE_DIR}/build/xinclude")
  
  # Copy GKlib headers to build/xinclude
  file(GLOB GKLIB_HEADERS "${gklib_SOURCE_DIR}/include/*.h")
  foreach(header ${GKLIB_HEADERS})
    get_filename_component(header_name ${header} NAME)
    configure_file(${header} "${metis_SOURCE_DIR}/build/xinclude/${header_name}" COPYONLY)
  endforeach()
  
  # Copy metis.h to build/xinclude
  configure_file(
    "${metis_SOURCE_DIR}/include/metis.h"
    "${metis_SOURCE_DIR}/build/xinclude/metis.h"
    COPYONLY
  )
  
  # Build METIS with GKlib path
  set(GKLIB_PATH "${gklib_SOURCE_DIR}" CACHE PATH "Path to GKlib" FORCE)
  set(SHARED FALSE CACHE BOOL "Build shared libraries for METIS" FORCE)
  
  # Set METIS data type widths (32 or 64 bit)
  # IDXTYPEWIDTH: 32 for int32_t indices, 64 for int64_t indices
  # REALTYPEWIDTH: 32 for float, 64 for double
  if(NOT DEFINED IDXTYPEWIDTH)
    set(IDXTYPEWIDTH 32)
  endif()
  if(NOT DEFINED REALTYPEWIDTH)
    set(REALTYPEWIDTH 64)
  endif()
  add_compile_definitions(IDXTYPEWIDTH=${IDXTYPEWIDTH})
  add_compile_definitions(REALTYPEWIDTH=${REALTYPEWIDTH})
  
  # Add METIS to the build
  add_subdirectory(${metis_SOURCE_DIR} ${metis_BINARY_DIR})
  set(metis_POPULATED TRUE)
endif()

# METIS doesn't export proper targets by default, so we create an alias
if(TARGET metis AND NOT TARGET METIS::METIS)
  # Make sure METIS links against GKlib since it depends on it
  target_link_libraries(metis PUBLIC GKlib)
  
  add_library(METIS::METIS ALIAS metis)
  # Set include directories for the alias
  target_include_directories(metis INTERFACE 
    $<BUILD_INTERFACE:${metis_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${metis_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${metis_SOURCE_DIR}/build/xinclude>
  )
endif()

message(STATUS "Using METIS via FetchContent with GKlib")
