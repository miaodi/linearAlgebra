cmake_minimum_required(VERSION 3.18...3.20)
project(enums CXX)

# Header-only library - no source files to compile
add_library(${PROJECT_NAME} INTERFACE)

# Make the header available to other targets
target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_LIST_DIR})
