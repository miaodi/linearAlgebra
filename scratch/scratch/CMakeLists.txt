cmake_minimum_required(VERSION 3.5...3.20)
project(scratch_test CXX)

# Simple test executable
add_executable(scratch_test test.cpp)

# Link with common dependencies if needed
# target_link_libraries(scratch_test PRIVATE utils)

# Enable C++20
target_compile_features(scratch_test PRIVATE cxx_std_20)
