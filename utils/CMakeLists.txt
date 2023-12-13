cmake_minimum_required(VERSION 3.5...3.20)
project(Utils CXX)

set(UtilsSRC utils.cpp)

add_library(${PROJECT_NAME} SHARED ${UtilsSRC})

target_include_directories(${PROJECT_NAME} PRIVATE)
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen fast_matrix_market)