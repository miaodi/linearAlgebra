cmake_minimum_required(VERSION 3.5...3.20)
project(Utils CXX)

set(UtilsSRC utils.cpp timer.cpp)

add_library(${PROJECT_NAME} ${UtilsSRC})

target_include_directories(${PROJECT_NAME} PUBLIC fast_matrix_market ${CMAKE_CURRENT_SOURCE_DIR} $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen fast_matrix_market OpenMP::OpenMP_CXX $<LINK_ONLY:MKL::MKL>)