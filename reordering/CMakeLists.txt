cmake_minimum_required(VERSION 3.5...3.20)
project(reordering CXX)

set(REORDERING_SRC BFS.cpp)

add_library(${PROJECT_NAME} SHARED ${REORDERING_SRC})

target_include_directories(${PROJECT_NAME} PUBLIC mkl_wrapper)
target_link_libraries(${PROJECT_NAME} PUBLIC mkl_wrapper)