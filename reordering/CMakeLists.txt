cmake_minimum_required(VERSION 3.5...3.20)
project(reordering CXX)

set(REORDERING_SRC BFS.cpp Reordering.cpp)

add_library(${PROJECT_NAME} SHARED ${REORDERING_SRC})

target_include_directories(${PROJECT_NAME} PUBLIC mkl_wrapper ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC mkl_wrapper Utils OpenMP::OpenMP_CXX)