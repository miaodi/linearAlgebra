cmake_minimum_required(VERSION 3.5...3.20)
project(graph CXX)

set(GRAPH_SRC graph_algs.cpp bfs.cpp tree.cpp)

set(DEPENDENCIES enums utils sparse_mat_op OpenMP::OpenMP_CXX)

add_library(${PROJECT_NAME} STATIC ${GRAPH_SRC})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC ${DEPENDENCIES})
target_enable_simd(${PROJECT_NAME})
