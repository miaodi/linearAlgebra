cmake_minimum_required(VERSION 3.5...3.20)
project(factorization CXX)

set(SRC cholesky_symbolic.cpp)

add_library(${PROJECT_NAME} STATIC ${SRC})

set(DEPENDENCIES utils sparse_mat_op graph OpenMP::OpenMP_CXX)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC ${DEPENDENCIES})
