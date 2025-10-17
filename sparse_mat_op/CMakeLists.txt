cmake_minimum_required(VERSION 3.5...3.20)
project(sparse_mat_op CXX)

set(SPARSE_MAT_OP_SRC matrix_utils.cpp triangle_solve.cpp precond.cpp permutation.cpp io.cpp)

set(DEPENDENCIES Utils OpenMP::OpenMP_CXX Eigen3::Eigen fast_matrix_market::fast_matrix_market TBB::tbb)

add_library(${PROJECT_NAME} STATIC ${SPARSE_MAT_OP_SRC})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC ${DEPENDENCIES})