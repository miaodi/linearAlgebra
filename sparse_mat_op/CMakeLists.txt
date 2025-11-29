cmake_minimum_required(VERSION 3.5...3.20)
project(sparse_mat_op CXX)

set(SPARSE_MAT_OP_SRC matrix_utils.cpp triangle_solve.cpp precond.cpp permutation.cpp io.cpp graph_algs.cpp sp_ops.cpp spgemm.cpp spadd.cpp ilum.cpp ruiz_scale.cpp)

set(DEPENDENCIES utils OpenMP::OpenMP_CXX Eigen3::Eigen fast_matrix_market::fast_matrix_market)

if(USE_METIS_LIB)
  set(DEPENDENCIES ${DEPENDENCIES} METIS::METIS)
endif()

add_library(${PROJECT_NAME} STATIC ${SPARSE_MAT_OP_SRC})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC ${DEPENDENCIES})
target_enable_simd(${PROJECT_NAME})
