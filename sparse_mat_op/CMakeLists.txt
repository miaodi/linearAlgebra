cmake_minimum_required(VERSION 3.5...3.20)
project(sparse_mat_op CXX)

set(SPARSE_MAT_OP_SRC matrix_utils.cpp triangle_solve.cpp precond.cpp permutation.cpp io.cpp sp_ops.cpp spgemm.cpp spadd.cpp ilum.cpp ruiz_scale.cpp scaling.cpp)

set(DEPENDENCIES enums utils graph OpenMP::OpenMP_CXX Eigen3::Eigen fast_matrix_market::fast_matrix_market stb_image_write)

# Add CUDA sublibrary if CUDA is enabled
if(USE_CUDA)
    enable_language(CUDA)
    add_subdirectory(cuda)
endif()

if(USE_METIS_LIB)
  set(DEPENDENCIES ${DEPENDENCIES} METIS::METIS)
endif()

if(USE_MKL)
  set(DEPENDENCIES ${DEPENDENCIES} MKL::MKL)
endif()

add_library(${PROJECT_NAME} STATIC ${SPARSE_MAT_OP_SRC})

target_include_directories(${PROJECT_NAME} PUBLIC 
  ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/spmv
)
target_link_libraries(${PROJECT_NAME} PUBLIC ${DEPENDENCIES})
target_enable_simd(${PROJECT_NAME})
