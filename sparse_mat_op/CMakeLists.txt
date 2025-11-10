cmake_minimum_required(VERSION 3.5...3.20)
project(sparse_mat_op CXX)

set(SPARSE_MAT_OP_SRC matrix_utils.cpp triangle_solve.cpp precond.cpp permutation.cpp io.cpp graph_algs.cpp sp_ops.cpp spgemm.cpp spadd.cpp ilum.cpp)

# Add CUDA sources if CUDA is enabled
if(USE_CUDA)
    enable_language(CUDA)
    list(APPEND SPARSE_MAT_OP_SRC cuda_gmres.cu cuda_gmres_device.cu)
    set(CUDA_DEPENDENCIES CUDA::cusparse CUDA::cublas CUDA::cudart MKL::MKL)
endif()

set(DEPENDENCIES Utils OpenMP::OpenMP_CXX Eigen3::Eigen fast_matrix_market::fast_matrix_market ${CUDA_DEPENDENCIES})

add_library(${PROJECT_NAME} STATIC ${SPARSE_MAT_OP_SRC})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC ${DEPENDENCIES})

# Set CUDA properties if CUDA is enabled
if(USE_CUDA)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
    )
endif()
