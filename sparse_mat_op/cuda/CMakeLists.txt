cmake_minimum_required(VERSION 3.5...3.20)
project(sparse_mat_op_cuda CUDA CXX)

set(CUDA_SOURCES
    cuda_gmres.cu
    cuda_gmres_device.cu
    cuda_bicgstab.cu
    cuda_spmv.cu
    cuda_preconditioner.cu
    cuda_kernels.cu
)

add_library(${PROJECT_NAME} STATIC ${CUDA_SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC 
  ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(${PROJECT_NAME} PUBLIC 
    CUDA::cusparse 
    CUDA::cublas 
    CUDA::cudart 
    LAPACK::LAPACK
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
)
