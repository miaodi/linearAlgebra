cmake_minimum_required(VERSION 3.18...3.20)
project(sparse_mat_op_cuda CUDA CXX)

find_package(BLAS REQUIRED)

if(TARGET BLAS::BLAS)
    set(LINEAR_ALGEBRA_BLAS_LIB BLAS::BLAS)
else()
    set(LINEAR_ALGEBRA_BLAS_LIB ${BLAS_LIBRARIES})
endif()

set(CUDA_SOURCES
    cuda_gmres.cu
    cuda_bicgstab.cu
    cuda_spmv.cu
    cuda_sptrsv.cu
    cuda_preconditioner.cu
    cuda_kernels.cu
    cuda_ilu_symbolic.cu
    cuda_ilu_symbolic_persistent.cu
    spgemm/spgemm_expand.cu
    spgemm/spgemm_sort.cu
    spgemm/spgemm_contract.cu
    cuda_csr_utils.cu
    cuda_spmm.cu
    cuda_iluk_U_symbolic.cu
    cuda_ruiz_scale_impl.cu
    cuda_tiled_sparse_mat.cu
)

add_library(${PROJECT_NAME} STATIC ${CUDA_SOURCES})
add_subdirectory(ilu)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    CCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
)
target_link_libraries(${PROJECT_NAME} PUBLIC
    CUDA::cusparse
    CUDA::cublas
    CUDA::cudart
    ${LINEAR_ALGEBRA_BLAS_LIB}
    LAPACK::LAPACK
    cuco
)
target_link_libraries(${PROJECT_NAME} PRIVATE
    OpenMP::OpenMP_CXX
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    CUDA_ARCHITECTURES "75;80;86;89"
)
