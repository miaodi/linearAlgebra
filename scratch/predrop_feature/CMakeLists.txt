cmake_minimum_required(VERSION 3.20)

# Create drop_global_precond executable
add_executable(drop_global_precond drop_global_precond.cpp)

# Link against project libraries
target_link_libraries(drop_global_precond PRIVATE
    utils
    fast_matrix_market::fast_matrix_market
    Eigen3::Eigen
    reordering
    sparse_mat_op
    factorization
    cxxopts
)

target_include_directories(drop_global_precond PRIVATE
    ${PROJECT_BINARY_DIR}
)

# Create jaccard_compare executable
add_executable(jaccard_compare jaccard_compare.cpp)
target_link_libraries(jaccard_compare PRIVATE
    utils
    fast_matrix_market::fast_matrix_market
    Eigen3::Eigen
    reordering
    sparse_mat_op
    factorization
    cxxopts
)
target_include_directories(jaccard_compare PRIVATE
    ${PROJECT_BINARY_DIR}
)

# Create reorder executable
add_executable(reorder reorder.cpp)
target_link_libraries(reorder PRIVATE
    utils
    fast_matrix_market::fast_matrix_market
    Eigen3::Eigen
    reordering
    sparse_mat_op
    factorization
    cxxopts
)
target_include_directories(reorder PRIVATE
    ${PROJECT_BINARY_DIR}
)

# Create cuda_ruiz_demo executable only when CUDA support is enabled.
if(USE_CUDA)
    add_executable(cuda_ruiz_demo cuda_ruiz_demo.cpp)
    target_link_libraries(cuda_ruiz_demo PRIVATE
        utils
        fast_matrix_market::fast_matrix_market
        reordering
        sparse_mat_op
        sparse_mat_op_cuda
        factorization
        cxxopts
        CUDA::cudart
    )
    target_include_directories(cuda_ruiz_demo PRIVATE
        ${PROJECT_BINARY_DIR}
    )
else()
    message(STATUS "Skipping cuda_ruiz_demo because USE_CUDA is OFF")
endif()
