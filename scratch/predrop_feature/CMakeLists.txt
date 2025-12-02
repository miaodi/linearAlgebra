cmake_minimum_required(VERSION 3.20)

# Create drop_global_precond executable
add_executable(drop_global_precond drop_global_precond.cpp)

# Link against project libraries
target_link_libraries(drop_global_precond PRIVATE
    utils
    fast_matrix_market::fast_matrix_market
    Eigen3::Eigen
    mkl_wrapper
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
    mkl_wrapper
    reordering
    sparse_mat_op
    factorization
    cxxopts
)
target_include_directories(jaccard_compare PRIVATE
    ${PROJECT_BINARY_DIR}
)
