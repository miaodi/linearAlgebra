cmake_minimum_required(VERSION 3.20)

# Create drop_global_precond executable
add_executable(drop_global_precond drop_global_precond.cpp)

# Link against project libraries
target_link_libraries(drop_global_precond PRIVATE
    Utils
    fast_matrix_market::fast_matrix_market
    Eigen3::Eigen
    mkl_wrapper
    reordering
    sparse_mat_op
    factorization
    cxxopts
)

# Include directories
target_include_directories(drop_global_precond PRIVATE
    ${PROJECT_SOURCE_DIR}/sparse_mat_op
    ${PROJECT_SOURCE_DIR}/mkl_wrapper
    ${PROJECT_SOURCE_DIR}/factorization
    ${PROJECT_SOURCE_DIR}/reordering
    ${PROJECT_SOURCE_DIR}/utils
)
