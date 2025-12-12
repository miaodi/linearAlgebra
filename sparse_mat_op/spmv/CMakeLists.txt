cmake_minimum_required(VERSION 3.5...3.20)
project(spmv CXX)

# Source files for SpMV implementations
set(SPMV_SRC 
    csr5_convert.cpp
)

# Create spmv library
add_library(${PROJECT_NAME} STATIC ${SPMV_SRC})

# Include directories
target_include_directories(${PROJECT_NAME} PUBLIC 
    ${CMAKE_CURRENT_LIST_DIR}
)

# Link dependencies
target_link_libraries(${PROJECT_NAME} PUBLIC 
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
)

# Enable SIMD optimizations
target_enable_simd(${PROJECT_NAME})
