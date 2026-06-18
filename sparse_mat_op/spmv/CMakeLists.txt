cmake_minimum_required(VERSION 3.5...3.20)
project(spmv CXX)

# SpMV implementations are header-only templates.
add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME} INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(${PROJECT_NAME} INTERFACE
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
    utils
)
