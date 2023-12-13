cmake_minimum_required(VERSION 3.5...3.20)
project(tests)

add_executable(test1 main.cpp)

target_include_directories(test1 PRIVATE Utils)
target_link_libraries(test1 PRIVATE Utils fast_matrix_market::fast_matrix_market Eigen3::Eigen mkl_wrapper)

add_executable(analysisBinary analysisBinary.cpp)

target_include_directories(analysisBinary PRIVATE Utils)
target_link_libraries(analysisBinary PRIVATE Utils fast_matrix_market::fast_matrix_market Eigen3::Eigen mkl_wrapper)