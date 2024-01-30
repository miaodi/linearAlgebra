cmake_minimum_required(VERSION 3.5...3.20)
project(tests)

add_executable(linearSolve linearSolve.cpp)
# target_include_directories(test1 PRIVATE Utils)
target_link_libraries(linearSolve PRIVATE Utils fast_matrix_market::fast_matrix_market Eigen3::Eigen mkl_wrapper TBB::tbb)

add_executable(analysisBinary analysisBinary.cpp)
# target_include_directories(analysisBinary PRIVATE Utils)
target_link_libraries(analysisBinary PRIVATE Utils fast_matrix_market::fast_matrix_market Eigen3::Eigen mkl_wrapper)

add_executable(eigen eigen.cpp)
target_link_libraries(eigen PRIVATE Utils fast_matrix_market::fast_matrix_market Eigen3::Eigen mkl_wrapper Spectra)

add_executable(tmp tmp.cpp)
target_include_directories(tmp PRIVATE ARPACK::ARPACK)
target_link_libraries(tmp PRIVATE Utils fast_matrix_market::fast_matrix_market Eigen3::Eigen mkl_wrapper Spectra ARPACK::ARPACK OpenMP::OpenMP_CXX)

add_executable(test1 test1.cpp)
# target_include_directories(test1 PRIVATE Utils)
target_link_libraries(test1 PRIVATE Utils fast_matrix_market::fast_matrix_market Eigen3::Eigen mkl_wrapper TBB::tbb)