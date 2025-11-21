cmake_minimum_required(VERSION 3.5...3.20)
project(tests)

set(libs Utils fast_matrix_market::fast_matrix_market Eigen3::Eigen mkl_wrapper reordering sparse_mat_op factorization cxxopts)

# add_executable(linearSolve linearSolve.cpp)
# target_link_libraries(linearSolve PRIVATE ${libs} TBB::tbb)

# add_executable(analysisBinary analysisBinary.cpp)
# target_link_libraries(analysisBinary PRIVATE ${libs})

# add_executable(eigen eigen.cpp)
# target_link_libraries(eigen PRIVATE ${libs} Spectra)

# add_executable(tmp tmp.cpp)
# target_link_libraries(tmp PRIVATE ${libs} OpenMP::OpenMP_CXX)

# add_executable(tmp2 tmp2.cpp)
# target_link_libraries(tmp2 PRIVATE ${libs} Spectra OpenMP::OpenMP_CXX)

# add_executable(test1 test1.cpp)
# target_link_libraries(test1 PRIVATE ${libs} TBB::tbb)

# add_executable(temp_test temp_test.cpp)
# target_link_libraries(temp_test PRIVATE ${libs})

add_executable(test_chol test_chol.cpp)
target_link_libraries(test_chol PRIVATE ${libs})

add_executable(test_gmres gmres.cpp)
target_link_libraries(test_gmres PRIVATE ${libs})

add_executable(test_unsymm unsymm.cpp)
target_link_libraries(test_unsymm PRIVATE ${libs})

add_executable(triangular_debug triangular_debug.cpp)
target_link_libraries(triangular_debug PRIVATE ${libs} OpenMP::OpenMP_CXX)

add_executable(ilu_graph_example ilu_graph_example.cpp)
target_link_libraries(ilu_graph_example PRIVATE ${libs})

add_executable(print_matrix_svg print_matrix_svg.cpp)
target_link_libraries(print_matrix_svg PRIVATE ${libs})

# CUDA executables (only build if CUDA is enabled)
if(USE_CUDA)
  add_executable(cuda_spsv_test cuda_spsv_test.cu)
  
  # Link CUDA libraries, project libraries, and cxxopts
  target_link_libraries(cuda_spsv_test PRIVATE 
    CUDA::cusparse 
    CUDA::cublas 
    CUDA::cudart
    sparse_mat_op
    cxxopts
  )
  
  # Set CUDA properties
  set_target_properties(cuda_spsv_test PROPERTIES
    CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
  )
  
  # Add CUDA GMRES test
  add_executable(test_cuda_gmres test_cuda_gmres.cu)
  
  target_link_libraries(test_cuda_gmres PRIVATE 
    CUDA::cusparse 
    CUDA::cublas 
    CUDA::cudart
    sparse_mat_op
    cxxopts
  )
  
  set_target_properties(test_cuda_gmres PROPERTIES
    CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
  )
endif()
