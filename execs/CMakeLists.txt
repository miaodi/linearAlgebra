cmake_minimum_required(VERSION 3.5...3.20)
project(execs)

set(libs Utils fast_matrix_market::fast_matrix_market Eigen3::Eigen mkl_wrapper reordering sparse_mat_op factorization cxxopts)

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

add_executable(test_ilum test_ilum.cpp)
target_link_libraries(test_ilum PRIVATE ${libs} OpenMP::OpenMP_CXX)

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
