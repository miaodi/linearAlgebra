cmake_minimum_required(VERSION 3.5...3.20)
project(scratch_test CXX CUDA)

# CUDA test executable for SpMMAnalyze
if(USE_CUDA)
  set_source_files_properties(test.cpp PROPERTIES LANGUAGE CUDA)
  add_executable(scratch_test test.cpp)
  
  # Link with CUDA libraries and sparse_mat_op_cuda
  target_link_libraries(scratch_test PRIVATE 
    CUDA::cusparse 
    CUDA::cublas 
    CUDA::cudart
    sparse_mat_op_cuda
  )
  
  # Set CUDA properties
  set_target_properties(scratch_test PROPERTIES
    CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
  )
  # CUDA test executable
  set_source_files_properties(cuda_test.cpp PROPERTIES LANGUAGE CUDA)
  add_executable(cuda_test cuda_test.cpp)
  target_compile_features(cuda_test PRIVATE cxx_std_20)
  target_link_libraries(cuda_test PRIVATE
    cxxopts::cxxopts
    sparse_mat_op
    sparse_mat_op_cuda
    CUDA::cudart
  )
else()
endif()
