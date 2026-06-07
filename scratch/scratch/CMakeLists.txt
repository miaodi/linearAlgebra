cmake_minimum_required(VERSION 3.18...3.20)
project(scratch_test CXX)

add_executable(ilu_factorization_levels ilu_factorization_levels.cpp)
target_compile_features(ilu_factorization_levels PRIVATE cxx_std_20)
target_link_libraries(ilu_factorization_levels PRIVATE
  cxxopts::cxxopts
  graph
  reordering
  sparse_mat_op
)

# CUDA test executable for SpMMAnalyze
if(USE_CUDA)
  enable_language(CUDA)

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

  set_source_files_properties(cuda_spgemm_phase_demo.cpp PROPERTIES LANGUAGE CUDA)
  add_executable(cuda_spgemm_phase_demo cuda_spgemm_phase_demo.cpp)
  target_compile_features(cuda_spgemm_phase_demo PRIVATE cxx_std_20)
  target_compile_definitions(cuda_spgemm_phase_demo PRIVATE
    CCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER
    CUDA_ALLOC_TRACE # Enable CUDA allocation tracing for debugging purposes
  )
  target_link_libraries(cuda_spgemm_phase_demo PRIVATE
    cxxopts::cxxopts
    sparse_mat_op
    sparse_mat_op_cuda
    CUDA::cusparse
    CUDA::cudart
  )
  set_target_properties(cuda_spgemm_phase_demo PROPERTIES
    CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
  )
else()
endif()
