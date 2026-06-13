cmake_minimum_required(VERSION 3.18...3.20)
project(tests)

include(FetchContent)

# Add Google Test
include(../cmake/GoogleTest.cmake)
include(../cmake/DownloadFile.cmake)
include(../cmake/ssget.cmake)
include(GoogleTest)

file(GLOB_RECURSE ALL_TEST_CPP *.cpp)
if(USE_CUDA)
   file(GLOB_RECURSE ALL_TEST_CU *.cu)
   list(APPEND ALL_TEST_CPP ${ALL_TEST_CU})
endif()

# List of tests to exclude
set(EXCLUDED_TESTS 
   #  "reordering_test.cpp"
   # "cuda_csr_utils_test.cu"
)

set(DISABLED_ILU_TESTS
   "cuda_ilu_symbolic_test.cpp"
)
list(APPEND EXCLUDED_TESTS ${DISABLED_ILU_TESTS})

# Conditionally exclude CUDA tests if CUDA is not enabled
if(NOT USE_CUDA)
   list(APPEND EXCLUDED_TESTS "cuda_csr_utils_test.cu")
endif()

set(MKL_REQUIRED_TESTS
   "mkl_sparse_mat_test.cpp"
   "precond_test.cpp"
   "solver_test.cpp"
   "uncatergorized_test.cpp"
)

if(NOT USE_MKL)
   list(APPEND EXCLUDED_TESTS ${MKL_REQUIRED_TESTS})
endif()

set(TEST_LIBS GTest::gtest_main GTest::gmock_main solver utils reordering factorization OpenMP::OpenMP_CXX ${LAPACK_LIBRARIES})
if(USE_MKL)
   list(APPEND TEST_LIBS mkl_wrapper)
endif()

foreach(ONE_TEST_CPP ${ALL_TEST_CPP})
   get_filename_component(ONE_TEST_EXEC ${ONE_TEST_CPP} NAME_WE)
   get_filename_component(ONE_TEST_FILENAME ${ONE_TEST_CPP} NAME)

   # Check if this is a CUDA test
   string(FIND ${ONE_TEST_FILENAME} "cuda" IS_CUDA_TEST)

   if(IS_CUDA_TEST GREATER -1 AND NOT USE_CUDA)
      message(STATUS "Skipping CUDA test because USE_CUDA is OFF: ${ONE_TEST_FILENAME}")
      continue()
   endif()
   
   # Skip excluded tests
   list(FIND EXCLUDED_TESTS ${ONE_TEST_FILENAME} SHOULD_EXCLUDE)
   if(SHOULD_EXCLUDE GREATER -1)
      message(STATUS "Skipping test: ${ONE_TEST_FILENAME}")
      continue()
   endif()
   
   # Avoid name collision 
   set(TARGET_NAME TEST_${ONE_TEST_EXEC})

   add_executable(${TARGET_NAME} ${ONE_TEST_CPP})
   set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_TEST_EXEC})
   
   # Link standard libraries
   target_link_libraries(${TARGET_NAME} PRIVATE ${TEST_LIBS})
   
   # Add CUDA-specific dependencies if this is a CUDA test
   if(IS_CUDA_TEST GREATER -1 AND USE_CUDA)
      target_link_libraries(${TARGET_NAME} PRIVATE sparse_mat_op_cuda CUDA::cudart)
      set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
   endif()

   # Use add_test instead of gtest_discover_tests for better compatibility
   add_test(NAME ${ONE_TEST_EXEC} COMMAND ${TARGET_NAME})
endforeach()

set(TEST_DATA_DIR ${CMAKE_CURRENT_BINARY_DIR}/data)
set(SPD_DATA_DIR ${TEST_DATA_DIR}/spd)
set(UNSYMM_WITH_DIAG_DATA_DIR ${TEST_DATA_DIR}/unsymm_with_diag)
set(NO_FULL_DIAG_DATA_DIR ${TEST_DATA_DIR}/no_full_diag)
set(OTHER_DATA_DIR ${TEST_DATA_DIR}/other)

# Local synthetic unsymmetric matrices with every diagonal entry explicitly present:
# local_4x4_unsymm_full_diag: 4 x 4, 9 nnz.
# local_7x7_unsymm_full_diag: 7 x 7, 18 nnz.
# local_12x12_unsymm_full_diag: 12 x 12, 36 nnz.
file(MAKE_DIRECTORY ${UNSYMM_WITH_DIAG_DATA_DIR})
file(GLOB UNSYMM_WITH_DIAG_LOCAL_MATRICES
   ${CMAKE_CURRENT_SOURCE_DIR}/data/unsymm_with_diag/*.mtx)
foreach(MATRIX_FILE ${UNSYMM_WITH_DIAG_LOCAL_MATRICES})
   get_filename_component(FILENAME ${MATRIX_FILE} NAME)
   configure_file(${MATRIX_FILE} ${UNSYMM_WITH_DIAG_DATA_DIR}/${FILENAME} COPYONLY)
endforeach()

# Unsymmetric SuiteSparse matrices with every diagonal entry explicitly present.
# HB/pores_1: 30 x 30, 180 nnz.
download_sparse_matrix("HB/pores_1" ${UNSYMM_WITH_DIAG_DATA_DIR})

# Bai/olm100: 100 x 100, 396 nnz.
download_sparse_matrix("Bai/olm100" ${UNSYMM_WITH_DIAG_DATA_DIR})

# Bai/lop163: 163 x 163, 935 nnz.
download_sparse_matrix("Bai/lop163" ${UNSYMM_WITH_DIAG_DATA_DIR})

# HB/pores_3: 532 x 532, 3,474 nnz.
download_sparse_matrix("HB/pores_3" ${UNSYMM_WITH_DIAG_DATA_DIR})

# HB/pores_2: 1,224 x 1,224, 9,613 nnz.
download_sparse_matrix("HB/pores_2" ${UNSYMM_WITH_DIAG_DATA_DIR})

# FIDAP/ex27: 974 x 974, 40,782 nnz.
download_sparse_matrix("FIDAP/ex27" ${UNSYMM_WITH_DIAG_DATA_DIR})

# SuiteSparse matrices with full structural rank but missing diagonal entries.
# These exercise structural matching beyond the trivial diagonal permutation.
# Grund/d_ss: 53 x 53, 149 pattern entries, 4 explicit diagonal entries.
# Grund/d_dyn: 87 x 87, 238 pattern entries, 4 explicit diagonal entries.
# VDOL/spaceStation_1: 99 x 99, 927 pattern entries, not all diagonal entries explicit.
# Morandini/robot: 120 x 120, 870 pattern entries, 80 explicit diagonal entries.
# Morandini/rotor2: 791 x 791, 10,685 pattern entries, 681 explicit diagonal entries.
# HB/west0067: 67 x 67, 294 pattern entries, 2 explicit diagonal entries.
# HB/west0132: 132 x 132, 414 pattern entries, 6 explicit diagonal entries.
download_sparse_matrix("Grund/d_ss" ${NO_FULL_DIAG_DATA_DIR})
download_sparse_matrix("Grund/d_dyn" ${NO_FULL_DIAG_DATA_DIR})
download_sparse_matrix("VDOL/spaceStation_1" ${NO_FULL_DIAG_DATA_DIR})
download_sparse_matrix("Morandini/robot" ${NO_FULL_DIAG_DATA_DIR})
download_sparse_matrix("Morandini/rotor2" ${NO_FULL_DIAG_DATA_DIR})
download_sparse_matrix("HB/west0067" ${NO_FULL_DIAG_DATA_DIR})
download_sparse_matrix("HB/west0132" ${NO_FULL_DIAG_DATA_DIR})

# SPD SuiteSparse matrices.
# HB/bcsstk17: 10,974 x 10,974, 428,650 nnz.
download_sparse_matrix("bcsstk17" ${SPD_DATA_DIR})

# Cylshell/s3rmt3m3: 5,357 x 5,357, 217,681 nnz.
download_sparse_matrix("s3rmt3m3" ${SPD_DATA_DIR})

# FIDAP/ex5: 27 x 27, 279 nnz.
download_sparse_matrix("ex5" ${SPD_DATA_DIR})

# HB/nos5: 468 x 468, 5,172 nnz.
download_sparse_matrix("nos5" ${SPD_DATA_DIR})

# Other non-SPD SuiteSparse matrices.
# HB/jgl009: 9 x 9, 50 nnz.
download_sparse_matrix("jgl009" ${OTHER_DATA_DIR})

# Zitney/rdist1: 4,134 x 4,134, 94,408 nnz.
download_sparse_matrix("rdist1" ${OTHER_DATA_DIR})
