cmake_minimum_required(VERSION 3.5...3.20)
project(tests)

include(FetchContent)

# Add Google Test
include(../cmake/GoogleTest.cmake)
include(../cmake/DownloadFile.cmake)
include(../cmake/ssget.cmake)
include(GoogleTest)

file(GLOB_RECURSE ALL_TEST_CPP *.cpp)

# List of tests to exclude
set(EXCLUDED_TESTS 
   #  "reordering_test.cpp"
)

# Conditionally exclude CUDA tests if CUDA is not enabled
if(NOT USE_CUDA)
   list(APPEND EXCLUDED_TESTS "cuda_ilu_symbolic_test.cpp")
endif()

foreach(ONE_TEST_CPP ${ALL_TEST_CPP})
   get_filename_component(ONE_TEST_EXEC ${ONE_TEST_CPP} NAME_WE)
   get_filename_component(ONE_TEST_FILENAME ${ONE_TEST_CPP} NAME)
   
   # Skip excluded tests
   list(FIND EXCLUDED_TESTS ${ONE_TEST_FILENAME} SHOULD_EXCLUDE)
   if(SHOULD_EXCLUDE GREATER -1)
      message(STATUS "Skipping test: ${ONE_TEST_FILENAME}")
      continue()
   endif()

   # Check if this is a CUDA test
   string(FIND ${ONE_TEST_FILENAME} "cuda" IS_CUDA_TEST)
   
   # Avoid name collision 
   set(TARGET_NAME TEST_${ONE_TEST_EXEC})

   add_executable(${TARGET_NAME} ${ONE_TEST_CPP})
   set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_TEST_EXEC})
   
   # Link standard libraries
   target_link_libraries(${TARGET_NAME} PRIVATE GTest::gtest_main GTest::gmock_main solver mkl_wrapper utils reordering OpenMP::OpenMP_CXX ${LAPACK_LIBRARIES})
   
   # Add CUDA-specific dependencies if this is a CUDA test
   if(IS_CUDA_TEST GREATER -1 AND USE_CUDA)
      target_link_libraries(${TARGET_NAME} PRIVATE sparse_mat_op_cuda CUDA::cudart)
      set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
   endif()

   # Use add_test instead of gtest_discover_tests for better compatibility
   add_test(NAME ${ONE_TEST_EXEC} COMMAND ${TARGET_NAME})
endforeach()

# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/data/ex5.mtx ${CMAKE_CURRENT_BINARY_DIR}/data/ex5.mtx COPYONLY)
# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/data/rdist1.mtx ${CMAKE_CURRENT_BINARY_DIR}/data/rdist1.mtx COPYONLY)

# file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/data)
# file(GLOB MATRIX_FILES ${CMAKE_CURRENT_SOURCE_DIR}/data/*.mtx)
# foreach(MATRIX_FILE ${MATRIX_FILES})
#    get_filename_component(FILENAME ${MATRIX_FILE} NAME)
#    file(CREATE_LINK ${MATRIX_FILE} ${CMAKE_CURRENT_BINARY_DIR}/data/${FILENAME} SYMBOLIC)
# endforeach()

# https://sparse.tamu.edu/HB/bcsstk17
# 10974
# spd
download_sparse_matrix("bcsstk17" ${CMAKE_CURRENT_BINARY_DIR}/data)

# https://sparse.tamu.edu/Cylshell/s3rmt3m3
# 5357
# spd
download_sparse_matrix("s3rmt3m3" ${CMAKE_CURRENT_BINARY_DIR}/data)

# https://sparse.tamu.edu/FIDAP/ex5
# 27
# spd
download_sparse_matrix("ex5" ${CMAKE_CURRENT_BINARY_DIR}/data)

# https://sparse.tamu.edu/HB/jgl009
# 9
# not spd
download_sparse_matrix("jgl009" ${CMAKE_CURRENT_BINARY_DIR}/data)

# https://sparse.tamu.edu/Zitney/rdist1
# 4134
# not spd
download_sparse_matrix("rdist1" ${CMAKE_CURRENT_BINARY_DIR}/data)

# https://sparse.tamu.edu/HB/nos5
# 468
# spd
download_sparse_matrix("nos5" ${CMAKE_CURRENT_BINARY_DIR}/data)

# https://sparse.tamu.edu/FIDAP/ex27
# 974
# not spd
download_sparse_matrix("ex27" ${CMAKE_CURRENT_BINARY_DIR}/data)