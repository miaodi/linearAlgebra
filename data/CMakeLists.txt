cmake_minimum_required(VERSION 3.5...3.20)
project(data)

include(${CMAKE_SOURCE_DIR}/cmake/ssget.cmake)

# Download some matrices from the SuiteSparse collection
# into the build directory
# You can add more matrices as needed
# The matrices will be downloaded into the build directory
download_sparse_matrix("msc00726" ${CMAKE_CURRENT_BINARY_DIR})
download_sparse_matrix("nasa2910" ${CMAKE_CURRENT_BINARY_DIR})
download_sparse_matrix("plat1919" ${CMAKE_CURRENT_BINARY_DIR})
download_sparse_matrix("bcsstk05" ${CMAKE_CURRENT_BINARY_DIR})
download_sparse_matrix("bcsstk01" ${CMAKE_CURRENT_BINARY_DIR})
download_sparse_matrix("ex5" ${CMAKE_CURRENT_BINARY_DIR})
download_sparse_matrix("mesh1em1" ${CMAKE_CURRENT_BINARY_DIR})
download_sparse_matrix("mesh1em6" ${CMAKE_CURRENT_BINARY_DIR})
download_sparse_matrix("barrier2-2" ${CMAKE_CURRENT_BINARY_DIR})
download_sparse_matrix("cvxbqp1" ${CMAKE_CURRENT_BINARY_DIR})
download_sparse_matrix("c-66b" ${CMAKE_CURRENT_BINARY_DIR})

# Create symbolic links for all files in the source directory
file(GLOB MATRIX_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(MATRIX_FILE ${MATRIX_FILES})
   if(IS_DIRECTORY ${MATRIX_FILE})
      continue()
   endif()
   get_filename_component(FILENAME ${MATRIX_FILE} NAME)
   file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME} ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME} SYMBOLIC)
endforeach()

add_subdirectory(unsymmetry)