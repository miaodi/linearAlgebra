cmake_minimum_required(VERSION 3.5...3.20)
project(data)

include(../cmake/ssget.cmake)


download_sparse_matrix("msc00726" ${CMAKE_CURRENT_BINARY_DIR})

download_sparse_matrix("nasa2910" ${CMAKE_CURRENT_BINARY_DIR})

download_sparse_matrix("plat1919" ${CMAKE_CURRENT_BINARY_DIR})

download_sparse_matrix("bcsstk05" ${CMAKE_CURRENT_BINARY_DIR})


message("CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
message("CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB MATRIX_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(MATRIX_FILE ${MATRIX_FILES})
   get_filename_component(FILENAME ${MATRIX_FILE} NAME)
   message("FILENAME: ${FILENAME}")
   file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME} ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME} SYMBOLIC)
endforeach()