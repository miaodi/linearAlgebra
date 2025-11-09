cmake_minimum_required(VERSION 3.5...3.20)
project(data)

include(${CMAKE_SOURCE_DIR}/cmake/ssget.cmake)

download_sparse_matrix("Hamrle1" ${CMAKE_CURRENT_BINARY_DIR})
download_sparse_matrix("d_ss" ${CMAKE_CURRENT_BINARY_DIR})
download_sparse_matrix("west0067" ${CMAKE_CURRENT_BINARY_DIR})
download_sparse_matrix("d_dyn" ${CMAKE_CURRENT_BINARY_DIR})
download_sparse_matrix("spaceStation_1" ${CMAKE_CURRENT_BINARY_DIR})
download_sparse_matrix("robot" ${CMAKE_CURRENT_BINARY_DIR})
download_sparse_matrix("impcol_b" ${CMAKE_CURRENT_BINARY_DIR})
download_sparse_matrix("west0132" ${CMAKE_CURRENT_BINARY_DIR})
download_sparse_matrix("mcfe" ${CMAKE_CURRENT_BINARY_DIR})
download_sparse_matrix("rotor2" ${CMAKE_CURRENT_BINARY_DIR})

# note: GD97_b is not structurally rank full
download_sparse_matrix("GD97_b" ${CMAKE_CURRENT_BINARY_DIR})
download_sparse_matrix("jpwh_991" ${CMAKE_CURRENT_BINARY_DIR})