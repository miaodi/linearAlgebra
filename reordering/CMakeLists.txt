cmake_minimum_required(VERSION 3.5...3.20)
project(reordering CXX)

set(REORDERING_SRC BFS.cpp Reordering.cpp UnionFind.cpp MinimumDegree.cpp quotient_graph.cpp UnsymmReordering.cpp)

add_library(${PROJECT_NAME} STATIC ${REORDERING_SRC})

set(DEPENDENCIES mkl_wrapper utils OpenMP::OpenMP_CXX)

if(USE_METIS_LIB)
  set(DEPENDENCIES ${DEPENDENCIES} METIS::METIS)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC ${DEPENDENCIES})