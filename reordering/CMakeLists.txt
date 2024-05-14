cmake_minimum_required(VERSION 3.5...3.20)
project(reordering CXX)

set(REORDERING_SRC BFS.cpp Reordering.cpp UnionFind.cpp)

add_library(${PROJECT_NAME} STATIC ${REORDERING_SRC})

set(DEPENDENCIES mkl_wrapper Utils OpenMP::OpenMP_CXX)

if (USE_METIS_LIB)
 set(DEPENDENCIES ${DEPENDENCIES} MORSE::METIS)
endif(USE_METIS_LIB)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC ${DEPENDENCIES})