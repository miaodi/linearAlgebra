cmake_minimum_required(VERSION 3.5...3.20)
project(solver CXX)

set(SOLVER_SRC 
    LinearSolverSystem.cpp
    Transformation.cpp
    TransformSeq.cpp
)

set(DEPENDENCIES utils sparse_mat_op graph reordering)

add_library(${PROJECT_NAME} STATIC ${SOLVER_SRC})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC ${DEPENDENCIES})
