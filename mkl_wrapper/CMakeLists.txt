cmake_minimum_required(VERSION 3.5...3.20)
project(mkl_wrapper CXX)

set(MKL_WRAPPER_SRC mkl_sparse_mat.cpp mkl_solver.cpp mkl_eigen.cpp)

add_library(${PROJECT_NAME} SHARED ${MKL_WRAPPER_SRC})
# target_sources(${PROJECT_NAME} PRIVATE  ${MKL_WRAPPER_SRC} PUBLIC ${MKL_WRAPPER_INC} )

target_compile_options(${PROJECT_NAME} PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(${PROJECT_NAME} PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES> Utils ${CMAKE_CURRENT_LIST_DIR}  ARPACK::ARPACK)
target_link_libraries(${PROJECT_NAME} PUBLIC $<LINK_ONLY:MKL::MKL> Utils  ARPACK::ARPACK TBB::tbb)