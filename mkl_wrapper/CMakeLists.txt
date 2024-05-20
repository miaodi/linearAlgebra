cmake_minimum_required(VERSION 3.5...3.20)
project(mkl_wrapper CXX)

set(MKL_WRAPPER_SRC mkl_sparse_mat.cpp mkl_solver.cpp mkl_eigen.cpp mumps_solver.cpp arpack_wrapper.cpp incomplete_cholesky.cpp incomplete_lu.cpp)

add_library(${PROJECT_NAME} STATIC ${MKL_WRAPPER_SRC})
# target_sources(${PROJECT_NAME} PRIVATE  ${MKL_WRAPPER_SRC} PUBLIC ${MKL_WRAPPER_INC} )

set(DEPENDENCIES $<LINK_ONLY:MKL::MKL> Utils TBB::tbb OpenMP::OpenMP_CXX)

if (USE_ARPACK_LIB)
 set(DEPENDENCIES ${DEPENDENCIES} ARPACK::ARPACK)
endif(USE_ARPACK_LIB)

if (USE_MUMPS_LIB)
 set(DEPENDENCIES ${DEPENDENCIES} MORSE::MUMPS -lgfortran)
endif(USE_MUMPS_LIB)

target_compile_options(${PROJECT_NAME} PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(${PROJECT_NAME} PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES> ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC ${DEPENDENCIES})