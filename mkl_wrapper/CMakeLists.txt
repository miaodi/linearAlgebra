cmake_minimum_required(VERSION 3.5...3.20)
project(mkl_wrapper CXX)

set(MKL_WRAPPER_SRC mkl_sparse_mat.cpp mkl_solver.cpp mkl_eigen.cpp mumps_solver.cpp arpack_wrapper.cpp incomplete_cholesky.cpp incomplete_lu.cpp amgcl_precond.cpp)

set(DEPENDENCIES $<LINK_ONLY:MKL::MKL> utils sparse_mat_op graph TBB::tbb OpenMP::OpenMP_CXX)

if(USE_ARPACK_LIB)
  set(DEPENDENCIES ${DEPENDENCIES} ARPACK::ARPACK)
endif()

if(USE_MUMPS_LIB)
  set(DEPENDENCIES ${DEPENDENCIES} MORSE::MUMPS -lgfortran)
endif()

if(USE_AMGCL_LIB)
  set(DEPENDENCIES ${DEPENDENCIES} amgcl::amgcl)
endif()

if(USE_BOOST_LIB)
  set(DEPENDENCIES ${DEPENDENCIES} Boost::boost)
endif()

if(USE_CUDA AND USE_CUDSS)
  set(MKL_WRAPPER_SRC ${MKL_WRAPPER_SRC} cudss_wrapper.cpp)
  set_source_files_properties(cudss_wrapper.cpp PROPERTIES LANGUAGE CUDA)
  set(DEPENDENCIES ${DEPENDENCIES} cudss)
endif()

add_library(${PROJECT_NAME} STATIC ${MKL_WRAPPER_SRC})

target_compile_options(${PROJECT_NAME} PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(${PROJECT_NAME} PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES> ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC ${DEPENDENCIES})