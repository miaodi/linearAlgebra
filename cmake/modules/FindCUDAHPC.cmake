# FindCUDAHPC.cmake
# 
# This module handles CUDA detection for both standard CUDA Toolkit and NVIDIA HPC SDK
# 
# Sets up the following targets if found:
#   CUDA::cusparse
#   CUDA::cublas  
#   CUDA::cudart
#
# Variables set by this module:
#   CUDAHPC_FOUND - True if CUDA libraries are found
#   CUDAHPC_IS_HPC_SDK - True if using NVIDIA HPC SDK

# Function to create manual CUDA targets for HPC SDK
function(create_hpc_cuda_targets)
    if(NOT DEFINED ENV{NVHPC_ROOT})
        return()
    endif()

    message(STATUS "Detected NVIDIA HPC SDK - setting up CUDA manually")
    
    # Set paths for HPC SDK
    set(NVHPC_CUDA_ROOT "$ENV{NVHPC_ROOT}/cuda")
    set(NVHPC_MATH_LIBS "$ENV{NVHPC_ROOT}/math_libs")
    
    # cusparse library
    find_library(CUSPARSE_LIBRARY cusparse
        PATHS "${NVHPC_MATH_LIBS}/lib64" "${NVHPC_CUDA_ROOT}/lib64"
        NO_DEFAULT_PATH
    )
    if(CUSPARSE_LIBRARY)
        add_library(CUDA::cusparse SHARED IMPORTED)
        set_target_properties(CUDA::cusparse PROPERTIES
            IMPORTED_LOCATION "${CUSPARSE_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${NVHPC_MATH_LIBS}/include;${NVHPC_CUDA_ROOT}/include"
        )
        message(STATUS "Found cusparse: ${CUSPARSE_LIBRARY}")
    endif()
    
    # cublas library
    find_library(CUBLAS_LIBRARY cublas
        PATHS "${NVHPC_MATH_LIBS}/lib64" "${NVHPC_CUDA_ROOT}/lib64"
        NO_DEFAULT_PATH
    )
    if(CUBLAS_LIBRARY)
        add_library(CUDA::cublas SHARED IMPORTED)
        set_target_properties(CUDA::cublas PROPERTIES
            IMPORTED_LOCATION "${CUBLAS_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${NVHPC_MATH_LIBS}/include;${NVHPC_CUDA_ROOT}/include"
        )
        message(STATUS "Found cublas: ${CUBLAS_LIBRARY}")
    endif()
    
    # cudart library
    find_library(CUDART_LIBRARY cudart
        PATHS "${NVHPC_CUDA_ROOT}/lib64" "${NVHPC_MATH_LIBS}/lib64"
        NO_DEFAULT_PATH
    )
    if(CUDART_LIBRARY)
        add_library(CUDA::cudart SHARED IMPORTED)
        set_target_properties(CUDA::cudart PROPERTIES
            IMPORTED_LOCATION "${CUDART_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${NVHPC_CUDA_ROOT}/include"
        )
        message(STATUS "Found cudart: ${CUDART_LIBRARY}")
    endif()

    # Set variables to indicate success
    if(CUSPARSE_LIBRARY AND CUBLAS_LIBRARY AND CUDART_LIBRARY)
        set(CUDAHPC_FOUND TRUE PARENT_SCOPE)
        set(CUDAHPC_IS_HPC_SDK TRUE PARENT_SCOPE)
    endif()
endfunction()

# Main detection logic
if(DEFINED ENV{NVHPC_ROOT})
    # Try HPC SDK approach first
    create_hpc_cuda_targets()
    
    if(NOT CUDAHPC_FOUND)
        message(STATUS "HPC SDK CUDA setup failed, falling back to standard CUDAToolkit")
        find_package(CUDAToolkit)
        if(CUDAToolkit_FOUND)
            set(CUDAHPC_FOUND TRUE)
            set(CUDAHPC_IS_HPC_SDK FALSE)
        endif()
    endif()
else()
    # Standard CUDA Toolkit installation
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        set(CUDAHPC_FOUND TRUE)
        set(CUDAHPC_IS_HPC_SDK FALSE)
    endif()
endif()

# Report results
if(CUDAHPC_FOUND)
    if(CUDAHPC_IS_HPC_SDK)
        message(STATUS "CUDA support enabled via NVIDIA HPC SDK")
    else()
        message(STATUS "CUDA support enabled via standard CUDA Toolkit")
    endif()
else()
    message(STATUS "CUDA support not found")
endif()