# FindMUMPS.cmake
# ---------------
# Find MUMPS (MUltifrontal Massively Parallel sparse direct Solver)
#
# This module defines:
#  MUMPS_FOUND          - True if MUMPS is found
#  MUMPS_INCLUDE_DIRS   - Include directories for MUMPS
#  MUMPS_LIBRARIES      - Libraries to link against
#  MUMPS_VERSION        - Version of MUMPS found
#  MUMPS::MUMPS         - Imported target (if found)
#
# Components (specify with find_package(MUMPS COMPONENTS ...)):
#  SEQ      - Sequential version (no MPI)
#  MPI      - MPI parallel version (default if no components specified)
#  OPENMP   - OpenMP parallel version
#  s        - Single precision
#  d        - Double precision (default)
#  c        - Complex single precision
#  z        - Complex double precision
#
# You can set the following variables to help find MUMPS:
#  MUMPS_ROOT           - Root directory of MUMPS installation
#  MUMPS_INCLUDE_DIR    - Directory containing MUMPS headers
#  MUMPS_LIBRARY_DIR    - Directory containing MUMPS libraries
#
# Environment variables:
#  MUMPS_ROOT, MUMPS_DIR - Root directory of MUMPS installation

# Default to double precision if no arithmetic type specified
set(MUMPS_FIND_COMPONENTS_PREC "d")
set(MUMPS_USE_SEQ FALSE)
set(MUMPS_USE_OPENMP FALSE)

if(MUMPS_FIND_COMPONENTS)
    foreach(comp ${MUMPS_FIND_COMPONENTS})
        if(comp STREQUAL "SEQ")
            set(MUMPS_USE_SEQ TRUE)
        elseif(comp STREQUAL "OPENMP")
            set(MUMPS_USE_OPENMP TRUE)
        elseif(comp MATCHES "^[sdcz]$")
            list(APPEND MUMPS_FIND_COMPONENTS_PREC ${comp})
        endif()
    endforeach()
    list(REMOVE_DUPLICATES MUMPS_FIND_COMPONENTS_PREC)
endif()

# Find the include directory
find_path(MUMPS_INCLUDE_DIR
    NAMES dmumps_c.h
    HINTS
        ${MUMPS_ROOT}
        ${MUMPS_DIR}
        $ENV{MUMPS_ROOT}
        $ENV{MUMPS_DIR}
    PATH_SUFFIXES
        include
        include/mumps
        MUMPS/include
)

# Find common library
find_library(MUMPS_COMMON_LIBRARY
    NAMES mumps_common
    HINTS
        ${MUMPS_ROOT}
        ${MUMPS_DIR}
        $ENV{MUMPS_ROOT}
        $ENV{MUMPS_DIR}
        ${MUMPS_LIBRARY_DIR}
    PATH_SUFFIXES
        lib
        lib64
        lib32
        MUMPS/lib
)

# Find pord library (ordering library)
find_library(MUMPS_PORD_LIBRARY
    NAMES pord
    HINTS
        ${MUMPS_ROOT}
        ${MUMPS_DIR}
        $ENV{MUMPS_ROOT}
        $ENV{MUMPS_DIR}
        ${MUMPS_LIBRARY_DIR}
    PATH_SUFFIXES
        lib
        lib64
        lib32
        MUMPS/lib
)

# Find sequential library if requested
if(MUMPS_USE_SEQ)
    find_library(MUMPS_SEQ_LIBRARY
        NAMES mpiseq
        HINTS
            ${MUMPS_ROOT}
            ${MUMPS_DIR}
            $ENV{MUMPS_ROOT}
            $ENV{MUMPS_DIR}
            ${MUMPS_LIBRARY_DIR}
        PATH_SUFFIXES
            lib
            lib64
            lib32
            libseq
            MUMPS/libseq
    )
endif()

# Find precision-specific libraries
set(MUMPS_PRECISION_LIBRARIES)
foreach(prec ${MUMPS_FIND_COMPONENTS_PREC})
    find_library(MUMPS_${prec}_LIBRARY
        NAMES ${prec}mumps
        HINTS
            ${MUMPS_ROOT}
            ${MUMPS_DIR}
            $ENV{MUMPS_ROOT}
            $ENV{MUMPS_DIR}
            ${MUMPS_LIBRARY_DIR}
        PATH_SUFFIXES
            lib
            lib64
            lib32
            MUMPS/lib
    )
    if(MUMPS_${prec}_LIBRARY)
        list(APPEND MUMPS_PRECISION_LIBRARIES ${MUMPS_${prec}_LIBRARY})
    endif()
endforeach()

# Extract version if possible
if(MUMPS_INCLUDE_DIR AND EXISTS "${MUMPS_INCLUDE_DIR}/dmumps_c.h")
    file(STRINGS "${MUMPS_INCLUDE_DIR}/dmumps_c.h" MUMPS_VERSION_LINE 
         REGEX "^#define[\t ]+MUMPS_VERSION[\t ]+\"")
    if(MUMPS_VERSION_LINE)
        string(REGEX REPLACE "^#define[\t ]+MUMPS_VERSION[\t ]+\"([0-9.]+).*" "\\1" 
               MUMPS_VERSION "${MUMPS_VERSION_LINE}")
    endif()
endif()

# Find dependencies
set(MUMPS_REQUIRED_VARS MUMPS_COMMON_LIBRARY MUMPS_INCLUDE_DIR)

# Find BLAS (required)
find_package(BLAS QUIET)
if(BLAS_FOUND)
    list(APPEND MUMPS_REQUIRED_VARS BLAS_LIBRARIES)
endif()

# Find OpenMP if requested
if(MUMPS_USE_OPENMP)
    find_package(OpenMP QUIET COMPONENTS C Fortran)
    if(OpenMP_FOUND)
        list(APPEND MUMPS_REQUIRED_VARS OpenMP_C_FOUND)
    endif()
endif()

# Find Threads (often needed)
find_package(Threads QUIET)

# Find METIS if it's being used
if("METIS" IN_LIST MUMPS_FIND_COMPONENTS OR USE_METIS_LIB)
    find_package(METIS QUIET)
endif()

# Handle sequential vs MPI
if(MUMPS_USE_SEQ)
    list(APPEND MUMPS_REQUIRED_VARS MUMPS_SEQ_LIBRARY)
endif()

# Add precision libraries to required vars
if(MUMPS_PRECISION_LIBRARIES)
    list(APPEND MUMPS_REQUIRED_VARS MUMPS_PRECISION_LIBRARIES)
endif()

# Test if MUMPS actually works by compiling a minimal example
if(MUMPS_COMMON_LIBRARY AND MUMPS_INCLUDE_DIR AND MUMPS_PRECISION_LIBRARIES)
    include(CheckCSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES ${MUMPS_INCLUDE_DIR})
    set(CMAKE_REQUIRED_LIBRARIES 
        ${MUMPS_PRECISION_LIBRARIES}
        ${MUMPS_COMMON_LIBRARY}
    )
    
    if(MUMPS_PORD_LIBRARY)
        list(APPEND CMAKE_REQUIRED_LIBRARIES ${MUMPS_PORD_LIBRARY})
    endif()
    
    if(MUMPS_USE_SEQ AND MUMPS_SEQ_LIBRARY)
        list(APPEND CMAKE_REQUIRED_LIBRARIES ${MUMPS_SEQ_LIBRARY})
    endif()
    
    if(BLAS_FOUND)
        list(APPEND CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
    endif()
    
    if(METIS_FOUND)
        list(APPEND CMAKE_REQUIRED_LIBRARIES ${METIS_LIBRARIES})
    endif()
    
    if(Threads_FOUND)
        list(APPEND CMAKE_REQUIRED_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
    endif()
    
    find_library(GFORTRAN_LIBRARY gfortran)
    if(GFORTRAN_LIBRARY)
        list(APPEND CMAKE_REQUIRED_LIBRARIES ${GFORTRAN_LIBRARY})
    endif()
    
    check_c_source_compiles("
        #include <dmumps_c.h>
        int main() {
            DMUMPS_STRUC_C id;
            id.comm_fortran = -987654; /* Sequential mode */
            id.par = 1; /* Host participates in factorization */
            id.sym = 0; /* Unsymmetric matrix */
            id.job = -1; /* Initialize */
            dmumps_c(&id);
            id.job = -2; /* Finalize */
            dmumps_c(&id);
            return 0;
        }
    " MUMPS_COMPILES)
    
    if(NOT MUMPS_COMPILES)
        set(MUMPS_COMMON_LIBRARY MUMPS_COMMON_LIBRARY-NOTFOUND)
        set(MUMPS_INCLUDE_DIR MUMPS_INCLUDE_DIR-NOTFOUND)
    endif()
endif()

# Handle standard arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MUMPS
    REQUIRED_VARS ${MUMPS_REQUIRED_VARS}
    VERSION_VAR MUMPS_VERSION
)

# Set output variables
if(MUMPS_FOUND)
    set(MUMPS_INCLUDE_DIRS ${MUMPS_INCLUDE_DIR})
    set(MUMPS_LIBRARIES 
        ${MUMPS_PRECISION_LIBRARIES}
        ${MUMPS_COMMON_LIBRARY}
    )
    
    if(MUMPS_PORD_LIBRARY)
        list(APPEND MUMPS_LIBRARIES ${MUMPS_PORD_LIBRARY})
    endif()
    
    if(MUMPS_USE_SEQ AND MUMPS_SEQ_LIBRARY)
        list(APPEND MUMPS_LIBRARIES ${MUMPS_SEQ_LIBRARY})
    endif()
    
    if(BLAS_FOUND)
        list(APPEND MUMPS_LIBRARIES ${BLAS_LIBRARIES})
    endif()
    
    if(METIS_FOUND)
        list(APPEND MUMPS_LIBRARIES ${METIS_LIBRARIES})
    endif()
    
    if(Threads_FOUND)
        list(APPEND MUMPS_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
    endif()
    
    # Find Fortran runtime libraries
    find_library(GFORTRAN_LIBRARY gfortran)
    if(GFORTRAN_LIBRARY)
        list(APPEND MUMPS_LIBRARIES ${GFORTRAN_LIBRARY})
    endif()
    
    # Create imported target
    if(NOT TARGET MUMPS::MUMPS)
        add_library(MUMPS::MUMPS INTERFACE IMPORTED)
        set_target_properties(MUMPS::MUMPS PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${MUMPS_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "${MUMPS_LIBRARIES}"
        )
        
        if(MUMPS_USE_OPENMP AND OpenMP_C_FOUND)
            target_link_libraries(MUMPS::MUMPS INTERFACE OpenMP::OpenMP_C OpenMP::OpenMP_Fortran)
        endif()
    endif()
    
    mark_as_advanced(
        MUMPS_INCLUDE_DIR
        MUMPS_COMMON_LIBRARY
        MUMPS_PORD_LIBRARY
        MUMPS_SEQ_LIBRARY
    )
    
    foreach(prec ${MUMPS_FIND_COMPONENTS_PREC})
        mark_as_advanced(MUMPS_${prec}_LIBRARY)
    endforeach()
endif()
