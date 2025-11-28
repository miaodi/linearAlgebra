# FindMETIS.cmake
# ---------------
# Find METIS library for graph partitioning and sparse matrix ordering
# Supports both standard METIS and mt-metis (multithreaded METIS)
#
# This module defines:
#  METIS_FOUND          - True if METIS is found
#  METIS_INCLUDE_DIRS   - Include directories for METIS
#  METIS_LIBRARIES      - Libraries to link against
#  METIS_VERSION        - Version of METIS found
#  METIS_IS_MTMETIS     - True if mt-metis (multithreaded) is found
#  METIS::METIS         - Imported target (if found)
#
# You can set the following variables to help find METIS:
#  METIS_ROOT           - Root directory of METIS installation
#  METIS_DIR            - Root directory of METIS installation
#  METIS_INCLUDE_DIR    - Directory containing metis.h or mtmetis.h
#  METIS_LIBRARY        - Path to METIS library
#
# Environment variables:
#  METIS_ROOT, METIS_DIR - Root directory of METIS installation

# Try to find via PkgConfig first
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_METIS QUIET metis)
endif()

# Try to find mt-metis first (multithreaded version), then fall back to standard METIS
find_path(METIS_INCLUDE_DIR
    NAMES mtmetis.h metis.h
    HINTS
        ${METIS_ROOT}
        ${METIS_DIR}
        $ENV{METIS_ROOT}
        $ENV{METIS_DIR}
        ${PC_METIS_INCLUDE_DIRS}
    PATH_SUFFIXES
        include
        include/metis
        include/mtmetis
)

# Check which variant we found
set(METIS_IS_MTMETIS FALSE)
if(METIS_INCLUDE_DIR)
    if(EXISTS "${METIS_INCLUDE_DIR}/mtmetis.h")
        set(METIS_IS_MTMETIS TRUE)
        set(METIS_HEADER_FILE "mtmetis.h")
        set(METIS_LIB_NAMES mtmetis)
    else()
        set(METIS_HEADER_FILE "metis.h")
        set(METIS_LIB_NAMES metis)
    endif()
endif()

# Find the library
find_library(METIS_LIBRARY
    NAMES ${METIS_LIB_NAMES}
    HINTS
        ${METIS_ROOT}
        ${METIS_DIR}
        $ENV{METIS_ROOT}
        $ENV{METIS_DIR}
        ${PC_METIS_LIBRARY_DIRS}
    PATH_SUFFIXES
        lib
        lib64
        lib32
)

# Find GKlib (dependency of standard METIS)
find_library(GKLIB_LIBRARY
    NAMES GKlib
    HINTS
        ${METIS_ROOT}
        ${METIS_DIR}
        $ENV{METIS_ROOT}
        $ENV{METIS_DIR}
        ${GKLIB_ROOT}
        ${GKLIB_DIR}
        $ENV{GKLIB_ROOT}
        $ENV{GKLIB_DIR}
        ${PC_METIS_LIBRARY_DIRS}
    PATH_SUFFIXES
        lib
        lib64
        lib32
        GKlib/lib
        gklib/lib
)

# Extract version from header if possible
if(METIS_INCLUDE_DIR AND EXISTS "${METIS_INCLUDE_DIR}/${METIS_HEADER_FILE}")
    if(METIS_IS_MTMETIS)
        # mt-metis version extraction
        file(STRINGS "${METIS_INCLUDE_DIR}/${METIS_HEADER_FILE}" METIS_VERSION_MAJOR_LINE REGEX "^#define[\t ]+MTMETIS_VER_MAJOR")
        file(STRINGS "${METIS_INCLUDE_DIR}/${METIS_HEADER_FILE}" METIS_VERSION_MINOR_LINE REGEX "^#define[\t ]+MTMETIS_VER_MINOR")
        file(STRINGS "${METIS_INCLUDE_DIR}/${METIS_HEADER_FILE}" METIS_VERSION_SUBMINOR_LINE REGEX "^#define[\t ]+MTMETIS_VER_SUBMINOR")
        
        if(METIS_VERSION_MAJOR_LINE)
            string(REGEX REPLACE "^#define[\t ]+MTMETIS_VER_MAJOR[\t ]+([0-9]+).*" "\\1" METIS_VERSION_MAJOR "${METIS_VERSION_MAJOR_LINE}")
            string(REGEX REPLACE "^#define[\t ]+MTMETIS_VER_MINOR[\t ]+([0-9]+).*" "\\1" METIS_VERSION_MINOR "${METIS_VERSION_MINOR_LINE}")
            string(REGEX REPLACE "^#define[\t ]+MTMETIS_VER_SUBMINOR[\t ]+([0-9]+).*" "\\1" METIS_VERSION_SUBMINOR "${METIS_VERSION_SUBMINOR_LINE}")
            set(METIS_VERSION "${METIS_VERSION_MAJOR}.${METIS_VERSION_MINOR}.${METIS_VERSION_SUBMINOR}")
        endif()
    else()
        # Standard METIS version extraction
        file(STRINGS "${METIS_INCLUDE_DIR}/${METIS_HEADER_FILE}" METIS_VERSION_MAJOR_LINE REGEX "^#define[\t ]+METIS_VER_MAJOR")
        file(STRINGS "${METIS_INCLUDE_DIR}/${METIS_HEADER_FILE}" METIS_VERSION_MINOR_LINE REGEX "^#define[\t ]+METIS_VER_MINOR")
        file(STRINGS "${METIS_INCLUDE_DIR}/${METIS_HEADER_FILE}" METIS_VERSION_SUBMINOR_LINE REGEX "^#define[\t ]+METIS_VER_SUBMINOR")
        
        if(METIS_VERSION_MAJOR_LINE)
            string(REGEX REPLACE "^#define[\t ]+METIS_VER_MAJOR[\t ]+([0-9]+).*" "\\1" METIS_VERSION_MAJOR "${METIS_VERSION_MAJOR_LINE}")
            string(REGEX REPLACE "^#define[\t ]+METIS_VER_MINOR[\t ]+([0-9]+).*" "\\1" METIS_VERSION_MINOR "${METIS_VERSION_MINOR_LINE}")
            string(REGEX REPLACE "^#define[\t ]+METIS_VER_SUBMINOR[\t ]+([0-9]+).*" "\\1" METIS_VERSION_SUBMINOR "${METIS_VERSION_SUBMINOR_LINE}")
            set(METIS_VERSION "${METIS_VERSION_MAJOR}.${METIS_VERSION_MINOR}.${METIS_VERSION_SUBMINOR}")
        endif()
    endif()
endif()

# Test if METIS actually works by compiling a minimal example
if(METIS_LIBRARY AND METIS_INCLUDE_DIR)
    # Enable C language for the compile test
    enable_language(C)
    
    include(CheckCSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES ${METIS_INCLUDE_DIR})
    set(CMAKE_REQUIRED_LIBRARIES "")
    set(CMAKE_REQUIRED_LINK_OPTIONS "")
    
    # Important: Link order matters for static libraries!
    # METIS must come before GKlib since METIS depends on GKlib
    list(APPEND CMAKE_REQUIRED_LIBRARIES ${METIS_LIBRARY})
    
    # Add GKlib if found (required for standard METIS)
    if(GKLIB_LIBRARY AND NOT METIS_IS_MTMETIS)
        list(APPEND CMAKE_REQUIRED_LIBRARIES ${GKLIB_LIBRARY})
    endif()
    
    # Add math library if found
    find_library(M_LIBRARY m)
    if(M_LIBRARY)
        list(APPEND CMAKE_REQUIRED_LIBRARIES ${M_LIBRARY})
    endif()
    
    # For mt-metis, add OpenMP and pthread dependencies for the compile test
    if(METIS_IS_MTMETIS)
        find_package(Threads QUIET)
        if(Threads_FOUND)
            list(APPEND CMAKE_REQUIRED_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
        endif()
        
        # mt-metis needs OpenMP - prefer C, but CXX works too
        set(OPENMP_FOUND FALSE)
        if(DEFINED OpenMP_C_FOUND AND OpenMP_C_FOUND)
            if(DEFINED OpenMP_C_FLAGS)
                list(APPEND CMAKE_REQUIRED_LINK_OPTIONS ${OpenMP_C_FLAGS})
                set(OPENMP_FOUND TRUE)
            endif()
        elseif(DEFINED OpenMP_CXX_FOUND AND OpenMP_CXX_FOUND)
            # Fall back to CXX if C not available (same -fopenmp flag typically)
            if(DEFINED OpenMP_CXX_FLAGS)
                list(APPEND CMAKE_REQUIRED_LINK_OPTIONS ${OpenMP_CXX_FLAGS})
                set(OPENMP_FOUND TRUE)
            endif()
        endif()
        
        # If OpenMP wasn't found in parent scope, try to find it ourselves
        if(NOT OPENMP_FOUND)
            find_package(OpenMP QUIET COMPONENTS C)
            if(OpenMP_C_FOUND AND DEFINED OpenMP_C_FLAGS)
                list(APPEND CMAKE_REQUIRED_LINK_OPTIONS ${OpenMP_C_FLAGS})
                set(OPENMP_FOUND TRUE)
            else()
                # Try CXX as fallback
                find_package(OpenMP QUIET COMPONENTS CXX)
                if(OpenMP_CXX_FOUND AND DEFINED OpenMP_CXX_FLAGS)
                    list(APPEND CMAKE_REQUIRED_LINK_OPTIONS ${OpenMP_CXX_FLAGS})
                endif()
            endif()
        endif()
    endif()
    
    if(METIS_IS_MTMETIS)
        # Test mt-metis with MTMETIS_NodeND
        check_c_source_compiles("
            #include <mtmetis.h>
            int main() {
                mtmetis_vtx_type nvtxs = 6;
                mtmetis_adj_type xadj[] = {0, 2, 5, 8, 11, 13, 15};
                mtmetis_vtx_type adjncy[] = {1, 3, 0, 2, 4, 1, 3, 5, 0, 2, 4, 1, 5, 2, 4};
                mtmetis_pid_type iperm[6];
                mtmetis_pid_type perm[6];
                double options[MTMETIS_NOPTIONS];
                MTMETIS_NodeND(&nvtxs, xadj, adjncy, NULL, options, perm, iperm);
                return 0;
            }
        " METIS_COMPILES)
    else()
        # Test standard METIS with METIS_NodeND
        check_c_source_compiles("
            #include <stddef.h>
            #include <metis.h>
            int main() {
                idx_t nvtxs = 6;
                idx_t xadj[] = {0, 2, 5, 8, 11, 13, 15};
                idx_t adjncy[] = {1, 3, 0, 2, 4, 1, 3, 5, 0, 2, 4, 1, 5, 2, 4};
                idx_t iperm[6];
                idx_t perm[6];
                idx_t options[METIS_NOPTIONS];
                METIS_SetDefaultOptions(options);
                METIS_NodeND(&nvtxs, xadj, adjncy, NULL, options, perm, iperm);
                return 0;
            }
        " METIS_COMPILES)
    endif()
    
    if(NOT METIS_COMPILES)
        set(METIS_LIBRARY METIS_LIBRARY-NOTFOUND)
        set(METIS_INCLUDE_DIR METIS_INCLUDE_DIR-NOTFOUND)
    endif()
endif()

# Handle standard arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(METIS
    REQUIRED_VARS METIS_LIBRARY METIS_INCLUDE_DIR
    VERSION_VAR METIS_VERSION
)

# Set output variables
if(METIS_FOUND)
    set(METIS_INCLUDE_DIRS ${METIS_INCLUDE_DIR})
    set(METIS_LIBRARIES ${METIS_LIBRARY})
    
    # Add GKlib for standard METIS
    if(GKLIB_LIBRARY AND NOT METIS_IS_MTMETIS)
        list(APPEND METIS_LIBRARIES ${GKLIB_LIBRARY})
    endif()
    
    # Add thread library for mt-metis
    if(METIS_IS_MTMETIS)
        find_package(Threads QUIET)
        if(Threads_FOUND)
            list(APPEND METIS_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
        endif()
        
        # mt-metis also needs OpenMP
        find_package(OpenMP QUIET COMPONENTS C)
        if(OpenMP_C_FOUND)
            list(APPEND METIS_LIBRARIES ${OpenMP_C_LIBRARIES})
        endif()
    endif()
    
    # Create imported target
    if(NOT TARGET METIS::METIS)
        add_library(METIS::METIS UNKNOWN IMPORTED)
        set_target_properties(METIS::METIS PROPERTIES
            IMPORTED_LOCATION "${METIS_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${METIS_INCLUDE_DIR}"
        )
        
        # Add compile definition for mt-metis
        if(METIS_IS_MTMETIS)
            set_target_properties(METIS::METIS PROPERTIES
                INTERFACE_COMPILE_DEFINITIONS "USE_MTMETIS"
            )
        endif()
        
        set(METIS_LINK_LIBRARIES "")
        
        # METIS needs GKlib
        if(GKLIB_LIBRARY AND NOT METIS_IS_MTMETIS)
            list(APPEND METIS_LINK_LIBRARIES ${GKLIB_LIBRARY})
        endif()
        
        # METIS may depend on math library
        find_library(M_LIBRARY m)
        if(M_LIBRARY)
            list(APPEND METIS_LINK_LIBRARIES ${M_LIBRARY})
        endif()
        
        # mt-metis needs pthread
        if(METIS_IS_MTMETIS AND Threads_FOUND)
            list(APPEND METIS_LINK_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
        endif()
        
        # mt-metis also needs OpenMP
        if(METIS_IS_MTMETIS)
            find_package(OpenMP QUIET COMPONENTS C)
            if(OpenMP_C_FOUND)
                list(APPEND METIS_LINK_LIBRARIES OpenMP::OpenMP_C)
            endif()
        endif()
        
        if(METIS_LINK_LIBRARIES)
            set_target_properties(METIS::METIS PROPERTIES
                INTERFACE_LINK_LIBRARIES "${METIS_LINK_LIBRARIES}"
            )
        endif()
    endif()
    
    if(METIS_IS_MTMETIS)
        message(STATUS "Found mt-metis (multithreaded): ${METIS_LIBRARY} (version ${METIS_VERSION})")
    endif()
    
    mark_as_advanced(METIS_INCLUDE_DIR METIS_LIBRARY)
endif()
