# FindMETIS.cmake
# ---------------
# Find METIS library for graph partitioning and sparse matrix ordering
#
# This module defines:
#  METIS_FOUND          - True if METIS is found
#  METIS_INCLUDE_DIRS   - Include directories for METIS
#  METIS_LIBRARIES      - Libraries to link against
#  METIS_VERSION        - Version of METIS found
#  METIS::METIS         - Imported target (if found)
#
# You can set the following variables to help find METIS:
#  METIS_ROOT           - Root directory of METIS installation
#  METIS_INCLUDE_DIR    - Directory containing metis.h
#  METIS_LIBRARY        - Path to METIS library
#
# Environment variables:
#  METIS_ROOT, METIS_DIR - Root directory of METIS installation

# Try to find via PkgConfig first
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_METIS QUIET metis)
endif()

# Find the header file
find_path(METIS_INCLUDE_DIR
    NAMES metis.h
    HINTS
        ${METIS_ROOT}
        ${METIS_DIR}
        $ENV{METIS_ROOT}
        $ENV{METIS_DIR}
        ${PC_METIS_INCLUDE_DIRS}
    PATH_SUFFIXES
        include
        include/metis
)

# Find the library
find_library(METIS_LIBRARY
    NAMES metis
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

# Extract version from header if possible
if(METIS_INCLUDE_DIR AND EXISTS "${METIS_INCLUDE_DIR}/metis.h")
    file(STRINGS "${METIS_INCLUDE_DIR}/metis.h" METIS_VERSION_MAJOR_LINE REGEX "^#define[\t ]+METIS_VER_MAJOR")
    file(STRINGS "${METIS_INCLUDE_DIR}/metis.h" METIS_VERSION_MINOR_LINE REGEX "^#define[\t ]+METIS_VER_MINOR")
    file(STRINGS "${METIS_INCLUDE_DIR}/metis.h" METIS_VERSION_SUBMINOR_LINE REGEX "^#define[\t ]+METIS_VER_SUBMINOR")
    
    if(METIS_VERSION_MAJOR_LINE)
        string(REGEX REPLACE "^#define[\t ]+METIS_VER_MAJOR[\t ]+([0-9]+).*" "\\1" METIS_VERSION_MAJOR "${METIS_VERSION_MAJOR_LINE}")
        string(REGEX REPLACE "^#define[\t ]+METIS_VER_MINOR[\t ]+([0-9]+).*" "\\1" METIS_VERSION_MINOR "${METIS_VERSION_MINOR_LINE}")
        string(REGEX REPLACE "^#define[\t ]+METIS_VER_SUBMINOR[\t ]+([0-9]+).*" "\\1" METIS_VERSION_SUBMINOR "${METIS_VERSION_SUBMINOR_LINE}")
        set(METIS_VERSION "${METIS_VERSION_MAJOR}.${METIS_VERSION_MINOR}.${METIS_VERSION_SUBMINOR}")
    endif()
endif()

# Test if METIS actually works by compiling a minimal example
if(METIS_LIBRARY AND METIS_INCLUDE_DIR)
    include(CheckCSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES ${METIS_INCLUDE_DIR})
    set(CMAKE_REQUIRED_LIBRARIES ${METIS_LIBRARY})
    
    # Add math library if found
    find_library(M_LIBRARY m)
    if(M_LIBRARY)
        list(APPEND CMAKE_REQUIRED_LIBRARIES ${M_LIBRARY})
    endif()
    
    check_c_source_compiles("
        #include <metis.h>
        int main() {
            idx_t nvtxs = 6;
            idx_t ncon = 1;
            idx_t xadj[] = {0, 2, 5, 8, 11, 13, 15};
            idx_t adjncy[] = {1, 3, 0, 2, 4, 1, 3, 5, 0, 2, 4, 1, 5, 2, 4};
            idx_t nparts = 2;
            idx_t objval;
            idx_t part[6];
            METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, NULL, NULL, NULL,
                                &nparts, NULL, NULL, NULL, &objval, part);
            return 0;
        }
    " METIS_COMPILES)
    
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
    
    # Create imported target
    if(NOT TARGET METIS::METIS)
        add_library(METIS::METIS UNKNOWN IMPORTED)
        set_target_properties(METIS::METIS PROPERTIES
            IMPORTED_LOCATION "${METIS_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${METIS_INCLUDE_DIR}"
        )
        
        # METIS may depend on math library
        find_library(M_LIBRARY m)
        if(M_LIBRARY)
            set_target_properties(METIS::METIS PROPERTIES
                INTERFACE_LINK_LIBRARIES "${M_LIBRARY}"
            )
        endif()
    endif()
    
    mark_as_advanced(METIS_INCLUDE_DIR METIS_LIBRARY)
endif()
