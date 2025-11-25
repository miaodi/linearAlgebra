# Scratch folder for temporary development binaries
# Automatically detects and builds all subdirectories with CMakeLists.txt

# Get all subdirectories
file(GLOB SCRATCH_SUBDIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

foreach(SUBDIR ${SCRATCH_SUBDIRS})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${SUBDIR})
        # Check if the subdirectory has a CMakeLists.txt
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${SUBDIR}/CMakeLists.txt)
            message(STATUS "Adding scratch subdirectory: ${SUBDIR}")
            add_subdirectory(${SUBDIR})
        endif()
    endif()
endforeach()
