include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/DownloadCache.cmake")

linear_algebra_set_default_cache_dir(
    LINEAR_ALGEBRA_MATRIX_CACHE_DIR
    "matrices"
    "Directory used to cache downloaded SuiteSparse Matrix Market files")

function(download_sparse_matrix matrix_name output_dir)
    # matrix_name: string like "HB/bcspwr01"
    # output_dir: where to expose the .mtx file in the current build tree
    linear_algebra_prefer_user_python()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    get_filename_component(_matrix_stem "${matrix_name}" NAME)
    set(_matrix_file "${_matrix_stem}.mtx")
    get_filename_component(_output_dir "${output_dir}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    set(_output_file "${_output_dir}/${_matrix_file}")
    set(_cache_file "${LINEAR_ALGEBRA_MATRIX_CACHE_DIR}/${_matrix_file}")

    if(EXISTS "${_output_file}")
        message(STATUS "Sparse matrix ${matrix_name} already available: ${_output_file}")
        return()
    endif()

    file(MAKE_DIRECTORY "${_output_dir}" "${LINEAR_ALGEBRA_MATRIX_CACHE_DIR}")

    if(NOT EXISTS "${_cache_file}")
        message(STATUS "Downloading sparse matrix ${matrix_name} into cache ${LINEAR_ALGEBRA_MATRIX_CACHE_DIR} ...")
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/cmake/download_matrix.py"
                    "${matrix_name}" "${LINEAR_ALGEBRA_MATRIX_CACHE_DIR}"
            RESULT_VARIABLE _result
            OUTPUT_VARIABLE _output
            ERROR_VARIABLE _error
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        )
        if(NOT _result EQUAL 0)
            message(FATAL_ERROR "Failed to download ${matrix_name}: ${_error}${_output}")
        endif()
    else()
        message(STATUS "Using cached sparse matrix ${matrix_name}: ${_cache_file}")
    endif()

    if(NOT EXISTS "${_cache_file}")
        message(FATAL_ERROR "Download for ${matrix_name} completed but did not produce ${_cache_file}")
    endif()

    if(IS_SYMLINK "${_output_file}" AND NOT EXISTS "${_output_file}")
        file(REMOVE "${_output_file}")
    endif()

    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E create_symlink "${_cache_file}" "${_output_file}"
        RESULT_VARIABLE _link_result
        ERROR_VARIABLE _link_error
    )
    if(NOT _link_result EQUAL 0)
        configure_file("${_cache_file}" "${_output_file}" COPYONLY)
    endif()

    message(STATUS "Sparse matrix ${matrix_name} available at ${_output_file}")
endfunction()
