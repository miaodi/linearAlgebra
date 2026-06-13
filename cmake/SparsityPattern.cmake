include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/DownloadCache.cmake")

option(LINEAR_ALGEBRA_GENERATE_SPARSITY_PATTERNS
    "Generate PNG sparsity pattern images for downloaded matrices" ON)

set(LINEAR_ALGEBRA_SPARSITY_IMAGE_SIZE "4096"
    CACHE STRING "Maximum width or height for generated sparsity pattern PNGs")

function(linear_algebra_sparsity_pattern_path matrix_file output_var)
    get_filename_component(_matrix_dir "${matrix_file}" DIRECTORY)
    get_filename_component(_matrix_stem "${matrix_file}" NAME_WE)
    set(${output_var} "${_matrix_dir}/${_matrix_stem}_pattern.png" PARENT_SCOPE)
endfunction()

function(linear_algebra_expose_file source_file output_file)
    if(NOT EXISTS "${source_file}")
        message(FATAL_ERROR "Cannot expose missing file: ${source_file}")
    endif()

    if("${source_file}" STREQUAL "${output_file}")
        return()
    endif()

    get_filename_component(_output_dir "${output_file}" DIRECTORY)
    file(MAKE_DIRECTORY "${_output_dir}")

    if(IS_SYMLINK "${output_file}" AND NOT EXISTS "${output_file}")
        file(REMOVE "${output_file}")
    endif()

    if(EXISTS "${output_file}")
        return()
    endif()

    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E create_symlink "${source_file}" "${output_file}"
        RESULT_VARIABLE _link_result
        ERROR_VARIABLE _link_error
    )
    if(NOT _link_result EQUAL 0)
        configure_file("${source_file}" "${output_file}" COPYONLY)
    endif()
endfunction()

function(linear_algebra_generate_sparsity_pattern matrix_file)
    linear_algebra_sparsity_pattern_path("${matrix_file}" _pattern_file)

    if(ARGC GREATER 1)
        set(${ARGV1} "${_pattern_file}" PARENT_SCOPE)
    endif()

    if(NOT LINEAR_ALGEBRA_GENERATE_SPARSITY_PATTERNS)
        return()
    endif()

    if(EXISTS "${_pattern_file}")
        return()
    endif()

    if(NOT EXISTS "${matrix_file}")
        message(FATAL_ERROR "Cannot draw missing matrix file: ${matrix_file}")
    endif()

    linear_algebra_prefer_user_python()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    message(STATUS "Generating sparsity pattern ${_pattern_file}")
    execute_process(
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/draw_sparsity_pattern.py"
                "${matrix_file}"
                "${_pattern_file}"
                "--max-size" "${LINEAR_ALGEBRA_SPARSITY_IMAGE_SIZE}"
        RESULT_VARIABLE _pattern_result
        OUTPUT_VARIABLE _pattern_output
        ERROR_VARIABLE _pattern_error
    )
    if(NOT _pattern_result EQUAL 0)
        message(FATAL_ERROR
            "Failed to generate sparsity pattern for ${matrix_file}:\n"
            "${_pattern_error}${_pattern_output}")
    endif()
endfunction()

function(linear_algebra_expose_sparsity_pattern matrix_file output_dir)
    if(NOT LINEAR_ALGEBRA_GENERATE_SPARSITY_PATTERNS)
        return()
    endif()

    linear_algebra_generate_sparsity_pattern("${matrix_file}" _source_pattern_file)
    get_filename_component(_matrix_stem "${matrix_file}" NAME_WE)
    get_filename_component(_output_dir "${output_dir}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    set(_output_pattern_file "${_output_dir}/${_matrix_stem}_pattern.png")
    linear_algebra_expose_file("${_source_pattern_file}" "${_output_pattern_file}")
endfunction()
