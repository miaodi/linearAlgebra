function(download_sparse_matrix matrix_name output_dir)
    # matrix_name: string like "HB/bcspwr01"
    # output_dir: where to store the .mtx file

    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    message("Downloading sparse matrix ${matrix_name} into ${output_dir} ...")
    # Make sure output dir exists
    file(MAKE_DIRECTORY ${output_dir})
    
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/download_matrix.py
                ${matrix_name} ${output_dir}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    if(NOT result EQUAL 0)
    message(FATAL_ERROR "Failed to download ${matrix_name}: ${error}")
else()
    message(STATUS "Downloaded ${matrix_name}: ${output}")
endif()
endfunction()