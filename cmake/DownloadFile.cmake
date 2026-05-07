include_guard(GLOBAL)

include(FetchContent)
include("${CMAKE_CURRENT_LIST_DIR}/DownloadCache.cmake")

option(CMAKE_TLS_VERIFY "Verify SSL certificates" ON)

linear_algebra_set_default_cache_dir(
    LINEAR_ALGEBRA_FILE_DOWNLOAD_CACHE_DIR
    "files"
    "Directory used to cache direct file downloads handled by FetchContent")

function(download_file url hash filename)
    message(STATUS "Preparing ${filename} from ${url}")

    set(_previous_fetchcontent_base_dir "${FETCHCONTENT_BASE_DIR}")
    set(FETCHCONTENT_BASE_DIR "${LINEAR_ALGEBRA_FILE_DOWNLOAD_CACHE_DIR}")

    if("${hash}" STREQUAL "NONE")
        FetchContent_Declare("${filename}"
            URL "${url}"
            DOWNLOAD_NO_EXTRACT true
        )
    else()
        FetchContent_Declare("${filename}"
            URL "${url}"
            URL_HASH "SHA256=${hash}"
            DOWNLOAD_NO_EXTRACT true
        )
    endif()

    FetchContent_GetProperties("${filename}" POPULATED _populated SOURCE_DIR _source_dir)
    if(NOT _populated)
        FetchContent_MakeAvailable("${filename}")
        FetchContent_GetProperties("${filename}" SOURCE_DIR _source_dir)
    endif()

    set(FETCHCONTENT_BASE_DIR "${_previous_fetchcontent_base_dir}")

    if(EXISTS "${_source_dir}/${filename}.tar.gz")
        if(NOT EXISTS "${_source_dir}/${filename}")
            file(ARCHIVE_EXTRACT INPUT "${_source_dir}/${filename}.tar.gz" DESTINATION "${_source_dir}")
        endif()
    elseif(EXISTS "${_source_dir}/${filename}.mtx.gz")
        if(NOT EXISTS "${_source_dir}/${filename}.mtx")
            execute_process(
                COMMAND gzip -cd "${_source_dir}/${filename}.mtx.gz"
                OUTPUT_FILE "${_source_dir}/${filename}.mtx"
                RESULT_VARIABLE _gzip_result
                ERROR_VARIABLE _gzip_error
            )
            if(NOT _gzip_result EQUAL 0)
                message(FATAL_ERROR "Failed to decompress ${_source_dir}/${filename}.mtx.gz: ${_gzip_error}")
            endif()
        endif()
    endif()

endfunction(download_file)

# # === example
# download_file(
#   https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg
#   12794390cce7d0682ffc783c785e4282305684431b30b29ed75c224da24035b4
# )
