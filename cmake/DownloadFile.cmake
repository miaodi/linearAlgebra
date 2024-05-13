include(FetchContent)

option(CMAKE_TLS_VERIFY "Verify SSL certificates" ON)

function(download_file url hash filename)
message("Downloading ${filename} from ${url} ${hash}")
if(${hash} STREQUAL "NONE")
    FetchContent_Declare(${filename}
    URL ${url}
    DOWNLOAD_NO_EXTRACT true
    )
else()
    FetchContent_Declare(${filename}
    URL ${url}
    URL_HASH SHA256=${hash}
    DOWNLOAD_NO_EXTRACT true
    )
endif()

if(NOT ${filename}_POPULATED)
    FetchContent_MakeAvailable(${filename})
    if(EXISTS ${${filename}_SOURCE_DIR}/${filename}.tar.gz)
        file(ARCHIVE_EXTRACT INPUT ${${filename}_SOURCE_DIR}/${filename}.tar.gz DESTINATION ${${filename}_SOURCE_DIR})
    elseif(EXISTS ${${filename}_SOURCE_DIR}/${filename}.mtx.gz)
        # file(ARCHIVE_EXTRACT INPUT ${${filename}_SOURCE_DIR}/${filename}.mtx.gz DESTINATION ${${filename}_SOURCE_DIR} COMPRESSION GZip)
        execute_process(COMMAND gunzip -d ${${filename}_SOURCE_DIR}/${filename}.mtx.gz)
    endif()
endif()
# message("src_folder: ${${filename}_SOURCE_DIR}")

endfunction(download_file)

# # === example
# download_file(
#   https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg
#   12794390cce7d0682ffc783c785e4282305684431b30b29ed75c224da24035b4
# )