include(FetchContent)

option(CMAKE_TLS_VERIFY "Verify SSL certificates" ON)

function(download_file url hash filename)

FetchContent_Declare(${filename}
URL ${url}
URL_HASH SHA256=${hash}
DOWNLOAD_NO_EXTRACT true
)

if(NOT ${filename}_POPULATED)
    FetchContent_MakeAvailable(${filename})
endif()
message("src_folder: ${${filename}_SOURCE_DIR}")
file(ARCHIVE_EXTRACT INPUT ${bcsstk26_SOURCE_DIR}/bcsstk26.mtx.gz)
endfunction(download_file)

# # === example
# download_file(
#   https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg
#   12794390cce7d0682ffc783c785e4282305684431b30b29ed75c224da24035b4
# )