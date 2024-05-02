cmake_minimum_required(VERSION 3.5...3.20)
project(benchmarks)

# Add Google Test
include(../cmake/GoogleTest.cmake)

include(../cmake/DownloadFile.cmake)

file(GLOB_RECURSE ALL_BENCH_CPP *.cpp)

foreach(ONE_BENCH_CPP ${ALL_BENCH_CPP})
   get_filename_component(ONE_BENCH_EXEC ${ONE_BENCH_CPP} NAME_WE)

   # Avoid name collision 
   set(TARGET_NAME Bench_${ONE_BENCH_EXEC})

   add_executable(${TARGET_NAME} ${ONE_BENCH_CPP})
   set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_BENCH_EXEC}) 
   target_link_libraries(${TARGET_NAME} PRIVATE benchmark::benchmark reordering Utils OpenMP::OpenMP_CXX)
   # add_test(${TARGET_NAME} ${ONE_BENCH_EXEC})
endforeach()
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/data)
download_file( https://suitesparse-collection-website.herokuapp.com/MM/VLSI/nv2.tar.gz
               b393f2abcb55732da1259c8c53b0a213e49f0c619511db68fa91f24104827cb0
               nv2
)

FetchContent_GetProperties(nv2 SOURCE_DIR NV2_SOURCE_DIR)
message(STATUS "NV2_SOURCE_DIR: ${NV2_SOURCE_DIR}")
file(CREATE_LINK ${NV2_SOURCE_DIR}/nv2/nv2.mtx ${CMAKE_CURRENT_BINARY_DIR}/data/nv2.mtx SYMBOLIC)

# find_file(EX5_MTX NAMES ex5.mtx PATHS ${CMAKE_BINARY_DIR}/tests NO_DEFAULT_PATH NO_CACHE REQUIRED)
# message(STATUS "EX5_MTX: ${EX5_MTX}")

# find_file(NV2_MTX NAMES nv2.mtx PATHS ${NV2_SOURCE_DIR} NO_DEFAULT_PATH)
# message(STATUS "NV2_MTX: ${NV2_MTX}")

# find_file(EX5_MTX NAMES ex5.mtx PATHS ${CMAKE_SOURCE_DIR}/tests NO_DEFAULT_PATH REQUIRED)
# message(${CMAKE_SOURCE_DIR}/tests/data)
# message(STATUS "EX5_MTX: ${EX5_MTX}")
download_file( 
   https://suitesparse-collection-website.herokuapp.com/MM/HB/mcfe.tar.gz
   70bde8afe35f5c1e1d822e5a1aa7a2a2a17da0a8622b1427af37a9b2be924970
   mcfe
)
# download_file( 
#    https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/bcsstruc4/bcsstk26.mtx.gz
#    ff4b1aed3d5ae52d7b7959363c195ac9e3413882f6be1fbb8ccccbcf99d0b5b4
#    bcsstk26
# )


# # === example
# download_file(
#   https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg
#   12794390cce7d0682ffc783c785e4282305684431b30b29ed75c224da24035b4
#   figure
# )