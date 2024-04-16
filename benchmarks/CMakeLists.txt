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
   target_link_libraries(${TARGET_NAME} PRIVATE benchmark::benchmark reordering)
   add_test(${TARGET_NAME} ${ONE_BENCH_EXEC})
endforeach()

# download_file( https://suitesparse-collection-website.herokuapp.com/MM/VLSI/nv2.tar.gz
#                2abe5e50db8eab756d2f3eb47267c983a8c83948ac02da7c896b2620729edb5c
# )

# download_file( 
#    https://suitesparse-collection-website.herokuapp.com/MM/HB/mcfe.tar.gz
#    b6f8b4c00a331befabe53f007bda8142ea450d6c6d26f922289e72f797f4ee36
#    mcfe
# )
download_file( 
   https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/bcsstruc4/bcsstk26.mtx.gz
   ff4b1aed3d5ae52d7b7959363c195ac9e3413882f6be1fbb8ccccbcf99d0b5b4
   bcsstk26
)


# # === example
# download_file(
#   https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg
#   12794390cce7d0682ffc783c785e4282305684431b30b29ed75c224da24035b4
#   figure
# )