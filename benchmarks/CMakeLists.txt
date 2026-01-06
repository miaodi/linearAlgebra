cmake_minimum_required(VERSION 3.5...3.20)
project(benchmarks)

# Add Google Benchmark
include(../cmake/GoogleBenchmark.cmake)
include(../cmake/DownloadFile.cmake)

file(GLOB_RECURSE ALL_BENCH_CPP *.cpp)

# List of benchmarks to exclude due to compilation issues
set(EXCLUDED_BENCHMARKS 
    "precond_bench.cpp"
    "reordering_bench.cpp"
    "unionfind_bench.cpp"
)

foreach(ONE_BENCH_CPP ${ALL_BENCH_CPP})
   get_filename_component(ONE_BENCH_FILENAME ${ONE_BENCH_CPP} NAME)
   get_filename_component(ONE_BENCH_EXEC ${ONE_BENCH_CPP} NAME_WE)
   
   # Skip excluded benchmarks
   list(FIND EXCLUDED_BENCHMARKS ${ONE_BENCH_FILENAME} SHOULD_EXCLUDE)
   if(SHOULD_EXCLUDE GREATER -1)
      message(STATUS "Skipping benchmark: ${ONE_BENCH_FILENAME} (compilation issues)")
      continue()
   endif()

   # Check if this is a CUDA benchmark
   string(FIND ${ONE_BENCH_FILENAME} "cuda" CUDA_POS_LOWER)
   string(FIND ${ONE_BENCH_FILENAME} "CUDA" CUDA_POS_UPPER)
   set(IS_CUDA_BENCH FALSE)
   if(NOT CUDA_POS_LOWER EQUAL -1 OR NOT CUDA_POS_UPPER EQUAL -1)
      set(IS_CUDA_BENCH TRUE)
   endif()

   # Skip CUDA benchmarks if CUDA is not enabled
   if(IS_CUDA_BENCH AND NOT USE_CUDA)
      message(STATUS "Skipping CUDA benchmark: ${ONE_BENCH_FILENAME} (CUDA not enabled)")
      continue()
   endif()

   # Avoid name collision 
   set(TARGET_NAME Bench_${ONE_BENCH_EXEC})

   add_executable(${TARGET_NAME} ${ONE_BENCH_CPP})
   set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_BENCH_EXEC})
   
   # CUDA benchmarks need different dependencies and settings
   if(IS_CUDA_BENCH)
      set_target_properties(${TARGET_NAME} PROPERTIES 
         CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
         CUDA_STANDARD_REQUIRED ON
      )
      target_link_libraries(${TARGET_NAME} PRIVATE 
         benchmark::benchmark 
         sparse_mat_op_cuda
         CUDA::cudart
      )
   else()
      target_link_libraries(${TARGET_NAME} PRIVATE benchmark::benchmark reordering graph utils sparse_mat_op OpenMP::OpenMP_CXX cxxopts)
      target_enable_simd(${TARGET_NAME})
   endif()
endforeach()

# https://sparse.tamu.edu/VLSI/nv2
# 1,453,908
# no symmetry
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/data)
download_file( https://suitesparse-collection-website.herokuapp.com/MM/VLSI/nv2.tar.gz
               b393f2abcb55732da1259c8c53b0a213e49f0c619511db68fa91f24104827cb0
               nv2
)

FetchContent_GetProperties(nv2 SOURCE_DIR NV2_SOURCE_DIR)
file(CREATE_LINK ${NV2_SOURCE_DIR}/nv2/nv2.mtx ${CMAKE_CURRENT_BINARY_DIR}/data/nv2.mtx SYMBOLIC)

download_file( 
   https://suitesparse-collection-website.herokuapp.com/MM/HB/mcfe.tar.gz
   70bde8afe35f5c1e1d822e5a1aa7a2a2a17da0a8622b1427af37a9b2be924970
   mcfe
)

FetchContent_GetProperties(mcfe SOURCE_DIR FILE_SOURCE_DIR)
file(CREATE_LINK ${FILE_SOURCE_DIR}/mcfe/mcfe.mtx ${CMAKE_CURRENT_BINARY_DIR}/data/mcfe.mtx SYMBOLIC)

# https://sparse.tamu.edu/GHS_psdef/ldoor
# 952,203
# spd
download_file( 
   https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/ldoor.tar.gz   
   bd50b2641d69adcda7be4f209939c5747af75b2bd7a2da8847f42ce328b2f98a
   ldoor
)

FetchContent_GetProperties(ldoor SOURCE_DIR FILE_SOURCE_DIR)
file(CREATE_LINK ${FILE_SOURCE_DIR}/ldoor/ldoor.mtx ${CMAKE_CURRENT_BINARY_DIR}/data/ldoor.mtx SYMBOLIC)


# https://sparse.tamu.edu/Schmid/thermal2
# 1228045,1228045
# spd
download_file( 
   https://suitesparse-collection-website.herokuapp.com/MM/Schmid/thermal2.tar.gz  
   02934a4b642b6829c33517e0b801b60ea894a6552c6cd7e3db6c709c776434ce
   thermal2
)

FetchContent_GetProperties(thermal2 SOURCE_DIR FILE_SOURCE_DIR)
file(CREATE_LINK ${FILE_SOURCE_DIR}/thermal2/thermal2.mtx ${CMAKE_CURRENT_BINARY_DIR}/data/thermal2.mtx SYMBOLIC)
