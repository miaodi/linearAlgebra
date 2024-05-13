cmake_minimum_required(VERSION 3.5...3.20)
project(tests)

include(FetchContent)

# Add Google Test
include(../cmake/GoogleTest.cmake)
include(../cmake/DownloadFile.cmake)

include(GoogleTest)

file(GLOB_RECURSE ALL_TEST_CPP *.cpp)

foreach(ONE_TEST_CPP ${ALL_TEST_CPP})
   get_filename_component(ONE_TEST_EXEC ${ONE_TEST_CPP} NAME_WE)

   # Avoid name collision 
   set(TARGET_NAME TEST_${ONE_TEST_EXEC})

   add_executable(${TARGET_NAME} ${ONE_TEST_CPP})
   set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_TEST_EXEC}) 
   target_link_libraries(${TARGET_NAME} PRIVATE GTest::gtest_main GTest::gmock_main mkl_wrapper Utils reordering  OpenMP::OpenMP_CXX)
   gtest_discover_tests(${TARGET_NAME})
endforeach()

# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/data/ex5.mtx ${CMAKE_CURRENT_BINARY_DIR}/data/ex5.mtx COPYONLY)
# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/data/rdist1.mtx ${CMAKE_CURRENT_BINARY_DIR}/data/rdist1.mtx COPYONLY)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/data)
file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/data/ex5.mtx ${CMAKE_CURRENT_BINARY_DIR}/data/ex5.mtx SYMBOLIC)
file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/data/rdist1.mtx ${CMAKE_CURRENT_BINARY_DIR}/data/rdist1.mtx SYMBOLIC)

download_file( 
   https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/lanpro/nos5.mtx.gz
   1317ee84c28a6380793e22ec6e70ceb75b1b5a74cb3765da7200b00e22615f16
   nos5
)

FetchContent_GetProperties(nos5 SOURCE_DIR FILE_SOURCE_DIR)
file(CREATE_LINK ${FILE_SOURCE_DIR}/nos5.mtx ${CMAKE_CURRENT_BINARY_DIR}/data/nos5.mtx SYMBOLIC)

download_file( 
   https://math.nist.gov/pub/MatrixMarket2/misc/cylshell/s3rmt3m3.mtx.gz
   ead9e4eaf44a0cd9048152f1308d9c38dedc1f76a07947960fbb5e8f4028585a
   s3rmt3m3
)

FetchContent_GetProperties(s3rmt3m3 SOURCE_DIR FILE_SOURCE_DIR)
if(NOT EXISTS ${FILE_SOURCE_DIR}/s3rmt3m3.mtx)
   message(FATAL_ERROR "File not found")
endif()
file(CREATE_LINK ${FILE_SOURCE_DIR}/s3rmt3m3.mtx ${CMAKE_CURRENT_BINARY_DIR}/data/s3rmt3m3.mtx SYMBOLIC)