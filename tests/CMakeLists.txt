cmake_minimum_required(VERSION 3.5...3.20)
project(tests)

include(FetchContent)

# Add Google Test
include(../cmake/GoogleTest.cmake)

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