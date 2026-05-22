add_library(stb_image_write STATIC stb_image_write.cpp)

target_include_directories(stb_image_write PUBLIC ${CMAKE_CURRENT_LIST_DIR})
