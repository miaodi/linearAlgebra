# Scratch Directory

This directory is for temporary development and testing binaries during feature development.

## Structure

- Create subdirectories for different features or experiments
- Each subdirectory should have its own `CMakeLists.txt` to build executables
- Subdirectories are automatically detected and compiled
- Delete subdirectories when development is complete

## Usage

1. Create a new subdirectory: `mkdir scratch/my_feature`
2. Add your source files and a `CMakeLists.txt`
3. Build as usual: `cmake --build build`
4. Clean up: `rm -rf scratch/my_feature` when done

## Example CMakeLists.txt

```cmake
# Create an executable from source files
add_executable(my_test_binary
    main.cpp
    helper.cpp
)

# Link against project libraries
target_link_libraries(my_test_binary PRIVATE
    sparse_mat_op
    mkl_wrapper
    ${MKL_LIBRARIES}
)
```
