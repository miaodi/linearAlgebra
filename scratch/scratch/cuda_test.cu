#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <iostream>
#include <vector>
#include <random>

// Kernel to sort an array using cub::WarpMergeSort with a single warp
template <int ITEMS_PER_THREAD, int WARP_THREADS = 32>
__global__ void warpMergeSortKernel(int* d_data, int num_items) {
    // Single warp, so use threadIdx.x directly as lane_id
    int lane_id = threadIdx.x;
    
    // Thread-private array to hold items
    int thread_data[ITEMS_PER_THREAD];
    
    // Load data into thread-private storage (blocked arrangement)
    // Each thread loads ITEMS_PER_THREAD consecutive items
    int thread_offset = lane_id * ITEMS_PER_THREAD;
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = thread_offset + i;
        thread_data[i] = (idx < num_items) ? d_data[idx] : INT_MAX;
    }
    
    // Create WarpMergeSort instance
    typedef cub::WarpMergeSort<int, ITEMS_PER_THREAD, WARP_THREADS> WarpMergeSort;
    
    // Allocate shared memory for WarpMergeSort
    __shared__ typename WarpMergeSort::TempStorage temp_storage;
    
    // Sort the data
    WarpMergeSort(temp_storage).Sort(thread_data, [](int a, int b) { return a < b; });
    
    // Write sorted data back to global memory (blocked arrangement)
    // Each thread writes ITEMS_PER_THREAD consecutive items
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = thread_offset + i;
        if (idx < num_items && thread_data[i] != INT_MAX) {
            d_data[idx] = thread_data[i];
        }
    }
}

// Helper function to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Dispatcher function to launch kernel with appropriate ITEMS_PER_THREAD
void launchKernelWithSize(int* d_data, int num_items, int items_per_thread) {
    switch(items_per_thread) {
        case 1: warpMergeSortKernel<1><<<1, 32>>>(d_data, num_items); break;
        case 2: warpMergeSortKernel<2><<<1, 32>>>(d_data, num_items); break;
        case 4: warpMergeSortKernel<4><<<1, 32>>>(d_data, num_items); break;
        case 8: warpMergeSortKernel<8><<<1, 32>>>(d_data, num_items); break;
        case 16: warpMergeSortKernel<16><<<1, 32>>>(d_data, num_items); break;
        case 32: warpMergeSortKernel<32><<<1, 32>>>(d_data, num_items); break;
        case 64: warpMergeSortKernel<64><<<1, 32>>>(d_data, num_items); break;
        case 128: warpMergeSortKernel<128><<<1, 32>>>(d_data, num_items); break;
        case 256: warpMergeSortKernel<256><<<1, 32>>>(d_data, num_items); break;
        default:
            std::cerr << "Error: ITEMS_PER_THREAD " << items_per_thread 
                      << " not supported (use power of 2 between 1 and 256)" << std::endl;
            exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    // Runtime-determined array size
    int num_items = 256;
    if (argc > 1) {
        num_items = std::atoi(argv[1]);
    }
    
    std::cout << "Sorting " << num_items << " elements using cub::WarpMergeSort with a single warp" << std::endl;
    
    // Configuration
    const int WARP_THREADS = 32;
    
    // Calculate required ITEMS_PER_THREAD using binary search approach
    // We need ITEMS_PER_THREAD * 32 >= num_items
    int required_items_per_thread = (num_items + WARP_THREADS - 1) / WARP_THREADS;
    
    // Round up to next power of 2 for better performance
    int items_per_thread = 1;
    while (items_per_thread < required_items_per_thread) {
        items_per_thread *= 2;
    }
    
    std::cout << "Using ITEMS_PER_THREAD = " << items_per_thread 
              << " (capacity: " << items_per_thread * WARP_THREADS << " elements)" << std::endl;
    
    // Prepare host data
    std::vector<int> h_data(num_items);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 1000);
    
    std::cout << "Original data (first 20): ";
    for (int i = 0; i < num_items; i++) {
        h_data[i] = dist(rng);
        if (i < 20) {
            std::cout << h_data[i] << " ";
        }
    }
    std::cout << std::endl;
    
    // Allocate device memory
    int* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, num_items * sizeof(int)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), num_items * sizeof(int), cudaMemcpyHostToDevice));
    
    // Launch kernel with appropriate ITEMS_PER_THREAD
    launchKernelWithSize(d_data, num_items, items_per_thread);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, num_items * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Verify results (each warp's segment should be sorted)
    std::cout << "Sorted data (first 20): ";
    for (int i = 0; i < std::min(20, num_items); i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;
    
    // Verify that the entire array is sorted
    bool all_sorted = true;
    for (int i = 0; i < num_items - 1; i++) {
        if (h_data[i] > h_data[i + 1]) {
            all_sorted = false;
            std::cerr << "Error: Array not sorted at index " << i 
                      << " (" << h_data[i] << " > " << h_data[i + 1] << ")" << std::endl;
            break;
        }
    }
    
    if (all_sorted) {
        std::cout << "Success! The entire array is properly sorted using a single warp." << std::endl;
    } else {
        std::cout << "Error: Array is not properly sorted." << std::endl;
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_data));
    
    return all_sorted ? 0 : 1;
}
