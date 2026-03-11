#include "cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace matrix_utils::sparse_cuda
{

// CUDA kernel for element-wise multiplication
// Each thread processes items_per_thread elements with coalesced access pattern
template <int items_per_thread>
__global__ void elementwise_multiply_kernel(const double* d_a, const double* d_b,
                                            double* d_output, size_t n)
{
    size_t block_start = blockIdx.x * blockDim.x * items_per_thread;
    
    #pragma unroll(8)
    for (int i = 0; i < items_per_thread; ++i) {
        size_t idx = block_start + threadIdx.x + i * blockDim.x;
        if (idx < n) {
            d_output[idx] = d_a[idx] * d_b[idx];
        }
    }
}

template <typename T>
__global__ void fill_array_kernel(T* d_data, size_t n, T value)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        d_data[idx] = value;
    }
}

template <int items_per_thread>
void elementwiseMultiply(const double* d_a, const double* d_b,
                         double* d_output, size_t n)
{
    const int block_size = 256;
    const int num_blocks = (n + block_size * items_per_thread - 1) / (block_size * items_per_thread);
    
    elementwise_multiply_kernel<items_per_thread><<<num_blocks, block_size>>>(
        d_a, d_b, d_output, n);
    
    // Error checking is done by the caller
}

template <typename T>
void fillArray(T* d_data, size_t n, T value)
{
    if (n == 0)
    {
        return;
    }

    const int block_size = 256;
    const int num_blocks = static_cast<int>((n + block_size - 1) / block_size);
    fill_array_kernel<<<num_blocks, block_size>>>(d_data, n, value);
}

// Explicit template instantiations for common values
template void elementwiseMultiply<1>(const double*, const double*, double*, size_t);
template void elementwiseMultiply<2>(const double*, const double*, double*, size_t);
template void elementwiseMultiply<4>(const double*, const double*, double*, size_t);
template void elementwiseMultiply<8>(const double*, const double*, double*, size_t);
template void elementwiseMultiply<16>(const double*, const double*, double*, size_t);
template void elementwiseMultiply<32>(const double*, const double*, double*, size_t);
template void fillArray<float>(float*, size_t, float);
template void fillArray<double>(double*, size_t, double);
template void fillArray<int>(int*, size_t, int);
template void fillArray<int64_t>(int64_t*, size_t, int64_t);

} // namespace matrix_utils::sparse_cuda
