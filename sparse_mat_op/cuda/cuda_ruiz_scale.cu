#include "cuda_kernels.cuh"
#include "cuda_ruiz_scale.cuh"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <type_traits>

namespace matrix_utils::sparse_cuda
{
namespace
{
inline void cuda_check(cudaError_t error, const char* message)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(std::string("CUDA error: ") + message + " - " + cudaGetErrorString(error));
    }
}
} // namespace

/// @brief Specialized kernel for double precision atomicMax
__device__ inline void atomicMax_device_double(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        double current = __longlong_as_double(assumed);
        if (current >= val)
            return;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

/// @brief Specialized kernel for float atomicMax
__device__ inline void atomicMax_device_float(float* address, float val)
{
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint, assumed;

    do
    {
        assumed = old;
        float current = __uint_as_float(assumed);
        if (current >= val)
            return;
        old = atomicCAS(address_as_uint, assumed, __float_as_uint(val));
    } while (assumed != old);
}

/// @brief Unified kernel to compute both row and column norms in a single pass
/// @tparam NORM The norm type (MaxNorm or L2Norm)
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, CudaRuizScalingNormType NORM>
__global__ void compute_norms(COLTYPE rows, const ROWTYPE* ai, const COLTYPE* aj, const VALTYPE* av,
                              VALTYPE* row_norms, VALTYPE* col_norms)
{
    COLTYPE row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
        return;

    const ROWTYPE row_start = ai[row];
    const ROWTYPE row_end = ai[row + 1];
    VALTYPE row_norm = 0.0;

    // Process each non-zero in the row
    for (ROWTYPE j = row_start; j < row_end; ++j)
    {
        COLTYPE col = aj[j];
        const VALTYPE val = av[j];

        if constexpr (NORM == CudaRuizScalingNormType::MaxNorm)
        {
            const VALTYPE abs_val = fabs(val);
            // Accumulate row norm
            row_norm = max(row_norm, abs_val);
            // Accumulate column norm via atomic
            if constexpr (std::is_same_v<VALTYPE, double>)
            {
                atomicMax_device_double(&col_norms[col], abs_val);
            }
            else
            {
                atomicMax_device_float(&col_norms[col], abs_val);
            }
        }
        else
        { // L2Norm
            // Accumulate row norm
            row_norm = fma(val, val, row_norm);
            // Accumulate column norm via atomic
            atomicAdd(&col_norms[col], val * val);
        }
    }

    // Store row norm for this row
    row_norms[row] = row_norm;
}

/// @brief Kernel to compute scaling factors from norms with template norm type
template <typename VALTYPE, CudaRuizScalingNormType NORM>
__global__ void compute_scaling_factors(int size, const VALTYPE* norms, VALTYPE* scale_factors)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
        return;

    VALTYPE norm_val = norms[i];

    if constexpr (NORM == CudaRuizScalingNormType::L2Norm)
    {
        norm_val = sqrt(norm_val);
    }

    if (norm_val > 0.0)
    {
        scale_factors[i] = 1.0 / sqrt(norm_val);
    }
    else
    {
        scale_factors[i] = 1.0;
    }
}

/// @brief Kernel to scale matrix values
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
__global__ void scale_matrix_and_track_change(COLTYPE rows, const ROWTYPE* ai, const COLTYPE* aj,
                                              VALTYPE* av, const VALTYPE* row_scale, const VALTYPE* col_scale)
{
    COLTYPE row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
        return;

    const ROWTYPE row_start = ai[row];
    const ROWTYPE row_end = ai[row + 1];
    const VALTYPE row_scale_factor = row_scale[row];

    for (ROWTYPE j = row_start; j < row_end; ++j)
    {
        COLTYPE col = aj[j];
        VALTYPE old_val = av[j];
        VALTYPE new_val = old_val * row_scale_factor * col_scale[col];
        av[j] = new_val;
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, CudaRuizScalingNormType NORM>
bool RuizScaleCuda(const COLTYPE rows, const COLTYPE cols, const ROWTYPE* d_ai, const COLTYPE* d_aj,
                   VALTYPE* d_av, VALTYPE* d_dr, VALTYPE* d_dc, const int max_iters)
{
    const int block_size = 256;
    const int grid_size_rows = (static_cast<int>(rows) + block_size - 1) / block_size;
    const int grid_size_cols = (static_cast<int>(cols) + block_size - 1) / block_size;

    // Create separate stream for accumulation operations
    cudaStream_t accum_stream;
    cuda_check(cudaStreamCreate(&accum_stream), "Failed to create accumulation stream");
    cudaEvent_t scale_factors_ready;
    cuda_check(cudaEventCreate(&scale_factors_ready), "Failed to create event");

    // Allocate temporary device arrays
    VALTYPE *d_row_norms, *d_col_norms, *d_row_scale, *d_col_scale;
    cuda_check(cudaMalloc(&d_row_norms, static_cast<size_t>(rows) * sizeof(VALTYPE)),
               "Failed to allocate row norms");
    cuda_check(cudaMalloc(&d_col_norms, static_cast<size_t>(cols) * sizeof(VALTYPE)),
               "Failed to allocate col norms");
    cuda_check(cudaMalloc(&d_row_scale, static_cast<size_t>(rows) * sizeof(VALTYPE)),
               "Failed to allocate row scale");
    cuda_check(cudaMalloc(&d_col_scale, static_cast<size_t>(cols) * sizeof(VALTYPE)),
               "Failed to allocate col scale");

    // Initialize scaling factors to 1
    thrust::device_ptr<VALTYPE> thrust_dr(d_dr);
    thrust::device_ptr<VALTYPE> thrust_dc(d_dc);
    thrust::fill(thrust_dr, thrust_dr + static_cast<int>(rows), static_cast<VALTYPE>(1));
    thrust::fill(thrust_dc, thrust_dc + static_cast<int>(cols), static_cast<VALTYPE>(1));

    for (int iter = 0; iter < max_iters; ++iter)
    {
        // Step 1: Compute row and column norms in a single kernel launch
        thrust::fill(thrust::device_ptr<VALTYPE>(d_row_norms),
                     thrust::device_ptr<VALTYPE>(d_row_norms) + static_cast<int>(rows),
                     static_cast<VALTYPE>(0));
        thrust::fill(thrust::device_ptr<VALTYPE>(d_col_norms),
                     thrust::device_ptr<VALTYPE>(d_col_norms) + static_cast<int>(cols),
                     static_cast<VALTYPE>(0));

        compute_norms<ROWTYPE, COLTYPE, VALTYPE, NORM>
            <<<grid_size_rows, block_size>>>(rows, d_ai, d_aj, d_av, d_row_norms, d_col_norms);
        cuda_check(cudaGetLastError(), "compute_norms failed");
        cuda_check(cudaDeviceSynchronize(), "Synchronization failed");

        // Step 2: Compute scaling factors from norms
        compute_scaling_factors<VALTYPE, NORM>
            <<<grid_size_rows, block_size>>>(static_cast<int>(rows), d_row_norms, d_row_scale);
        compute_scaling_factors<VALTYPE, NORM>
            <<<grid_size_cols, block_size>>>(static_cast<int>(cols), d_col_norms, d_col_scale);
        cuda_check(cudaGetLastError(), "compute_scaling_factors failed");
        
        // Record event after scaling factors are computed
        cuda_check(cudaEventRecord(scale_factors_ready, 0), "Failed to record event");

        // Update accumulated scaling factors on separate stream
        cuda_check(cudaStreamWaitEvent(accum_stream, scale_factors_ready, 0), "Stream wait failed");
        thrust::device_ptr<VALTYPE> thrust_row_scale(d_row_scale);
        thrust::device_ptr<VALTYPE> thrust_col_scale(d_col_scale);
        thrust::transform(thrust::cuda::par.on(accum_stream),
                          thrust_dr, thrust_dr + static_cast<int>(rows), thrust_row_scale,
                          thrust_dr, thrust::multiplies<VALTYPE>());
        thrust::transform(thrust::cuda::par.on(accum_stream),
                          thrust_dc, thrust_dc + static_cast<int>(cols), thrust_col_scale,
                          thrust_dc, thrust::multiplies<VALTYPE>());

        // Step 3: Scale matrix (can run in parallel with accumulation)
        scale_matrix_and_track_change<<<grid_size_rows, block_size>>>(rows, d_ai, d_aj, d_av,
                                                                      d_row_scale, d_col_scale);
        cuda_check(cudaGetLastError(), "scale_matrix_and_track_change failed");
        cuda_check(cudaDeviceSynchronize(), "Synchronization failed");

        // // Compute max row norm and max col norm via reduction (min scale factor = max norm)
        // VALTYPE min_row_scale = thrust::reduce(thrust_row_scale, thrust_row_scale + static_cast<int>(rows),
        //                                        static_cast<VALTYPE>(1.0), thrust::minimum<VALTYPE>());
        // VALTYPE min_col_scale = thrust::reduce(thrust_col_scale, thrust_col_scale + static_cast<int>(cols),
        //                                        static_cast<VALTYPE>(1.0), thrust::minimum<VALTYPE>());
        // VALTYPE max_row_norm = 1.0 / (min_row_scale * min_row_scale);
        // VALTYPE max_col_norm = 1.0 / (min_col_scale * min_col_scale);
        // std::cout << "Iteration " << iter << ": max_row_norm = " << max_row_norm
        //           << ", max_col_norm = " << max_col_norm << std::endl;
    }

    std::cout << "Completed " << max_iters << " iterations" << std::endl;

    // Cleanup
    cudaFree(d_row_norms);
    cudaFree(d_col_norms);
    cudaFree(d_row_scale);
    cudaFree(d_col_scale);
    cuda_check(cudaStreamDestroy(accum_stream), "Failed to destroy accumulation stream");
    cuda_check(cudaEventDestroy(scale_factors_ready), "Failed to destroy event");

    return true;
}

// Explicit template instantiations
#define INSTANTIATE_RUIZ_SCALE_CUDA(ROWTYPE, COLTYPE, VALTYPE)                                                  \
    template bool RuizScaleCuda<ROWTYPE, COLTYPE, VALTYPE, CudaRuizScalingNormType::MaxNorm>(                   \
        const COLTYPE, const COLTYPE, const ROWTYPE*, const COLTYPE*, VALTYPE*, VALTYPE*, VALTYPE*, const int); \
    template bool RuizScaleCuda<ROWTYPE, COLTYPE, VALTYPE, CudaRuizScalingNormType::L2Norm>(                    \
        const COLTYPE, const COLTYPE, const ROWTYPE*, const COLTYPE*, VALTYPE*, VALTYPE*, VALTYPE*, const int);

// int32, float
INSTANTIATE_RUIZ_SCALE_CUDA(int32_t, int32_t, float)
// int32, double
INSTANTIATE_RUIZ_SCALE_CUDA(int32_t, int32_t, double)
// int64, float
INSTANTIATE_RUIZ_SCALE_CUDA(int64_t, int64_t, float)
// int64, double
INSTANTIATE_RUIZ_SCALE_CUDA(int64_t, int64_t, double)

#undef INSTANTIATE_RUIZ_SCALE_CUDA

} // namespace matrix_utils::sparse_cuda
