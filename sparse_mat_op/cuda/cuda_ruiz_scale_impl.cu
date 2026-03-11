#include "cuda_kernels.cuh"
#include "cuda_ruiz_scale.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <type_traits>

namespace matrix_utils::sparse_cuda
{
namespace
{
template <typename CountType, typename SizeType>
inline __host__ __device__ bool should_use_shared_norms(CountType tile_nnz, SizeType tile_size)
{
    return tile_nnz >= static_cast<CountType>(6 * tile_size);
}

template <typename CountType, typename SizeType>
inline __host__ __device__ bool should_use_shared_scales(CountType tile_nnz, SizeType tile_size)
{
    return tile_nnz >= static_cast<CountType>(8 * tile_size);
}

inline void cuda_check(cudaError_t error, const char* message)
{
    if (error != cudaSuccess)
    {
        char full_message[512];
        std::snprintf(full_message, sizeof(full_message), "CUDA error: %s - %s", message,
                      cudaGetErrorString(error));
        throw std::runtime_error(full_message);
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

/// @brief Tile-COO norm computation: one block handles one tile.
///
/// The whole block cooperates on a single tile so row/column norm buffers are
/// shared across all participating threads before a single global flush.
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, CudaRuizScalingNormType NORM>
__global__ void compute_norms_tiled(COLTYPE rows, COLTYPE cols, COLTYPE n_tiles, int tile_k,
                                    ROWTYPE base, const COLTYPE* tile_row_ind,
                                    const COLTYPE* tile_col_ind, const ROWTYPE* tile_nnz_prefix,
                                    const COLTYPE* row_ind, const COLTYPE* col_ind,
                                    const VALTYPE* values, VALTYPE* row_norms, VALTYPE* col_norms)
{
    extern __shared__ unsigned char shared_norms_raw[];

    const int tid = static_cast<int>(threadIdx.x);
    const COLTYPE tile = static_cast<COLTYPE>(blockIdx.x);
    if (tile >= n_tiles)
        return;

    const COLTYPE tile_size = static_cast<COLTYPE>(COLTYPE{1} << tile_k);
    VALTYPE* shared_norms = reinterpret_cast<VALTYPE*>(shared_norms_raw);
    VALTYPE* tile_row_norms = shared_norms;
    VALTYPE* tile_col_norms = shared_norms + static_cast<size_t>(tile_size);

    const COLTYPE tile_row0 = static_cast<COLTYPE>(tile_row_ind[tile] * tile_size);
    const COLTYPE tile_col0 = static_cast<COLTYPE>(tile_col_ind[tile] * tile_size);
    const ROWTYPE start = tile_nnz_prefix[tile];
    const ROWTYPE end = tile_nnz_prefix[tile + 1];
    const ROWTYPE tile_nnz = static_cast<ROWTYPE>(end - start);
    const bool use_shared = should_use_shared_norms(tile_nnz, tile_size);

    if (use_shared)
    {
        // Initialize tile-local row/column accumulators in shared memory.
        for (COLTYPE i = static_cast<COLTYPE>(tid); i < tile_size;
             i += static_cast<COLTYPE>(blockDim.x))
        {
            tile_row_norms[i] = static_cast<VALTYPE>(0);
            tile_col_norms[i] = static_cast<VALTYPE>(0);
        }
        __syncthreads();

        // Sweep nnz entries that belong to this tile and accumulate into tile-local norms.
        for (ROWTYPE i = static_cast<ROWTYPE>(start + tid); i < end;
             i = static_cast<ROWTYPE>(i + blockDim.x))
        {
            const COLTYPE row_global = static_cast<COLTYPE>(row_ind[i] - static_cast<COLTYPE>(base));
            const COLTYPE col_global = static_cast<COLTYPE>(col_ind[i] - static_cast<COLTYPE>(base));
            const COLTYPE row_local = static_cast<COLTYPE>(row_global - tile_row0);
            const COLTYPE col_local = static_cast<COLTYPE>(col_global - tile_col0);
            const VALTYPE val = values[i];

            if constexpr (NORM == CudaRuizScalingNormType::MaxNorm)
            {
                const VALTYPE abs_val = fabs(val);
                if (row_local >= 0 && row_local < tile_size)
                {
                    if constexpr (std::is_same_v<VALTYPE, double>)
                    {
                        atomicMax_device_double(&tile_row_norms[row_local], abs_val);
                    }
                    else
                    {
                        atomicMax_device_float(&tile_row_norms[row_local], abs_val);
                    }
                }
                if (col_local >= 0 && col_local < tile_size)
                {
                    if constexpr (std::is_same_v<VALTYPE, double>)
                    {
                        atomicMax_device_double(&tile_col_norms[col_local], abs_val);
                    }
                    else
                    {
                        atomicMax_device_float(&tile_col_norms[col_local], abs_val);
                    }
                }
            }
            else
            {
                const VALTYPE sq = val * val;
                if (row_local >= 0 && row_local < tile_size)
                {
                    atomicAdd(&tile_row_norms[row_local], sq);
                }
                if (col_local >= 0 && col_local < tile_size)
                {
                    atomicAdd(&tile_col_norms[col_local], sq);
                }
            }
        }
        __syncthreads();

        // Flush tile-local row/column norms to global arrays via atomics.
        for (COLTYPE i = static_cast<COLTYPE>(tid); i < tile_size;
             i += static_cast<COLTYPE>(blockDim.x))
        {
            const COLTYPE row_global = static_cast<COLTYPE>(tile_row0 + i);
            if (row_global < rows)
            {
                const VALTYPE value = tile_row_norms[i];
                if constexpr (std::is_same_v<VALTYPE, double>)
                {
                    if constexpr (NORM == CudaRuizScalingNormType::MaxNorm)
                    {
                        atomicMax_device_double(&row_norms[row_global], value);
                    }
                    else
                    {
                        atomicAdd(&row_norms[row_global], value);
                    }
                }
                else
                {
                    if constexpr (NORM == CudaRuizScalingNormType::MaxNorm)
                    {
                        atomicMax_device_float(&row_norms[row_global], value);
                    }
                    else
                    {
                        atomicAdd(&row_norms[row_global], value);
                    }
                }
            }

            const COLTYPE col_global = static_cast<COLTYPE>(tile_col0 + i);
            if (col_global < cols)
            {
                const VALTYPE value = tile_col_norms[i];
                if constexpr (std::is_same_v<VALTYPE, double>)
                {
                    if constexpr (NORM == CudaRuizScalingNormType::MaxNorm)
                    {
                        atomicMax_device_double(&col_norms[col_global], value);
                    }
                    else
                    {
                        atomicAdd(&col_norms[col_global], value);
                    }
                }
                else
                {
                    if constexpr (NORM == CudaRuizScalingNormType::MaxNorm)
                    {
                        atomicMax_device_float(&col_norms[col_global], value);
                    }
                    else
                    {
                        atomicAdd(&col_norms[col_global], value);
                    }
                }
            }
        }
    }
    else
    {
        for (ROWTYPE i = static_cast<ROWTYPE>(start + tid); i < end;
             i = static_cast<ROWTYPE>(i + blockDim.x))
        {
            const COLTYPE row_global = static_cast<COLTYPE>(row_ind[i] - static_cast<COLTYPE>(base));
            const COLTYPE col_global = static_cast<COLTYPE>(col_ind[i] - static_cast<COLTYPE>(base));
            const VALTYPE val = values[i];

            if constexpr (NORM == CudaRuizScalingNormType::MaxNorm)
            {
                const VALTYPE abs_val = fabs(val);
                if constexpr (std::is_same_v<VALTYPE, double>)
                {
                    atomicMax_device_double(&row_norms[row_global], abs_val);
                    atomicMax_device_double(&col_norms[col_global], abs_val);
                }
                else
                {
                    atomicMax_device_float(&row_norms[row_global], abs_val);
                    atomicMax_device_float(&col_norms[col_global], abs_val);
                }
            }
            else
            {
                const VALTYPE sq = val * val;
                atomicAdd(&row_norms[row_global], sq);
                atomicAdd(&col_norms[col_global], sq);
            }
        }
    }
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

template <typename VALTYPE>
__global__ void elementwise_multiply_inplace(int size, VALTYPE* accum, const VALTYPE* factors)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
        return;

    accum[i] *= factors[i];
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

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
__global__ void scale_tiled_values(COLTYPE rows, COLTYPE cols, COLTYPE n_tiles, int tile_k,
                                   ROWTYPE base, const COLTYPE* tile_row_ind,
                                   const COLTYPE* tile_col_ind, const ROWTYPE* tile_nnz_prefix,
                                   const COLTYPE* row_ind, const COLTYPE* col_ind, VALTYPE* values,
                                   const VALTYPE* row_scale, const VALTYPE* col_scale)
{
    extern __shared__ unsigned char shared_scales_raw[];

    const int tid = static_cast<int>(threadIdx.x);
    const COLTYPE tile = static_cast<COLTYPE>(blockIdx.x);
    if (tile >= n_tiles)
        return;

    const COLTYPE tile_size = static_cast<COLTYPE>(COLTYPE{1} << tile_k);
    VALTYPE* shared_scales = reinterpret_cast<VALTYPE*>(shared_scales_raw);
    VALTYPE* tile_row_scale = shared_scales;
    VALTYPE* tile_col_scale = shared_scales + static_cast<size_t>(tile_size);

    const COLTYPE tile_row0 = static_cast<COLTYPE>(tile_row_ind[tile] * tile_size);
    const COLTYPE tile_col0 = static_cast<COLTYPE>(tile_col_ind[tile] * tile_size);
    const ROWTYPE start = tile_nnz_prefix[tile];
    const ROWTYPE end = tile_nnz_prefix[tile + 1];
    const ROWTYPE tile_nnz = static_cast<ROWTYPE>(end - start);
    const bool use_shared = should_use_shared_scales(tile_nnz, tile_size);

    if (use_shared)
    {
        // Stage this tile's row/column scale vectors in shared memory.
        for (COLTYPE i = static_cast<COLTYPE>(tid); i < tile_size;
             i += static_cast<COLTYPE>(blockDim.x))
        {
            const COLTYPE row = static_cast<COLTYPE>(tile_row0 + i);
            const COLTYPE col = static_cast<COLTYPE>(tile_col0 + i);
            tile_row_scale[i] = (row < rows) ? row_scale[row] : static_cast<VALTYPE>(1);
            tile_col_scale[i] = (col < cols) ? col_scale[col] : static_cast<VALTYPE>(1);
        }
        __syncthreads();

        for (ROWTYPE i = static_cast<ROWTYPE>(start + tid); i < end;
             i = static_cast<ROWTYPE>(i + blockDim.x))
        {
            const COLTYPE row_global = static_cast<COLTYPE>(row_ind[i] - static_cast<COLTYPE>(base));
            const COLTYPE col_global = static_cast<COLTYPE>(col_ind[i] - static_cast<COLTYPE>(base));
            const COLTYPE row_local = static_cast<COLTYPE>(row_global - tile_row0);
            const COLTYPE col_local = static_cast<COLTYPE>(col_global - tile_col0);
            const bool in_row_tile = (row_local >= 0 && row_local < tile_size);
            const bool in_col_tile = (col_local >= 0 && col_local < tile_size);
            const VALTYPE rs = in_row_tile ? tile_row_scale[row_local] : row_scale[row_global];
            const VALTYPE cs = in_col_tile ? tile_col_scale[col_local] : col_scale[col_global];
            values[i] = values[i] * rs * cs;
        }
    }
    else
    {
        for (ROWTYPE i = static_cast<ROWTYPE>(start + tid); i < end;
             i = static_cast<ROWTYPE>(i + blockDim.x))
        {
            const COLTYPE row_global = static_cast<COLTYPE>(row_ind[i] - static_cast<COLTYPE>(base));
            const COLTYPE col_global = static_cast<COLTYPE>(col_ind[i] - static_cast<COLTYPE>(base));
            const VALTYPE rs = row_scale[row_global];
            const VALTYPE cs = col_scale[col_global];
            values[i] = values[i] * rs * cs;
        }
    }
}

namespace
{
inline int choose_tile_block_size(const size_t tile_size)
{
    if (tile_size <= 32)
        return 32;
    if (tile_size <= 64)
        return 64;
    if (tile_size <= 128)
        return 128;
    return 256;
}
} // namespace

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, CudaRuizScalingNormType NORM>
bool RuizScaleCudaCSRTemplate(const COLTYPE rows, const COLTYPE cols, const ROWTYPE* d_ai, const COLTYPE* d_aj,
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

    // Initialize scaling factors to 1.
    fillArray(d_dr, static_cast<size_t>(rows), static_cast<VALTYPE>(1));
    fillArray(d_dc, static_cast<size_t>(cols), static_cast<VALTYPE>(1));

    for (int iter = 0; iter < max_iters; ++iter)
    {
        // Step 1: Compute row and column norms in a single kernel launch
        fillArray(d_row_norms, static_cast<size_t>(rows), static_cast<VALTYPE>(0));
        fillArray(d_col_norms, static_cast<size_t>(cols), static_cast<VALTYPE>(0));

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
        elementwise_multiply_inplace<<<grid_size_rows, block_size, 0, accum_stream>>>(
            static_cast<int>(rows), d_dr, d_row_scale);
        elementwise_multiply_inplace<<<grid_size_cols, block_size, 0, accum_stream>>>(
            static_cast<int>(cols), d_dc, d_col_scale);

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

    // std::cout << "Completed " << max_iters << " iterations" << std::endl;

    // Cleanup
    cudaFree(d_row_norms);
    cudaFree(d_col_norms);
    cudaFree(d_row_scale);
    cudaFree(d_col_scale);
    cuda_check(cudaStreamDestroy(accum_stream), "Failed to destroy accumulation stream");
    cuda_check(cudaEventDestroy(scale_factors_ready), "Failed to destroy event");

    return true;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, CudaRuizScalingNormType NORM>
bool RuizScaleCudaTileTemplate(DeviceTileCOOMatrix<ROWTYPE, COLTYPE, VALTYPE>& tile_mat,
                               VALTYPE* d_dr, VALTYPE* d_dc, const int max_iters)
{
    const COLTYPE rows = tile_mat.n_rows;
    const COLTYPE cols = tile_mat.n_cols;
    const COLTYPE n_tiles = tile_mat.n_tiles;
    const size_t nnz = tile_mat.nnz();

    if (!d_dr || !d_dc || !tile_mat.tileNnzPrefixData() || !tile_mat.tileRowIndData() ||
        !tile_mat.tileColIndData() || !tile_mat.rowIndData() || !tile_mat.colIndData() ||
        !tile_mat.valuesData())
    {
        throw std::invalid_argument("RuizScaleCuda(tile) received null pointer");
    }
    if (tile_mat.tile_k < 0 || tile_mat.tile_k >= 30)
    {
        throw std::invalid_argument("RuizScaleCuda(tile) requires 0 <= tile_k < 30");
    }

    const int block_size = 256;
    const int grid_size_rows = (static_cast<int>(rows) + block_size - 1) / block_size;
    const int grid_size_cols = (static_cast<int>(cols) + block_size - 1) / block_size;

    const size_t tile_size = size_t{1} << tile_mat.tile_k;
    const int tile_block_size = choose_tile_block_size(tile_size);
    const int tile_grid_size = static_cast<int>(n_tiles);
    const size_t shared_bytes = 2 * tile_size * sizeof(VALTYPE);

    int max_shared_per_block = 0;
    cuda_check(cudaDeviceGetAttribute(&max_shared_per_block, cudaDevAttrMaxSharedMemoryPerBlock, 0),
               "Failed to query max shared memory");
    if (shared_bytes > static_cast<size_t>(max_shared_per_block))
    {
        throw std::invalid_argument(
            "RuizScaleCuda(tile) shared memory requirement exceeds device limit");
    }

    VALTYPE *d_row_norms = nullptr, *d_col_norms = nullptr, *d_row_scale = nullptr, *d_col_scale = nullptr;
    cuda_check(cudaMalloc(&d_row_norms, static_cast<size_t>(rows) * sizeof(VALTYPE)),
               "Failed to allocate row norms");
    cuda_check(cudaMalloc(&d_col_norms, static_cast<size_t>(cols) * sizeof(VALTYPE)),
               "Failed to allocate col norms");
    cuda_check(cudaMalloc(&d_row_scale, static_cast<size_t>(rows) * sizeof(VALTYPE)),
               "Failed to allocate row scale");
    cuda_check(cudaMalloc(&d_col_scale, static_cast<size_t>(cols) * sizeof(VALTYPE)),
               "Failed to allocate col scale");

    fillArray(d_dr, static_cast<size_t>(rows), static_cast<VALTYPE>(1));
    fillArray(d_dc, static_cast<size_t>(cols), static_cast<VALTYPE>(1));

    cudaStream_t accum_stream;
    cuda_check(cudaStreamCreate(&accum_stream), "Failed to create accumulation stream");
    cudaEvent_t scale_factors_ready;
    cuda_check(cudaEventCreate(&scale_factors_ready), "Failed to create event");

    for (int iter = 0; iter < max_iters; ++iter)
    {
        fillArray(d_row_norms, static_cast<size_t>(rows), static_cast<VALTYPE>(0));
        fillArray(d_col_norms, static_cast<size_t>(cols), static_cast<VALTYPE>(0));

        compute_norms_tiled<ROWTYPE, COLTYPE, VALTYPE, NORM>
            <<<tile_grid_size, tile_block_size, shared_bytes>>>(
                rows, cols, n_tiles, tile_mat.tile_k, tile_mat.base, tile_mat.tileRowIndData(),
                tile_mat.tileColIndData(), tile_mat.tileNnzPrefixData(), tile_mat.rowIndData(),
                tile_mat.colIndData(), tile_mat.valuesData(), d_row_norms, d_col_norms);
        cuda_check(cudaGetLastError(), "compute_norms_tiled failed");
        cuda_check(cudaDeviceSynchronize(), "compute_norms_tiled sync failed");

        compute_scaling_factors<VALTYPE, NORM>
            <<<grid_size_rows, block_size>>>(static_cast<int>(rows), d_row_norms, d_row_scale);
        cuda_check(cudaGetLastError(), "tiled row compute_scaling_factors failed");

        compute_scaling_factors<VALTYPE, NORM>
            <<<grid_size_cols, block_size>>>(static_cast<int>(cols), d_col_norms, d_col_scale);
        cuda_check(cudaGetLastError(), "tiled col compute_scaling_factors failed");
        cuda_check(cudaDeviceSynchronize(), "tiled col compute_scaling_factors sync failed");

        cuda_check(cudaEventRecord(scale_factors_ready, 0), "Failed to record event");
        cuda_check(cudaStreamWaitEvent(accum_stream, scale_factors_ready, 0), "Stream wait failed");

        elementwise_multiply_inplace<<<grid_size_rows, block_size, 0, accum_stream>>>(
            static_cast<int>(rows), d_dr, d_row_scale);
        cuda_check(cudaGetLastError(), "tiled row accumulation launch failed");
        elementwise_multiply_inplace<<<grid_size_cols, block_size, 0, accum_stream>>>(
            static_cast<int>(cols), d_dc, d_col_scale);
        cuda_check(cudaGetLastError(), "tiled col accumulation launch failed");

        scale_tiled_values<<<tile_grid_size, tile_block_size, shared_bytes>>>(
            rows, cols, n_tiles, tile_mat.tile_k, tile_mat.base, tile_mat.tileRowIndData(),
            tile_mat.tileColIndData(), tile_mat.tileNnzPrefixData(), tile_mat.rowIndData(),
            tile_mat.colIndData(), tile_mat.valuesData(), d_row_scale, d_col_scale);
        cuda_check(cudaGetLastError(), "scale_tiled_values failed");
        cuda_check(cudaDeviceSynchronize(), "scale_tiled_values sync failed");
    }

    cudaFree(d_row_norms);
    cudaFree(d_col_norms);
    cudaFree(d_row_scale);
    cudaFree(d_col_scale);
    cuda_check(cudaStreamDestroy(accum_stream), "Failed to destroy accumulation stream");
    cuda_check(cudaEventDestroy(scale_factors_ready), "Failed to destroy event");

    return true;
}

#define DEFINE_RUIZ_SCALE_CUDA_OVERLOADS(ROWTYPE, COLTYPE, VALTYPE)                                             \
    bool detail::RuizScaleCudaCSRImplMaxNorm(const COLTYPE rows, const COLTYPE cols,                            \
                                             const ROWTYPE* d_ai, const COLTYPE* d_aj, VALTYPE* d_av,           \
                                             VALTYPE* d_dr, VALTYPE* d_dc, const int max_iters)                 \
    {                                                                                                           \
        return RuizScaleCudaCSRTemplate<ROWTYPE, COLTYPE, VALTYPE, CudaRuizScalingNormType::MaxNorm>(           \
            rows, cols, d_ai, d_aj, d_av, d_dr, d_dc, max_iters);                                               \
    }                                                                                                           \
    bool detail::RuizScaleCudaCSRImplL2Norm(const COLTYPE rows, const COLTYPE cols,                             \
                                            const ROWTYPE* d_ai, const COLTYPE* d_aj, VALTYPE* d_av,            \
                                            VALTYPE* d_dr, VALTYPE* d_dc, const int max_iters)                  \
    {                                                                                                           \
        return RuizScaleCudaCSRTemplate<ROWTYPE, COLTYPE, VALTYPE, CudaRuizScalingNormType::L2Norm>(            \
            rows, cols, d_ai, d_aj, d_av, d_dr, d_dc, max_iters);                                               \
    }                                                                                                           \
    bool detail::RuizScaleCudaTileImplMaxNorm(DeviceTileCOOMatrix<ROWTYPE, COLTYPE, VALTYPE>& tile_mat,         \
                                              VALTYPE* d_dr, VALTYPE* d_dc, const int max_iters)                \
    {                                                                                                           \
        return RuizScaleCudaTileTemplate<ROWTYPE, COLTYPE, VALTYPE, CudaRuizScalingNormType::MaxNorm>(          \
            tile_mat, d_dr, d_dc, max_iters);                                                                   \
    }                                                                                                           \
    bool detail::RuizScaleCudaTileImplL2Norm(DeviceTileCOOMatrix<ROWTYPE, COLTYPE, VALTYPE>& tile_mat,          \
                                             VALTYPE* d_dr, VALTYPE* d_dc, const int max_iters)                 \
    {                                                                                                           \
        return RuizScaleCudaTileTemplate<ROWTYPE, COLTYPE, VALTYPE, CudaRuizScalingNormType::L2Norm>(           \
            tile_mat, d_dr, d_dc, max_iters);                                                                   \
    }                                                                                                           \
    template bool RuizScaleCudaCSRTemplate<ROWTYPE, COLTYPE, VALTYPE, CudaRuizScalingNormType::MaxNorm>(        \
        const COLTYPE, const COLTYPE, const ROWTYPE*, const COLTYPE*, VALTYPE*, VALTYPE*, VALTYPE*, const int); \
    template bool RuizScaleCudaCSRTemplate<ROWTYPE, COLTYPE, VALTYPE, CudaRuizScalingNormType::L2Norm>(         \
        const COLTYPE, const COLTYPE, const ROWTYPE*, const COLTYPE*, VALTYPE*, VALTYPE*, VALTYPE*, const int); \
    template bool RuizScaleCudaTileTemplate<ROWTYPE, COLTYPE, VALTYPE, CudaRuizScalingNormType::MaxNorm>(       \
        DeviceTileCOOMatrix<ROWTYPE, COLTYPE, VALTYPE>&, VALTYPE*, VALTYPE*, const int);                        \
    template bool RuizScaleCudaTileTemplate<ROWTYPE, COLTYPE, VALTYPE, CudaRuizScalingNormType::L2Norm>(        \
        DeviceTileCOOMatrix<ROWTYPE, COLTYPE, VALTYPE>&, VALTYPE*, VALTYPE*, const int);

DEFINE_RUIZ_SCALE_CUDA_OVERLOADS(int32_t, int32_t, float)
DEFINE_RUIZ_SCALE_CUDA_OVERLOADS(int32_t, int32_t, double)
DEFINE_RUIZ_SCALE_CUDA_OVERLOADS(int64_t, int64_t, float)
DEFINE_RUIZ_SCALE_CUDA_OVERLOADS(int64_t, int64_t, double)

#undef DEFINE_RUIZ_SCALE_CUDA_OVERLOADS

} // namespace matrix_utils::sparse_cuda
