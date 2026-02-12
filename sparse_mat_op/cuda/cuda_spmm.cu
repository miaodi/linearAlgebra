#include "cuda_spmm.cuh"
#include <algorithm>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <thrust/functional.h>

namespace cuda_iterative_solver
{
// ============================================================================
// STEP 1: Compute workload prefix sum and memory requirements
// ============================================================================

// Kernel: Compute workload for each row (nnz_A[i] * nnz_B[i])
template <typename ROWTYPE, typename COLTYPE>
__global__ void compute_workload_kernel(const ROWTYPE* d_ai_A, const ROWTYPE* d_ai_B, COLTYPE n_rows, ROWTYPE base,
                                        ROWTYPE* d_workloads) // Output: workload per row
{
    COLTYPE row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= n_rows)
        return;

    ROWTYPE row_start_A = d_ai_A[row_idx] - base;
    ROWTYPE row_end_A = d_ai_A[row_idx + 1] - base;
    ROWTYPE row_start_B = d_ai_B[row_idx] - base;
    ROWTYPE row_end_B = d_ai_B[row_idx + 1] - base;

    ROWTYPE nnz_A = row_end_A - row_start_A;
    ROWTYPE nnz_B = row_end_B - row_start_B;

    d_workloads[row_idx] = nnz_A * nnz_B;
}

// Host function: Step 1 - Compute prefix sum and return memory requirements
template <typename ROWTYPE, typename COLTYPE>
bool SpMMAnalyze(COLTYPE n_rows, const ROWTYPE* d_ai_A, const ROWTYPE* d_ai_B, ROWTYPE base, ROWTYPE* d_workload_prefix)
{
    if (n_rows <= 0)
        return false;

    // Allocate workload array
    ROWTYPE* d_workloads;
    cudaMalloc(&d_workloads, (n_rows + 1) * sizeof(ROWTYPE));
    cudaMemset(d_workloads, 0, (n_rows + 1) * sizeof(ROWTYPE));

    // Launch kernel to compute workloads
    int threads = 256;
    int blocks = (n_rows + threads - 1) / threads;
    compute_workload_kernel<<<blocks, threads>>>(d_ai_A, d_ai_B, n_rows, base, d_workloads);

    // Compute prefix sum using CUB
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, d_workloads,
                                   d_workload_prefix, thrust::plus<ROWTYPE>(), base, n_rows + 1);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, d_workloads,
                                   d_workload_prefix, thrust::plus<ROWTYPE>(), base, n_rows + 1);

    // Cleanup temporary storage
    cudaFree(d_workloads);
    cudaFree(d_temp_storage);

    return true;
}

// ============================================================================
// Template instantiations
// ============================================================================

// Step 1: Compute workload
template bool SpMMAnalyze<int, int>(int n_rows, const int* d_ai_A, const int* d_ai_B, int base,
                                    int* d_workload_prefix);
template bool SpMMAnalyze<int64_t, int>(int n_rows, const int64_t* d_ai_A, const int64_t* d_ai_B,
                                        int64_t base, int64_t* d_workload_prefix);

} // namespace cuda_iterative_solver
