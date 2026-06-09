#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace matrix_utils::sparse_cuda
{

struct ILUPersistentLaunchConfig
{
    int block_size = 0;
    int grid_blocks = 0;
    int blocks_per_sm = 0;
    int resident_warps = 0;
};

/**
 * @brief Persistent CUDA ILU numeric factorization without host-side level scheduling.
 *
 * Resident warps pull rows from a monotonic global row counter. Before row i uses
 * a lower dependency row k, it spin-waits until d_row_done[k] is published. The
 * monotonic row assignment keeps every lower-index dependency scheduled before
 * any dependent row can wait on it, avoiding the deadlock risk of launching one
 * independent block per row.
 *
 * The caller must initialize d_lu_av first, usually with ILUEmbedAValuesToLUAsync.
 * d_next_row is a device scalar and d_row_done has size n. Both are initialized
 * by this function on the supplied stream.
 */
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
cudaError_t ILUBaseNumericFactorizationPersistentAsync( COLTYPE n,
                                                        const ROWTYPE* d_lu_ai,
                                                        const COLTYPE* d_lu_aj,
                                                        const ROWTYPE* d_lu_diag,
                                                        COLTYPE base,
                                                        VALTYPE* d_lu_av,
                                                        int* d_status,
                                                        COLTYPE* d_next_row,
                                                        int* d_row_done,
                                                        cudaStream_t stream = nullptr,
                                                        ILUPersistentLaunchConfig* h_launch_config = nullptr );

extern template cudaError_t ILUBaseNumericFactorizationPersistentAsync<int, int, float>( int,
                                                                                         const int*,
                                                                                         const int*,
                                                                                         const int*,
                                                                                         int,
                                                                                         float*,
                                                                                         int*,
                                                                                         int*,
                                                                                         int*,
                                                                                         cudaStream_t,
                                                                                         ILUPersistentLaunchConfig* );

extern template cudaError_t ILUBaseNumericFactorizationPersistentAsync<int, int, double>( int,
                                                                                          const int*,
                                                                                          const int*,
                                                                                          const int*,
                                                                                          int,
                                                                                          double*,
                                                                                          int*,
                                                                                          int*,
                                                                                          int*,
                                                                                          cudaStream_t,
                                                                                          ILUPersistentLaunchConfig* );

extern template cudaError_t ILUBaseNumericFactorizationPersistentAsync<std::int64_t, int, double>(
    int,
    const std::int64_t*,
    const int*,
    const std::int64_t*,
    int,
    double*,
    int*,
    int*,
    int*,
    cudaStream_t,
    ILUPersistentLaunchConfig* );

} // namespace matrix_utils::sparse_cuda
