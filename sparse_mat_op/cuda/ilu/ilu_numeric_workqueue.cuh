#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace matrix_utils::sparse_cuda
{

/**
 * @brief Experimental no-cache CUDA ILU numeric factorization with a per-level work queue.
 *
 * The host still walks topological levels, preserving the exact dependency
 * boundary used by ILUBaseNumericFactorizationAsync. Inside each level,
 * resident warps dynamically pull rows from d_level_row_counter instead of
 * using a static row-to-warp assignment. The row update itself still uses the
 * original global-memory binary search path; no update cache is required.
 *
 * @param d_level_row_counter Device scalar used as the current level's row counter.
 *                            The caller owns this storage and must keep it valid
 *                            until the stream reaches this work.
 * @param blocks_per_sm Maximum persistent blocks to launch per SM for each level.
 */
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
cudaError_t ILUBaseNumericFactorizationWorkQueueAsync( COLTYPE n,
                                                       const ROWTYPE* d_a_ai,
                                                       const COLTYPE* d_a_aj,
                                                       const VALTYPE* d_a_av,
                                                       const ROWTYPE* d_lu_ai,
                                                       const COLTYPE* d_lu_aj,
                                                       const ROWTYPE* d_lu_diag,
                                                       const COLTYPE* d_level_perm,
                                                       const COLTYPE* h_level_prefix,
                                                       COLTYPE levels,
                                                       COLTYPE base,
                                                       VALTYPE* d_lu_av,
                                                       int* d_status,
                                                       COLTYPE* d_level_row_counter,
                                                       int blocks_per_sm = 4,
                                                       cudaStream_t stream = nullptr );

extern template cudaError_t ILUBaseNumericFactorizationWorkQueueAsync<int, int, float>( int,
                                                                                        const int*,
                                                                                        const int*,
                                                                                        const float*,
                                                                                        const int*,
                                                                                        const int*,
                                                                                        const int*,
                                                                                        const int*,
                                                                                        const int*,
                                                                                        int,
                                                                                        int,
                                                                                        float*,
                                                                                        int*,
                                                                                        int*,
                                                                                        int,
                                                                                        cudaStream_t );

extern template cudaError_t ILUBaseNumericFactorizationWorkQueueAsync<int, int, double>( int,
                                                                                         const int*,
                                                                                         const int*,
                                                                                         const double*,
                                                                                         const int*,
                                                                                         const int*,
                                                                                         const int*,
                                                                                         const int*,
                                                                                         const int*,
                                                                                         int,
                                                                                         int,
                                                                                         double*,
                                                                                         int*,
                                                                                         int*,
                                                                                         int,
                                                                                         cudaStream_t );

extern template cudaError_t ILUBaseNumericFactorizationWorkQueueAsync<std::int64_t, int, double>(
    int,
    const std::int64_t*,
    const int*,
    const double*,
    const std::int64_t*,
    const int*,
    const std::int64_t*,
    const int*,
    const int*,
    int,
    int,
    double*,
    int*,
    int*,
    int,
    cudaStream_t );

} // namespace matrix_utils::sparse_cuda
