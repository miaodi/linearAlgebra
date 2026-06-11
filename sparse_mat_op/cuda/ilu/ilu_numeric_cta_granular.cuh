#pragma once

#include "cuda_memory.cuh"
#include "ilu_numeric.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace matrix_utils::sparse_cuda
{

struct ILUCtaGranularLaunchConfig
{
    int warps_per_block = 0;
    int block_size = 0;
    int kernel_launches = 0;
    int total_blocks = 0;
    int hollow_warps = 0;
};

struct ILUCtaGranularScratch
{
    DeviceArray<int> row_done;
    DeviceArray<int> next_row;

    std::size_t bytes() const { return ( row_done.size() + next_row.size() ) * sizeof( int ); }
};

/**
 * @brief ILU numeric factorization using CTA-sized row-permutation chunks.
 *
 * One grid block is launched for each 8-row chunk. Blocks atomically claim
 * chunks from row_perm in order, run up to one row per warp, and then retire.
 * Before a row uses a lower dependency row, its warp waits for that row's
 * completion flag. The caller owns d_diag_inv scratch storage of size n.
 */
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
cudaError_t ILUBaseNumericFactorizationCtaGranularAsync( COLTYPE n,
                                                         const ROWTYPE* d_lu_ai,
                                                         const COLTYPE* d_lu_aj,
                                                         const ROWTYPE* d_lu_diag,
                                                         const COLTYPE* d_row_perm,
                                                         COLTYPE base,
                                                         VALTYPE* d_lu_av,
                                                         VALTYPE* d_diag_inv,
                                                         int* d_status,
                                                         ILUNumericRowLookup row_lookup,
                                                         ILUNumericRowUpdateStrategy row_update,
                                                         ILUCtaGranularScratch& scratch,
                                                         cudaStream_t stream = nullptr,
                                                         ILUCtaGranularLaunchConfig* h_launch_config = nullptr );

/**
 * @brief CTA-granular ILU numeric factorization using precomputed update positions.
 *
 * This keeps the same 8-row CTA work-claim model as
 * ILUBaseNumericFactorizationCtaGranularAsync, but replaces per-update row-column
 * searches with the lower-only update cache built by BuildILUUpdateCacheAsync.
 */
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
cudaError_t ILUBaseNumericFactorizationCtaGranularCachedAsync( COLTYPE n,
                                                               const ROWTYPE* d_lu_ai,
                                                               const COLTYPE* d_lu_aj,
                                                               const ROWTYPE* d_lu_diag,
                                                               const ROWTYPE* d_lower_row_ptr,
                                                               const ROWTYPE* d_update_ptr,
                                                               const ROWTYPE* d_update_jpos,
                                                               const ROWTYPE* d_update_pos,
                                                               const COLTYPE* d_row_perm,
                                                               COLTYPE base,
                                                               VALTYPE* d_lu_av,
                                                               VALTYPE* d_diag_inv,
                                                               int* d_status,
                                                               ILUCtaGranularScratch& scratch,
                                                               cudaStream_t stream = nullptr,
                                                               ILUCtaGranularLaunchConfig* h_launch_config = nullptr );

extern template cudaError_t ILUBaseNumericFactorizationCtaGranularAsync<int, int, float>(
    int,
    const int*,
    const int*,
    const int*,
    const int*,
    int,
    float*,
    float*,
    int*,
    ILUNumericRowLookup,
    ILUNumericRowUpdateStrategy,
    ILUCtaGranularScratch&,
    cudaStream_t,
    ILUCtaGranularLaunchConfig* );

extern template cudaError_t ILUBaseNumericFactorizationCtaGranularAsync<int, int, double>(
    int,
    const int*,
    const int*,
    const int*,
    const int*,
    int,
    double*,
    double*,
    int*,
    ILUNumericRowLookup,
    ILUNumericRowUpdateStrategy,
    ILUCtaGranularScratch&,
    cudaStream_t,
    ILUCtaGranularLaunchConfig* );

extern template cudaError_t ILUBaseNumericFactorizationCtaGranularAsync<std::int64_t, int, double>(
    int,
    const std::int64_t*,
    const int*,
    const std::int64_t*,
    const int*,
    int,
    double*,
    double*,
    int*,
    ILUNumericRowLookup,
    ILUNumericRowUpdateStrategy,
    ILUCtaGranularScratch&,
    cudaStream_t,
    ILUCtaGranularLaunchConfig* );

extern template cudaError_t ILUBaseNumericFactorizationCtaGranularCachedAsync<int, int, float>(
    int,
    const int*,
    const int*,
    const int*,
    const int*,
    const int*,
    const int*,
    const int*,
    const int*,
    int,
    float*,
    float*,
    int*,
    ILUCtaGranularScratch&,
    cudaStream_t,
    ILUCtaGranularLaunchConfig* );

extern template cudaError_t ILUBaseNumericFactorizationCtaGranularCachedAsync<int, int, double>(
    int,
    const int*,
    const int*,
    const int*,
    const int*,
    const int*,
    const int*,
    const int*,
    const int*,
    int,
    double*,
    double*,
    int*,
    ILUCtaGranularScratch&,
    cudaStream_t,
    ILUCtaGranularLaunchConfig* );

extern template cudaError_t ILUBaseNumericFactorizationCtaGranularCachedAsync<std::int64_t, int, double>(
    int,
    const std::int64_t*,
    const int*,
    const std::int64_t*,
    const std::int64_t*,
    const std::int64_t*,
    const std::int64_t*,
    const std::int64_t*,
    const int*,
    int,
    double*,
    double*,
    int*,
    ILUCtaGranularScratch&,
    cudaStream_t,
    ILUCtaGranularLaunchConfig* );

} // namespace matrix_utils::sparse_cuda
