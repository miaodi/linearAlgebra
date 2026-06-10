#pragma once

#include "cuda_memory.cuh"
#include "ilu_numeric.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace matrix_utils::sparse_cuda
{

struct ILULevelCtaLaunchConfig
{
    int warps_per_block = 0;
    int block_size = 0;
    int level_launches = 0;
    int total_blocks = 0;
    int hollow_warps = 0;
};

struct ILULevelCtaScratch
{
    DeviceArray<int> row_done;
    DeviceArray<int> next_cta;

    std::size_t bytes() const { return ( row_done.size() + next_cta.size() ) * sizeof( int ); }
};

template <typename ROWTYPE, typename COLTYPE>
struct ILULevelCtaSchedule
{
    int rows_per_cta = 0;
    COLTYPE row_count = 0;
    COLTYPE level_count = 0;
    int cta_count = 0;
    int cta_edge_count = 0;
    int hollow_warps = 0;

    // CTA -> rows. Rows are stored as zero-based matrix row ids.
    std::vector<COLTYPE> cta_row_ptr;
    std::vector<COLTYPE> cta_rows;
    // Highest topological level represented in each CTA.
    std::vector<int> cta_level;

    // Row -> CTA id, indexed by zero-based row id.
    std::vector<int> row_to_cta;

    // Cross-CTA predecessor CSR. cta_preds[cta_pred_ptr[c]..cta_pred_ptr[c+1])
    // are dependency CTAs that must complete before CTA c can run. Row
    // dependencies inside the same CTA are handled by row_done in the kernel.
    std::vector<int> cta_pred_ptr;
    std::vector<int> cta_preds;

    // Successor CSR. cta_succs[cta_succ_ptr[c]..cta_succ_ptr[c+1])
    // are dependent CTAs released after CTA c completes.
    std::vector<int> cta_succ_ptr;
    std::vector<int> cta_succs;

    // Initial predecessor count per CTA, retained for schedule diagnostics.
    std::vector<int> cta_indegree;

    // Source CTAs with zero initial indegree, retained for schedule diagnostics.
    std::vector<int> initial_ready_ctas;
};

template <typename ROWTYPE, typename COLTYPE>
struct DeviceILULevelCtaSchedule
{
    int rows_per_cta = 0;
    COLTYPE row_count = 0;
    COLTYPE level_count = 0;
    int cta_count = 0;
    int cta_edge_count = 0;
    int hollow_warps = 0;

    DeviceArray<COLTYPE> cta_row_ptr;
    DeviceArray<COLTYPE> cta_rows;
    DeviceArray<int> cta_level;
    DeviceArray<int> row_to_cta;

    // Device copies of the host CTA DAG CSR arrays above.
    DeviceArray<int> cta_pred_ptr;
    DeviceArray<int> cta_preds;
    DeviceArray<int> cta_succ_ptr;
    DeviceArray<int> cta_succs;
    DeviceArray<int> cta_indegree;
    DeviceArray<int> initial_ready_ctas;

    std::size_t bytes() const
    {
        return ( cta_row_ptr.size() + cta_rows.size() ) * sizeof( COLTYPE ) +
               ( cta_level.size() + row_to_cta.size() + cta_pred_ptr.size() + cta_preds.size() +
                 cta_succ_ptr.size() + cta_succs.size() + cta_indegree.size() + initial_ready_ctas.size() ) *
                   sizeof( int );
    }
};

/**
 * @brief Build a packed topological row-to-CTA schedule and its CTA-level dependency DAG.
 *
 * The row DAG is extracted from strict-lower LU entries: each lower entry
 * A(i,k), k < i, creates a dependency edge row k -> row i. Rows are packed in
 * level_perm order across level boundaries, so only the final CTA should have
 * hollow warps. Row-level completion flags preserve dependencies between rows
 * inside the same CTA.
 */
template <typename ROWTYPE, typename COLTYPE>
ILULevelCtaSchedule<ROWTYPE, COLTYPE> BuildILULevelCtaSchedule( COLTYPE n,
                                                                const ROWTYPE* lu_ai,
                                                                const COLTYPE* lu_aj,
                                                                const ROWTYPE* lu_diag,
                                                                const COLTYPE* level_perm,
                                                                const COLTYPE* level_prefix,
                                                                COLTYPE levels,
                                                                COLTYPE base,
                                                                int rows_per_cta = 8 );

template <typename ROWTYPE, typename COLTYPE>
cudaError_t UploadILULevelCtaSchedule( const ILULevelCtaSchedule<ROWTYPE, COLTYPE>& schedule,
                                       DeviceILULevelCtaSchedule<ROWTYPE, COLTYPE>& device_schedule );

/**
 * @brief ILU numeric factorization using the CTA-level DAG schedule.
 *
 * Resident blocks consume CTA tasks in the topological order produced by
 * BuildILULevelCtaSchedule. Before a row uses a lower dependency row, its warp
 * waits for that row's completion flag. This avoids the ready-queue and
 * successor-release atomics while preserving row dependencies inside one launch.
 * The caller owns d_diag_inv scratch storage of size schedule.row_count.
 */
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
cudaError_t ILUBaseNumericFactorizationLevelCtaAsync( const DeviceILULevelCtaSchedule<ROWTYPE, COLTYPE>& schedule,
                                                      const ROWTYPE* d_lu_ai,
                                                      const COLTYPE* d_lu_aj,
                                                      const ROWTYPE* d_lu_diag,
                                                      COLTYPE base,
                                                      VALTYPE* d_lu_av,
                                                      VALTYPE* d_diag_inv,
                                                      int* d_status,
                                                      ILUNumericRowLookup row_lookup,
                                                      ILUNumericRowUpdateStrategy row_update,
                                                      ILULevelCtaScratch& scratch,
                                                      cudaStream_t stream = nullptr,
                                                      ILULevelCtaLaunchConfig* h_launch_config = nullptr );

extern template cudaError_t ILUBaseNumericFactorizationLevelCtaAsync<int, int, float>(
    const DeviceILULevelCtaSchedule<int, int>&,
    const int*,
    const int*,
    const int*,
    int,
    float*,
    float*,
    int*,
    ILUNumericRowLookup,
    ILUNumericRowUpdateStrategy,
    ILULevelCtaScratch&,
    cudaStream_t,
    ILULevelCtaLaunchConfig* );

extern template cudaError_t ILUBaseNumericFactorizationLevelCtaAsync<int, int, double>(
    const DeviceILULevelCtaSchedule<int, int>&,
    const int*,
    const int*,
    const int*,
    int,
    double*,
    double*,
    int*,
    ILUNumericRowLookup,
    ILUNumericRowUpdateStrategy,
    ILULevelCtaScratch&,
    cudaStream_t,
    ILULevelCtaLaunchConfig* );

extern template cudaError_t ILUBaseNumericFactorizationLevelCtaAsync<std::int64_t, int, double>(
    const DeviceILULevelCtaSchedule<std::int64_t, int>&,
    const std::int64_t*,
    const int*,
    const std::int64_t*,
    int,
    double*,
    double*,
    int*,
    ILUNumericRowLookup,
    ILUNumericRowUpdateStrategy,
    ILULevelCtaScratch&,
    cudaStream_t,
    ILULevelCtaLaunchConfig* );

extern template ILULevelCtaSchedule<int, int> BuildILULevelCtaSchedule<int, int>( int,
                                                                                  const int*,
                                                                                  const int*,
                                                                                  const int*,
                                                                                  const int*,
                                                                                  const int*,
                                                                                  int,
                                                                                  int,
                                                                                  int );

extern template ILULevelCtaSchedule<std::int64_t, int> BuildILULevelCtaSchedule<std::int64_t, int>(
    int,
    const std::int64_t*,
    const int*,
    const std::int64_t*,
    const int*,
    const int*,
    int,
    int,
    int );

extern template cudaError_t UploadILULevelCtaSchedule<int, int>( const ILULevelCtaSchedule<int, int>&,
                                                                 DeviceILULevelCtaSchedule<int, int>& );

extern template cudaError_t UploadILULevelCtaSchedule<std::int64_t, int>(
    const ILULevelCtaSchedule<std::int64_t, int>&,
    DeviceILULevelCtaSchedule<std::int64_t, int>& );

} // namespace matrix_utils::sparse_cuda
