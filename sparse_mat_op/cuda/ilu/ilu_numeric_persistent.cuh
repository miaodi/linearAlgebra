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
 * d_diag_inv is caller-owned scratch storage of size n. d_next_row is a device
 * scalar and d_row_done has size n. d_next_row and d_row_done are initialized by
 * this function on the supplied stream.
 */
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
cudaError_t ILUBaseNumericFactorizationPersistentAsync( COLTYPE n,
                                                        const ROWTYPE* d_lu_ai,
                                                        const COLTYPE* d_lu_aj,
                                                        const ROWTYPE* d_lu_diag,
                                                        COLTYPE base,
                                                        VALTYPE* d_lu_av,
                                                        VALTYPE* d_diag_inv,
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
    double*,
    int*,
    int*,
    int*,
    cudaStream_t,
    ILUPersistentLaunchConfig* );

/**
 * @brief Persistent CUDA ILU numeric factorization using a row permutation.
 *
 * This is the low-overhead experimental variant of
 * ILUBaseNumericFactorizationPersistentAsync. Resident warps still claim one
 * monotonic work slot at a time, but map that slot through d_row_perm before
 * factoring the original matrix row. d_row_perm is expected to be a dependency
 * topological order such as the level permutation produced by TopologicalSort2.
 */
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
cudaError_t ILUBaseNumericFactorizationPersistentPermAsync( COLTYPE n,
                                                            const ROWTYPE* d_lu_ai,
                                                            const COLTYPE* d_lu_aj,
                                                            const ROWTYPE* d_lu_diag,
                                                            const COLTYPE* d_row_perm,
                                                            COLTYPE base,
                                                            VALTYPE* d_lu_av,
                                                            VALTYPE* d_diag_inv,
                                                            int* d_status,
                                                            COLTYPE* d_next_row,
                                                            int* d_row_done,
                                                            cudaStream_t stream = nullptr,
                                                            ILUPersistentLaunchConfig* h_launch_config = nullptr );

extern template cudaError_t ILUBaseNumericFactorizationPersistentPermAsync<int, int, float>( int,
                                                                                             const int*,
                                                                                             const int*,
                                                                                             const int*,
                                                                                             const int*,
                                                                                             int,
                                                                                             float*,
                                                                                             float*,
                                                                                             int*,
                                                                                             int*,
                                                                                             int*,
                                                                                             cudaStream_t,
                                                                                             ILUPersistentLaunchConfig* );

extern template cudaError_t ILUBaseNumericFactorizationPersistentPermAsync<int, int, double>( int,
                                                                                              const int*,
                                                                                              const int*,
                                                                                              const int*,
                                                                                              const int*,
                                                                                              int,
                                                                                              double*,
                                                                                              double*,
                                                                                              int*,
                                                                                              int*,
                                                                                              int*,
                                                                                              cudaStream_t,
                                                                                              ILUPersistentLaunchConfig* );

extern template cudaError_t ILUBaseNumericFactorizationPersistentPermAsync<std::int64_t, int, double>(
    int,
    const std::int64_t*,
    const int*,
    const std::int64_t*,
    const int*,
    int,
    double*,
    double*,
    int*,
    int*,
    int*,
    cudaStream_t,
    ILUPersistentLaunchConfig* );

/**
 * @brief Persistent CUDA ILU numeric factorization using a precomputed update cache.
 *
 * This keeps the persistent row scheduler from ILUBaseNumericFactorizationPersistentAsync
 * but replaces per-update binary searches with the lower-only update cache used by
 * ILUBaseNumericFactorizationCachedAsync. The caller owns d_diag_inv, d_next_row,
 * and d_row_done scratch storage.
 */
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
cudaError_t ILUBaseNumericFactorizationPersistentCachedAsync( COLTYPE n,
                                                              const ROWTYPE* d_lu_ai,
                                                              const COLTYPE* d_lu_aj,
                                                              const ROWTYPE* d_lu_diag,
                                                              const ROWTYPE* d_lower_row_ptr,
                                                              const ROWTYPE* d_update_ptr,
                                                              const ROWTYPE* d_update_jpos,
                                                              const ROWTYPE* d_update_pos,
                                                              COLTYPE base,
                                                              VALTYPE* d_lu_av,
                                                              VALTYPE* d_diag_inv,
                                                              int* d_status,
                                                              COLTYPE* d_next_row,
                                                              int* d_row_done,
                                                              cudaStream_t stream = nullptr,
                                                              ILUPersistentLaunchConfig* h_launch_config = nullptr );

extern template cudaError_t ILUBaseNumericFactorizationPersistentCachedAsync<int, int, float>(
    int,
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
    int*,
    int*,
    cudaStream_t,
    ILUPersistentLaunchConfig* );

extern template cudaError_t ILUBaseNumericFactorizationPersistentCachedAsync<int, int, double>(
    int,
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
    int*,
    int*,
    cudaStream_t,
    ILUPersistentLaunchConfig* );

extern template cudaError_t ILUBaseNumericFactorizationPersistentCachedAsync<std::int64_t, int, double>(
    int,
    const std::int64_t*,
    const int*,
    const std::int64_t*,
    const std::int64_t*,
    const std::int64_t*,
    const std::int64_t*,
    const std::int64_t*,
    int,
    double*,
    double*,
    int*,
    int*,
    int*,
    cudaStream_t,
    ILUPersistentLaunchConfig* );

/**
 * @brief Persistent cached CUDA ILU numeric factorization using a row permutation.
 *
 * This keeps the precomputed update cache but claims work slots in dependency
 * topological order through d_row_perm. Use a separate entry point so the raw
 * persistent cached path remains branch-free.
 */
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
cudaError_t ILUBaseNumericFactorizationPersistentCachedPermAsync( COLTYPE n,
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
                                                                  COLTYPE* d_next_row,
                                                                  int* d_row_done,
                                                                  cudaStream_t stream = nullptr,
                                                                  ILUPersistentLaunchConfig* h_launch_config = nullptr );

extern template cudaError_t ILUBaseNumericFactorizationPersistentCachedPermAsync<int, int, float>(
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
    int*,
    int*,
    cudaStream_t,
    ILUPersistentLaunchConfig* );

extern template cudaError_t ILUBaseNumericFactorizationPersistentCachedPermAsync<int, int, double>(
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
    int*,
    int*,
    cudaStream_t,
    ILUPersistentLaunchConfig* );

extern template cudaError_t ILUBaseNumericFactorizationPersistentCachedPermAsync<std::int64_t, int, double>(
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
    int*,
    int*,
    cudaStream_t,
    ILUPersistentLaunchConfig* );

} // namespace matrix_utils::sparse_cuda
