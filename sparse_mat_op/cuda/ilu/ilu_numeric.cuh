#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace matrix_utils::sparse_cuda
{

/**
 * @brief Base CUDA numerical ILU factorization over a precomputed LU pattern.
 *
 * The host traverses topological levels. For each level, one warp is assigned
 * to each row. Within a row, the warp processes lower entries sequentially and
 * updates row values in parallel.
 *
 * @param n Matrix order.
 * @param d_a_ai Input A row pointers on device.
 * @param d_a_aj Input A column indices on device.
 * @param d_a_av Input A values on device.
 * @param d_lu_ai Symbolic LU row pointers on device.
 * @param d_lu_aj Symbolic LU column indices on device.
 * @param d_lu_diag Diagonal positions in LU, with the same base as d_lu_ai.
 * @param d_level_perm Topological row ordering on device.
 * @param h_level_prefix Topological level boundaries on host.
 * @param levels Number of topological levels.
 * @param base CSR base, usually 0.
 * @param d_lu_av Output LU values on device, size d_lu_ai[n] - base.
 * @param d_status Device status flag, set to 0 before factorization and to 1
 *                 if a zero pivot is found. The caller owns this storage and
 *                 must keep it valid until the stream reaches this work.
 * @param stream CUDA stream.
 * @return cudaSuccess if work was enqueued successfully. This does not imply
 *         the factorization has completed successfully; after successful stream
 *         completion, inspect d_status to detect numerical failure.
 */
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
cudaError_t ILUBaseNumericFactorizationAsync( COLTYPE n,
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
                                              cudaStream_t stream = nullptr );

/**
 * @brief CUDA numerical ILU factorization using a precomputed update cache.
 *
 * The cache removes numeric-phase binary searches. For every lower entry
 * position k_pos in row i, update_ptr[k_pos]..update_ptr[k_pos + 1] stores the
 * compact list of source U positions and destination row-i positions that
 * survive the ILU sparsity pattern.
 *
 * @param d_update_ptr Device update offsets, size d_lu_ai[n] - base + 1.
 * @param d_update_jpos Device source LU value positions for U(k,j).
 * @param d_update_pos Device destination LU value positions for row i.
 */
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
cudaError_t ILUBaseNumericFactorizationCachedAsync( COLTYPE n,
                                                    const ROWTYPE* d_a_ai,
                                                    const COLTYPE* d_a_aj,
                                                    const VALTYPE* d_a_av,
                                                    const ROWTYPE* d_lu_ai,
                                                    const COLTYPE* d_lu_aj,
                                                    const ROWTYPE* d_lu_diag,
                                                    const ROWTYPE* d_update_ptr,
                                                    const ROWTYPE* d_update_jpos,
                                                    const ROWTYPE* d_update_pos,
                                                    const COLTYPE* d_level_perm,
                                                    const COLTYPE* h_level_prefix,
                                                    COLTYPE levels,
                                                    COLTYPE base,
                                                    VALTYPE* d_lu_av,
                                                    int* d_status,
                                                    cudaStream_t stream = nullptr );

/**
 * @brief Blocking convenience wrapper for ILUBaseNumericFactorizationAsync.
 *
 * This function synchronizes the supplied stream before returning so it can
 * report zero-pivot status as a host bool.
 *
 * @return true on success, false if launch/copy fails or a zero pivot is found.
 */
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
bool ILUBaseNumericFactorization( COLTYPE n,
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
                                  cudaStream_t stream = nullptr );

extern template cudaError_t ILUBaseNumericFactorizationAsync<int, int, float>( int,
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
                                                                               cudaStream_t );

extern template cudaError_t ILUBaseNumericFactorizationAsync<int, int, double>( int,
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
                                                                                cudaStream_t );

extern template cudaError_t ILUBaseNumericFactorizationAsync<std::int64_t, int, double>( int,
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
                                                                                         cudaStream_t );

extern template cudaError_t ILUBaseNumericFactorizationCachedAsync<int, int, float>( int,
                                                                                     const int*,
                                                                                     const int*,
                                                                                     const float*,
                                                                                     const int*,
                                                                                     const int*,
                                                                                     const int*,
                                                                                     const int*,
                                                                                     const int*,
                                                                                     const int*,
                                                                                     const int*,
                                                                                     const int*,
                                                                                     int,
                                                                                     int,
                                                                                     float*,
                                                                                     int*,
                                                                                     cudaStream_t );

extern template cudaError_t ILUBaseNumericFactorizationCachedAsync<int, int, double>( int,
                                                                                      const int*,
                                                                                      const int*,
                                                                                      const double*,
                                                                                      const int*,
                                                                                      const int*,
                                                                                      const int*,
                                                                                      const int*,
                                                                                      const int*,
                                                                                      const int*,
                                                                                      const int*,
                                                                                      const int*,
                                                                                      int,
                                                                                      int,
                                                                                      double*,
                                                                                      int*,
                                                                                      cudaStream_t );

extern template cudaError_t ILUBaseNumericFactorizationCachedAsync<std::int64_t, int, double>(
    int,
    const std::int64_t*,
    const int*,
    const double*,
    const std::int64_t*,
    const int*,
    const std::int64_t*,
    const std::int64_t*,
    const std::int64_t*,
    const std::int64_t*,
    const int*,
    const int*,
    int,
    int,
    double*,
    int*,
    cudaStream_t );

extern template bool ILUBaseNumericFactorization<int, int, float>( int,
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
                                                                   cudaStream_t );

extern template bool ILUBaseNumericFactorization<int, int, double>( int,
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
                                                                    cudaStream_t );

extern template bool ILUBaseNumericFactorization<std::int64_t, int, double>( int,
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
                                                                             cudaStream_t );

} // namespace matrix_utils::sparse_cuda
