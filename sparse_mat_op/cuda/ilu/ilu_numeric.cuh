#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace matrix_utils::sparse_cuda
{

enum class ILUNumericRowLookup
{
    Global,
    Shared
};

enum class ILUNumericRowUpdateStrategy
{
    BinarySearch,
    Merge
};

/**
 * @brief Embed A values into a precomputed LU sparsity pattern.
 *
 * For every entry in the LU pattern, this copies the matching A value when the
 * entry exists in A and writes zero for fill-ins. A and LU column indices must
 * be sorted within each row because the device kernel uses binary search.
 *
 * @param n Matrix order.
 * @param d_a_ai Input A row pointers on device.
 * @param d_a_aj Input A column indices on device.
 * @param d_a_av Input A values on device.
 * @param d_lu_ai Symbolic LU row pointers on device.
 * @param d_lu_aj Symbolic LU column indices on device.
 * @param base CSR base, usually 0.
 * @param d_lu_av Output LU values on device, size d_lu_ai[n] - base.
 * @param stream CUDA stream.
 */
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
cudaError_t ILUEmbedAValuesToLUAsync( COLTYPE n,
                                      const ROWTYPE* d_a_ai,
                                      const COLTYPE* d_a_aj,
                                      const VALTYPE* d_a_av,
                                      const ROWTYPE* d_lu_ai,
                                      const COLTYPE* d_lu_aj,
                                      COLTYPE base,
                                      VALTYPE* d_lu_av,
                                      cudaStream_t stream = nullptr );

/**
 * @brief Base CUDA numerical ILU factorization with explicit row lookup/update modes.
 *
 * The host traverses topological levels. For each level, one warp is assigned
 * to each row. Within a row, the warp processes lower entries sequentially and
 * updates row values in parallel. The caller must initialize d_lu_av first,
 * usually with ILUEmbedAValuesToLUAsync.
 *
 * Global reads row indices from global memory. Shared caches current-row indices
 * in shared memory when the row fits the internal per-warp capacity and falls
 * back to global-memory row lookup for longer rows. BinarySearch updates use a
 * per-reference-entry binary search into the current row. Merge updates use
 * warp-sized sorted-row intersection tiles.
 */
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
cudaError_t ILUBaseNumericFactorizationAsync( COLTYPE n,
                                              const ROWTYPE* d_lu_ai,
                                              const COLTYPE* d_lu_aj,
                                              const ROWTYPE* d_lu_diag,
                                              const COLTYPE* d_level_perm,
                                              const COLTYPE* h_level_prefix,
                                              COLTYPE levels,
                                              COLTYPE base,
                                              VALTYPE* d_lu_av,
                                              int* d_status,
                                              ILUNumericRowLookup row_lookup,
                                              ILUNumericRowUpdateStrategy row_update,
                                              cudaStream_t stream = nullptr );

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
cudaError_t ILUBaseNumericFactorizationAsync( COLTYPE n,
                                              const ROWTYPE* d_lu_ai,
                                              const COLTYPE* d_lu_aj,
                                              const ROWTYPE* d_lu_diag,
                                              const COLTYPE* d_level_perm,
                                              const COLTYPE* h_level_prefix,
                                              COLTYPE levels,
                                              COLTYPE base,
                                              VALTYPE* d_lu_av,
                                              int* d_status,
                                              ILUNumericRowLookup row_lookup,
                                              cudaStream_t stream = nullptr );

#if 0
// Disabled while focusing on the base global/shared binary-search numeric path.
/**
 * @brief CUDA numerical ILU factorization using a precomputed update cache.
 *
 * The cache removes numeric-phase binary searches. For every lower entry
 * position k_pos in row i, update_ptr[k_pos]..update_ptr[k_pos + 1] stores the
 * compact list of source U positions and destination row-i positions that
 * survive the ILU sparsity pattern.
 *
 * The caller must initialize d_lu_av first, usually with
 * ILUEmbedAValuesToLUAsync.
 *
 * @param d_update_ptr Device update offsets, size d_lu_ai[n] - base + 1.
 * @param d_update_jpos Device source LU value positions for U(k,j).
 * @param d_update_pos Device destination LU value positions for row i.
 */
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
cudaError_t ILUBaseNumericFactorizationCachedAsync( COLTYPE n,
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
#endif

extern template cudaError_t ILUEmbedAValuesToLUAsync<int, int, float>( int,
                                                                       const int*,
                                                                       const int*,
                                                                       const float*,
                                                                       const int*,
                                                                       const int*,
                                                                       int,
                                                                       float*,
                                                                       cudaStream_t );

extern template cudaError_t ILUEmbedAValuesToLUAsync<int, int, double>( int,
                                                                        const int*,
                                                                        const int*,
                                                                        const double*,
                                                                        const int*,
                                                                        const int*,
                                                                        int,
                                                                        double*,
                                                                        cudaStream_t );

extern template cudaError_t ILUEmbedAValuesToLUAsync<std::int64_t, int, double>( int,
                                                                                 const std::int64_t*,
                                                                                 const int*,
                                                                                 const double*,
                                                                                 const std::int64_t*,
                                                                                 const int*,
                                                                                 int,
                                                                                 double*,
                                                                                 cudaStream_t );

extern template cudaError_t ILUBaseNumericFactorizationAsync<int, int, float>( int,
                                                                               const int*,
                                                                               const int*,
                                                                               const int*,
                                                                               const int*,
                                                                               const int*,
                                                                               int,
                                                                               int,
                                                                               float*,
                                                                               int*,
                                                                               ILUNumericRowLookup,
                                                                               ILUNumericRowUpdateStrategy,
                                                                               cudaStream_t );

extern template cudaError_t ILUBaseNumericFactorizationAsync<int, int, double>( int,
                                                                                const int*,
                                                                                const int*,
                                                                                const int*,
                                                                                const int*,
                                                                                const int*,
                                                                                int,
                                                                                int,
                                                                                double*,
                                                                                int*,
                                                                                ILUNumericRowLookup,
                                                                                ILUNumericRowUpdateStrategy,
                                                                                cudaStream_t );

extern template cudaError_t ILUBaseNumericFactorizationAsync<std::int64_t, int, double>( int,
                                                                                         const std::int64_t*,
                                                                                         const int*,
                                                                                         const std::int64_t*,
                                                                                         const int*,
                                                                                         const int*,
                                                                                         int,
                                                                                         int,
                                                                                         double*,
                                                                                         int*,
                                                                                         ILUNumericRowLookup,
                                                                                         ILUNumericRowUpdateStrategy,
                                                                                         cudaStream_t );

extern template cudaError_t ILUBaseNumericFactorizationAsync<int, int, float>( int,
                                                                               const int*,
                                                                               const int*,
                                                                               const int*,
                                                                               const int*,
                                                                               const int*,
                                                                               int,
                                                                               int,
                                                                               float*,
                                                                               int*,
                                                                               ILUNumericRowLookup,
                                                                               cudaStream_t );

extern template cudaError_t ILUBaseNumericFactorizationAsync<int, int, double>( int,
                                                                                const int*,
                                                                                const int*,
                                                                                const int*,
                                                                                const int*,
                                                                                const int*,
                                                                                int,
                                                                                int,
                                                                                double*,
                                                                                int*,
                                                                                ILUNumericRowLookup,
                                                                                cudaStream_t );

extern template cudaError_t ILUBaseNumericFactorizationAsync<std::int64_t, int, double>( int,
                                                                                         const std::int64_t*,
                                                                                         const int*,
                                                                                         const std::int64_t*,
                                                                                         const int*,
                                                                                         const int*,
                                                                                         int,
                                                                                         int,
                                                                                         double*,
                                                                                         int*,
                                                                                         ILUNumericRowLookup,
                                                                                         cudaStream_t );

#if 0
// Disabled while focusing on the base global/shared binary-search numeric path.
extern template cudaError_t ILUBaseNumericFactorizationCachedAsync<int, int, float>( int,
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
#endif

} // namespace matrix_utils::sparse_cuda
