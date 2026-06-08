#include "ilu_numeric.cuh"
#include "ilu_numeric_common.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace matrix_utils::sparse_cuda
{
namespace
{
using ilu_detail::kThreadsPerBlock;
using ilu_detail::kWarpSize;
using ilu_detail::kWarpsPerBlock;

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, ilu_detail::RowIndexLookup Lookup, ilu_detail::RowUpdateStrategy Update>
__global__ void ilu_level_factor_kernel( COLTYPE level_rows,
                                         const COLTYPE* level_rows_perm,
                                         const ROWTYPE* lu_ai,
                                         const COLTYPE* lu_aj,
                                         const ROWTYPE* lu_diag,
                                         COLTYPE base,
                                         VALTYPE* lu_av,
                                         int* status )
{
    const int warp_in_block = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x & ( kWarpSize - 1 );
    const COLTYPE level_row = static_cast<COLTYPE>( blockIdx.x * kWarpsPerBlock + warp_in_block );
    if ( level_row >= level_rows )
    {
        return;
    }

    const COLTYPE i = level_rows_perm[level_row] - base;

    COLTYPE* shared_row_cols = nullptr;
    COLTYPE* shared_ref_cols = nullptr;
    if constexpr ( ( Lookup == ilu_detail::RowIndexLookup::Shared &&
                     Update == ilu_detail::RowUpdateStrategy::BinarySearch ) ||
                   Update == ilu_detail::RowUpdateStrategy::Merge )
    {
        extern __shared__ unsigned char shared_storage[];
        COLTYPE* shared_cols = reinterpret_cast<COLTYPE*>( shared_storage );
        int shared_offset = 0;
        if constexpr ( Lookup == ilu_detail::RowIndexLookup::Shared &&
                       Update == ilu_detail::RowUpdateStrategy::BinarySearch )
        {
            shared_row_cols = shared_cols + shared_offset + warp_in_block * ilu_detail::kSharedRowColumnsPerWarp;
            shared_offset += ilu_detail::kWarpsPerBlock * ilu_detail::kSharedRowColumnsPerWarp;
        }
        if constexpr ( Update == ilu_detail::RowUpdateStrategy::Merge )
        {
            shared_ref_cols = shared_cols + shared_offset + warp_in_block * ilu_detail::kMergeReferenceColumnsPerWarp;
        }
    }

    if constexpr ( Update == ilu_detail::RowUpdateStrategy::Merge )
    {
        ilu_detail::FactorLURowMerge<ROWTYPE, COLTYPE, VALTYPE, Lookup>(
            i, lu_ai, lu_aj, lu_diag, base, lu_av, status, lane, shared_ref_cols );
    }
    else if constexpr ( Lookup == ilu_detail::RowIndexLookup::Shared )
    {
        const ROWTYPE row_len = ( lu_ai[i + 1] - base ) - ( lu_ai[i] - base );
        if ( row_len <= static_cast<ROWTYPE>( ilu_detail::kSharedRowColumnsPerWarp ) )
        {
            ilu_detail::FactorLURowBinarySearch<ROWTYPE, COLTYPE, VALTYPE, ilu_detail::RowIndexLookup::Shared>(
                i, lu_ai, lu_aj, lu_diag, base, lu_av, status, lane, shared_row_cols );
        }
        else
        {
            ilu_detail::FactorLURowBinarySearch<ROWTYPE, COLTYPE, VALTYPE, ilu_detail::RowIndexLookup::Global>(
                i, lu_ai, lu_aj, lu_diag, base, lu_av, status, lane );
        }
    }
    else
    {
        ilu_detail::FactorLURowBinarySearch<ROWTYPE, COLTYPE, VALTYPE, ilu_detail::RowIndexLookup::Global>(
            i, lu_ai, lu_aj, lu_diag, base, lu_av, status, lane );
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, ilu_detail::RowIndexLookup Lookup, ilu_detail::RowUpdateStrategy Update>
cudaError_t launch_level_factor_kernel( const int blocks,
                                        const COLTYPE level_rows,
                                        const COLTYPE* level_rows_perm,
                                        const ROWTYPE* lu_ai,
                                        const COLTYPE* lu_aj,
                                        const ROWTYPE* lu_diag,
                                        const COLTYPE base,
                                        VALTYPE* lu_av,
                                        int* status,
                                        cudaStream_t stream )
{
    const auto shared_bytes = ilu_detail::SharedFactorRowBytes<COLTYPE>( Lookup, Update );
    ilu_level_factor_kernel<ROWTYPE, COLTYPE, VALTYPE, Lookup, Update>
        <<<blocks, kThreadsPerBlock, shared_bytes, stream>>>( level_rows, level_rows_perm, lu_ai,
                                                              lu_aj, lu_diag, base, lu_av, status );
    return ilu_detail::CudaLaunchStatus();
}

#if 0
// Disabled while focusing on the base global/shared binary-search numeric path.
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
__global__ void ilu_level_factor_cached_kernel( COLTYPE level_rows,
                                                const COLTYPE* level_rows_perm,
                                                const ROWTYPE* lu_ai,
                                                const COLTYPE* lu_aj,
                                                const ROWTYPE* lu_diag,
                                                const ROWTYPE* update_ptr,
                                                const ROWTYPE* update_jpos,
                                                const ROWTYPE* update_pos,
                                                COLTYPE base,
                                                VALTYPE* lu_av,
                                                int* status )
{
    const int warp_in_block = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x & ( kWarpSize - 1 );
    const COLTYPE level_row = static_cast<COLTYPE>( blockIdx.x * kWarpsPerBlock + warp_in_block );
    if ( level_row >= level_rows )
    {
        return;
    }

    const COLTYPE i = level_rows_perm[level_row] - base;
    const ROWTYPE row_begin = lu_ai[i] - base;
    const ROWTYPE lower_end = lu_diag[i] - base;

    for ( ROWTYPE k_pos = row_begin; k_pos < lower_end; ++k_pos )
    {
        const COLTYPE k = lu_aj[k_pos] - base;
        const VALTYPE aik = ilu_detail::NormalizeLowerEntry<ROWTYPE, COLTYPE, VALTYPE>(
            k_pos, k, lu_diag, base, lu_av, status, lane );
        if ( aik == VALTYPE( 0 ) )
        {
            continue;
        }

        const ROWTYPE update_begin = update_ptr[k_pos];
        const ROWTYPE update_end = update_ptr[k_pos + 1];
        for ( ROWTYPE update = update_begin + lane; update < update_end; update += kWarpSize )
        {
            lu_av[update_pos[update]] -= aik * lu_av[update_jpos[update]];
        }
    }
}
#endif

} // namespace

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
cudaError_t ILUEmbedAValuesToLUAsync( COLTYPE n,
                                      const ROWTYPE* d_a_ai,
                                      const COLTYPE* d_a_aj,
                                      const VALTYPE* d_a_av,
                                      const ROWTYPE* d_lu_ai,
                                      const COLTYPE* d_lu_aj,
                                      COLTYPE base,
                                      VALTYPE* d_lu_av,
                                      cudaStream_t stream )
{
    if ( n <= 0 || d_a_ai == nullptr || d_a_aj == nullptr || d_a_av == nullptr ||
         d_lu_ai == nullptr || d_lu_aj == nullptr || d_lu_av == nullptr )
    {
        return cudaErrorInvalidValue;
    }

    const int init_blocks = ( n + kWarpsPerBlock - 1 ) / kWarpsPerBlock;
    ilu_detail::InitLUValuesKernel<<<init_blocks, kThreadsPerBlock, 0, stream>>>(
        n, d_a_ai, d_a_aj, d_a_av, d_lu_ai, d_lu_aj, base, d_lu_av );
    return ilu_detail::CudaLaunchStatus();
}

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
                                              cudaStream_t stream )
{
    if ( n <= 0 || levels < 0 || d_lu_ai == nullptr || d_lu_aj == nullptr || d_lu_diag == nullptr ||
         d_level_perm == nullptr || h_level_prefix == nullptr || d_lu_av == nullptr || d_status == nullptr )
    {
        return cudaErrorInvalidValue;
    }

    cudaError_t status = cudaMemsetAsync( d_status, 0, sizeof( int ), stream );
    if ( status != cudaSuccess )
    {
        return status;
    }

    for ( COLTYPE level = 0; level < levels; ++level )
    {
        const COLTYPE level_begin = h_level_prefix[level] - base;
        const COLTYPE level_end = h_level_prefix[level + 1] - base;
        const COLTYPE level_rows = level_end - level_begin;
        if ( level_rows <= 0 )
        {
            continue;
        }

        const int blocks = ( level_rows + kWarpsPerBlock - 1 ) / kWarpsPerBlock;
        switch ( row_lookup )
        {
        case ILUNumericRowLookup::Global:
            switch ( row_update )
            {
            case ILUNumericRowUpdateStrategy::BinarySearch:
                status = launch_level_factor_kernel<ROWTYPE, COLTYPE, VALTYPE, ilu_detail::RowIndexLookup::Global,
                                                    ilu_detail::RowUpdateStrategy::BinarySearch>(
                    blocks, level_rows, d_level_perm + level_begin, d_lu_ai, d_lu_aj, d_lu_diag,
                    base, d_lu_av, d_status, stream );
                break;
            case ILUNumericRowUpdateStrategy::Merge:
                status = launch_level_factor_kernel<ROWTYPE, COLTYPE, VALTYPE, ilu_detail::RowIndexLookup::Global,
                                                    ilu_detail::RowUpdateStrategy::Merge>(
                    blocks, level_rows, d_level_perm + level_begin, d_lu_ai, d_lu_aj, d_lu_diag,
                    base, d_lu_av, d_status, stream );
                break;
            default:
                return cudaErrorInvalidValue;
            }
            break;
        case ILUNumericRowLookup::Shared:
            switch ( row_update )
            {
            case ILUNumericRowUpdateStrategy::BinarySearch:
                status = launch_level_factor_kernel<ROWTYPE, COLTYPE, VALTYPE, ilu_detail::RowIndexLookup::Shared,
                                                    ilu_detail::RowUpdateStrategy::BinarySearch>(
                    blocks, level_rows, d_level_perm + level_begin, d_lu_ai, d_lu_aj, d_lu_diag,
                    base, d_lu_av, d_status, stream );
                break;
            case ILUNumericRowUpdateStrategy::Merge:
                status = launch_level_factor_kernel<ROWTYPE, COLTYPE, VALTYPE, ilu_detail::RowIndexLookup::Shared,
                                                    ilu_detail::RowUpdateStrategy::Merge>(
                    blocks, level_rows, d_level_perm + level_begin, d_lu_ai, d_lu_aj, d_lu_diag,
                    base, d_lu_av, d_status, stream );
                break;
            default:
                return cudaErrorInvalidValue;
            }
            break;
        default:
            return cudaErrorInvalidValue;
        }
        if ( status != cudaSuccess )
        {
            return status;
        }
    }

    return cudaSuccess;
}

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
                                              cudaStream_t stream )
{
    return ILUBaseNumericFactorizationAsync<ROWTYPE, COLTYPE, VALTYPE>(
        n, d_lu_ai, d_lu_aj, d_lu_diag, d_level_perm, h_level_prefix, levels, base, d_lu_av,
        d_status, row_lookup, ILUNumericRowUpdateStrategy::BinarySearch, stream );
}

#if 0
// Disabled while focusing on the base global/shared binary-search numeric path.
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
                                                    cudaStream_t stream )
{
    if ( n <= 0 || levels < 0 || d_lu_ai == nullptr || d_lu_aj == nullptr || d_lu_diag == nullptr ||
         d_update_ptr == nullptr || d_update_jpos == nullptr || d_update_pos == nullptr ||
         d_level_perm == nullptr || h_level_prefix == nullptr || d_lu_av == nullptr || d_status == nullptr )
    {
        return cudaErrorInvalidValue;
    }

    cudaError_t status = cudaMemsetAsync( d_status, 0, sizeof( int ), stream );
    if ( status != cudaSuccess )
    {
        return status;
    }

    for ( COLTYPE level = 0; level < levels; ++level )
    {
        const COLTYPE level_begin = h_level_prefix[level] - base;
        const COLTYPE level_end = h_level_prefix[level + 1] - base;
        const COLTYPE level_rows = level_end - level_begin;
        if ( level_rows <= 0 )
        {
            continue;
        }

        const int blocks = ( level_rows + kWarpsPerBlock - 1 ) / kWarpsPerBlock;
        ilu_level_factor_cached_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
            level_rows, d_level_perm + level_begin, d_lu_ai, d_lu_aj, d_lu_diag, d_update_ptr,
            d_update_jpos, d_update_pos, base, d_lu_av, d_status );
        status = ilu_detail::CudaLaunchStatus();
        if ( status != cudaSuccess )
        {
            return status;
        }
    }

    return cudaSuccess;
}
#endif

template cudaError_t ILUEmbedAValuesToLUAsync<int, int, float>( int,
                                                                const int*,
                                                                const int*,
                                                                const float*,
                                                                const int*,
                                                                const int*,
                                                                int,
                                                                float*,
                                                                cudaStream_t );

template cudaError_t ILUEmbedAValuesToLUAsync<int, int, double>( int,
                                                                 const int*,
                                                                 const int*,
                                                                 const double*,
                                                                 const int*,
                                                                 const int*,
                                                                 int,
                                                                 double*,
                                                                 cudaStream_t );

template cudaError_t ILUEmbedAValuesToLUAsync<std::int64_t, int, double>( int,
                                                                          const std::int64_t*,
                                                                          const int*,
                                                                          const double*,
                                                                          const std::int64_t*,
                                                                          const int*,
                                                                          int,
                                                                          double*,
                                                                          cudaStream_t );

template cudaError_t ILUBaseNumericFactorizationAsync<int, int, float>( int,
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

template cudaError_t ILUBaseNumericFactorizationAsync<int, int, double>( int,
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

template cudaError_t ILUBaseNumericFactorizationAsync<std::int64_t, int, double>( int,
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

template cudaError_t ILUBaseNumericFactorizationAsync<int, int, float>( int,
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

template cudaError_t ILUBaseNumericFactorizationAsync<int, int, double>( int,
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

template cudaError_t ILUBaseNumericFactorizationAsync<std::int64_t, int, double>( int,
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
template cudaError_t ILUBaseNumericFactorizationCachedAsync<int, int, float>( int,
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

template cudaError_t ILUBaseNumericFactorizationCachedAsync<int, int, double>( int,
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

template cudaError_t ILUBaseNumericFactorizationCachedAsync<std::int64_t, int, double>( int,
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
