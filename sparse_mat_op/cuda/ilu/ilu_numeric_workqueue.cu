#include "ilu_numeric_workqueue.cuh"
#include "ilu_numeric_common.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#if 0
// Disabled while focusing on the base global/shared binary-search numeric path.
namespace matrix_utils::sparse_cuda
{
namespace
{
using ilu_detail::kThreadsPerBlock;
using ilu_detail::kWarpSize;
using ilu_detail::kWarpsPerBlock;

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
__global__ void ilu_level_factor_workqueue_kernel( COLTYPE level_rows,
                                                   const COLTYPE* level_rows_perm,
                                                   const ROWTYPE* lu_ai,
                                                   const COLTYPE* lu_aj,
                                                   const ROWTYPE* lu_diag,
                                                   COLTYPE base,
                                                   VALTYPE* lu_av,
                                                   int* status,
                                                   COLTYPE* level_row_counter )
{
    const int warp_in_block = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x & ( kWarpSize - 1 );
    extern __shared__ unsigned char shared_storage[];
    COLTYPE* shared_row_cols = reinterpret_cast<COLTYPE*>( shared_storage ) +
                               warp_in_block * ilu_detail::kSharedRowColumnsPerWarp;

    while ( true )
    {
        COLTYPE level_row = level_rows;
        if ( lane == 0 )
        {
            level_row = ( *status == 0 ) ? atomicAdd( level_row_counter, COLTYPE( 1 ) ) : level_rows;
        }
        level_row = __shfl_sync( 0xffffffffu, level_row, 0 );
        if ( level_row >= level_rows )
        {
            return;
        }

        const COLTYPE i = level_rows_perm[level_row] - base;
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
}

inline cudaError_t compute_persistent_block_limit( const int blocks_per_sm, int* block_limit )
{
    if ( block_limit == nullptr || blocks_per_sm <= 0 )
    {
        return cudaErrorInvalidValue;
    }

    int device = 0;
    cudaError_t status = cudaGetDevice( &device );
    if ( status != cudaSuccess )
    {
        return status;
    }

    int sm_count = 0;
    status = cudaDeviceGetAttribute( &sm_count, cudaDevAttrMultiProcessorCount, device );
    if ( status != cudaSuccess )
    {
        return status;
    }
    sm_count = std::max( sm_count, 1 );

    const long long persistent_blocks = static_cast<long long>( sm_count ) * blocks_per_sm;
    *block_limit = static_cast<int>( std::min<long long>(
        std::max<long long>( persistent_blocks, 1 ), std::numeric_limits<int>::max() ) );
    return cudaSuccess;
}

inline int select_workqueue_blocks( const long long level_rows, const int block_limit )
{
    const long long static_blocks = ( level_rows + kWarpsPerBlock - 1 ) / kWarpsPerBlock;
    const long long selected_blocks =
        std::max<long long>( 1, std::min<long long>( static_blocks, block_limit ) );
    return static_cast<int>( std::min<long long>( selected_blocks, std::numeric_limits<int>::max() ) );
}
} // namespace

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
cudaError_t ILUBaseNumericFactorizationWorkQueueAsync( COLTYPE n,
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
                                                       int blocks_per_sm,
                                                       cudaStream_t stream )
{
    if ( n <= 0 || levels < 0 || d_lu_ai == nullptr || d_lu_aj == nullptr || d_lu_diag == nullptr ||
         d_level_perm == nullptr || h_level_prefix == nullptr || d_lu_av == nullptr ||
         d_status == nullptr || d_level_row_counter == nullptr || blocks_per_sm <= 0 )
    {
        return cudaErrorInvalidValue;
    }

    cudaError_t status = cudaMemsetAsync( d_status, 0, sizeof( int ), stream );
    if ( status != cudaSuccess )
    {
        return status;
    }

    int persistent_block_limit = 0;
    status = compute_persistent_block_limit( blocks_per_sm, &persistent_block_limit );
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

        status = cudaMemsetAsync( d_level_row_counter, 0, sizeof( COLTYPE ), stream );
        if ( status != cudaSuccess )
        {
            return status;
        }

        const int blocks =
            select_workqueue_blocks( static_cast<long long>( level_rows ), persistent_block_limit );

        const auto shared_bytes = ilu_detail::SharedRowIndexCacheBytes<COLTYPE>();
        ilu_level_factor_workqueue_kernel<<<blocks, kThreadsPerBlock, shared_bytes, stream>>>(
            level_rows, d_level_perm + level_begin, d_lu_ai, d_lu_aj, d_lu_diag, base, d_lu_av,
            d_status, d_level_row_counter );
        status = ilu_detail::CudaLaunchStatus();
        if ( status != cudaSuccess )
        {
            return status;
        }
    }

    return cudaSuccess;
}

template cudaError_t ILUBaseNumericFactorizationWorkQueueAsync<int, int, float>( int,
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

template cudaError_t ILUBaseNumericFactorizationWorkQueueAsync<int, int, double>( int,
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

template cudaError_t ILUBaseNumericFactorizationWorkQueueAsync<std::int64_t, int, double>( int,
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
#endif
