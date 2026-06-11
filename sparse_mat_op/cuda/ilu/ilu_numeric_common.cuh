#pragma once

#include <cuda/atomic>
#include <cuda_runtime.h>

#include <cstddef>

namespace matrix_utils::sparse_cuda::ilu_detail
{

inline constexpr int kWarpSize = 32;
inline constexpr int kWarpsPerBlock = 4;
inline constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
inline constexpr int kSharedRowColumnsPerWarp = 256;
inline constexpr int kMergeReferenceColumnsPerWarp = kWarpSize;

enum class RowIndexLookup
{
    Global,
    Shared
};

enum class RowUpdateStrategy
{
    BinarySearch,
    Merge
};

enum class RowWaitSleepPolicy
{
    Fixed64,
    Adaptive
};

template <typename COLTYPE>
inline constexpr std::size_t SharedRowIndexCacheBytes()
{
    return static_cast<std::size_t>( kWarpsPerBlock ) * kSharedRowColumnsPerWarp * sizeof( COLTYPE );
}

template <typename COLTYPE>
inline constexpr std::size_t SharedMergeReferenceCacheBytes()
{
    return static_cast<std::size_t>( kWarpsPerBlock ) * kMergeReferenceColumnsPerWarp * sizeof( COLTYPE );
}

template <typename COLTYPE>
inline constexpr std::size_t SharedFactorRowBytes( const RowIndexLookup lookup, const RowUpdateStrategy update )
{
    std::size_t bytes = 0;
    if ( lookup == RowIndexLookup::Shared && update == RowUpdateStrategy::BinarySearch )
    {
        bytes += SharedRowIndexCacheBytes<COLTYPE>();
    }
    if ( update == RowUpdateStrategy::Merge )
    {
        bytes += SharedMergeReferenceCacheBytes<COLTYPE>();
    }
    return bytes;
}

template <typename ROWTYPE, typename COLTYPE>
__device__ __forceinline__ ROWTYPE BinarySearchRow( const COLTYPE target,
                                                    const ROWTYPE row_begin,
                                                    const ROWTYPE row_end,
                                                    const COLTYPE* cols,
                                                    const COLTYPE base,
                                                    const ROWTYPE index_offset = ROWTYPE( 0 ) )
{
    ROWTYPE left = row_begin - index_offset;
    const ROWTYPE local_end = row_end - index_offset;
    ROWTYPE right = local_end;
    while ( left < right )
    {
        const ROWTYPE mid = left + ( right - left ) / 2;
        const COLTYPE col = cols[mid] - base;
        if ( col < target )
        {
            left = mid + 1;
        }
        else
        {
            right = mid;
        }
    }
    return ( left < local_end && cols[left] - base == target ) ? left + index_offset : ROWTYPE( -1 );
}

__device__ __forceinline__ int LoadDeviceInt( const int* value )
{
    cuda::atomic_ref<int, cuda::thread_scope_device> device_value( *const_cast<int*>( value ) );
    return device_value.load( cuda::memory_order_relaxed );
}

template <typename COLTYPE, RowWaitSleepPolicy SleepPolicy = RowWaitSleepPolicy::Fixed64>
__device__ __forceinline__ bool WaitForRowDone( const COLTYPE row, const int* row_done, const int* status, const int lane )
{
    int success = 1;
    if ( lane == 0 )
    {
        cuda::atomic_ref<int, cuda::thread_scope_device> ready( *const_cast<int*>( row_done + row ) );
        int spins = 0;
        while ( ready.load( cuda::memory_order_acquire ) == 0 )
        {
            if ( ( spins++ & 0xff ) == 0 && LoadDeviceInt( status ) != 0 )
            {
                success = 0;
                break;
            }
#if __CUDA_ARCH__ >= 700
            if ( spins > 64 )
            {
                if constexpr ( SleepPolicy == RowWaitSleepPolicy::Adaptive )
                {
                    __nanosleep( spins < 256 ? spins : 256 );
                }
                else
                {
                    __nanosleep( 64 );
                }
            }
#endif
        }
    }
    return __shfl_sync( 0xffffffffu, success, 0 ) != 0;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
__device__ __forceinline__ VALTYPE NormalizeLowerEntryWithDiagInv( const ROWTYPE k_pos,
                                                                   const COLTYPE k,
                                                                   VALTYPE* lu_av,
                                                                   const VALTYPE* diag_inv,
                                                                   const int lane )
{
    VALTYPE aik = lu_av[k_pos];
    if ( aik == VALTYPE( 0 ) )
    {
        return VALTYPE( 0 );
    }

    if ( lane == 0 )
    {
        aik *= diag_inv[k];
        lu_av[k_pos] = aik;
    }

    return __shfl_sync( 0xffffffffu, aik, 0 );
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
__device__ __forceinline__ void PublishRowDone( const COLTYPE row,
                                                const ROWTYPE diag_pos,
                                                const bool needs_fence,
                                                VALTYPE* lu_av,
                                                VALTYPE* diag_inv,
                                                int* status,
                                                int* row_done,
                                                const int lane )
{
    if ( needs_fence )
    {
        __threadfence();
        __syncwarp();
    }
    if ( lane == 0 && LoadDeviceInt( status ) == 0 )
    {
        const VALTYPE diagonal = lu_av[diag_pos];
        if ( diagonal == VALTYPE( 0 ) )
        {
            atomicCAS( status, 0, 1 );
        }
        else
        {
            diag_inv[row] = VALTYPE( 1 ) / diagonal;
            cuda::atomic_ref<int, cuda::thread_scope_device> ready( row_done[row] );
            ready.store( 1, cuda::memory_order_release );
        }
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, RowIndexLookup Lookup, RowWaitSleepPolicy SleepPolicy = RowWaitSleepPolicy::Fixed64, bool SyncAfterWait = true>
__device__ void FactorLURowBinarySearchWithRowDone( const ROWTYPE row_begin,
                                                    const ROWTYPE row_end,
                                                    const ROWTYPE lower_end,
                                                    const ROWTYPE* lu_ai,
                                                    const COLTYPE* lu_aj,
                                                    const ROWTYPE* lu_diag,
                                                    const COLTYPE base,
                                                    VALTYPE* lu_av,
                                                    const VALTYPE* diag_inv,
                                                    const int* status,
                                                    const int* row_done,
                                                    const int lane,
                                                    COLTYPE* shared_row_cols = nullptr )
{
    if constexpr ( Lookup == RowIndexLookup::Shared )
    {
        const ROWTYPE row_len = row_end - row_begin;
        for ( ROWTYPE offset = lane; offset < row_len; offset += kWarpSize )
        {
            shared_row_cols[offset] = lu_aj[row_begin + offset] - base;
        }
        __syncwarp();
    }

    for ( ROWTYPE k_pos = row_begin; k_pos < lower_end; ++k_pos )
    {
        const COLTYPE k = lu_aj[k_pos] - base;
        if ( !WaitForRowDone<COLTYPE, SleepPolicy>( k, row_done, status, lane ) )
        {
            return;
        }
        if constexpr ( SyncAfterWait )
        {
            __syncwarp();
        }

        const VALTYPE aik =
            NormalizeLowerEntryWithDiagInv<ROWTYPE, COLTYPE, VALTYPE>( k_pos, k, lu_av, diag_inv, lane );
        if ( aik == VALTYPE( 0 ) )
        {
            continue;
        }

        const ROWTYPE k_u_begin = ( lu_diag[k] - base ) + 1;
        const ROWTYPE k_u_end = lu_ai[k + 1] - base;
        for ( ROWTYPE j_pos = k_u_begin + lane; j_pos < k_u_end; j_pos += kWarpSize )
        {
            const COLTYPE j = lu_aj[j_pos] - base;
            ROWTYPE pos_i = ROWTYPE( -1 );
            if constexpr ( Lookup == RowIndexLookup::Shared )
            {
                pos_i = BinarySearchRow<ROWTYPE, COLTYPE>( j, k_pos + 1, row_end, shared_row_cols,
                                                           COLTYPE( 0 ), row_begin );
            }
            else
            {
                pos_i = BinarySearchRow<ROWTYPE, COLTYPE>( j, k_pos + 1, row_end, lu_aj, base );
            }
            if ( pos_i >= 0 )
            {
                lu_av[pos_i] -= aik * lu_av[j_pos];
            }
        }
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
static __global__ void InitLUValuesKernel( COLTYPE n,
                                           const ROWTYPE* a_ai,
                                           const COLTYPE* a_aj,
                                           const VALTYPE* a_av,
                                           const ROWTYPE* lu_ai,
                                           const COLTYPE* lu_aj,
                                           COLTYPE base,
                                           VALTYPE* lu_av )
{
    const int warp_in_block = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x & ( kWarpSize - 1 );
    const COLTYPE row = static_cast<COLTYPE>( blockIdx.x * kWarpsPerBlock + warp_in_block );
    if ( row >= n )
    {
        return;
    }

    const ROWTYPE lu_begin = lu_ai[row] - base;
    const ROWTYPE lu_end = lu_ai[row + 1] - base;
    const ROWTYPE a_begin = a_ai[row] - base;
    const ROWTYPE a_end = a_ai[row + 1] - base;

    for ( ROWTYPE pos = lu_begin + lane; pos < lu_end; pos += kWarpSize )
    {
        const COLTYPE col = lu_aj[pos] - base;
        const ROWTYPE a_pos = BinarySearchRow<ROWTYPE, COLTYPE>( col, a_begin, a_end, a_aj, base );
        lu_av[pos] = ( a_pos >= 0 ) ? a_av[a_pos] : VALTYPE( 0 );
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
__device__ __forceinline__ VALTYPE NormalizeLowerEntry( const ROWTYPE k_pos,
                                                        const COLTYPE k,
                                                        const ROWTYPE* lu_diag,
                                                        const COLTYPE base,
                                                        VALTYPE* lu_av,
                                                        int* status,
                                                        const int lane )
{
    VALTYPE aik = lu_av[k_pos];
    if ( aik == VALTYPE( 0 ) )
    {
        return VALTYPE( 0 );
    }

    if ( lane == 0 )
    {
        const VALTYPE akk = lu_av[lu_diag[k] - base];
        if ( akk == VALTYPE( 0 ) )
        {
            atomicCAS( status, 0, 1 );
            aik = VALTYPE( 0 );
        }
        else
        {
            aik /= akk;
            lu_av[k_pos] = aik;
        }
    }

    aik = __shfl_sync( 0xffffffffu, aik, 0 );
    return ( *status == 0 ) ? aik : VALTYPE( 0 );
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, RowIndexLookup Lookup = RowIndexLookup::Global>
__device__ __forceinline__ void FactorLURowBinarySearch( const COLTYPE i,
                                                         const ROWTYPE* lu_ai,
                                                         const COLTYPE* lu_aj,
                                                         const ROWTYPE* lu_diag,
                                                         const COLTYPE base,
                                                         VALTYPE* lu_av,
                                                         int* status,
                                                         const int lane,
                                                         COLTYPE* shared_row_cols = nullptr )
{
    const ROWTYPE row_begin = lu_ai[i] - base;
    const ROWTYPE row_end = lu_ai[i + 1] - base;
    const ROWTYPE lower_end = lu_diag[i] - base;
    const ROWTYPE row_len = row_end - row_begin;

    if ( row_begin == lower_end )
    {
        return;
    }

    if constexpr ( Lookup == RowIndexLookup::Shared )
    {
        for ( ROWTYPE offset = lane; offset < row_len; offset += kWarpSize )
        {
            shared_row_cols[offset] = lu_aj[row_begin + offset] - base;
        }
        __syncwarp();
    }

    // Walk the strictly lower part of the current i row. Each entry gives a
    // previously factored k row used to eliminate/update row i.
    for ( ROWTYPE k_pos = row_begin; k_pos < lower_end; ++k_pos )
    {
        const COLTYPE k = lu_aj[k_pos] - base;
        const VALTYPE aik =
            NormalizeLowerEntry<ROWTYPE, COLTYPE, VALTYPE>( k_pos, k, lu_diag, base, lu_av, status, lane );
        if ( aik == VALTYPE( 0 ) )
        {
            continue;
        }

        // Read U(k, j) from the already factored k row, then find the matching
        // column j in the current i row before applying A(i, j) -= L(i, k) * U(k, j).
        const ROWTYPE k_u_begin = ( lu_diag[k] - base ) + 1;
        const ROWTYPE k_u_end = lu_ai[k + 1] - base;
        for ( ROWTYPE j_pos = k_u_begin + lane; j_pos < k_u_end; j_pos += kWarpSize )
        {
            const COLTYPE j = lu_aj[j_pos] - base;
            ROWTYPE pos_i = ROWTYPE( -1 );
            if constexpr ( Lookup == RowIndexLookup::Shared )
            {
                pos_i = BinarySearchRow<ROWTYPE, COLTYPE>( j, k_pos + 1, row_end, shared_row_cols,
                                                           COLTYPE( 0 ), row_begin );
            }
            else
            {
                pos_i = BinarySearchRow<ROWTYPE, COLTYPE>( j, k_pos + 1, row_end, lu_aj, base );
            }
            if ( pos_i >= 0 )
            {
                lu_av[pos_i] -= aik * lu_av[j_pos];
            }
        }
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, RowIndexLookup Lookup = RowIndexLookup::Global>
__device__ __forceinline__ void FactorLURowMerge( const COLTYPE i,
                                                  const ROWTYPE* lu_ai,
                                                  const COLTYPE* lu_aj,
                                                  const ROWTYPE* lu_diag,
                                                  const COLTYPE base,
                                                  VALTYPE* lu_av,
                                                  int* status,
                                                  const int lane,
                                                  COLTYPE* shared_ref_cols )
{
    const ROWTYPE row_begin = lu_ai[i] - base;
    const ROWTYPE row_end = lu_ai[i + 1] - base;
    const ROWTYPE lower_end = lu_diag[i] - base;

    if ( row_begin == lower_end )
    {
        return;
    }

    // Walk the strictly lower part of the current i row. For each reference row,
    // intersect the sorted current-row suffix with U(k, :) using warp-sized tiles.
    for ( ROWTYPE k_pos = row_begin; k_pos < lower_end; ++k_pos )
    {
        const COLTYPE k = lu_aj[k_pos] - base;
        const VALTYPE aik =
            NormalizeLowerEntry<ROWTYPE, COLTYPE, VALTYPE>( k_pos, k, lu_diag, base, lu_av, status, lane );
        if ( aik == VALTYPE( 0 ) )
        {
            continue;
        }

        ROWTYPE curr_head = k_pos + 1;
        ROWTYPE ref_head = ( lu_diag[k] - base ) + 1;
        const ROWTYPE ref_end = lu_ai[k + 1] - base;

        while ( curr_head < row_end && ref_head < ref_end )
        {
            const ROWTYPE curr_tile_end = ( curr_head + kWarpSize < row_end ) ? curr_head + kWarpSize : row_end;
            const ROWTYPE ref_tile_end = ( ref_head + kWarpSize < ref_end ) ? ref_head + kWarpSize : ref_end;
            const int ref_count = static_cast<int>( ref_tile_end - ref_head );

            const ROWTYPE ref_pos = ref_head + lane;
            if ( ref_pos < ref_tile_end )
            {
                shared_ref_cols[lane] = lu_aj[ref_pos] - base;
            }
            __syncwarp();

            const ROWTYPE curr_pos = curr_head + lane;
            if ( curr_pos < curr_tile_end )
            {
                const COLTYPE curr_col = lu_aj[curr_pos] - base;
                for ( int ref_lane = 0; ref_lane < ref_count; ++ref_lane )
                {
                    if ( curr_col == shared_ref_cols[ref_lane] )
                    {
                        lu_av[curr_pos] -= aik * lu_av[ref_head + ref_lane];
                        break;
                    }
                }
            }

            const COLTYPE curr_last_col = lu_aj[curr_tile_end - 1] - base;
            const COLTYPE ref_last_col = lu_aj[ref_tile_end - 1] - base;
            if ( curr_last_col <= ref_last_col )
            {
                curr_head += kWarpSize;
            }
            if ( ref_last_col <= curr_last_col )
            {
                ref_head += kWarpSize;
            }
            __syncwarp();
        }
    }
}

inline cudaError_t CudaLaunchStatus()
{
    return cudaGetLastError();
}

} // namespace matrix_utils::sparse_cuda::ilu_detail
