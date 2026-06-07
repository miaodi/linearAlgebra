#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace matrix_utils::sparse_cuda::ilu_detail
{

inline constexpr int kWarpSize = 32;
inline constexpr int kWarpsPerBlock = 4;
inline constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
inline constexpr int kSharedRowColumnsPerWarp = 256;

enum class RowIndexLookup
{
    Global,
    Shared
};

template <typename COLTYPE>
inline constexpr std::size_t SharedRowIndexCacheBytes()
{
    return static_cast<std::size_t>( kWarpsPerBlock ) * kSharedRowColumnsPerWarp * sizeof( COLTYPE );
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

inline cudaError_t CudaLaunchStatus()
{
    return cudaGetLastError();
}

} // namespace matrix_utils::sparse_cuda::ilu_detail
