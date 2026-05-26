#pragma once

#include <cuda_runtime.h>

namespace matrix_utils::sparse_cuda::ilu_detail
{

inline constexpr int kWarpSize = 32;
inline constexpr int kWarpsPerBlock = 4;
inline constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;

template <typename ROWTYPE, typename COLTYPE>
__device__ __forceinline__ ROWTYPE BinarySearchRow( const COLTYPE target,
                                                    const ROWTYPE row_begin,
                                                    const ROWTYPE row_end,
                                                    const COLTYPE* cols,
                                                    const COLTYPE base )
{
    ROWTYPE left = row_begin;
    ROWTYPE right = row_end;
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
    return ( left < row_end && cols[left] - base == target ) ? left : ROWTYPE( -1 );
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
__device__ __forceinline__ void InitializeLUValuesRow( const COLTYPE row,
                                                       const int lane,
                                                       const ROWTYPE* a_ai,
                                                       const COLTYPE* a_aj,
                                                       const VALTYPE* a_av,
                                                       const ROWTYPE* lu_ai,
                                                       const COLTYPE* lu_aj,
                                                       const COLTYPE base,
                                                       VALTYPE* lu_av )
{
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

    InitializeLUValuesRow<ROWTYPE, COLTYPE, VALTYPE>( row, lane, a_ai, a_aj, a_av, lu_ai, lu_aj, base, lu_av );
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

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
__device__ __forceinline__ void FactorLURowBinarySearch( const COLTYPE i,
                                                         const ROWTYPE* lu_ai,
                                                         const COLTYPE* lu_aj,
                                                         const ROWTYPE* lu_diag,
                                                         const COLTYPE base,
                                                         VALTYPE* lu_av,
                                                         int* status,
                                                         const int lane )
{
    const ROWTYPE row_begin = lu_ai[i] - base;
    const ROWTYPE row_end = lu_ai[i + 1] - base;
    const ROWTYPE lower_end = lu_diag[i] - base;

    for ( ROWTYPE k_pos = row_begin; k_pos < lower_end; ++k_pos )
    {
        const COLTYPE k = lu_aj[k_pos] - base;
        const VALTYPE aik =
            NormalizeLowerEntry<ROWTYPE, COLTYPE, VALTYPE>( k_pos, k, lu_diag, base, lu_av, status, lane );
        if ( aik == VALTYPE( 0 ) )
        {
            continue;
        }

        const ROWTYPE k_u_begin = ( lu_diag[k] - base ) + 1;
        const ROWTYPE k_u_end = lu_ai[k + 1] - base;
        for ( ROWTYPE j_pos = k_u_begin + lane; j_pos < k_u_end; j_pos += kWarpSize )
        {
            const COLTYPE j = lu_aj[j_pos] - base;
            const ROWTYPE pos_i = BinarySearchRow<ROWTYPE, COLTYPE>( j, row_begin, row_end, lu_aj, base );
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
