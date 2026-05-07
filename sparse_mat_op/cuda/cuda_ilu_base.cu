#include "cuda_ilu_base.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace matrix_utils::sparse_cuda
{
namespace
{
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;

template <typename ROWTYPE, typename COLTYPE>
__device__ ROWTYPE binary_search_row( const COLTYPE target,
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
__global__ void init_lu_values_kernel( COLTYPE n,
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
        const ROWTYPE a_pos = binary_search_row<ROWTYPE, COLTYPE>( col, a_begin, a_end, a_aj, base );
        lu_av[pos] = ( a_pos >= 0 ) ? a_av[a_pos] : VALTYPE( 0 );
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
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
    const ROWTYPE row_begin = lu_ai[i] - base;
    const ROWTYPE row_end = lu_ai[i + 1] - base;
    const ROWTYPE lower_end = lu_diag[i] - base;

    for ( ROWTYPE k_pos = row_begin; k_pos < lower_end; ++k_pos )
    {
        const COLTYPE k = lu_aj[k_pos] - base;
        VALTYPE aik = lu_av[k_pos];
        if ( aik == VALTYPE( 0 ) )
        {
            continue;
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
        if ( *status != 0 || aik == VALTYPE( 0 ) )
        {
            continue;
        }

        const ROWTYPE k_u_begin = ( lu_diag[k] - base ) + 1;
        const ROWTYPE k_u_end = lu_ai[k + 1] - base;
        for ( ROWTYPE j_pos = k_u_begin + lane; j_pos < k_u_end; j_pos += kWarpSize )
        {
            const COLTYPE j = lu_aj[j_pos] - base;
            const ROWTYPE pos_i = binary_search_row<ROWTYPE, COLTYPE>( j, row_begin, row_end, lu_aj, base );
            if ( pos_i >= 0 )
            {
                lu_av[pos_i] -= aik * lu_av[j_pos];
            }
        }
    }
}

inline bool cuda_ok( cudaError_t status )
{
    return status == cudaSuccess;
}

inline cudaError_t cuda_launch_status()
{
    return cudaGetLastError();
}

inline cudaError_t allocate_status( int** d_status, cudaStream_t stream, bool* async_allocated )
{
#if defined( CUDART_VERSION ) && CUDART_VERSION >= 11020
    cudaError_t status = cudaMallocAsync( reinterpret_cast<void**>( d_status ), sizeof( int ), stream );
    if ( status == cudaSuccess )
    {
        *async_allocated = true;
        return cudaSuccess;
    }
    if ( status != cudaErrorNotSupported )
    {
        return status;
    }
    cudaGetLastError();
#endif
    *async_allocated = false;
    return cudaMalloc( reinterpret_cast<void**>( d_status ), sizeof( int ) );
}

inline cudaError_t free_status( int* d_status, cudaStream_t stream, bool async_allocated )
{
#if defined( CUDART_VERSION ) && CUDART_VERSION >= 11020
    if ( async_allocated )
    {
        return cudaFreeAsync( d_status, stream );
    }
#endif
    return cudaFree( d_status );
}
} // namespace

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
                                              cudaStream_t stream )
{
    if ( n <= 0 || levels < 0 || d_a_ai == nullptr || d_a_aj == nullptr || d_a_av == nullptr ||
         d_lu_ai == nullptr || d_lu_aj == nullptr || d_lu_diag == nullptr ||
         d_level_perm == nullptr || h_level_prefix == nullptr || d_lu_av == nullptr ||
         d_status == nullptr )
    {
        return cudaErrorInvalidValue;
    }

    cudaError_t status = cudaMemsetAsync( d_status, 0, sizeof( int ), stream );
    if ( status != cudaSuccess )
    {
        return status;
    }

    const int init_blocks = ( n + kWarpsPerBlock - 1 ) / kWarpsPerBlock;
    init_lu_values_kernel<<<init_blocks, kThreadsPerBlock, 0, stream>>>(
        n, d_a_ai, d_a_aj, d_a_av, d_lu_ai, d_lu_aj, base, d_lu_av );
    status = cuda_launch_status();
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
        ilu_level_factor_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
            level_rows, d_level_perm + level_begin, d_lu_ai, d_lu_aj, d_lu_diag, base, d_lu_av, d_status );
        status = cuda_launch_status();
        if ( status != cudaSuccess )
        {
            return status;
        }
    }

    return cudaSuccess;
}

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
                                  cudaStream_t stream )
{
    int* d_status = nullptr;
    bool async_allocated = false;
    if ( !cuda_ok( allocate_status( &d_status, stream, &async_allocated ) ) )
    {
        return false;
    }

    const cudaError_t enqueue_status = ILUBaseNumericFactorizationAsync(
        n, d_a_ai, d_a_aj, d_a_av, d_lu_ai, d_lu_aj, d_lu_diag, d_level_perm,
        h_level_prefix, levels, base, d_lu_av, d_status, stream );
    if ( enqueue_status != cudaSuccess )
    {
        free_status( d_status, stream, async_allocated );
        if ( async_allocated )
        {
            cudaStreamSynchronize( stream );
        }
        return false;
    }

    int h_status = 0;
    const bool copied_status =
        cuda_ok( cudaMemcpyAsync( &h_status, d_status, sizeof( int ), cudaMemcpyDeviceToHost, stream ) );
    const bool synced = copied_status && cuda_ok( cudaStreamSynchronize( stream ) );
    free_status( d_status, stream, async_allocated );
    if ( async_allocated )
    {
        cudaStreamSynchronize( stream );
    }
    if ( !synced )
    {
        return false;
    }
    return h_status == 0;
}

template cudaError_t ILUBaseNumericFactorizationAsync<int, int, float>( int,
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

template cudaError_t ILUBaseNumericFactorizationAsync<int, int, double>( int,
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

template cudaError_t ILUBaseNumericFactorizationAsync<std::int64_t, int, double>( int,
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

template bool ILUBaseNumericFactorization<int, int, float>( int,
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

template bool ILUBaseNumericFactorization<int, int, double>( int,
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

template bool ILUBaseNumericFactorization<std::int64_t, int, double>( int,
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
