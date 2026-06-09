#include "ilu_update_cache.hpp"

#include <cub/cub.cuh>

#include <chrono>
#include <cstdint>
#include <limits>

namespace matrix_utils::sparse_cuda
{
namespace
{
inline constexpr int kThreadsPerBlock = 256;

template <typename ROWTYPE, typename COLTYPE>
__global__ void compute_lower_counts_kernel( COLTYPE n, const ROWTYPE* lu_ai, const ROWTYPE* lu_diag, ROWTYPE* lower_counts )
{
    const COLTYPE idx = static_cast<COLTYPE>( blockIdx.x * blockDim.x + threadIdx.x );
    if ( idx < n )
    {
        lower_counts[idx] = lu_diag[idx] - lu_ai[idx];
    }
    else if ( idx == n )
    {
        lower_counts[idx] = ROWTYPE( 0 );
    }
}

template <typename ROWTYPE, typename COLTYPE>
__device__ COLTYPE find_lower_row( const ROWTYPE lower_id, COLTYPE n, const ROWTYPE* lower_row_ptr )
{
    COLTYPE left = 0;
    COLTYPE right = n;
    while ( left < right )
    {
        const COLTYPE mid = left + ( right - left ) / 2;
        if ( lower_row_ptr[mid] <= lower_id )
        {
            left = mid + 1;
        }
        else
        {
            right = mid;
        }
    }
    return left - 1;
}

template <typename ROWTYPE, typename COLTYPE>
__device__ ROWTYPE count_update_intersections_device( const ROWTYPE row_end,
                                                      const ROWTYPE k_pos,
                                                      const ROWTYPE k_u_begin,
                                                      const ROWTYPE k_u_end,
                                                      const COLTYPE* cols )
{
    ROWTYPE row_pos = k_pos + 1;
    ROWTYPE j_pos = k_u_begin;
    ROWTYPE count = 0;
    while ( row_pos < row_end && j_pos < k_u_end )
    {
        const COLTYPE row_col = cols[row_pos];
        const COLTYPE u_col = cols[j_pos];
        if ( row_col < u_col )
        {
            ++row_pos;
        }
        else if ( u_col < row_col )
        {
            ++j_pos;
        }
        else
        {
            ++count;
            ++row_pos;
            ++j_pos;
        }
    }
    return count;
}

template <typename ROWTYPE, typename COLTYPE>
__device__ void fill_update_intersections_device( const ROWTYPE row_end,
                                                  const ROWTYPE k_pos,
                                                  const ROWTYPE k_u_begin,
                                                  const ROWTYPE k_u_end,
                                                  const COLTYPE* cols,
                                                  ROWTYPE* update_jpos,
                                                  ROWTYPE* update_pos,
                                                  ROWTYPE write )
{
    ROWTYPE row_pos = k_pos + 1;
    ROWTYPE j_pos = k_u_begin;
    while ( row_pos < row_end && j_pos < k_u_end )
    {
        const COLTYPE row_col = cols[row_pos];
        const COLTYPE u_col = cols[j_pos];
        if ( row_col < u_col )
        {
            ++row_pos;
        }
        else if ( u_col < row_col )
        {
            ++j_pos;
        }
        else
        {
            update_jpos[write] = j_pos;
            update_pos[write] = row_pos;
            ++write;
            ++row_pos;
            ++j_pos;
        }
    }
}

template <typename ROWTYPE, typename COLTYPE>
__global__ void count_update_intersections_kernel( ROWTYPE strict_lower_nnz,
                                                   COLTYPE n,
                                                   const ROWTYPE* lu_ai,
                                                   const COLTYPE* lu_aj,
                                                   const ROWTYPE* lu_diag,
                                                   const ROWTYPE* lower_row_ptr,
                                                   COLTYPE base,
                                                   ROWTYPE* update_counts )
{
    const ROWTYPE lower_id = static_cast<ROWTYPE>( blockIdx.x * blockDim.x + threadIdx.x );
    if ( lower_id >= strict_lower_nnz )
    {
        return;
    }

    const COLTYPE i = find_lower_row<ROWTYPE, COLTYPE>( lower_id, n, lower_row_ptr );
    const ROWTYPE row_begin = lu_ai[i] - base;
    const ROWTYPE row_end = lu_ai[i + 1] - base;
    const ROWTYPE k_pos = row_begin + ( lower_id - lower_row_ptr[i] );
    const COLTYPE k = lu_aj[k_pos] - base;
    const ROWTYPE k_u_begin = ( lu_diag[k] - base ) + 1;
    const ROWTYPE k_u_end = lu_ai[k + 1] - base;
    update_counts[lower_id] = count_update_intersections_device( row_end, k_pos, k_u_begin, k_u_end, lu_aj );
}

template <typename ROWTYPE, typename COLTYPE>
__global__ void fill_update_intersections_kernel( ROWTYPE strict_lower_nnz,
                                                  COLTYPE n,
                                                  const ROWTYPE* lu_ai,
                                                  const COLTYPE* lu_aj,
                                                  const ROWTYPE* lu_diag,
                                                  const ROWTYPE* lower_row_ptr,
                                                  const ROWTYPE* update_ptr,
                                                  COLTYPE base,
                                                  ROWTYPE* update_jpos,
                                                  ROWTYPE* update_pos )
{
    const ROWTYPE lower_id = static_cast<ROWTYPE>( blockIdx.x * blockDim.x + threadIdx.x );
    if ( lower_id >= strict_lower_nnz )
    {
        return;
    }

    const COLTYPE i = find_lower_row<ROWTYPE, COLTYPE>( lower_id, n, lower_row_ptr );
    const ROWTYPE row_begin = lu_ai[i] - base;
    const ROWTYPE row_end = lu_ai[i + 1] - base;
    const ROWTYPE k_pos = row_begin + ( lower_id - lower_row_ptr[i] );
    const COLTYPE k = lu_aj[k_pos] - base;
    const ROWTYPE k_u_begin = ( lu_diag[k] - base ) + 1;
    const ROWTYPE k_u_end = lu_ai[k + 1] - base;
    fill_update_intersections_device( row_end, k_pos, k_u_begin, k_u_end, lu_aj, update_jpos,
                                      update_pos, update_ptr[lower_id] );
}

template <typename ROWTYPE>
cudaError_t exclusive_sum( const ROWTYPE* input, ROWTYPE* output, int items, cudaStream_t stream )
{
    void* temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;
    cudaError_t status =
        cub::DeviceScan::ExclusiveSum( temp_storage, temp_storage_bytes, input, output, items, stream );
    if ( status != cudaSuccess )
    {
        return status;
    }

    AsyncDeviceArray<std::uint8_t> scan_storage( AsyncDeviceAllocator{ stream } );
    scan_storage.resize( temp_storage_bytes );
    return cub::DeviceScan::ExclusiveSum( scan_storage.data(), temp_storage_bytes, input, output, items, stream );
}

template <typename T>
bool fits_cub_items( T value )
{
    return value <= static_cast<T>( std::numeric_limits<int>::max() );
}
} // namespace

template <typename ROWTYPE, typename COLTYPE>
cudaError_t BuildILUUpdateCacheAsync( const COLTYPE n,
                                      const ROWTYPE* d_lu_ai,
                                      const COLTYPE* d_lu_aj,
                                      const ROWTYPE* d_lu_diag,
                                      const COLTYPE base,
                                      DeviceILUUpdateCache<ROWTYPE>& cache,
                                      cudaStream_t stream )
{
    if ( n <= 0 || d_lu_ai == nullptr || d_lu_aj == nullptr || d_lu_diag == nullptr )
    {
        return cudaErrorInvalidValue;
    }
    if ( !fits_cub_items( static_cast<std::int64_t>( n ) + 1 ) )
    {
        return cudaErrorInvalidValue;
    }

    const auto build_start = std::chrono::steady_clock::now();
    const AsyncDeviceAllocator pool{ stream };

    cache.lower_row_ptr.resize( static_cast<std::size_t>( n ) + 1 );
    AsyncDeviceArray<ROWTYPE> lower_counts( pool );
    lower_counts.resize( static_cast<std::size_t>( n ) + 1 );

    const int lower_count_blocks = ( static_cast<int>( n ) + 1 + kThreadsPerBlock - 1 ) / kThreadsPerBlock;
    compute_lower_counts_kernel<<<lower_count_blocks, kThreadsPerBlock, 0, stream>>>(
        n, d_lu_ai, d_lu_diag, lower_counts.data() );
    cudaError_t status = cudaGetLastError();
    if ( status != cudaSuccess )
    {
        return status;
    }

    status = exclusive_sum( lower_counts.data(), cache.lower_row_ptr.data(), static_cast<int>( n ) + 1, stream );
    if ( status != cudaSuccess )
    {
        return status;
    }

    ROWTYPE strict_lower_nnz = 0;
    status = cudaMemcpyAsync( &strict_lower_nnz, cache.lower_row_ptr.data() + n, sizeof( ROWTYPE ),
                              cudaMemcpyDeviceToHost, stream );
    if ( status != cudaSuccess )
    {
        return status;
    }
    status = cudaStreamSynchronize( stream );
    if ( status != cudaSuccess )
    {
        return status;
    }
    if ( strict_lower_nnz < 0 || !fits_cub_items( static_cast<std::int64_t>( strict_lower_nnz ) + 1 ) )
    {
        return cudaErrorInvalidValue;
    }

    cache.strict_lower_nnz = strict_lower_nnz;
    cache.update_ptr.resize( static_cast<std::size_t>( strict_lower_nnz ) + 1 );
    if ( strict_lower_nnz == 0 )
    {
        status = cudaMemsetAsync( cache.update_ptr.data(), 0, sizeof( ROWTYPE ), stream );
        if ( status != cudaSuccess )
        {
            return status;
        }
        cache.update_jpos.resize( 0 );
        cache.update_pos.resize( 0 );
        cache.total_updates = 0;
        status = cudaStreamSynchronize( stream );
        if ( status != cudaSuccess )
        {
            return status;
        }
        const auto build_end = std::chrono::steady_clock::now();
        cache.build_ms = std::chrono::duration<double, std::milli>( build_end - build_start ).count();
        return cudaSuccess;
    }

    AsyncDeviceArray<ROWTYPE> update_counts( pool );
    update_counts.resize( static_cast<std::size_t>( strict_lower_nnz ) + 1 );
    status = cudaMemsetAsync( update_counts.data() + strict_lower_nnz, 0, sizeof( ROWTYPE ), stream );
    if ( status != cudaSuccess )
    {
        return status;
    }

    const int count_blocks = static_cast<int>( ( strict_lower_nnz + kThreadsPerBlock - 1 ) / kThreadsPerBlock );
    count_update_intersections_kernel<<<count_blocks, kThreadsPerBlock, 0, stream>>>(
        strict_lower_nnz, n, d_lu_ai, d_lu_aj, d_lu_diag, cache.lower_row_ptr.data(), base,
        update_counts.data() );
    status = cudaGetLastError();
    if ( status != cudaSuccess )
    {
        return status;
    }

    status = exclusive_sum( update_counts.data(), cache.update_ptr.data(),
                            static_cast<int>( strict_lower_nnz ) + 1, stream );
    if ( status != cudaSuccess )
    {
        return status;
    }

    ROWTYPE total_updates = 0;
    status = cudaMemcpyAsync( &total_updates, cache.update_ptr.data() + strict_lower_nnz,
                              sizeof( ROWTYPE ), cudaMemcpyDeviceToHost, stream );
    if ( status != cudaSuccess )
    {
        return status;
    }
    status = cudaStreamSynchronize( stream );
    if ( status != cudaSuccess )
    {
        return status;
    }
    if ( total_updates < 0 )
    {
        return cudaErrorInvalidValue;
    }

    cache.total_updates = total_updates;
    cache.update_jpos.resize( static_cast<std::size_t>( total_updates ) );
    cache.update_pos.resize( static_cast<std::size_t>( total_updates ) );
    if ( total_updates > 0 )
    {
        fill_update_intersections_kernel<<<count_blocks, kThreadsPerBlock, 0, stream>>>(
            strict_lower_nnz, n, d_lu_ai, d_lu_aj, d_lu_diag, cache.lower_row_ptr.data(),
            cache.update_ptr.data(), base, cache.update_jpos.data(), cache.update_pos.data() );
        status = cudaGetLastError();
        if ( status != cudaSuccess )
        {
            return status;
        }
    }

    status = cudaStreamSynchronize( stream );
    if ( status != cudaSuccess )
    {
        return status;
    }

    const auto build_end = std::chrono::steady_clock::now();
    cache.build_ms = std::chrono::duration<double, std::milli>( build_end - build_start ).count();
    return cudaSuccess;
}

template cudaError_t BuildILUUpdateCacheAsync<int, int>( int,
                                                         const int*,
                                                         const int*,
                                                         const int*,
                                                         int,
                                                         DeviceILUUpdateCache<int>&,
                                                         cudaStream_t );

template cudaError_t BuildILUUpdateCacheAsync<std::int64_t, int>( int,
                                                                  const std::int64_t*,
                                                                  const int*,
                                                                  const std::int64_t*,
                                                                  int,
                                                                  DeviceILUUpdateCache<std::int64_t>&,
                                                                  cudaStream_t );

} // namespace matrix_utils::sparse_cuda
