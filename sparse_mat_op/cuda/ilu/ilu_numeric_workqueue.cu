#include "ilu_numeric_workqueue.cuh"
#include "ilu_numeric_common.cuh"

#include <cuda/atomic>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace matrix_utils::sparse_cuda
{
namespace
{
using ilu_detail::kSharedRowColumnsPerWarp;
using ilu_detail::kWarpSize;

__device__ __forceinline__ int load_status( const int* status )
{
    cuda::atomic_ref<int, cuda::thread_scope_device> status_value( *const_cast<int*>( status ) );
    return status_value.load( cuda::memory_order_relaxed );
}

template <typename COLTYPE>
__device__ __forceinline__ bool wait_for_row_done( const COLTYPE row, const int* row_done, const int* status, const int lane )
{
    int success = 1;
    if ( lane == 0 )
    {
        cuda::atomic_ref<int, cuda::thread_scope_device> ready( *const_cast<int*>( row_done + row ) );
        int spins = 0;
        while ( ready.load( cuda::memory_order_acquire ) == 0 )
        {
            if ( ( spins++ & 0xff ) == 0 && load_status( status ) != 0 )
            {
                success = 0;
                break;
            }
#if __CUDA_ARCH__ >= 700
            if ( spins > 64 )
            {
                __nanosleep( spins < 256 ? spins : 256 );
            }
#endif
        }
    }
    return __shfl_sync( 0xffffffffu, success, 0 ) != 0;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
__device__ __forceinline__ VALTYPE normalize_lower_entry_with_diag_inv( const ROWTYPE k_pos,
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
__device__ __forceinline__ void publish_row_done( const COLTYPE row,
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
    if ( lane == 0 && load_status( status ) == 0 )
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

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, ilu_detail::RowIndexLookup Lookup>
__device__ void factor_lu_row_binary_search_persistent( const ROWTYPE row_begin,
                                                        const ROWTYPE row_end,
                                                        const ROWTYPE lower_end,
                                                        const ROWTYPE* lu_ai,
                                                        const COLTYPE* lu_aj,
                                                        const ROWTYPE* lu_diag,
                                                        const COLTYPE base,
                                                        VALTYPE* lu_av,
                                                        const VALTYPE* diag_inv,
                                                        int* status,
                                                        const int* row_done,
                                                        const int lane,
                                                        COLTYPE* shared_row_cols = nullptr )
{
    if constexpr ( Lookup == ilu_detail::RowIndexLookup::Shared )
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
        if ( !wait_for_row_done( k, row_done, status, lane ) )
        {
            return;
        }

        const VALTYPE aik = normalize_lower_entry_with_diag_inv<ROWTYPE, COLTYPE, VALTYPE>(
            k_pos, k, lu_av, diag_inv, lane );
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
            if constexpr ( Lookup == ilu_detail::RowIndexLookup::Shared )
            {
                pos_i = ilu_detail::BinarySearchRow<ROWTYPE, COLTYPE>(
                    j, k_pos + 1, row_end, shared_row_cols, COLTYPE( 0 ), row_begin );
            }
            else
            {
                pos_i = ilu_detail::BinarySearchRow<ROWTYPE, COLTYPE>( j, k_pos + 1, row_end, lu_aj, base );
            }
            if ( pos_i >= 0 )
            {
                lu_av[pos_i] -= aik * lu_av[j_pos];
            }
        }
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
__device__ void factor_lu_row_cached_persistent( const ROWTYPE row_begin,
                                                 const ROWTYPE lower_end,
                                                 const ROWTYPE lower_begin,
                                                 const COLTYPE* lu_aj,
                                                 VALTYPE* lu_av,
                                                 const VALTYPE* diag_inv,
                                                 const ROWTYPE* update_ptr,
                                                 const ROWTYPE* update_jpos,
                                                 const ROWTYPE* update_pos,
                                                 const COLTYPE base,
                                                 const int* row_done,
                                                 const int* status,
                                                 const int lane )
{
    for ( ROWTYPE k_pos = row_begin; k_pos < lower_end; ++k_pos )
    {
        const COLTYPE k = lu_aj[k_pos] - base;
        if ( !wait_for_row_done( k, row_done, status, lane ) )
        {
            return;
        }

        const VALTYPE aik = normalize_lower_entry_with_diag_inv<ROWTYPE, COLTYPE, VALTYPE>(
            k_pos, k, lu_av, diag_inv, lane );
        if ( aik == VALTYPE( 0 ) )
        {
            continue;
        }

        const ROWTYPE lower_id = lower_begin + ( k_pos - row_begin );
        const ROWTYPE update_begin = update_ptr[lower_id];
        const ROWTYPE update_end = update_ptr[lower_id + 1];
        for ( ROWTYPE update = update_begin + lane; update < update_end; update += kWarpSize )
        {
            lu_av[update_pos[update]] -= aik * lu_av[update_jpos[update]];
        }
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
__global__ void ilu_persistent_spin_kernel( COLTYPE n,
                                            const ROWTYPE* lu_ai,
                                            const COLTYPE* lu_aj,
                                            const ROWTYPE* lu_diag,
                                            COLTYPE base,
                                            VALTYPE* lu_av,
                                            VALTYPE* diag_inv,
                                            int* status,
                                            COLTYPE* next_row,
                                            int* row_done )
{
    const int warp_in_block = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x & ( kWarpSize - 1 );
    extern __shared__ unsigned char shared_storage[];
    COLTYPE* shared_row_cols = reinterpret_cast<COLTYPE*>( shared_storage ) +
                               static_cast<std::size_t>( warp_in_block ) * kSharedRowColumnsPerWarp;

    while ( true )
    {
        COLTYPE row = n;
        if ( lane == 0 )
        {
            row = ( load_status( status ) == 0 ) ? atomicAdd( next_row, COLTYPE( 1 ) ) : n;
        }
        row = __shfl_sync( 0xffffffffu, row, 0 );
        if ( row >= n )
        {
            return;
        }

        const ROWTYPE row_begin = lu_ai[row] - base;
        const ROWTYPE row_end = lu_ai[row + 1] - base;
        const ROWTYPE lower_end = lu_diag[row] - base;
        const bool has_lower = row_begin < lower_end;
        if ( has_lower && row_end - row_begin <= static_cast<ROWTYPE>( kSharedRowColumnsPerWarp ) )
        {
            factor_lu_row_binary_search_persistent<ROWTYPE, COLTYPE, VALTYPE, ilu_detail::RowIndexLookup::Shared>(
                row_begin, row_end, lower_end, lu_ai, lu_aj, lu_diag, base, lu_av, diag_inv, status,
                row_done, lane, shared_row_cols );
        }
        else if ( has_lower )
        {
            factor_lu_row_binary_search_persistent<ROWTYPE, COLTYPE, VALTYPE, ilu_detail::RowIndexLookup::Global>(
                row_begin, row_end, lower_end, lu_ai, lu_aj, lu_diag, base, lu_av, diag_inv, status,
                row_done, lane );
        }

        publish_row_done<ROWTYPE, COLTYPE, VALTYPE>( row, lower_end, has_lower, lu_av, diag_inv,
                                                     status, row_done, lane );
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
__global__ void ilu_persistent_cached_kernel( COLTYPE n,
                                              const ROWTYPE* lu_ai,
                                              const COLTYPE* lu_aj,
                                              const ROWTYPE* lu_diag,
                                              const ROWTYPE* lower_row_ptr,
                                              const ROWTYPE* update_ptr,
                                              const ROWTYPE* update_jpos,
                                              const ROWTYPE* update_pos,
                                              COLTYPE base,
                                              VALTYPE* lu_av,
                                              VALTYPE* diag_inv,
                                              int* status,
                                              COLTYPE* next_row,
                                              int* row_done )
{
    const int lane = threadIdx.x & ( kWarpSize - 1 );

    while ( true )
    {
        COLTYPE row = n;
        if ( lane == 0 )
        {
            row = ( load_status( status ) == 0 ) ? atomicAdd( next_row, COLTYPE( 1 ) ) : n;
        }
        row = __shfl_sync( 0xffffffffu, row, 0 );
        if ( row >= n )
        {
            return;
        }

        const ROWTYPE row_begin = lu_ai[row] - base;
        const ROWTYPE lower_end = lu_diag[row] - base;
        const bool has_lower = row_begin < lower_end;
        if ( has_lower )
        {
            factor_lu_row_cached_persistent<ROWTYPE, COLTYPE, VALTYPE>(
                row_begin, lower_end, lower_row_ptr[row], lu_aj, lu_av, diag_inv, update_ptr,
                update_jpos, update_pos, base, row_done, status, lane );
        }

        publish_row_done<ROWTYPE, COLTYPE, VALTYPE>( row, lower_end, has_lower, lu_av, diag_inv,
                                                     status, row_done, lane );
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
cudaError_t select_persistent_launch_config( const COLTYPE n, ILUPersistentLaunchConfig* config )
{
    if ( n <= 0 || config == nullptr )
    {
        return cudaErrorInvalidValue;
    }

    int device = 0;
    cudaError_t status = cudaGetDevice( &device );
    if ( status != cudaSuccess )
    {
        return status;
    }

    cudaDeviceProp prop{};
    status = cudaGetDeviceProperties( &prop, device );
    if ( status != cudaSuccess )
    {
        return status;
    }

    constexpr int candidates[] = { 64, 128, 256, 512 };
    int best_block_size = 0;
    int best_blocks_per_sm = 0;
    int best_resident_warps_per_sm = -1;

    for ( const int block_size : candidates )
    {
        if ( block_size > prop.maxThreadsPerBlock )
        {
            continue;
        }

        const int warps_per_block = block_size / kWarpSize;
        const std::size_t shared_bytes =
            static_cast<std::size_t>( warps_per_block ) * kSharedRowColumnsPerWarp * sizeof( COLTYPE );
        int blocks_per_sm = 0;
        status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, ilu_persistent_spin_kernel<ROWTYPE, COLTYPE, VALTYPE>, block_size, shared_bytes );
        if ( status != cudaSuccess )
        {
            return status;
        }
        if ( blocks_per_sm <= 0 )
        {
            continue;
        }

        const int resident_warps_per_sm = blocks_per_sm * warps_per_block;
        if ( resident_warps_per_sm > best_resident_warps_per_sm ||
             ( resident_warps_per_sm == best_resident_warps_per_sm && block_size < best_block_size ) )
        {
            best_block_size = block_size;
            best_blocks_per_sm = blocks_per_sm;
            best_resident_warps_per_sm = resident_warps_per_sm;
        }
    }

    if ( best_block_size <= 0 || best_blocks_per_sm <= 0 )
    {
        return cudaErrorInvalidValue;
    }

    const int warps_per_block = best_block_size / kWarpSize;
    const long long static_blocks = ( static_cast<long long>( n ) + warps_per_block - 1 ) / warps_per_block;
    const long long occupancy_blocks =
        static_cast<long long>( std::max( prop.multiProcessorCount, 1 ) ) * best_blocks_per_sm;
    const long long grid_blocks = std::max<long long>( 1, std::min( static_blocks, occupancy_blocks ) );

    config->block_size = best_block_size;
    config->grid_blocks =
        static_cast<int>( std::min<long long>( grid_blocks, std::numeric_limits<int>::max() ) );
    config->blocks_per_sm = best_blocks_per_sm;
    config->resident_warps = config->grid_blocks * warps_per_block;
    return cudaSuccess;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
cudaError_t select_persistent_cached_launch_config( const COLTYPE n, ILUPersistentLaunchConfig* config )
{
    if ( n <= 0 || config == nullptr )
    {
        return cudaErrorInvalidValue;
    }

    int device = 0;
    cudaError_t status = cudaGetDevice( &device );
    if ( status != cudaSuccess )
    {
        return status;
    }

    cudaDeviceProp prop{};
    status = cudaGetDeviceProperties( &prop, device );
    if ( status != cudaSuccess )
    {
        return status;
    }

    constexpr int candidates[] = { 64, 128, 256, 512 };
    int best_block_size = 0;
    int best_blocks_per_sm = 0;
    int best_resident_warps_per_sm = -1;

    for ( const int block_size : candidates )
    {
        if ( block_size > prop.maxThreadsPerBlock )
        {
            continue;
        }

        const int warps_per_block = block_size / kWarpSize;
        int blocks_per_sm = 0;
        status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, ilu_persistent_cached_kernel<ROWTYPE, COLTYPE, VALTYPE>, block_size, 0 );
        if ( status != cudaSuccess )
        {
            return status;
        }
        if ( blocks_per_sm <= 0 )
        {
            continue;
        }

        const int resident_warps_per_sm = blocks_per_sm * warps_per_block;
        if ( resident_warps_per_sm > best_resident_warps_per_sm ||
             ( resident_warps_per_sm == best_resident_warps_per_sm && block_size < best_block_size ) )
        {
            best_block_size = block_size;
            best_blocks_per_sm = blocks_per_sm;
            best_resident_warps_per_sm = resident_warps_per_sm;
        }
    }

    if ( best_block_size <= 0 || best_blocks_per_sm <= 0 )
    {
        return cudaErrorInvalidValue;
    }

    const int warps_per_block = best_block_size / kWarpSize;
    const long long static_blocks = ( static_cast<long long>( n ) + warps_per_block - 1 ) / warps_per_block;
    const long long occupancy_blocks =
        static_cast<long long>( std::max( prop.multiProcessorCount, 1 ) ) * best_blocks_per_sm;
    const long long grid_blocks = std::max<long long>( 1, std::min( static_blocks, occupancy_blocks ) );

    config->block_size = best_block_size;
    config->grid_blocks =
        static_cast<int>( std::min<long long>( grid_blocks, std::numeric_limits<int>::max() ) );
    config->blocks_per_sm = best_blocks_per_sm;
    config->resident_warps = config->grid_blocks * warps_per_block;
    return cudaSuccess;
}
} // namespace

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
                                                        cudaStream_t stream,
                                                        ILUPersistentLaunchConfig* h_launch_config )
{
    if ( n <= 0 || d_lu_ai == nullptr || d_lu_aj == nullptr || d_lu_diag == nullptr || d_lu_av == nullptr ||
         d_diag_inv == nullptr || d_status == nullptr || d_next_row == nullptr || d_row_done == nullptr )
    {
        return cudaErrorInvalidValue;
    }

    ILUPersistentLaunchConfig config;
    cudaError_t status = select_persistent_launch_config<ROWTYPE, COLTYPE, VALTYPE>( n, &config );
    if ( status != cudaSuccess )
    {
        return status;
    }

    status = cudaMemsetAsync( d_status, 0, sizeof( int ), stream );
    if ( status != cudaSuccess )
    {
        return status;
    }
    status = cudaMemsetAsync( d_next_row, 0, sizeof( COLTYPE ), stream );
    if ( status != cudaSuccess )
    {
        return status;
    }
    status = cudaMemsetAsync( d_row_done, 0, static_cast<std::size_t>( n ) * sizeof( int ), stream );
    if ( status != cudaSuccess )
    {
        return status;
    }

    const int warps_per_block = config.block_size / kWarpSize;
    const std::size_t shared_bytes =
        static_cast<std::size_t>( warps_per_block ) * kSharedRowColumnsPerWarp * sizeof( COLTYPE );
    ilu_persistent_spin_kernel<<<config.grid_blocks, config.block_size, shared_bytes, stream>>>(
        n, d_lu_ai, d_lu_aj, d_lu_diag, base, d_lu_av, d_diag_inv, d_status, d_next_row, d_row_done );
    status = ilu_detail::CudaLaunchStatus();
    if ( status != cudaSuccess )
    {
        return status;
    }

    if ( h_launch_config != nullptr )
    {
        *h_launch_config = config;
    }
    return cudaSuccess;
}

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
                                                              cudaStream_t stream,
                                                              ILUPersistentLaunchConfig* h_launch_config )
{
    if ( n <= 0 || d_lu_ai == nullptr || d_lu_aj == nullptr || d_lu_diag == nullptr ||
         d_lower_row_ptr == nullptr || d_update_ptr == nullptr || d_update_jpos == nullptr ||
         d_update_pos == nullptr || d_lu_av == nullptr || d_diag_inv == nullptr ||
         d_status == nullptr || d_next_row == nullptr || d_row_done == nullptr )
    {
        return cudaErrorInvalidValue;
    }

    ILUPersistentLaunchConfig config;
    cudaError_t status = select_persistent_cached_launch_config<ROWTYPE, COLTYPE, VALTYPE>( n, &config );
    if ( status != cudaSuccess )
    {
        return status;
    }

    status = cudaMemsetAsync( d_status, 0, sizeof( int ), stream );
    if ( status != cudaSuccess )
    {
        return status;
    }
    status = cudaMemsetAsync( d_next_row, 0, sizeof( COLTYPE ), stream );
    if ( status != cudaSuccess )
    {
        return status;
    }
    status = cudaMemsetAsync( d_row_done, 0, static_cast<std::size_t>( n ) * sizeof( int ), stream );
    if ( status != cudaSuccess )
    {
        return status;
    }

    ilu_persistent_cached_kernel<<<config.grid_blocks, config.block_size, 0, stream>>>(
        n, d_lu_ai, d_lu_aj, d_lu_diag, d_lower_row_ptr, d_update_ptr, d_update_jpos, d_update_pos,
        base, d_lu_av, d_diag_inv, d_status, d_next_row, d_row_done );
    status = ilu_detail::CudaLaunchStatus();
    if ( status != cudaSuccess )
    {
        return status;
    }

    if ( h_launch_config != nullptr )
    {
        *h_launch_config = config;
    }
    return cudaSuccess;
}

template cudaError_t ILUBaseNumericFactorizationPersistentAsync<int, int, float>( int,
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

template cudaError_t ILUBaseNumericFactorizationPersistentAsync<int, int, double>( int,
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

template cudaError_t ILUBaseNumericFactorizationPersistentAsync<std::int64_t, int, double>(
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

template cudaError_t ILUBaseNumericFactorizationPersistentCachedAsync<int, int, float>( int,
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

template cudaError_t ILUBaseNumericFactorizationPersistentCachedAsync<int, int, double>( int,
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

template cudaError_t ILUBaseNumericFactorizationPersistentCachedAsync<std::int64_t, int, double>(
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

} // namespace matrix_utils::sparse_cuda
