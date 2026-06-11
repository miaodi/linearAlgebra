#include "ilu_numeric_persistent.cuh"
#include "ilu_numeric_common.cuh"

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
        if ( !ilu_detail::WaitForRowDone<COLTYPE, ilu_detail::RowWaitSleepPolicy::Adaptive>(
                 k, row_done, status, lane ) )
        {
            return;
        }

        const VALTYPE aik = ilu_detail::NormalizeLowerEntryWithDiagInv<ROWTYPE, COLTYPE, VALTYPE>(
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

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, bool UseRowPerm>
__global__ void ilu_persistent_spin_kernel( COLTYPE n,
                                            const ROWTYPE* lu_ai,
                                            const COLTYPE* lu_aj,
                                            const ROWTYPE* lu_diag,
                                            const COLTYPE* row_perm,
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
        COLTYPE row_slot = n;
        if ( lane == 0 )
        {
            row_slot = ( ilu_detail::LoadDeviceInt( status ) == 0 ) ? atomicAdd( next_row, COLTYPE( 1 ) ) : n;
        }
        row_slot = __shfl_sync( 0xffffffffu, row_slot, 0 );
        if ( row_slot >= n )
        {
            return;
        }

        COLTYPE row = row_slot;
        if constexpr ( UseRowPerm )
        {
            row = row_perm[row_slot] - base;
        }

        const ROWTYPE row_begin = lu_ai[row] - base;
        const ROWTYPE row_end = lu_ai[row + 1] - base;
        const ROWTYPE lower_end = lu_diag[row] - base;
        const bool has_lower = row_begin < lower_end;
        if ( has_lower )
        {
            if ( row_end - row_begin <= static_cast<ROWTYPE>( kSharedRowColumnsPerWarp ) )
            {
                ilu_detail::FactorLURowBinarySearchWithRowDone<ROWTYPE, COLTYPE, VALTYPE, ilu_detail::RowIndexLookup::Shared,
                                                               ilu_detail::RowWaitSleepPolicy::Adaptive, false>(
                    row_begin, row_end, lower_end, lu_ai, lu_aj, lu_diag, base, lu_av, diag_inv,
                    status, row_done, lane, shared_row_cols );
            }
            else
            {
                ilu_detail::FactorLURowBinarySearchWithRowDone<ROWTYPE, COLTYPE, VALTYPE, ilu_detail::RowIndexLookup::Global,
                                                               ilu_detail::RowWaitSleepPolicy::Adaptive, false>(
                    row_begin, row_end, lower_end, lu_ai, lu_aj, lu_diag, base, lu_av, diag_inv,
                    status, row_done, lane );
            }
        }

        ilu_detail::PublishRowDone<ROWTYPE, COLTYPE, VALTYPE>( row, lower_end, has_lower, lu_av,
                                                               diag_inv, status, row_done, lane );
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, bool UseRowPerm>
__global__ void ilu_persistent_cached_kernel( COLTYPE n,
                                              const ROWTYPE* lu_ai,
                                              const COLTYPE* lu_aj,
                                              const ROWTYPE* lu_diag,
                                              const ROWTYPE* lower_row_ptr,
                                              const ROWTYPE* update_ptr,
                                              const ROWTYPE* update_jpos,
                                              const ROWTYPE* update_pos,
                                              const COLTYPE* row_perm,
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
        COLTYPE row_slot = n;
        if ( lane == 0 )
        {
            row_slot = ( ilu_detail::LoadDeviceInt( status ) == 0 ) ? atomicAdd( next_row, COLTYPE( 1 ) ) : n;
        }
        row_slot = __shfl_sync( 0xffffffffu, row_slot, 0 );
        if ( row_slot >= n )
        {
            return;
        }

        COLTYPE row = row_slot;
        if constexpr ( UseRowPerm )
        {
            row = row_perm[row_slot] - base;
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

        ilu_detail::PublishRowDone<ROWTYPE, COLTYPE, VALTYPE>( row, lower_end, has_lower, lu_av,
                                                               diag_inv, status, row_done, lane );
    }
}

template <typename COLTYPE, typename Kernel, typename DynamicSharedBytes>
cudaError_t select_persistent_launch_config_impl( const COLTYPE n,
                                                  ILUPersistentLaunchConfig* config,
                                                  Kernel kernel,
                                                  DynamicSharedBytes dynamic_shared_bytes )
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
        const std::size_t shared_bytes = dynamic_shared_bytes( block_size );
        int blocks_per_sm = 0;
        status = cudaOccupancyMaxActiveBlocksPerMultiprocessor( &blocks_per_sm, kernel, block_size, shared_bytes );
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

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, bool UseRowPerm>
cudaError_t select_persistent_launch_config( const COLTYPE n, ILUPersistentLaunchConfig* config )
{
    return select_persistent_launch_config_impl<COLTYPE>(
        n, config, ilu_persistent_spin_kernel<ROWTYPE, COLTYPE, VALTYPE, UseRowPerm>,
        []( const int block_size ) -> std::size_t
        {
            const int warps_per_block = block_size / kWarpSize;
            return static_cast<std::size_t>( warps_per_block ) * kSharedRowColumnsPerWarp * sizeof( COLTYPE );
        } );
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, bool UseRowPerm>
cudaError_t select_persistent_cached_launch_config( const COLTYPE n, ILUPersistentLaunchConfig* config )
{
    return select_persistent_launch_config_impl<COLTYPE>(
        n, config, ilu_persistent_cached_kernel<ROWTYPE, COLTYPE, VALTYPE, UseRowPerm>,
        []( const int ) -> std::size_t { return 0; } );
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, bool UseRowPerm>
cudaError_t launch_persistent_async( COLTYPE n,
                                     const ROWTYPE* d_lu_ai,
                                     const COLTYPE* d_lu_aj,
                                     const ROWTYPE* d_lu_diag,
                                     const COLTYPE* d_row_perm,
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
    if constexpr ( UseRowPerm )
    {
        if ( d_row_perm == nullptr )
        {
            return cudaErrorInvalidValue;
        }
    }

    ILUPersistentLaunchConfig config;
    cudaError_t status = select_persistent_launch_config<ROWTYPE, COLTYPE, VALTYPE, UseRowPerm>( n, &config );
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
    ilu_persistent_spin_kernel<ROWTYPE, COLTYPE, VALTYPE, UseRowPerm>
        <<<config.grid_blocks, config.block_size, shared_bytes, stream>>>(
            n, d_lu_ai, d_lu_aj, d_lu_diag, d_row_perm, base, d_lu_av, d_diag_inv, d_status,
            d_next_row, d_row_done );
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

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, bool UseRowPerm>
cudaError_t launch_persistent_cached_async( COLTYPE n,
                                            const ROWTYPE* d_lu_ai,
                                            const COLTYPE* d_lu_aj,
                                            const ROWTYPE* d_lu_diag,
                                            const ROWTYPE* d_lower_row_ptr,
                                            const ROWTYPE* d_update_ptr,
                                            const ROWTYPE* d_update_jpos,
                                            const ROWTYPE* d_update_pos,
                                            const COLTYPE* d_row_perm,
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
    if constexpr ( UseRowPerm )
    {
        if ( d_row_perm == nullptr )
        {
            return cudaErrorInvalidValue;
        }
    }

    ILUPersistentLaunchConfig config;
    cudaError_t status =
        select_persistent_cached_launch_config<ROWTYPE, COLTYPE, VALTYPE, UseRowPerm>( n, &config );
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

    ilu_persistent_cached_kernel<ROWTYPE, COLTYPE, VALTYPE, UseRowPerm>
        <<<config.grid_blocks, config.block_size, 0, stream>>>(
            n, d_lu_ai, d_lu_aj, d_lu_diag, d_lower_row_ptr, d_update_ptr, d_update_jpos,
            d_update_pos, d_row_perm, base, d_lu_av, d_diag_inv, d_status, d_next_row, d_row_done );
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
    return launch_persistent_async<ROWTYPE, COLTYPE, VALTYPE, false>(
        n, d_lu_ai, d_lu_aj, d_lu_diag, nullptr, base, d_lu_av, d_diag_inv, d_status, d_next_row,
        d_row_done, stream, h_launch_config );
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
cudaError_t ILUBaseNumericFactorizationPersistentPermAsync( COLTYPE n,
                                                            const ROWTYPE* d_lu_ai,
                                                            const COLTYPE* d_lu_aj,
                                                            const ROWTYPE* d_lu_diag,
                                                            const COLTYPE* d_row_perm,
                                                            COLTYPE base,
                                                            VALTYPE* d_lu_av,
                                                            VALTYPE* d_diag_inv,
                                                            int* d_status,
                                                            COLTYPE* d_next_row,
                                                            int* d_row_done,
                                                            cudaStream_t stream,
                                                            ILUPersistentLaunchConfig* h_launch_config )
{
    return launch_persistent_async<ROWTYPE, COLTYPE, VALTYPE, true>(
        n, d_lu_ai, d_lu_aj, d_lu_diag, d_row_perm, base, d_lu_av, d_diag_inv, d_status, d_next_row,
        d_row_done, stream, h_launch_config );
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
    return launch_persistent_cached_async<ROWTYPE, COLTYPE, VALTYPE, false>(
        n, d_lu_ai, d_lu_aj, d_lu_diag, d_lower_row_ptr, d_update_ptr, d_update_jpos, d_update_pos,
        nullptr, base, d_lu_av, d_diag_inv, d_status, d_next_row, d_row_done, stream, h_launch_config );
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
cudaError_t ILUBaseNumericFactorizationPersistentCachedPermAsync( COLTYPE n,
                                                                  const ROWTYPE* d_lu_ai,
                                                                  const COLTYPE* d_lu_aj,
                                                                  const ROWTYPE* d_lu_diag,
                                                                  const ROWTYPE* d_lower_row_ptr,
                                                                  const ROWTYPE* d_update_ptr,
                                                                  const ROWTYPE* d_update_jpos,
                                                                  const ROWTYPE* d_update_pos,
                                                                  const COLTYPE* d_row_perm,
                                                                  COLTYPE base,
                                                                  VALTYPE* d_lu_av,
                                                                  VALTYPE* d_diag_inv,
                                                                  int* d_status,
                                                                  COLTYPE* d_next_row,
                                                                  int* d_row_done,
                                                                  cudaStream_t stream,
                                                                  ILUPersistentLaunchConfig* h_launch_config )
{
    return launch_persistent_cached_async<ROWTYPE, COLTYPE, VALTYPE, true>(
        n, d_lu_ai, d_lu_aj, d_lu_diag, d_lower_row_ptr, d_update_ptr, d_update_jpos, d_update_pos,
        d_row_perm, base, d_lu_av, d_diag_inv, d_status, d_next_row, d_row_done, stream, h_launch_config );
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

template cudaError_t ILUBaseNumericFactorizationPersistentPermAsync<int, int, float>( int,
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

template cudaError_t ILUBaseNumericFactorizationPersistentPermAsync<int, int, double>( int,
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

template cudaError_t ILUBaseNumericFactorizationPersistentPermAsync<std::int64_t, int, double>(
    int,
    const std::int64_t*,
    const int*,
    const std::int64_t*,
    const int*,
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

template cudaError_t ILUBaseNumericFactorizationPersistentCachedPermAsync<int, int, float>( int,
                                                                                            const int*,
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

template cudaError_t ILUBaseNumericFactorizationPersistentCachedPermAsync<int, int, double>(
    int,
    const int*,
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

template cudaError_t ILUBaseNumericFactorizationPersistentCachedPermAsync<std::int64_t, int, double>(
    int,
    const std::int64_t*,
    const int*,
    const std::int64_t*,
    const std::int64_t*,
    const std::int64_t*,
    const std::int64_t*,
    const std::int64_t*,
    const int*,
    int,
    double*,
    double*,
    int*,
    int*,
    int*,
    cudaStream_t,
    ILUPersistentLaunchConfig* );

} // namespace matrix_utils::sparse_cuda
