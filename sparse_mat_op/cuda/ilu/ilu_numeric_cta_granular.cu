#include "ilu_numeric_cta_granular.cuh"
#include "ilu_numeric_common.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace matrix_utils::sparse_cuda
{
namespace
{
using ilu_detail::kMergeReferenceColumnsPerWarp;
using ilu_detail::kSharedRowColumnsPerWarp;
using ilu_detail::kWarpSize;

inline constexpr int kCtaGranularWarpsPerBlock = 8;
inline constexpr int kCtaGranularThreadsPerBlock = kWarpSize * kCtaGranularWarpsPerBlock;
#if defined( __CUDA_ARCH__ ) && __CUDA_ARCH__ >= 800
inline constexpr int kCtaGranularMinBlocksPerSm = 6;
#else
inline constexpr int kCtaGranularMinBlocksPerSm = 4;
#endif

template <typename COLTYPE>
std::size_t CtaGranularSharedFactorRowBytes( const ilu_detail::RowIndexLookup lookup,
                                             const ilu_detail::RowUpdateStrategy update )
{
    std::size_t bytes = 0;
    if ( lookup == ilu_detail::RowIndexLookup::Shared && update == ilu_detail::RowUpdateStrategy::BinarySearch )
    {
        bytes += static_cast<std::size_t>( kCtaGranularWarpsPerBlock ) * kSharedRowColumnsPerWarp *
                 sizeof( COLTYPE );
    }
    if ( update == ilu_detail::RowUpdateStrategy::Merge )
    {
        bytes += static_cast<std::size_t>( kCtaGranularWarpsPerBlock ) *
                 kMergeReferenceColumnsPerWarp * sizeof( COLTYPE );
    }
    return bytes;
}

void initializeLaunchConfig( ILUCtaGranularLaunchConfig* config )
{
    if ( config == nullptr )
    {
        return;
    }
    config->warps_per_block = kCtaGranularWarpsPerBlock;
    config->block_size = kCtaGranularThreadsPerBlock;
    config->kernel_launches = 0;
    config->total_blocks = 0;
    config->hollow_warps = 0;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, ilu_detail::RowIndexLookup Lookup>
__device__ void factor_lu_row_merge_wait( const ROWTYPE row_begin,
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
                                          COLTYPE* shared_ref_cols )
{
    for ( ROWTYPE k_pos = row_begin; k_pos < lower_end; ++k_pos )
    {
        const COLTYPE k = lu_aj[k_pos] - base;
        if ( !ilu_detail::WaitForRowDone<COLTYPE>( k, row_done, status, lane ) )
        {
            return;
        }
        __syncwarp();

        const VALTYPE aik = ilu_detail::NormalizeLowerEntryWithDiagInv<ROWTYPE, COLTYPE, VALTYPE>(
            k_pos, k, lu_av, diag_inv, lane );
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

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, ilu_detail::RowIndexLookup Lookup, ilu_detail::RowUpdateStrategy Update>
__device__ void factor_cta_granular_row_slot( const COLTYPE row_slot,
                                              const COLTYPE n,
                                              const COLTYPE* row_perm,
                                              const ROWTYPE* lu_ai,
                                              const COLTYPE* lu_aj,
                                              const ROWTYPE* lu_diag,
                                              const COLTYPE base,
                                              VALTYPE* lu_av,
                                              VALTYPE* diag_inv,
                                              int* status,
                                              int* row_done,
                                              const int lane,
                                              COLTYPE* shared_row_cols,
                                              COLTYPE* shared_ref_cols )
{
    if ( row_slot >= n || ilu_detail::LoadDeviceInt( status ) != 0 )
    {
        return;
    }

    const COLTYPE i = row_perm[row_slot] - base;
    const ROWTYPE row_begin = lu_ai[i] - base;
    const ROWTYPE row_end = lu_ai[i + 1] - base;
    const ROWTYPE lower_end = lu_diag[i] - base;

    if ( row_begin < lower_end )
    {
        if constexpr ( Update == ilu_detail::RowUpdateStrategy::Merge )
        {
            factor_lu_row_merge_wait<ROWTYPE, COLTYPE, VALTYPE, Lookup>(
                row_begin, row_end, lower_end, lu_ai, lu_aj, lu_diag, base, lu_av, diag_inv, status,
                row_done, lane, shared_ref_cols );
        }
        else if constexpr ( Lookup == ilu_detail::RowIndexLookup::Shared )
        {
            if ( row_end - row_begin <= static_cast<ROWTYPE>( kSharedRowColumnsPerWarp ) )
            {
                ilu_detail::FactorLURowBinarySearchWithRowDone<ROWTYPE, COLTYPE, VALTYPE, ilu_detail::RowIndexLookup::Shared>(
                    row_begin, row_end, lower_end, lu_ai, lu_aj, lu_diag, base, lu_av, diag_inv,
                    status, row_done, lane, shared_row_cols );
            }
            else
            {
                ilu_detail::FactorLURowBinarySearchWithRowDone<ROWTYPE, COLTYPE, VALTYPE, ilu_detail::RowIndexLookup::Global>(
                    row_begin, row_end, lower_end, lu_ai, lu_aj, lu_diag, base, lu_av, diag_inv,
                    status, row_done, lane );
            }
        }
        else
        {
            ilu_detail::FactorLURowBinarySearchWithRowDone<ROWTYPE, COLTYPE, VALTYPE, ilu_detail::RowIndexLookup::Global>(
                row_begin, row_end, lower_end, lu_ai, lu_aj, lu_diag, base, lu_av, diag_inv, status,
                row_done, lane );
        }
    }

    ilu_detail::PublishRowDone<ROWTYPE, COLTYPE, VALTYPE>( i, lower_end, true, lu_av, diag_inv,
                                                           status, row_done, lane );
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, ilu_detail::RowIndexLookup Lookup, ilu_detail::RowUpdateStrategy Update>
__global__ __launch_bounds__( kCtaGranularThreadsPerBlock, kCtaGranularMinBlocksPerSm ) void ilu_cta_granular_kernel(
    COLTYPE n,
    const COLTYPE* row_perm,
    const ROWTYPE* lu_ai,
    const COLTYPE* lu_aj,
    const ROWTYPE* lu_diag,
    COLTYPE base,
    VALTYPE* lu_av,
    VALTYPE* diag_inv,
    int* status,
    int* row_done,
    int* next_row )
{
    const int warp_in_block = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x & ( kWarpSize - 1 );
    __shared__ int shared_row_begin;

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
            shared_row_cols = shared_cols + shared_offset + warp_in_block * kSharedRowColumnsPerWarp;
            shared_offset += kCtaGranularWarpsPerBlock * kSharedRowColumnsPerWarp;
        }
        if constexpr ( Update == ilu_detail::RowUpdateStrategy::Merge )
        {
            shared_ref_cols = shared_cols + shared_offset + warp_in_block * kMergeReferenceColumnsPerWarp;
        }
    }

    if ( threadIdx.x == 0 )
    {
        shared_row_begin = ( ilu_detail::LoadDeviceInt( status ) == 0 )
                               ? atomicAdd( next_row, kCtaGranularWarpsPerBlock )
                               : static_cast<int>( n );
    }
    __syncthreads();

    const COLTYPE row_slot = static_cast<COLTYPE>( shared_row_begin + warp_in_block );
    if ( row_slot >= n )
    {
        return;
    }

    factor_cta_granular_row_slot<ROWTYPE, COLTYPE, VALTYPE, Lookup, Update>(
        row_slot, n, row_perm, lu_ai, lu_aj, lu_diag, base, lu_av, diag_inv, status, row_done, lane,
        shared_row_cols, shared_ref_cols );
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, ilu_detail::RowIndexLookup Lookup, ilu_detail::RowUpdateStrategy Update>
cudaError_t select_cta_granular_launch_config( const COLTYPE n,
                                               const std::size_t shared_bytes,
                                               ILUCtaGranularLaunchConfig* config )
{
    if ( n <= 0 || config == nullptr )
    {
        return cudaErrorInvalidValue;
    }

    int blocks_per_sm = 0;
    cudaError_t status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, ilu_cta_granular_kernel<ROWTYPE, COLTYPE, VALTYPE, Lookup, Update>,
        kCtaGranularThreadsPerBlock, shared_bytes );
    if ( status != cudaSuccess )
    {
        return status;
    }
    if ( blocks_per_sm <= 0 )
    {
        return cudaErrorInvalidValue;
    }

    config->warps_per_block = kCtaGranularWarpsPerBlock;
    config->block_size = kCtaGranularThreadsPerBlock;
    config->kernel_launches = 1;
    config->total_blocks = static_cast<int>( ( n + kCtaGranularWarpsPerBlock - 1 ) / kCtaGranularWarpsPerBlock );
    config->hollow_warps = config->total_blocks * kCtaGranularWarpsPerBlock - n;
    return cudaSuccess;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, ilu_detail::RowIndexLookup Lookup, ilu_detail::RowUpdateStrategy Update>
cudaError_t launch_cta_granular_kernel( const COLTYPE n,
                                        const ROWTYPE* lu_ai,
                                        const COLTYPE* lu_aj,
                                        const ROWTYPE* lu_diag,
                                        const COLTYPE* row_perm,
                                        const COLTYPE base,
                                        VALTYPE* lu_av,
                                        VALTYPE* diag_inv,
                                        int* status,
                                        ILUCtaGranularScratch& scratch,
                                        cudaStream_t stream,
                                        ILUCtaGranularLaunchConfig* h_launch_config )
{
    const auto shared_bytes = CtaGranularSharedFactorRowBytes<COLTYPE>( Lookup, Update );
    ILUCtaGranularLaunchConfig config;
    cudaError_t launch_status =
        select_cta_granular_launch_config<ROWTYPE, COLTYPE, VALTYPE, Lookup, Update>( n, shared_bytes, &config );
    if ( launch_status != cudaSuccess )
    {
        return launch_status;
    }

    ilu_cta_granular_kernel<ROWTYPE, COLTYPE, VALTYPE, Lookup, Update>
        <<<config.total_blocks, kCtaGranularThreadsPerBlock, shared_bytes, stream>>>(
            n, row_perm, lu_ai, lu_aj, lu_diag, base, lu_av, diag_inv, status,
            scratch.row_done.data(), scratch.next_row.data() );
    launch_status = ilu_detail::CudaLaunchStatus();
    if ( launch_status != cudaSuccess )
    {
        return launch_status;
    }

    if ( h_launch_config != nullptr )
    {
        *h_launch_config = config;
    }
    return cudaSuccess;
}

} // namespace

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
cudaError_t ILUBaseNumericFactorizationCtaGranularAsync( COLTYPE n,
                                                         const ROWTYPE* d_lu_ai,
                                                         const COLTYPE* d_lu_aj,
                                                         const ROWTYPE* d_lu_diag,
                                                         const COLTYPE* d_row_perm,
                                                         COLTYPE base,
                                                         VALTYPE* d_lu_av,
                                                         VALTYPE* d_diag_inv,
                                                         int* d_status,
                                                         ILUNumericRowLookup row_lookup,
                                                         ILUNumericRowUpdateStrategy row_update,
                                                         ILUCtaGranularScratch& scratch,
                                                         cudaStream_t stream,
                                                         ILUCtaGranularLaunchConfig* h_launch_config )
{
    if ( n <= 0 || d_lu_ai == nullptr || d_lu_aj == nullptr || d_lu_diag == nullptr ||
         d_row_perm == nullptr || d_lu_av == nullptr || d_diag_inv == nullptr || d_status == nullptr )
    {
        return cudaErrorInvalidValue;
    }

    initializeLaunchConfig( h_launch_config );
    scratch.row_done.resize( static_cast<std::size_t>( n ) );
    scratch.next_row.resize( 1 );

    cudaError_t status = cudaMemsetAsync( d_status, 0, sizeof( int ), stream );
    if ( status != cudaSuccess )
    {
        return status;
    }
    status = cudaMemsetAsync( scratch.row_done.data(), 0, static_cast<std::size_t>( n ) * sizeof( int ), stream );
    if ( status != cudaSuccess )
    {
        return status;
    }
    status = cudaMemsetAsync( scratch.next_row.data(), 0, sizeof( int ), stream );
    if ( status != cudaSuccess )
    {
        return status;
    }
    switch ( row_lookup )
    {
    case ILUNumericRowLookup::Global:
        switch ( row_update )
        {
        case ILUNumericRowUpdateStrategy::BinarySearch:
            status = launch_cta_granular_kernel<ROWTYPE, COLTYPE, VALTYPE, ilu_detail::RowIndexLookup::Global,
                                                ilu_detail::RowUpdateStrategy::BinarySearch>(
                n, d_lu_ai, d_lu_aj, d_lu_diag, d_row_perm, base, d_lu_av, d_diag_inv, d_status,
                scratch, stream, h_launch_config );
            break;
        case ILUNumericRowUpdateStrategy::Merge:
            status =
                launch_cta_granular_kernel<ROWTYPE, COLTYPE, VALTYPE, ilu_detail::RowIndexLookup::Global, ilu_detail::RowUpdateStrategy::Merge>(
                    n, d_lu_ai, d_lu_aj, d_lu_diag, d_row_perm, base, d_lu_av, d_diag_inv, d_status,
                    scratch, stream, h_launch_config );
            break;
        default:
            return cudaErrorInvalidValue;
        }
        break;
    case ILUNumericRowLookup::Shared:
        switch ( row_update )
        {
        case ILUNumericRowUpdateStrategy::BinarySearch:
            status = launch_cta_granular_kernel<ROWTYPE, COLTYPE, VALTYPE, ilu_detail::RowIndexLookup::Shared,
                                                ilu_detail::RowUpdateStrategy::BinarySearch>(
                n, d_lu_ai, d_lu_aj, d_lu_diag, d_row_perm, base, d_lu_av, d_diag_inv, d_status,
                scratch, stream, h_launch_config );
            break;
        case ILUNumericRowUpdateStrategy::Merge:
            status =
                launch_cta_granular_kernel<ROWTYPE, COLTYPE, VALTYPE, ilu_detail::RowIndexLookup::Shared, ilu_detail::RowUpdateStrategy::Merge>(
                    n, d_lu_ai, d_lu_aj, d_lu_diag, d_row_perm, base, d_lu_av, d_diag_inv, d_status,
                    scratch, stream, h_launch_config );
            break;
        default:
            return cudaErrorInvalidValue;
        }
        break;
    default:
        return cudaErrorInvalidValue;
    }

    return status;
}

template cudaError_t ILUBaseNumericFactorizationCtaGranularAsync<int, int, float>( int,
                                                                                   const int*,
                                                                                   const int*,
                                                                                   const int*,
                                                                                   const int*,
                                                                                   int,
                                                                                   float*,
                                                                                   float*,
                                                                                   int*,
                                                                                   ILUNumericRowLookup,
                                                                                   ILUNumericRowUpdateStrategy,
                                                                                   ILUCtaGranularScratch&,
                                                                                   cudaStream_t,
                                                                                   ILUCtaGranularLaunchConfig* );

template cudaError_t ILUBaseNumericFactorizationCtaGranularAsync<int, int, double>( int,
                                                                                    const int*,
                                                                                    const int*,
                                                                                    const int*,
                                                                                    const int*,
                                                                                    int,
                                                                                    double*,
                                                                                    double*,
                                                                                    int*,
                                                                                    ILUNumericRowLookup,
                                                                                    ILUNumericRowUpdateStrategy,
                                                                                    ILUCtaGranularScratch&,
                                                                                    cudaStream_t,
                                                                                    ILUCtaGranularLaunchConfig* );

template cudaError_t ILUBaseNumericFactorizationCtaGranularAsync<std::int64_t, int, double>(
    int,
    const std::int64_t*,
    const int*,
    const std::int64_t*,
    const int*,
    int,
    double*,
    double*,
    int*,
    ILUNumericRowLookup,
    ILUNumericRowUpdateStrategy,
    ILUCtaGranularScratch&,
    cudaStream_t,
    ILUCtaGranularLaunchConfig* );

} // namespace matrix_utils::sparse_cuda
