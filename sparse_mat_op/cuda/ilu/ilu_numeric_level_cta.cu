#include "ilu_numeric_level_cta.cuh"
#include "ilu_numeric_common.cuh"

#include <cuda/atomic>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace matrix_utils::sparse_cuda
{
namespace
{
using ilu_detail::kMergeReferenceColumnsPerWarp;
using ilu_detail::kSharedRowColumnsPerWarp;
using ilu_detail::kWarpSize;

inline constexpr int kLevelCtaWarpsPerBlock = 8;
inline constexpr int kLevelCtaThreadsPerBlock = kWarpSize * kLevelCtaWarpsPerBlock;
#if defined( __CUDA_ARCH__ ) && __CUDA_ARCH__ >= 800
inline constexpr int kLevelCtaMinBlocksPerSm = 6;
#else
inline constexpr int kLevelCtaMinBlocksPerSm = 4;
#endif

template <typename COLTYPE>
std::size_t LevelCtaSharedFactorRowBytes( const ilu_detail::RowIndexLookup lookup,
                                          const ilu_detail::RowUpdateStrategy update )
{
    std::size_t bytes = 0;
    if ( lookup == ilu_detail::RowIndexLookup::Shared && update == ilu_detail::RowUpdateStrategy::BinarySearch )
    {
        bytes += static_cast<std::size_t>( kLevelCtaWarpsPerBlock ) * kSharedRowColumnsPerWarp * sizeof( COLTYPE );
    }
    if ( update == ilu_detail::RowUpdateStrategy::Merge )
    {
        bytes += static_cast<std::size_t>( kLevelCtaWarpsPerBlock ) *
                 kMergeReferenceColumnsPerWarp * sizeof( COLTYPE );
    }
    return bytes;
}

void initializeLaunchConfig( ILULevelCtaLaunchConfig* config )
{
    if ( config == nullptr )
    {
        return;
    }
    config->warps_per_block = kLevelCtaWarpsPerBlock;
    config->block_size = kLevelCtaThreadsPerBlock;
    config->level_launches = 0;
    config->total_blocks = 0;
    config->hollow_warps = 0;
}

template <typename ROWTYPE, typename COLTYPE>
COLTYPE checkedZeroBasedIndex( const COLTYPE value, const COLTYPE base, const COLTYPE limit, const char* name )
{
    const COLTYPE index = value - base;
    if ( index < 0 || index >= limit )
    {
        throw std::runtime_error( std::string( "invalid " ) + name + " in ILU level CTA schedule" );
    }
    return index;
}

__device__ __forceinline__ int load_device_int( const int* value )
{
    cuda::atomic_ref<int, cuda::thread_scope_device> device_value( *const_cast<int*>( value ) );
    return device_value.load( cuda::memory_order_relaxed );
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
            if ( ( spins++ & 0xff ) == 0 && load_device_int( status ) != 0 )
            {
                success = 0;
                break;
            }
#if __CUDA_ARCH__ >= 700
            if ( spins > 64 )
            {
                __nanosleep( 64 );
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
                                                  VALTYPE* lu_av,
                                                  VALTYPE* diag_inv,
                                                  int* status,
                                                  int* row_done,
                                                  const int lane )
{
    __threadfence();
    __syncwarp();
    if ( lane == 0 && load_device_int( status ) == 0 )
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
__device__ void factor_lu_row_binary_search_wait( const ROWTYPE row_begin,
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
    const ROWTYPE row_len = row_end - row_begin;

    if constexpr ( Lookup == ilu_detail::RowIndexLookup::Shared )
    {
        for ( ROWTYPE offset = lane; offset < row_len; offset += kWarpSize )
        {
            shared_row_cols[offset] = lu_aj[row_begin + offset] - base;
        }
        __syncwarp();
    }

    for ( ROWTYPE k_pos = row_begin; k_pos < lower_end; ++k_pos )
    {
        const COLTYPE k = lu_aj[k_pos] - base;
        if ( !wait_for_row_done<COLTYPE>( k, row_done, status, lane ) )
        {
            return;
        }
        __syncwarp();

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
        if ( !wait_for_row_done<COLTYPE>( k, row_done, status, lane ) )
        {
            return;
        }
        __syncwarp();

        const VALTYPE aik = normalize_lower_entry_with_diag_inv<ROWTYPE, COLTYPE, VALTYPE>(
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
__device__ void factor_level_cta_task( const int cta,
                                       const COLTYPE* cta_row_ptr,
                                       const COLTYPE* cta_rows,
                                       const ROWTYPE* lu_ai,
                                       const COLTYPE* lu_aj,
                                       const ROWTYPE* lu_diag,
                                       const COLTYPE base,
                                       VALTYPE* lu_av,
                                       VALTYPE* diag_inv,
                                       int* status,
                                       int* row_done,
                                       const int warp_in_block,
                                       const int lane,
                                       COLTYPE* shared_row_cols,
                                       COLTYPE* shared_ref_cols )
{
    const COLTYPE cta_begin = cta_row_ptr[cta];
    const COLTYPE cta_end = cta_row_ptr[cta + 1];
    const COLTYPE row_slot = cta_begin + static_cast<COLTYPE>( warp_in_block );
    if ( row_slot >= cta_end || load_device_int( status ) != 0 )
    {
        return;
    }

    const COLTYPE i = cta_rows[row_slot];
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
                factor_lu_row_binary_search_wait<ROWTYPE, COLTYPE, VALTYPE, ilu_detail::RowIndexLookup::Shared>(
                    row_begin, row_end, lower_end, lu_ai, lu_aj, lu_diag, base, lu_av, diag_inv,
                    status, row_done, lane, shared_row_cols );
            }
            else
            {
                factor_lu_row_binary_search_wait<ROWTYPE, COLTYPE, VALTYPE, ilu_detail::RowIndexLookup::Global>(
                    row_begin, row_end, lower_end, lu_ai, lu_aj, lu_diag, base, lu_av, diag_inv,
                    status, row_done, lane );
            }
        }
        else
        {
            factor_lu_row_binary_search_wait<ROWTYPE, COLTYPE, VALTYPE, ilu_detail::RowIndexLookup::Global>(
                row_begin, row_end, lower_end, lu_ai, lu_aj, lu_diag, base, lu_av, diag_inv, status,
                row_done, lane );
        }
    }

    publish_row_done<ROWTYPE, COLTYPE, VALTYPE>( i, lower_end, lu_av, diag_inv, status, row_done, lane );
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, ilu_detail::RowIndexLookup Lookup, ilu_detail::RowUpdateStrategy Update>
__global__ __launch_bounds__( kLevelCtaThreadsPerBlock,
                              kLevelCtaMinBlocksPerSm ) void ilu_level_cta_kernel( int cta_count,
                                                                                   const COLTYPE* cta_row_ptr,
                                                                                   const COLTYPE* cta_rows,
                                                                                   const ROWTYPE* lu_ai,
                                                                                   const COLTYPE* lu_aj,
                                                                                   const ROWTYPE* lu_diag,
                                                                                   COLTYPE base,
                                                                                   VALTYPE* lu_av,
                                                                                   VALTYPE* diag_inv,
                                                                                   int* status,
                                                                                   int* row_done,
                                                                                   int* next_cta )
{
    const int warp_in_block = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x & ( kWarpSize - 1 );
    __shared__ int shared_cta;

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
            shared_offset += kLevelCtaWarpsPerBlock * kSharedRowColumnsPerWarp;
        }
        if constexpr ( Update == ilu_detail::RowUpdateStrategy::Merge )
        {
            shared_ref_cols = shared_cols + shared_offset + warp_in_block * kMergeReferenceColumnsPerWarp;
        }
    }

    while ( true )
    {
        if ( threadIdx.x == 0 )
        {
            shared_cta = ( load_device_int( status ) == 0 ) ? atomicAdd( next_cta, 1 ) : cta_count;
        }
        __syncthreads();

        const int cta = shared_cta;
        if ( cta >= cta_count )
        {
            return;
        }

        factor_level_cta_task<ROWTYPE, COLTYPE, VALTYPE, Lookup, Update>(
            cta, cta_row_ptr, cta_rows, lu_ai, lu_aj, lu_diag, base, lu_av, diag_inv, status,
            row_done, warp_in_block, lane, shared_row_cols, shared_ref_cols );
        __syncthreads();
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, ilu_detail::RowIndexLookup Lookup, ilu_detail::RowUpdateStrategy Update>
cudaError_t select_level_cta_launch_config( const int cta_count,
                                            const std::size_t shared_bytes,
                                            ILULevelCtaLaunchConfig* config )
{
    if ( cta_count <= 0 || config == nullptr )
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

    int blocks_per_sm = 0;
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, ilu_level_cta_kernel<ROWTYPE, COLTYPE, VALTYPE, Lookup, Update>,
        kLevelCtaThreadsPerBlock, shared_bytes );
    if ( status != cudaSuccess )
    {
        return status;
    }
    if ( blocks_per_sm <= 0 )
    {
        return cudaErrorInvalidValue;
    }

    const int occupancy_blocks = std::max( prop.multiProcessorCount, 1 ) * blocks_per_sm;
    config->warps_per_block = kLevelCtaWarpsPerBlock;
    config->block_size = kLevelCtaThreadsPerBlock;
    config->level_launches = 1;
    config->total_blocks = std::max( 1, std::min( cta_count, occupancy_blocks ) );
    return cudaSuccess;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, ilu_detail::RowIndexLookup Lookup, ilu_detail::RowUpdateStrategy Update>
cudaError_t launch_level_cta_kernel( const DeviceILULevelCtaSchedule<ROWTYPE, COLTYPE>& schedule,
                                     const ROWTYPE* lu_ai,
                                     const COLTYPE* lu_aj,
                                     const ROWTYPE* lu_diag,
                                     const COLTYPE base,
                                     VALTYPE* lu_av,
                                     VALTYPE* diag_inv,
                                     int* status,
                                     ILULevelCtaScratch& scratch,
                                     cudaStream_t stream,
                                     ILULevelCtaLaunchConfig* h_launch_config )
{
    const auto shared_bytes = LevelCtaSharedFactorRowBytes<COLTYPE>( Lookup, Update );
    ILULevelCtaLaunchConfig config;
    cudaError_t launch_status = select_level_cta_launch_config<ROWTYPE, COLTYPE, VALTYPE, Lookup, Update>(
        schedule.cta_count, shared_bytes, &config );
    if ( launch_status != cudaSuccess )
    {
        return launch_status;
    }
    config.hollow_warps = schedule.hollow_warps;

    ilu_level_cta_kernel<ROWTYPE, COLTYPE, VALTYPE, Lookup, Update>
        <<<config.total_blocks, kLevelCtaThreadsPerBlock, shared_bytes, stream>>>(
            schedule.cta_count, schedule.cta_row_ptr.data(), schedule.cta_rows.data(), lu_ai, lu_aj,
            lu_diag, base, lu_av, diag_inv, status, scratch.row_done.data(), scratch.next_cta.data() );
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

template <typename ROWTYPE, typename COLTYPE>
ILULevelCtaSchedule<ROWTYPE, COLTYPE> BuildILULevelCtaSchedule( COLTYPE n,
                                                                const ROWTYPE* lu_ai,
                                                                const COLTYPE* lu_aj,
                                                                const ROWTYPE* lu_diag,
                                                                const COLTYPE* level_perm,
                                                                const COLTYPE* level_prefix,
                                                                COLTYPE levels,
                                                                COLTYPE base,
                                                                int rows_per_cta )
{
    if ( n <= 0 || levels < 0 || rows_per_cta <= 0 || lu_ai == nullptr || lu_aj == nullptr ||
         lu_diag == nullptr || level_perm == nullptr || level_prefix == nullptr )
    {
        throw std::runtime_error( "invalid input to BuildILULevelCtaSchedule" );
    }

    ILULevelCtaSchedule<ROWTYPE, COLTYPE> schedule;
    schedule.rows_per_cta = rows_per_cta;
    schedule.row_count = n;
    schedule.level_count = levels;
    schedule.row_to_cta.assign( static_cast<std::size_t>( n ), -1 );
    schedule.cta_row_ptr.push_back( 0 );

    int rows_in_current_cta = 0;
    for ( COLTYPE level = 0; level < levels; ++level )
    {
        const COLTYPE level_begin = level_prefix[level] - base;
        const COLTYPE level_end = level_prefix[level + 1] - base;
        if ( level_begin < 0 || level_end < level_begin || level_end > n )
        {
            throw std::runtime_error( "invalid level prefix in ILU level CTA schedule" );
        }

        for ( COLTYPE pos = level_begin; pos < level_end; ++pos )
        {
            if ( rows_in_current_cta == 0 )
            {
                ++schedule.cta_count;
                schedule.cta_level.push_back( static_cast<int>( level ) );
            }

            schedule.cta_level.back() = static_cast<int>( level );
            const int cta = schedule.cta_count - 1;
            const COLTYPE row =
                checkedZeroBasedIndex<ROWTYPE, COLTYPE>( level_perm[pos], base, n, "level row" );
            schedule.cta_rows.push_back( row );
            schedule.row_to_cta[static_cast<std::size_t>( row )] = cta;

            ++rows_in_current_cta;
            if ( rows_in_current_cta == rows_per_cta )
            {
                schedule.cta_row_ptr.push_back( static_cast<COLTYPE>( schedule.cta_rows.size() ) );
                rows_in_current_cta = 0;
            }
        }
    }
    if ( rows_in_current_cta != 0 )
    {
        schedule.hollow_warps += rows_per_cta - rows_in_current_cta;
        schedule.cta_row_ptr.push_back( static_cast<COLTYPE>( schedule.cta_rows.size() ) );
    }

    if ( static_cast<COLTYPE>( schedule.cta_rows.size() ) != n )
    {
        throw std::runtime_error( "level schedule does not cover all rows" );
    }
    for ( COLTYPE row = 0; row < n; ++row )
    {
        if ( schedule.row_to_cta[static_cast<std::size_t>( row )] < 0 )
        {
            throw std::runtime_error( "row missing from ILU level CTA schedule" );
        }
    }

    std::vector<int> succ_counts( static_cast<std::size_t>( schedule.cta_count ), 0 );
    schedule.cta_pred_ptr.reserve( static_cast<std::size_t>( schedule.cta_count ) + 1 );
    schedule.cta_pred_ptr.push_back( 0 );

    std::vector<int> cta_predecessors;
    for ( int dst_cta = 0; dst_cta < schedule.cta_count; ++dst_cta )
    {
        const COLTYPE cta_row_begin = schedule.cta_row_ptr[static_cast<std::size_t>( dst_cta )];
        const COLTYPE cta_row_end = schedule.cta_row_ptr[static_cast<std::size_t>( dst_cta + 1 )];
        cta_predecessors.clear();
        for ( COLTYPE row_slot = cta_row_begin; row_slot < cta_row_end; ++row_slot )
        {
            const COLTYPE row = schedule.cta_rows[static_cast<std::size_t>( row_slot )];
            const ROWTYPE row_begin = lu_ai[row] - base;
            const ROWTYPE lower_end = lu_diag[row] - base;
            if ( row_begin < 0 || lower_end < row_begin || lower_end > lu_ai[row + 1] - base )
            {
                throw std::runtime_error( "invalid LU row bounds in ILU level CTA schedule" );
            }

            for ( ROWTYPE pos = row_begin; pos < lower_end; ++pos )
            {
                const COLTYPE dep_row =
                    checkedZeroBasedIndex<ROWTYPE, COLTYPE>( lu_aj[pos], base, n, "dependency row" );
                const int src_cta = schedule.row_to_cta[static_cast<std::size_t>( dep_row )];
                if ( src_cta == dst_cta )
                {
                    continue;
                }
                if ( src_cta > dst_cta )
                {
                    throw std::runtime_error( "CTA edge violates topological schedule order" );
                }
                cta_predecessors.push_back( src_cta );
            }
        }

        std::sort( cta_predecessors.begin(), cta_predecessors.end() );
        cta_predecessors.erase( std::unique( cta_predecessors.begin(), cta_predecessors.end() ),
                                cta_predecessors.end() );
        for ( const int src_cta : cta_predecessors )
        {
            schedule.cta_preds.push_back( src_cta );
            ++succ_counts[static_cast<std::size_t>( src_cta )];
        }
        schedule.cta_pred_ptr.push_back( static_cast<int>( schedule.cta_preds.size() ) );
    }

    schedule.cta_edge_count = static_cast<int>( schedule.cta_preds.size() );
    schedule.cta_indegree.resize( static_cast<std::size_t>( schedule.cta_count ) );
    schedule.cta_succ_ptr.resize( static_cast<std::size_t>( schedule.cta_count ) + 1 );
    for ( int cta = 0; cta < schedule.cta_count; ++cta )
    {
        schedule.cta_indegree[static_cast<std::size_t>( cta )] =
            schedule.cta_pred_ptr[static_cast<std::size_t>( cta + 1 )] -
            schedule.cta_pred_ptr[static_cast<std::size_t>( cta )];
        schedule.cta_succ_ptr[static_cast<std::size_t>( cta + 1 )] =
            schedule.cta_succ_ptr[static_cast<std::size_t>( cta )] +
            succ_counts[static_cast<std::size_t>( cta )];
    }

    schedule.cta_succs.resize( static_cast<std::size_t>( schedule.cta_edge_count ) );
    for ( int cta = 0; cta < schedule.cta_count; ++cta )
    {
        if ( schedule.cta_indegree[static_cast<std::size_t>( cta )] == 0 )
        {
            schedule.initial_ready_ctas.push_back( cta );
        }
    }

    std::vector<int> succ_offsets = schedule.cta_succ_ptr;
    for ( int dst_cta = 0; dst_cta < schedule.cta_count; ++dst_cta )
    {
        const int pred_begin = schedule.cta_pred_ptr[static_cast<std::size_t>( dst_cta )];
        const int pred_end = schedule.cta_pred_ptr[static_cast<std::size_t>( dst_cta + 1 )];
        for ( int pos = pred_begin; pos < pred_end; ++pos )
        {
            const int src_cta = schedule.cta_preds[static_cast<std::size_t>( pos )];
            schedule.cta_succs[static_cast<std::size_t>( succ_offsets[static_cast<std::size_t>( src_cta )]++ )] =
                dst_cta;
        }
    }

    return schedule;
}

template <typename ROWTYPE, typename COLTYPE>
cudaError_t UploadILULevelCtaSchedule( const ILULevelCtaSchedule<ROWTYPE, COLTYPE>& schedule,
                                       DeviceILULevelCtaSchedule<ROWTYPE, COLTYPE>& device_schedule )
{
    device_schedule.rows_per_cta = schedule.rows_per_cta;
    device_schedule.row_count = schedule.row_count;
    device_schedule.level_count = schedule.level_count;
    device_schedule.cta_count = schedule.cta_count;
    device_schedule.cta_edge_count = schedule.cta_edge_count;
    device_schedule.hollow_warps = schedule.hollow_warps;

    device_schedule.cta_row_ptr.copyFromHost( schedule.cta_row_ptr.data(), schedule.cta_row_ptr.size() );
    device_schedule.cta_rows.copyFromHost( schedule.cta_rows.data(), schedule.cta_rows.size() );
    device_schedule.cta_level.copyFromHost( schedule.cta_level.data(), schedule.cta_level.size() );
    device_schedule.row_to_cta.copyFromHost( schedule.row_to_cta.data(), schedule.row_to_cta.size() );
    device_schedule.cta_pred_ptr.copyFromHost( schedule.cta_pred_ptr.data(), schedule.cta_pred_ptr.size() );
    device_schedule.cta_preds.copyFromHost( schedule.cta_preds.data(), schedule.cta_preds.size() );
    device_schedule.cta_succ_ptr.copyFromHost( schedule.cta_succ_ptr.data(), schedule.cta_succ_ptr.size() );
    device_schedule.cta_succs.copyFromHost( schedule.cta_succs.data(), schedule.cta_succs.size() );
    device_schedule.cta_indegree.copyFromHost( schedule.cta_indegree.data(), schedule.cta_indegree.size() );
    device_schedule.initial_ready_ctas.copyFromHost( schedule.initial_ready_ctas.data(),
                                                     schedule.initial_ready_ctas.size() );
    return cudaSuccess;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
cudaError_t ILUBaseNumericFactorizationLevelCtaAsync( const DeviceILULevelCtaSchedule<ROWTYPE, COLTYPE>& schedule,
                                                      const ROWTYPE* d_lu_ai,
                                                      const COLTYPE* d_lu_aj,
                                                      const ROWTYPE* d_lu_diag,
                                                      COLTYPE base,
                                                      VALTYPE* d_lu_av,
                                                      VALTYPE* d_diag_inv,
                                                      int* d_status,
                                                      ILUNumericRowLookup row_lookup,
                                                      ILUNumericRowUpdateStrategy row_update,
                                                      ILULevelCtaScratch& scratch,
                                                      cudaStream_t stream,
                                                      ILULevelCtaLaunchConfig* h_launch_config )
{
    if ( schedule.cta_count <= 0 || schedule.row_count <= 0 || d_lu_ai == nullptr || d_lu_aj == nullptr ||
         d_lu_diag == nullptr || d_lu_av == nullptr || d_diag_inv == nullptr || d_status == nullptr ||
         schedule.cta_row_ptr.data() == nullptr || schedule.cta_rows.data() == nullptr )
    {
        return cudaErrorInvalidValue;
    }

    initializeLaunchConfig( h_launch_config );
    scratch.row_done.resize( static_cast<std::size_t>( schedule.row_count ) );
    scratch.next_cta.resize( 1 );

    cudaError_t status = cudaMemsetAsync( d_status, 0, sizeof( int ), stream );
    if ( status != cudaSuccess )
    {
        return status;
    }
    status = cudaMemsetAsync( scratch.row_done.data(), 0,
                              static_cast<std::size_t>( schedule.row_count ) * sizeof( int ), stream );
    if ( status != cudaSuccess )
    {
        return status;
    }
    status = cudaMemsetAsync( scratch.next_cta.data(), 0, sizeof( int ), stream );
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
            status =
                launch_level_cta_kernel<ROWTYPE, COLTYPE, VALTYPE, ilu_detail::RowIndexLookup::Global, ilu_detail::RowUpdateStrategy::BinarySearch>(
                    schedule, d_lu_ai, d_lu_aj, d_lu_diag, base, d_lu_av, d_diag_inv, d_status,
                    scratch, stream, h_launch_config );
            break;
        case ILUNumericRowUpdateStrategy::Merge:
            status =
                launch_level_cta_kernel<ROWTYPE, COLTYPE, VALTYPE, ilu_detail::RowIndexLookup::Global, ilu_detail::RowUpdateStrategy::Merge>(
                    schedule, d_lu_ai, d_lu_aj, d_lu_diag, base, d_lu_av, d_diag_inv, d_status,
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
            status =
                launch_level_cta_kernel<ROWTYPE, COLTYPE, VALTYPE, ilu_detail::RowIndexLookup::Shared, ilu_detail::RowUpdateStrategy::BinarySearch>(
                    schedule, d_lu_ai, d_lu_aj, d_lu_diag, base, d_lu_av, d_diag_inv, d_status,
                    scratch, stream, h_launch_config );
            break;
        case ILUNumericRowUpdateStrategy::Merge:
            status =
                launch_level_cta_kernel<ROWTYPE, COLTYPE, VALTYPE, ilu_detail::RowIndexLookup::Shared, ilu_detail::RowUpdateStrategy::Merge>(
                    schedule, d_lu_ai, d_lu_aj, d_lu_diag, base, d_lu_av, d_diag_inv, d_status,
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

template cudaError_t ILUBaseNumericFactorizationLevelCtaAsync<int, int, float>(
    const DeviceILULevelCtaSchedule<int, int>&,
    const int*,
    const int*,
    const int*,
    int,
    float*,
    float*,
    int*,
    ILUNumericRowLookup,
    ILUNumericRowUpdateStrategy,
    ILULevelCtaScratch&,
    cudaStream_t,
    ILULevelCtaLaunchConfig* );

template cudaError_t ILUBaseNumericFactorizationLevelCtaAsync<int, int, double>(
    const DeviceILULevelCtaSchedule<int, int>&,
    const int*,
    const int*,
    const int*,
    int,
    double*,
    double*,
    int*,
    ILUNumericRowLookup,
    ILUNumericRowUpdateStrategy,
    ILULevelCtaScratch&,
    cudaStream_t,
    ILULevelCtaLaunchConfig* );

template cudaError_t ILUBaseNumericFactorizationLevelCtaAsync<std::int64_t, int, double>(
    const DeviceILULevelCtaSchedule<std::int64_t, int>&,
    const std::int64_t*,
    const int*,
    const std::int64_t*,
    int,
    double*,
    double*,
    int*,
    ILUNumericRowLookup,
    ILUNumericRowUpdateStrategy,
    ILULevelCtaScratch&,
    cudaStream_t,
    ILULevelCtaLaunchConfig* );

template ILULevelCtaSchedule<int, int> BuildILULevelCtaSchedule<int, int>( int,
                                                                           const int*,
                                                                           const int*,
                                                                           const int*,
                                                                           const int*,
                                                                           const int*,
                                                                           int,
                                                                           int,
                                                                           int );

template ILULevelCtaSchedule<std::int64_t, int> BuildILULevelCtaSchedule<std::int64_t, int>( int,
                                                                                             const std::int64_t*,
                                                                                             const int*,
                                                                                             const std::int64_t*,
                                                                                             const int*,
                                                                                             const int*,
                                                                                             int,
                                                                                             int,
                                                                                             int );

template cudaError_t UploadILULevelCtaSchedule<int, int>( const ILULevelCtaSchedule<int, int>&,
                                                          DeviceILULevelCtaSchedule<int, int>& );

template cudaError_t UploadILULevelCtaSchedule<std::int64_t, int>( const ILULevelCtaSchedule<std::int64_t, int>&,
                                                                   DeviceILULevelCtaSchedule<std::int64_t, int>& );

} // namespace matrix_utils::sparse_cuda
