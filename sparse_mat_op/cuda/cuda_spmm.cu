#include "cuda_spmm.cuh"
#include <algorithm>
#include <cub/cub.cuh>
#include <cuda/std/utility>
#include <cuda_runtime.h>
#include <limits>
#include <thrust/functional.h>

namespace matrix_utils::sparse_cuda
{
// ============================================================================
// STEP 1: Compute workload prefix sum and memory requirements
// ============================================================================

// Kernel: Compute workload for each row (nnz_A[i] * nnz_B[i])
template <typename ROWTYPE, typename COLTYPE>
__global__ void compute_workload_kernel( const ROWTYPE* d_ai_A,
                                         const ROWTYPE* d_ai_B,
                                         COLTYPE n_rows,
                                         ROWTYPE base,
                                         ROWTYPE* d_workloads ) // Output: workload per row
{
    COLTYPE row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( row_idx >= n_rows )
        return;

    ROWTYPE row_start_A = d_ai_A[row_idx] - base;
    ROWTYPE row_end_A = d_ai_A[row_idx + 1] - base;
    ROWTYPE row_start_B = d_ai_B[row_idx] - base;
    ROWTYPE row_end_B = d_ai_B[row_idx + 1] - base;

    ROWTYPE nnz_A = row_end_A - row_start_A;
    ROWTYPE nnz_B = row_end_B - row_start_B;

    d_workloads[row_idx] = nnz_A * nnz_B;
}

// Host function: Step 1 - Compute prefix sum and return memory requirements
template <typename ROWTYPE, typename COLTYPE>
bool SpMMAnalyze( COLTYPE n_rows, const ROWTYPE* d_ai_A, const ROWTYPE* d_ai_B, ROWTYPE base, ROWTYPE* d_workload_prefix, ROWTYPE* required_array_size )
{
    if ( n_rows <= 0 || d_workload_prefix == nullptr || required_array_size == nullptr )
        return false;

    // Allocate workload array
    ROWTYPE* d_workloads;
    cudaMalloc( &d_workloads, ( n_rows + 1 ) * sizeof( ROWTYPE ) );
    cudaMemset( d_workloads, 0, ( n_rows + 1 ) * sizeof( ROWTYPE ) );

    // Launch kernel to compute workloads
    int threads = 256;
    int blocks = ( n_rows + threads - 1 ) / threads;
    compute_workload_kernel<<<blocks, threads>>>( d_ai_A, d_ai_B, n_rows, base, d_workloads );

    // Compute prefix sum using CUB
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceScan::ExclusiveScan( d_temp_storage, temp_storage_bytes, d_workloads, d_workload_prefix,
                                    ::cuda::std::plus<ROWTYPE>(), base, n_rows + 1 );

    cudaMalloc( &d_temp_storage, temp_storage_bytes );
    cub::DeviceScan::ExclusiveScan( d_temp_storage, temp_storage_bytes, d_workloads, d_workload_prefix,
                                    ::cuda::std::plus<ROWTYPE>(), base, n_rows + 1 );
    ROWTYPE last_prefix = 0;
    cudaMemcpy( &last_prefix, d_workload_prefix + n_rows, sizeof( ROWTYPE ), cudaMemcpyDeviceToHost );
    *required_array_size = last_prefix - base;

    // Cleanup temporary storage
    cudaFree( d_workloads );
    cudaFree( d_temp_storage );

    return true;
}

// ============================================================================
// STEP 2: Build COO sparsity structure from outer products
// ============================================================================

template <typename ROWTYPE, typename COLTYPE>
__global__ void build_outer_product_pairs( const ROWTYPE* d_ai_A,
                                           const COLTYPE* d_aj_A,
                                           const ROWTYPE* d_ai_B,
                                           const COLTYPE* d_aj_B,
                                           COLTYPE n,
                                           ROWTYPE base,
                                           const ROWTYPE* d_workload_prefix,
                                           ROWTYPE total_pairs,
                                           uint64_t* d_keys )
{
    ROWTYPE global_pos = static_cast<ROWTYPE>( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( global_pos >= total_pairs )
        return;

    ROWTYPE target = global_pos + base;

    // Binary search to find row index in workload_prefix
    COLTYPE low = 0, high = n;
    while ( low < high )
    {
        COLTYPE mid = ( low + high ) / 2;
        if ( d_workload_prefix[mid] <= target )
        {
            low = mid + 1;
        }
        else
        {
            high = mid;
        }
    }
    COLTYPE row_i = low - 1;

    ROWTYPE row_start_A = d_ai_A[row_i] - base;
    ROWTYPE row_start_B = d_ai_B[row_i] - base;
    ROWTYPE nnz_A = d_ai_A[row_i + 1] - d_ai_A[row_i];
    ROWTYPE nnz_B = d_ai_B[row_i + 1] - d_ai_B[row_i];

    ROWTYPE local_pos = target - d_workload_prefix[row_i];
    ROWTYPE a_offset = local_pos / nnz_B;
    ROWTYPE b_offset = local_pos - a_offset * nnz_B;

    COLTYPE row = d_aj_A[row_start_A + a_offset];
    COLTYPE col = d_aj_B[row_start_B + b_offset];

    // Pack row in upper 32 bits, col in lower 32 bits
    uint64_t key = ( static_cast<uint64_t>( row ) << 32 ) | static_cast<uint32_t>( col );
    d_keys[global_pos] = key;
}

template <typename ROWTYPE, typename COLTYPE, int WARP_TILE_COLS = 128>
__global__ void build_outer_product_pairs_shared( const ROWTYPE* d_ai_A,
                                                  const COLTYPE* d_aj_A,
                                                  const ROWTYPE* d_ai_B,
                                                  const COLTYPE* d_aj_B,
                                                  COLTYPE n,
                                                  ROWTYPE base,
                                                  const ROWTYPE* d_workload_prefix,
                                                  ROWTYPE total_pairs,
                                                  uint64_t* d_keys )
{
    constexpr int warp_size = 32;
    int lane = threadIdx.x & ( warp_size - 1 );
    int warp_in_block = threadIdx.x / warp_size;
    int warps_per_block = blockDim.x / warp_size;
    int global_warp_id = blockIdx.x * warps_per_block + warp_in_block;
    int total_warps = gridDim.x * warps_per_block;
    if ( total_warps <= 0 || global_warp_id >= total_warps || total_pairs <= 0 )
        return;

    ROWTYPE seg_begin =
        ( total_pairs * static_cast<ROWTYPE>( global_warp_id ) ) / static_cast<ROWTYPE>( total_warps );
    ROWTYPE seg_end = ( total_pairs * static_cast<ROWTYPE>( global_warp_id + 1 ) ) /
                      static_cast<ROWTYPE>( total_warps );
    if ( seg_begin >= seg_end )
        return;

    extern __shared__ COLTYPE smem_cols[];
    COLTYPE* warp_smem = smem_cols + warp_in_block * WARP_TILE_COLS;

    ROWTYPE target = seg_begin + base;
    COLTYPE low = 0, high = n;
    while ( low < high )
    {
        COLTYPE mid = ( low + high ) / 2;
        if ( d_workload_prefix[mid] <= target )
            low = mid + 1;
        else
            high = mid;
    }
    COLTYPE row_i = low - 1;

    ROWTYPE cursor = seg_begin;
    while ( cursor < seg_end && row_i < n )
    {
        ROWTYPE row_prefix_start = d_workload_prefix[row_i] - base;
        ROWTYPE row_prefix_end = d_workload_prefix[row_i + 1] - base;
        if ( row_prefix_end <= cursor )
        {
            ++row_i;
            continue;
        }

        ROWTYPE row_start_A = d_ai_A[row_i] - base;
        ROWTYPE row_start_B = d_ai_B[row_i] - base;
        ROWTYPE nnz_A = d_ai_A[row_i + 1] - d_ai_A[row_i];
        ROWTYPE nnz_B = d_ai_B[row_i + 1] - d_ai_B[row_i];

        ROWTYPE row_seg_begin = cursor > row_prefix_start ? cursor : row_prefix_start;
        ROWTYPE row_seg_end = seg_end < row_prefix_end ? seg_end : row_prefix_end;
        if ( row_seg_begin >= row_seg_end || nnz_A <= 0 || nnz_B <= 0 )
        {
            cursor = row_seg_end;
            if ( cursor >= row_prefix_end )
                ++row_i;
            continue;
        }

        ROWTYPE local_begin = row_seg_begin - row_prefix_start;
        ROWTYPE local_end = row_seg_end - row_prefix_start;
        ROWTYPE a_begin = local_begin / nnz_B;
        ROWTYPE a_end = ( local_end - 1 ) / nnz_B;

        for ( ROWTYPE a_offset = a_begin; a_offset <= a_end; ++a_offset )
        {
            ROWTYPE b_begin = ( a_offset == a_begin ) ? ( local_begin - a_offset * nnz_B ) : 0;
            ROWTYPE b_end = ( a_offset == a_end ) ? ( local_end - a_offset * nnz_B ) : nnz_B;

            COLTYPE row = 0;
            if ( lane == 0 )
                row = d_aj_A[row_start_A + a_offset];
            row = __shfl_sync( 0xFFFFFFFFu, row, 0 );

            for ( ROWTYPE tile_begin = b_begin; tile_begin < b_end; tile_begin += WARP_TILE_COLS )
            {
                ROWTYPE remaining = b_end - tile_begin;
                ROWTYPE tile_count = remaining < static_cast<ROWTYPE>( WARP_TILE_COLS )
                                         ? remaining
                                         : static_cast<ROWTYPE>( WARP_TILE_COLS );
                for ( ROWTYPE t = lane; t < tile_count; t += warp_size )
                    warp_smem[t] = d_aj_B[row_start_B + tile_begin + t];
                __syncwarp();

                for ( ROWTYPE t = lane; t < tile_count; t += warp_size )
                {
                    COLTYPE col = warp_smem[t];
                    ROWTYPE b_offset = tile_begin + t;
                    ROWTYPE local_pos = a_offset * nnz_B + b_offset;
                    ROWTYPE global_pos = row_prefix_start + local_pos;
                    uint64_t key = ( static_cast<uint64_t>( row ) << 32 ) | static_cast<uint32_t>( col );
                    d_keys[global_pos] = key;
                }
                __syncwarp();
            }
        }

        cursor = row_seg_end;
        if ( cursor >= row_prefix_end )
            ++row_i;
    }
}

// Unpack uint64_t keys into separate row and column arrays
template <typename COLTYPE>
__global__ void packed_to_split_coo( const uint64_t* d_keys, COLTYPE unique_nnz, COLTYPE* d_rows, COLTYPE* d_cols )
{
    COLTYPE idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < unique_nnz )
    {
        uint64_t key = d_keys[idx];
        d_rows[idx] = static_cast<COLTYPE>( key >> 32 );
        d_cols[idx] = static_cast<COLTYPE>( key & 0xFFFFFFFFU );
    }
}

// Scatter counts to row positions
template <typename ROWTYPE, typename COLTYPE>
__global__ void scatter_counts( const COLTYPE* d_unique_rows,
                                const ROWTYPE* d_counts,
                                COLTYPE num_runs,
                                ROWTYPE base,
                                ROWTYPE* d_row_count )
{
    COLTYPE idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < num_runs )
    {
        COLTYPE row_id = d_unique_rows[idx] - base;
        d_row_count[row_id] = d_counts[idx];
    }
}

/**
 * @brief Convert packed COO (uint64_t keys) to CSR format
 *
 * Takes sorted and deduplicated uint64_t keys (row in upper 32 bits, col in lower 32 bits)
 * and converts to CSR format using RLE and scan.
 *
 * @tparam ROWTYPE Type for row pointers (int or int64_t)
 * @tparam COLTYPE Type for column indices (int)
 *
 * @param d_keys Device pointer to packed uint64_t keys (row||col)
 * @param unique_nnz Number of unique keys
 * @param n_rows Number of rows in output matrix
 * @param base Index base (0 or 1)
 * @param output [Output] DeviceCSRMatrix to hold output CSR structure
 *
 * @return true if successful, false otherwise
 */
template <typename ROWTYPE, typename COLTYPE>
bool PackedCOOtoCSR( const uint64_t* d_keys,
                     ROWTYPE unique_nnz,
                     COLTYPE n_rows,
                     ROWTYPE base,
                     DeviceCSRMatrix<ROWTYPE, COLTYPE>& output )
{
    if ( d_keys == nullptr || unique_nnz <= 0 || n_rows <= 0 )
    {
        return false;
    }

    int threads = 256;

    // Allocate output arrays
    output.ai.resize( static_cast<size_t>( n_rows + 1 ) );
    output.aj.resize( static_cast<size_t>( unique_nnz ) );
    output.n_rows = n_rows;
    output.base = base;

    int csr_blocks = ( unique_nnz + threads - 1 ) / threads;
    if ( csr_blocks == 0 )
        csr_blocks = 1;

    // Unpack keys into separate row and column arrays
    DeviceArray<COLTYPE> d_rows;
    d_rows.resize( static_cast<size_t>( unique_nnz ) );
    packed_to_split_coo<COLTYPE><<<csr_blocks, threads>>>(
        d_keys, static_cast<COLTYPE>( unique_nnz ), d_rows.data(), output.aj.data() );

    // Run-length encode to identify unique rows and their counts
    DeviceArray<COLTYPE> d_unique_rows;
    d_unique_rows.resize( static_cast<size_t>( unique_nnz ) ); // Max possible unique rows
    DeviceArray<ROWTYPE> d_counts;
    d_counts.resize( static_cast<size_t>( unique_nnz ) ); // Max possible runs
    DeviceArray<COLTYPE> d_num_runs;
    d_num_runs.resize( 1 );

    void* d_rle_temp = nullptr;
    size_t rle_temp_bytes = 0;
    cub::DeviceRunLengthEncode::Encode( d_rle_temp, rle_temp_bytes, d_rows.data(),
                                        d_unique_rows.data(), d_counts.data(), d_num_runs.data(),
                                        static_cast<COLTYPE>( unique_nnz ) );
    DeviceArray<uint8_t> d_rle_storage;
    d_rle_storage.resize( rle_temp_bytes );
    cub::DeviceRunLengthEncode::Encode( d_rle_storage.data(), rle_temp_bytes, d_rows.data(),
                                        d_unique_rows.data(), d_counts.data(), d_num_runs.data(),
                                        static_cast<COLTYPE>( unique_nnz ) );

    COLTYPE num_runs = 0;
    cudaMemcpy( &num_runs, d_num_runs.data(), sizeof( COLTYPE ), cudaMemcpyDeviceToHost );

    // Create d_row_count array of size n+1 initialized to 0
    DeviceArray<ROWTYPE> d_row_count;
    d_row_count.resize( static_cast<size_t>( n_rows + 1 ) );
    cudaMemset( d_row_count.data(), 0, ( n_rows + 1 ) * sizeof( ROWTYPE ) );

    // Scatter d_counts to d_row_count based on d_unique_rows
    int scatter_blocks = ( num_runs + threads - 1 ) / threads;
    if ( scatter_blocks == 0 )
        scatter_blocks = 1;
    scatter_counts<ROWTYPE, COLTYPE><<<scatter_blocks, threads>>>(
        d_unique_rows.data(), d_counts.data(), num_runs, static_cast<ROWTYPE>( base ), d_row_count.data() );

    // Exclusive scan of d_row_count with initial value base to obtain d_ai
    void* d_scan_temp = nullptr;
    size_t scan_temp_bytes = 0;
    cub::DeviceScan::ExclusiveScan( d_scan_temp, scan_temp_bytes, d_row_count.data(), output.ai.data(),
                                    ::cuda::std::plus<ROWTYPE>(), static_cast<ROWTYPE>( base ), n_rows + 1 );
    DeviceArray<uint8_t> d_scan_storage;
    d_scan_storage.resize( scan_temp_bytes );
    cub::DeviceScan::ExclusiveScan( d_scan_storage.data(), scan_temp_bytes, d_row_count.data(),
                                    output.ai.data(), ::cuda::std::plus<ROWTYPE>(),
                                    static_cast<ROWTYPE>( base ), n_rows + 1 );

    cudaDeviceSynchronize();

    return true;
}

/**
 * @brief Build packed COO sparsity pattern for C = A * B from outer products
 *
 * Directly uses uint64_t keys from build_outer_product_pairs for efficient sorting.
 */
template <typename ROWTYPE, typename COLTYPE>
bool SpMMStruct( COLTYPE n,
                 const ROWTYPE* d_ai_A,
                 const COLTYPE* d_aj_A,
                 const ROWTYPE* d_ai_B,
                 const COLTYPE* d_aj_B,
                 ROWTYPE base,
                 DeviceArray<uint64_t>& packed_coo,
                 OuterProductBuildMethod method )
{
    static_assert( sizeof( COLTYPE ) <= 4, "SpMMStruct requires COLTYPE <= 32 bits." );

    if ( n <= 0 || d_ai_A == nullptr || d_aj_A == nullptr || d_ai_B == nullptr || d_aj_B == nullptr )
    {
        return false;
    }

    // Step 0: Allocate and compute workload prefix
    DeviceArray<ROWTYPE> d_workload_prefix;
    d_workload_prefix.resize( static_cast<size_t>( n + 1 ) );
    ROWTYPE total_pairs = 0;

    if ( !SpMMAnalyze<ROWTYPE, COLTYPE>( n, d_ai_A, d_ai_B, base, d_workload_prefix.data(), &total_pairs ) )
    {
        return false;
    }

    // Step 2: Allocate temporary DeviceArray for uint64_t keys
    DeviceArray<uint64_t>& d_keys = packed_coo;
    DeviceArray<uint64_t> d_keys_sorted;
    d_keys.resize( static_cast<size_t>( total_pairs ) );
    d_keys_sorted.resize( static_cast<size_t>( total_pairs ) );

    if ( total_pairs <= 0 )
    {
        return true;
    }

    int threads = 256;
    int blocks = static_cast<int>( ( total_pairs + threads - 1 ) / threads );

    // Step 3: Build outer products directly to uint64_t keys
    if ( method == OuterProductBuildMethod::SharedMemoryWarp )
    {
        constexpr int kWarpTileCols = 128;
        int warps_per_block = threads / 32;
        int warp_blocks = blocks;
        int smem_bytes = warps_per_block * kWarpTileCols * static_cast<int>( sizeof( COLTYPE ) );
        build_outer_product_pairs_shared<ROWTYPE, COLTYPE, kWarpTileCols>
            <<<warp_blocks, threads, smem_bytes>>>( d_ai_A, d_aj_A, d_ai_B, d_aj_B, n, base,
                                                    d_workload_prefix.data(), total_pairs, d_keys.data() );
    }
    else
    {
        build_outer_product_pairs<ROWTYPE, COLTYPE><<<blocks, threads>>>(
            d_ai_A, d_aj_A, d_ai_B, d_aj_B, n, base, d_workload_prefix.data(), total_pairs, d_keys.data() );
    }

    // Step 4 & 5: Sort and deduplicate using shared temp buffer
    DeviceArray<ROWTYPE> d_unique_count;
    d_unique_count.resize( 1 );

    // Query memory requirements for both operations
    void* d_temp_storage = nullptr;
    size_t sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys( d_temp_storage, sort_temp_bytes, d_keys.data(),
                                    d_keys_sorted.data(), total_pairs );

    size_t unique_temp_bytes = 0;
    cub::DeviceSelect::Unique( d_temp_storage, unique_temp_bytes, d_keys_sorted.data(),
                               d_keys.data(), d_unique_count.data(), total_pairs );

    // Allocate single temp buffer with maximum required size
    size_t temp_storage_bytes = std::max( sort_temp_bytes, unique_temp_bytes );
    DeviceArray<uint8_t> d_temp_buffer;
    d_temp_buffer.resize( temp_storage_bytes );
    d_temp_storage = d_temp_buffer.data();

    // Execute Sort
    cub::DeviceRadixSort::SortKeys( d_temp_storage, sort_temp_bytes, d_keys.data(),
                                    d_keys_sorted.data(), total_pairs );

    // Execute Unique (reuse d_keys for output)
    cub::DeviceSelect::Unique( d_temp_storage, unique_temp_bytes, d_keys_sorted.data(),
                               d_keys.data(), d_unique_count.data(), total_pairs );

    // Get unique count
    ROWTYPE unique_nnz = 0;
    cudaMemcpy( &unique_nnz, d_unique_count.data(), sizeof( ROWTYPE ), cudaMemcpyDeviceToHost );

    // Resize d_keys to actual unique size
    d_keys.resize( static_cast<size_t>( unique_nnz ) );
    return true;
}

// ============================================================================
// Template instantiations
// ============================================================================

// Step 1: Compute workload
template bool SpMMAnalyze<int, int>( int n_rows,
                                     const int* d_ai_A,
                                     const int* d_ai_B,
                                     int base,
                                     int* d_workload_prefix,
                                     int* required_array_size );
template bool SpMMAnalyze<int64_t, int>( int n_rows,
                                         const int64_t* d_ai_A,
                                         const int64_t* d_ai_B,
                                         int64_t base,
                                         int64_t* d_workload_prefix,
                                         int64_t* required_array_size );

// Step 2: Build packed COO sparsity pattern
template bool SpMMStruct<int, int>( int n,
                                    const int* d_ai_A,
                                    const int* d_aj_A,
                                    const int* d_ai_B,
                                    const int* d_aj_B,
                                    int base,
                                    DeviceArray<uint64_t>& packed_coo,
                                    OuterProductBuildMethod method );
template bool SpMMStruct<int64_t, int>( int n,
                                        const int64_t* d_ai_A,
                                        const int* d_aj_A,
                                        const int64_t* d_ai_B,
                                        const int* d_aj_B,
                                        int64_t base,
                                        DeviceArray<uint64_t>& packed_coo,
                                        OuterProductBuildMethod method );

// Step 3: Convert packed COO to CSR
template bool PackedCOOtoCSR<int, int>( const uint64_t* d_keys,
                                        int unique_nnz,
                                        int n_rows,
                                        int base,
                                        DeviceCSRMatrix<int, int>& output );
template bool PackedCOOtoCSR<int64_t, int>( const uint64_t* d_keys,
                                            int64_t unique_nnz,
                                            int n_rows,
                                            int64_t base,
                                            DeviceCSRMatrix<int64_t, int>& output );

// Helper kernel instantiations
template __global__ void packed_to_split_coo<int>( const uint64_t* d_keys, int unique_nnz, int* d_rows, int* d_cols );
template __global__ void scatter_counts<int, int>( const int* d_unique_rows,
                                                   const int* d_counts,
                                                   int num_runs,
                                                   int base,
                                                   int* d_row_count );
template __global__ void scatter_counts<int64_t, int>( const int* d_unique_rows,
                                                       const int64_t* d_counts,
                                                       int num_runs,
                                                       int64_t base,
                                                       int64_t* d_row_count );

} // namespace matrix_utils::sparse_cuda
