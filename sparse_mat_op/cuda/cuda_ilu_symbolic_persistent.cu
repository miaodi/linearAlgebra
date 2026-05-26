#include "cuda_ilu_symbolic.cuh"
#include <cuda/iterator>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/scatter.h>
#include <thrust/reduce.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include <array>
#include <limits>
#include <vector>
#include <iostream>
namespace matrix_utils::sparse_cuda
{

template <typename COLTYPE>
struct Pair
{
    COLTYPE src;
    COLTYPE cur;

    __host__ __device__ bool operator==( const Pair& other ) const
    {
        return src == other.src && cur == other.cur;
    }

    __host__ __device__ bool operator<( const Pair& other ) const
    {
        return src < other.src || ( src == other.src && cur < other.cur );
    }
};

template <typename COLTYPE>
struct PairSrc
{
    __host__ __device__ COLTYPE operator()( const Pair<COLTYPE>& p ) const { return p.src; }
};

template <typename COLTYPE>
struct PairCurPlusBase
{
    COLTYPE base;
    __host__ __device__ explicit PairCurPlusBase( COLTYPE base_in ) : base( base_in ) {}
    __host__ __device__ COLTYPE operator()( const Pair<COLTYPE>& p ) const { return p.cur + base; }
};

template <typename T>
struct LessThan
{
    __device__ bool operator()( const T& a, const T& b ) const { return a < b; }
};

__device__ int g_ilu_row_counter;

enum class OverflowCode : int
{
    None = 0,
    FrontierBuffer = 1,
    TempBuffer = 2,
    HashTable = 3,
    GlobalPairs = 4
};

__host__ __device__ constexpr int overflow_to_int( OverflowCode code )
{
    return static_cast<int>( code );
}

__host__ __device__ constexpr OverflowCode overflow_from_int( int value )
{
    return static_cast<OverflowCode>( value );
}

struct OverflowPositive
{
    __host__ __device__ bool operator()( int code ) const { return code > 0; }
};

// Kernel: Precompute degree (neighbor count) for all nodes
template <typename ROWTYPE, typename COLTYPE>
__global__ void compute_all_degrees_kernel( const ROWTYPE* d_ai, COLTYPE n, COLTYPE base, int* degrees )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= n )
        return;

    ROWTYPE row_start = d_ai[idx] - base;
    ROWTYPE row_end = d_ai[idx + 1] - base;
    degrees[idx] = static_cast<int>( row_end - row_start );
}

template <typename COLTYPE>
__device__ __forceinline__ unsigned int hash_node( COLTYPE v )
{
    unsigned int x = static_cast<unsigned int>( v );
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

template <typename COLTYPE>
__device__ __forceinline__ bool hash_insert_if_absent( COLTYPE* table, int size, COLTYPE key )
{
    const unsigned int mask = static_cast<unsigned int>( size - 1 );
    unsigned int pos = hash_node( key ) & mask;
    for ( int probe = 0; probe < size; ++probe )
    {
        COLTYPE val = table[pos];
        if ( val == key )
        {
            return false;
        }
        if ( val == static_cast<COLTYPE>( -1 ) )
        {
            table[pos] = key;
            return true;
        }
        pos = ( pos + 1 ) & mask;
    }
    return false;
}

template <typename COLTYPE>
__device__ __forceinline__ int hash_insert_or_find_atomic( COLTYPE* table, int size, COLTYPE key )
{
    // Lock-free linear-probing insert/find:
    // returns 1 if this thread inserted key, 0 if key already present, -1 if table is full.
    // Uses -1 sentinel in the table and atomicCAS so multiple lanes can race safely.
    static_assert( sizeof( COLTYPE ) == sizeof( int ),
                   "hash_insert_or_find_atomic expects 32-bit COLTYPE" );
    const unsigned int mask = static_cast<unsigned int>( size - 1 );
    unsigned int pos = hash_node( key ) & mask;
    for ( int probe = 0; probe < size; ++probe )
    {
        COLTYPE val = table[pos];
        if ( val == key )
        {
            return 0;
        }
        if ( val == static_cast<COLTYPE>( -1 ) )
        {
            int old = atomicCAS( reinterpret_cast<int*>( &table[pos] ), static_cast<int>( -1 ),
                                 static_cast<int>( key ) );
            if ( old == static_cast<int>( -1 ) )
            {
                return 1;
            }
            if ( static_cast<COLTYPE>( old ) == key )
            {
                return 0;
            }
        }
        pos = ( pos + 1 ) & mask;
    }
    return -1;
}

/**
 * @brief Process a single row for ILU(k) symbolic factorization using warp-level parallelism
 *
 * This function computes the symbolic pattern for one row of the ILU(k) upper triangular factor.
 * It performs a BFS starting from the diagonal element, exploring neighbors up to k levels.
 * The algorithm maintains:
 * - A frontier of nodes to explore (nodes reachable at current level)
 * - A hash table to track visited nodes
 * - A temporary buffer for output pairs where column >= row
 *
 * Key invariant: Only explore neighbors through intermediate nodes smaller than the current row.
 * This ensures we only capture fill-in from previously processed rows.
 *
 * @tparam ROWTYPE Row pointer type (int or int64_t)
 * @tparam COLTYPE Column index type
 * @param row Current row index being processed
 * @param d_ai CSR row pointers
 * @param d_aj CSR column indices
 * @param lvl Fill level k (0 = ILU(0), 1 = ILU(1), etc.)
 * @param base Index base (0 or 1)
 * @param keepdiag If true, include diagonal element in output
 * @param frontier Warp-local buffer for current BFS frontier
 * @param next_frontier Warp-local buffer for next BFS level
 * @param temp Warp-local buffer for output pairs (columns >= row)
 * @param frontier_cap Capacity of frontier buffers
 * @param temp_cap Capacity of temp buffer
 * @param hash Warp-local hash table for visited tracking
 * @param hash_cap Hash table capacity (must be power of 2)
 * @param out_pairs Global output buffer for all pairs
 * @param out_count Global counter for output pairs
 * @param out_cap Maximum capacity of out_pairs
 * @param lane Warp lane ID (0-31)
 * @param overflow_vec Per-row overflow codes (size >= n)
 * @param degrees Precomputed degree array for all nodes
 */
template <typename ROWTYPE, typename COLTYPE>
__device__ void ProcessNode( COLTYPE row,
                             const ROWTYPE* d_ai,
                             const COLTYPE* d_aj,
                             int lvl,
                             COLTYPE base,
                             bool keepdiag,
                             COLTYPE* frontier,
                             COLTYPE* next_frontier,
                             COLTYPE* temp,
                             size_t frontier_cap,
                             size_t temp_cap,
                             COLTYPE* hash,
                             size_t hash_cap,
                             Pair<COLTYPE>* out_pairs,
                             unsigned long long* out_count,
                             unsigned long long out_cap,
                             int lane,
                             int* overflow_vec,
                             const int* degrees )
{
    constexpr int warp_size = 32;
    const unsigned int mask = 0xffffffffu;
    const size_t row_index = static_cast<size_t>( row );
    int overflow_int = 0;

    auto sync_overflow = [&]() -> int
    {
        overflow_int = __shfl_sync( mask, overflow_int, 0 );
        return overflow_int;
    };

    auto set_overflow = [&]( OverflowCode code )
    {
        if ( lane == 0 && overflow_int == 0 )
        {
            overflow_int = overflow_to_int( code );
        }
    };

    auto finalize_if_overflow = [&]() -> bool
    {
        if ( sync_overflow() != 0 )
        {
            if ( lane == 0 )
            {
                overflow_vec[row_index] = overflow_int;
            }
            return true;
        }
        return false;
    };

    // Fast exit if overflow already flagged (should be zero for fresh rows)
    if ( finalize_if_overflow() )
    {
        return;
    }

    // Initialize hash table with sentinel value -1 (parallel across warp)
    for ( int idx = lane; idx < hash_cap; idx += warp_size )
    {
        hash[idx] = static_cast<COLTYPE>( -1 );
    }
    __syncwarp( mask );

    // Initialize frontier with the diagonal element (row, row)
    // Only lane 0 performs initialization, others wait
    int frontier_size = 0;
    int temp_count = 0;
    if ( lane == 0 )
    {
        frontier[0] = row;
        frontier_size = 1;

        // Mark diagonal as visited (should always succeed on first insert)
        hash_insert_if_absent( hash, hash_cap, row );

        // If keeping diagonal, add (row, row) to output immediately
        if ( keepdiag )
        {
            if ( temp_count < temp_cap )
            {
                temp[temp_count++] = row;
            }
            else
            {
                set_overflow( OverflowCode::TempBuffer );
            }
        }
    }

    if ( finalize_if_overflow() )
    {
        return;
    }

    // Broadcast frontier_size and temp_count to all lanes
    frontier_size = __shfl_sync( mask, frontier_size, 0 );
    temp_count = __shfl_sync( mask, temp_count, 0 );

    // BFS loop: iterate through k levels
    for ( int level = 0; level <= lvl; ++level )
    {
        if ( frontier_size == 0 )
        {
            break;
        }

        // Expand frontier using 3-step parallel approach within the warp:
        // Step 1: Each thread looks up precomputed degrees for assigned frontier nodes
        // Step 2: Prefix sum to determine write positions (process in chunks)
        // Step 3: Each thread writes neighbors to allocated positions in next_frontier

        int total_neighbors = 0;

        // Process frontier in chunks of warp_size to avoid memory conflicts
        typedef cub::WarpScan<int> WarpScan;
        __shared__ typename WarpScan::TempStorage temp_storage[8]; // Support up to 8 warps per block
        int warp_in_block = threadIdx.x / warp_size;

        for ( int chunk_start = 0; chunk_start < frontier_size; chunk_start += warp_size )
        {
            int f = chunk_start + lane;

            // Step 1: Lookup degree for this thread's frontier node
            int degree = 0;
            COLTYPE node = -1;
            if ( f < frontier_size )
            {
                node = frontier[f];
                degree = degrees[node];
            }

            // Step 2: Compute exclusive prefix sum for write offsets
            int my_offset;
            WarpScan( temp_storage[warp_in_block] ).ExclusiveSum( degree, my_offset );
            my_offset += total_neighbors; // Add base offset from previous chunks

            // Update total for next chunk
            int chunk_total = __shfl_sync( mask, my_offset + degree, warp_size - 1 );
            total_neighbors = chunk_total;

            __syncwarp( mask );

            // Step 3: Copy neighbors to next_frontier
            if ( f < frontier_size && node != static_cast<COLTYPE>( -1 ) )
            {
                ROWTYPE row_start = d_ai[node] - base;
                ROWTYPE row_end = d_ai[node + 1] - base;
                int write_offset = my_offset;

                // Copy all neighbors of this node
                for ( ROWTYPE jj = row_start; jj < row_end; ++jj )
                {
                    COLTYPE neighbor = d_aj[jj] - base;
                    next_frontier[write_offset++] = neighbor;
                }
            }

            __syncwarp( mask );
        }

        // Check if we have enough space
        if ( lane == 0 && total_neighbors > frontier_cap )
        {
            set_overflow( OverflowCode::FrontierBuffer );
        }
        if ( finalize_if_overflow() )
        {
            return;
        }

        int adj_count = total_neighbors;

        // Warp-parallel dedupe + partition without a full sort:
        // each lane probes/inserts into the hash table and then uses ballots to compact
        // new frontier (v < row) and temp (v >= row) outputs.
        int new_frontier_size = 0;
        int new_temp_count = temp_count;
        for ( int chunk_start = 0; chunk_start < adj_count; chunk_start += warp_size )
        {
            int idx = chunk_start + lane;
            bool in_range = idx < adj_count;
            COLTYPE v = in_range ? next_frontier[idx] : static_cast<COLTYPE>( -1 );

            int status = 0;
            if ( in_range )
            {
                // Atomic insert ensures only the first lane inserting v "owns" it.
                status = hash_insert_or_find_atomic( hash, hash_cap, v );
            }

            // Any lane reporting -1 means the hash table is saturated; abort this row.
            int overflow = __any_sync( mask, status < 0 ) ? overflow_to_int( OverflowCode::HashTable )
                                                          : overflow_to_int( OverflowCode::None );
            if ( overflow != overflow_to_int( OverflowCode::None ) )
            {
                set_overflow( OverflowCode::HashTable );
                if ( finalize_if_overflow() )
                {
                    return;
                }
            }

            // Only newly inserted nodes are allowed to emit output or be revisited.
            bool inserted = status > 0;
            bool to_frontier = in_range && inserted && ( v < row );
            bool to_temp = in_range && inserted && ( v >= row );

            // Compact frontier writes in-lane order using ballot/popcount.
            unsigned int front_mask = __ballot_sync( mask, to_frontier );
            int front_count = __popc( front_mask );
            int front_rank = __popc( front_mask & ( ( 1u << lane ) - 1 ) );
            int front_base = __shfl_sync( mask, new_frontier_size, 0 );
            if ( lane == 0 && front_base + front_count > frontier_cap )
            {
                set_overflow( OverflowCode::FrontierBuffer );
            }
            if ( finalize_if_overflow() )
            {
                return;
            }
            if ( to_frontier )
            {
                frontier[front_base + front_rank] = v;
            }
            if ( lane == 0 )
            {
                new_frontier_size = front_base + front_count;
            }
            new_frontier_size = __shfl_sync( mask, new_frontier_size, 0 );

            // Compact temp writes (U-pattern columns) similarly.
            unsigned int temp_mask = __ballot_sync( mask, to_temp );
            int temp_count_chunk = __popc( temp_mask );
            int temp_rank = __popc( temp_mask & ( ( 1u << lane ) - 1 ) );
            int temp_base = __shfl_sync( mask, new_temp_count, 0 );
            if ( lane == 0 && temp_base + temp_count_chunk > temp_cap )
            {
                set_overflow( OverflowCode::TempBuffer );
            }
            if ( finalize_if_overflow() )
            {
                return;
            }
            if ( to_temp )
            {
                temp[temp_base + temp_rank] = v;
            }
            if ( lane == 0 )
            {
                new_temp_count = temp_base + temp_count_chunk;
            }
            new_temp_count = __shfl_sync( mask, new_temp_count, 0 );
        }

        if ( lane == 0 )
        {
            frontier_size = new_frontier_size;
            temp_count = new_temp_count;
        }

        // Broadcast updated sizes to all lanes
        frontier_size = __shfl_sync( mask, frontier_size, 0 );
        temp_count = __shfl_sync( mask, temp_count, 0 );

        // Check overflow after each level
        if ( finalize_if_overflow() )
        {
            return;
        }
    }

    // Atomically allocate space in global output buffer
    unsigned long long base_offset = 0;
    if ( lane == 0 )
    {
        base_offset = atomicAdd( out_count, static_cast<unsigned long long>( temp_count ) );
        if ( base_offset + static_cast<unsigned long long>( temp_count ) > out_cap )
        {
            set_overflow( OverflowCode::GlobalPairs );
        }
    }
    base_offset = __shfl_sync( mask, base_offset, 0 );

    if ( finalize_if_overflow() )
    {
        return;
    }

    // Write output pairs to global memory (parallel across warp)
    // Each pair (row, col) represents a nonzero in the ILU(k) upper triangular factor
    for ( int idx = lane; idx < temp_count; idx += warp_size )
    {
        out_pairs[base_offset + idx].src = row;
        out_pairs[base_offset + idx].cur = temp[idx];
    }
}

template <typename ROWTYPE, typename COLTYPE>
__global__ void ilu_symbolic_u_persistent_kernel( COLTYPE n,
                                                  const ROWTYPE* d_ai,
                                                  const COLTYPE* d_aj,
                                                  int lvl,
                                                  COLTYPE base,
                                                  bool keepdiag,
                                                  COLTYPE* frontier,
                                                  COLTYPE* next_frontier,
                                                  COLTYPE* temp,
                                                  COLTYPE* hash,
                                                  size_t frontier_cap,
                                                  size_t temp_cap,
                                                  size_t hash_cap,
                                                  Pair<COLTYPE>* out_pairs,
                                                  unsigned long long out_cap,
                                                  unsigned long long* out_count,
                                                  int* overflow_vec,
                                                  int total_warps,
                                                  const int* degrees )
{
    constexpr int warp_size = 32;
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = global_thread / warp_size;
    const int lane = threadIdx.x & ( warp_size - 1 );

    if ( warp_id >= total_warps )
    {
        return;
    }

    COLTYPE* warp_frontier = frontier + static_cast<size_t>( warp_id ) * frontier_cap;
    COLTYPE* warp_next_frontier = next_frontier + static_cast<size_t>( warp_id ) * frontier_cap;
    COLTYPE* warp_temp = temp + static_cast<size_t>( warp_id ) * temp_cap;
    COLTYPE* warp_hash = hash + static_cast<size_t>( warp_id ) * hash_cap;

    while ( true )
    {
        COLTYPE row = 0;
        if ( lane == 0 )
        {
            int next = atomicAdd( &g_ilu_row_counter, 1 );
            row = static_cast<COLTYPE>( next );
        }
        row = __shfl_sync( 0xffffffffu, row, 0 );
        if ( row >= n )
        {
            return;
        }

        ProcessNode( row, d_ai, d_aj, lvl, base, keepdiag, warp_frontier, warp_next_frontier,
                     warp_temp, frontier_cap, temp_cap, warp_hash, hash_cap, out_pairs, out_count,
                     out_cap, lane, overflow_vec, degrees );
    }
}

template <typename ROWTYPE, typename COLTYPE>
static int compute_total_warps( COLTYPE n, int threads_per_block, size_t per_warp_bytes )
{
    // Pick a warp count based on occupancy and then cap by estimated memory footprint.
    if ( n <= 0 )
    {
        return 0;
    }

    constexpr int warp_size = 32;
    int max_warps = static_cast<int>( n );
    cudaDeviceProp prop{};
    int device = 0;
    if ( cudaGetDevice( &device ) == cudaSuccess && cudaGetDeviceProperties( &prop, device ) == cudaSuccess )
    {
        int warps_per_sm = 4;
        int max_blocks_per_sm = 0;
        // Use occupancy to estimate max active blocks/SM for this kernel and block size.
        const cudaError_t occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, ilu_symbolic_u_persistent_kernel<ROWTYPE, COLTYPE>, threads_per_block, 0 );
        if ( occ_err == cudaSuccess && max_blocks_per_sm > 0 )
        {
            warps_per_sm = max_blocks_per_sm * ( threads_per_block / warp_size );
        }
        max_warps = prop.multiProcessorCount * warps_per_sm;

        if ( per_warp_bytes > 0 )
        {
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            // Cap by a conservative fraction of free memory to reduce allocation risk.
            if ( cudaMemGetInfo( &free_bytes, &total_bytes ) == cudaSuccess )
            {
                const size_t usable_bytes = static_cast<size_t>( free_bytes * 0.8 );
                const int mem_cap = static_cast<int>( usable_bytes / per_warp_bytes );
                if ( mem_cap > 0 )
                {
                    max_warps = std::min( max_warps, mem_cap );
                }
            }
        }
    }

    return std::max( 1, std::min( static_cast<int>( n ), max_warps ) );
}

static int next_pow2( int v )
{
    int p = 1;
    while ( p < v )
    {
        p <<= 1;
    }
    return p;
}

template <typename ROWTYPE, typename COLTYPE>
bool ILUSymbolicU_CUDA_Persistent( COLTYPE n,
                                   const ROWTYPE* d_ai,
                                   const COLTYPE* d_aj,
                                   int lvl,
                                   COLTYPE base,
                                   bool keepdiag,
                                   ROWTYPE* d_u_ai,
                                   COLTYPE** d_u_aj,
                                   ROWTYPE* u_nnz )
{
    std::cout << "[DEBUG] ILUSymbolicU_CUDA_Persistent called: n=" << n << ", lvl=" << lvl << std::endl;
    if ( n <= 0 || lvl < 0 )
    {
        std::cout << "[DEBUG] Invalid parameters" << std::endl;
        return false;
    }

    ROWTYPE nnz = 0;
    cudaMemcpy( &nnz, &d_ai[n], sizeof( ROWTYPE ), cudaMemcpyDeviceToHost );
    nnz -= base;

    constexpr int threads_per_block = 256;
    constexpr int warp_size = 32;
    const int warps_per_block = threads_per_block / warp_size;

    COLTYPE* d_frontier = nullptr;
    COLTYPE* d_next_frontier = nullptr;
    COLTYPE* d_temp = nullptr;
    COLTYPE* d_hash = nullptr;
    Pair<COLTYPE>* d_pairs = nullptr;
    unsigned long long* d_pair_count = nullptr;
    int* d_overflow_vec = nullptr;
    int* d_degrees = nullptr;

    // Allocate device memory for degrees first
    cudaError_t err = cudaMalloc( &d_degrees, static_cast<size_t>( n ) * sizeof( int ) );
    if ( err != cudaSuccess )
    {
        std::cout << "[DEBUG] cudaMalloc for d_degrees failed: " << cudaGetErrorString( err ) << std::endl;
        return false;
    }

    // Precompute degrees for all nodes
    int degree_threads = 256;
    int degree_blocks = ( n + degree_threads - 1 ) / degree_threads;
    compute_all_degrees_kernel<<<degree_blocks, degree_threads>>>( d_ai, n, base, d_degrees );
    cudaDeviceSynchronize();

    // Find maximum degree using Thrust
    thrust::device_ptr<int> d_degrees_ptr( d_degrees );
    int max_degree = *thrust::max_element( d_degrees_ptr, d_degrees_ptr + n );

    const int avg_degree = ( n > 0 ) ? static_cast<int>( nnz / n ) : 0;
    const int base_degree = std::max( 1, std::max( max_degree, avg_degree ) );

    // Buffer sizing: scale with level and degree
    // For higher ILU levels, the frontier can grow exponentially
    // Use more aggressive sizing for medium/large matrices
    auto mul_saturate = []( size_t a, size_t b ) -> size_t
    {
        if ( a == 0 || b == 0 )
        {
            return 0;
        }
        const size_t max_size = std::numeric_limits<size_t>::max();
        if ( a > max_size / b )
        {
            return max_size;
        }
        return a * b;
    };

    size_t level_multiplier = 1;
    for ( int i = 0; i <= lvl; ++i )
    {
        level_multiplier = mul_saturate( level_multiplier, 4u ); // Exponential growth: 4^(lvl+1)
    }

    // Increase frontier capacity significantly - it's the bottleneck
    const size_t frontier_scaled =
        mul_saturate( mul_saturate( static_cast<size_t>( base_degree ), level_multiplier ), 2u );
    const size_t temp_scaled =
        mul_saturate( mul_saturate( static_cast<size_t>( base_degree ), level_multiplier ), 4u );
    size_t frontier_per_warp =
        std::min<size_t>( static_cast<size_t>( 32768u ), std::max<size_t>( 2048u, frontier_scaled ) );
    size_t temp_per_warp =
        std::min<size_t>( static_cast<size_t>( 65536u ), std::max<size_t>( 4096u, temp_scaled ) );
    // Hash table sized for a modest load factor without extreme memory use.
    // Smaller than before to reduce footprint; may increase collisions on dense/fill-heavy cases.
    size_t hash_per_warp = std::max<size_t>(
        static_cast<size_t>( 4096 ),
        next_pow2( std::min( static_cast<size_t>( 4194304 ), frontier_per_warp * 32 ) ) ); // Cap at 4M

    // Estimate per-warp memory so warp count can be capped by available device memory.
    const size_t per_warp_bytes =
        ( frontier_per_warp * 2u + temp_per_warp + hash_per_warp ) * sizeof( COLTYPE );
    const int total_warps = compute_total_warps<ROWTYPE, COLTYPE>(
        n, // Use a fraction of n to avoid excessive memory use on large matrices
        threads_per_block, per_warp_bytes );
    const int num_blocks = ( total_warps + warps_per_block - 1 ) / warps_per_block;

    const size_t bytes_per_gb = 1024ull * 1024ull * 1024ull;
    const size_t frontier_cap_bytes = 5ull * bytes_per_gb;
    const size_t temp_cap_bytes = 5ull * bytes_per_gb;
    const size_t hash_cap_bytes = 40ull * bytes_per_gb;

    auto clamp_total_bytes = [&]( const char* label, size_t cap_bytes, size_t& per_warp )
    {
        if ( total_warps <= 0 || cap_bytes == 0 )
        {
            return;
        }
        const size_t cap_entries = cap_bytes / sizeof( COLTYPE );
        if ( cap_entries == 0 )
        {
            per_warp = 0;
            return;
        }
        const size_t total_entries = static_cast<size_t>( total_warps ) * per_warp;
        if ( total_entries <= cap_entries )
        {
            return;
        }
        const size_t max_per_warp = cap_entries / static_cast<size_t>( total_warps );
        if ( max_per_warp == 0 )
        {
            per_warp = 1;
        }
        else
        {
            per_warp = std::max<size_t>( 1u, max_per_warp );
        }
        std::cout << "[DEBUG] Clamped " << label << " per warp to " << per_warp << " to satisfy global memory cap of "
                  << ( cap_bytes / bytes_per_gb ) << " GB" << std::endl;
    };

    clamp_total_bytes( "frontier", frontier_cap_bytes, frontier_per_warp );
    clamp_total_bytes( "temp", temp_cap_bytes, temp_per_warp );
    clamp_total_bytes( "hash", hash_cap_bytes, hash_per_warp );

    std::cout << "[DEBUG] Buffer sizing: max_degree=" << max_degree << ", avg_degree=" << avg_degree
              << std::endl;
    std::cout << "[DEBUG] level_multiplier=" << level_multiplier << ", base_degree=" << base_degree
              << std::endl;
    std::cout << "[DEBUG] frontier_per_warp=" << frontier_per_warp << ", temp_per_warp=" << temp_per_warp
              << ", hash_per_warp=" << hash_per_warp << std::endl;
    std::cout << "[DEBUG] total_warps=" << total_warps << ", num_blocks=" << num_blocks << std::endl;

    const size_t max_size = std::numeric_limits<size_t>::max();
    if ( static_cast<size_t>( total_warps ) > max_size / static_cast<size_t>( frontier_per_warp ) ||
         static_cast<size_t>( total_warps ) > max_size / static_cast<size_t>( temp_per_warp ) ||
         static_cast<size_t>( total_warps ) > max_size / static_cast<size_t>( hash_per_warp ) )
    {
        return false;
    }

    const size_t total_frontier = static_cast<size_t>( total_warps ) * frontier_per_warp;
    const size_t total_temp = static_cast<size_t>( total_warps ) * temp_per_warp;
    const size_t total_hash = static_cast<size_t>( total_warps ) * hash_per_warp;

    std::cout << "[DEBUG] Memory allocation sizes:" << std::endl;
    std::cout << "  total_frontier: " << total_frontier * sizeof( COLTYPE ) / ( 1024.0 * 1024.0 )
              << " MB" << std::endl;
    std::cout << "  total_temp: " << total_temp * sizeof( COLTYPE ) / ( 1024.0 * 1024.0 ) << " MB" << std::endl;
    std::cout << "  total_hash: " << total_hash * sizeof( COLTYPE ) / ( 1024.0 * 1024.0 ) << " MB" << std::endl;

    if ( static_cast<size_t>( n ) > std::numeric_limits<size_t>::max() / static_cast<size_t>( temp_per_warp ) )
    {
        cudaFree( d_degrees );
        return false;
    }
    const size_t max_pairs = static_cast<size_t>( n ) * static_cast<size_t>( temp_per_warp );
    const unsigned long long out_cap = static_cast<unsigned long long>( max_pairs );

    if ( ( err = cudaMalloc( &d_frontier, total_frontier * sizeof( COLTYPE ) ) ) != cudaSuccess ||
         ( err = cudaMalloc( &d_next_frontier, total_frontier * sizeof( COLTYPE ) ) ) != cudaSuccess ||
         ( err = cudaMalloc( &d_temp, total_temp * sizeof( COLTYPE ) ) ) != cudaSuccess ||
         ( err = cudaMalloc( &d_hash, total_hash * sizeof( COLTYPE ) ) ) != cudaSuccess ||
         ( err = cudaMalloc( &d_pairs, max_pairs * sizeof( Pair<COLTYPE> ) ) ) != cudaSuccess ||
         ( err = cudaMalloc( &d_pair_count, sizeof( unsigned long long ) ) ) != cudaSuccess ||
         ( err = cudaMalloc( &d_overflow_vec, static_cast<size_t>( n ) * sizeof( int ) ) ) != cudaSuccess )
    {
        std::cout << "[DEBUG] cudaMalloc failed: " << cudaGetErrorString( err ) << std::endl;
        std::cout << "[DEBUG] max_pairs size: " << max_pairs * sizeof( Pair<COLTYPE> ) / ( 1024.0 * 1024.0 )
                  << " MB" << std::endl;
        if ( d_frontier )
            cudaFree( d_frontier );
        if ( d_next_frontier )
            cudaFree( d_next_frontier );
        if ( d_temp )
            cudaFree( d_temp );
        if ( d_hash )
            cudaFree( d_hash );
        if ( d_pairs )
            cudaFree( d_pairs );
        if ( d_pair_count )
            cudaFree( d_pair_count );
        if ( d_overflow_vec )
            cudaFree( d_overflow_vec );
        if ( d_degrees )
            cudaFree( d_degrees );
        return false;
    }

    unsigned long long zero_u64 = 0;
    int zero_i32 = 0;
    cudaMemcpy( d_pair_count, &zero_u64, sizeof( unsigned long long ), cudaMemcpyHostToDevice );
    cudaMemset( d_overflow_vec, 0, static_cast<size_t>( n ) * sizeof( int ) );
    cudaMemcpyToSymbol( g_ilu_row_counter, &zero_i32, sizeof( int ) );

    ilu_symbolic_u_persistent_kernel<<<num_blocks, threads_per_block>>>(
        n, d_ai, d_aj, lvl, base, keepdiag, d_frontier, d_next_frontier, d_temp, d_hash,
        frontier_per_warp, temp_per_warp, hash_per_warp, d_pairs, out_cap, d_pair_count,
        d_overflow_vec, total_warps, d_degrees );

    thrust::device_ptr<int> overflow_begin( d_overflow_vec );
    const size_t overflow_rows = static_cast<size_t>(
        thrust::count_if( thrust::device, overflow_begin, overflow_begin + n, OverflowPositive() ) );

    if ( overflow_rows > 0 )
    {
        std::vector<int> h_overflow( static_cast<size_t>( n ) );
        cudaMemcpy( h_overflow.data(), d_overflow_vec, static_cast<size_t>( n ) * sizeof( int ),
                    cudaMemcpyDeviceToHost );

        std::array<size_t, 5> overflow_hist{};
        size_t unknown_overflow = 0;
        size_t overflow_total = 0;
        for ( size_t idx = 0; idx < static_cast<size_t>( n ); ++idx )
        {
            int code = h_overflow[idx];
            if ( code <= 0 )
            {
                continue;
            }
            ++overflow_total;
            if ( static_cast<size_t>( code ) < overflow_hist.size() )
            {
                overflow_hist[static_cast<size_t>( code )]++;
            }
            else
            {
                ++unknown_overflow;
            }
        }

        if ( overflow_total > 0 )
        {
            std::cout << "[DEBUG] Overflow detected on " << overflow_total << " rows" << std::endl;
            const auto frontier_idx = overflow_to_int( OverflowCode::FrontierBuffer );
            const auto temp_idx = overflow_to_int( OverflowCode::TempBuffer );
            const auto hash_idx = overflow_to_int( OverflowCode::HashTable );
            const auto global_idx = overflow_to_int( OverflowCode::GlobalPairs );

            if ( overflow_hist[frontier_idx] > 0 )
            {
                std::cout << "[DEBUG] FRONTIER buffer overflow in " << overflow_hist[frontier_idx]
                          << " rows: frontier_per_warp=" << frontier_per_warp
                          << ". Try increasing frontier size or reducing active warps." << std::endl;
            }
            if ( overflow_hist[temp_idx] > 0 )
            {
                std::cout << "[DEBUG] TEMP buffer overflow in " << overflow_hist[temp_idx]
                          << " rows: temp_per_warp=" << temp_per_warp << " (currently sized for "
                          << ( out_cap / n )
                          << " entries per row). Consider enlarging temp storage." << std::endl;
            }
            if ( overflow_hist[hash_idx] > 0 )
            {
                std::cout << "[DEBUG] HASH TABLE overflow in " << overflow_hist[hash_idx]
                          << " rows: hash_per_warp=" << hash_per_warp
                          << ". Increase hash capacity (power of two)." << std::endl;
            }
            if ( overflow_hist[global_idx] > 0 )
            {
                std::cout << "[DEBUG] GLOBAL PAIRS buffer overflow in " << overflow_hist[global_idx]
                          << " rows: out_cap=" << out_cap
                          << " is insufficient for generated entries." << std::endl;
            }
            if ( unknown_overflow > 0 )
            {
                std::cout << "[DEBUG] Unknown overflow codes encountered: " << unknown_overflow << std::endl;
            }
            cudaFree( d_frontier );
            cudaFree( d_next_frontier );
            cudaFree( d_temp );
            cudaFree( d_hash );
            cudaFree( d_pairs );
            cudaFree( d_pair_count );
            cudaFree( d_overflow_vec );
            cudaFree( d_degrees );
            return false;
        }
    }

    unsigned long long pair_count_u64 = 0;
    cudaMemcpy( &pair_count_u64, d_pair_count, sizeof( unsigned long long ), cudaMemcpyDeviceToHost );

    std::cout << "[DEBUG] Generated " << pair_count_u64 << " pairs total" << std::endl;

    if ( pair_count_u64 > static_cast<unsigned long long>( std::numeric_limits<size_t>::max() ) )
    {
        cudaFree( d_frontier );
        cudaFree( d_next_frontier );
        cudaFree( d_temp );
        cudaFree( d_hash );
        cudaFree( d_pairs );
        cudaFree( d_pair_count );
        cudaFree( d_overflow_vec );
        cudaFree( d_degrees );
        return false;
    }

    size_t pair_count = static_cast<size_t>( pair_count_u64 );
    thrust::device_ptr<Pair<COLTYPE>> pairs_begin( d_pairs );
    thrust::device_ptr<Pair<COLTYPE>> pairs_end = pairs_begin + pair_count;

    if ( pair_count == 0 )
    {
        thrust::device_vector<ROWTYPE> u_ai_dev( static_cast<size_t>( n ) + 1, ROWTYPE( base ) );
        cudaMemcpy( d_u_ai, thrust::raw_pointer_cast( u_ai_dev.data() ),
                    ( static_cast<size_t>( n ) + 1 ) * sizeof( ROWTYPE ), cudaMemcpyDeviceToDevice );
        *u_nnz = 0;
        *d_u_aj = nullptr;
        cudaFree( d_frontier );
        cudaFree( d_next_frontier );
        cudaFree( d_temp );
        cudaFree( d_hash );
        cudaFree( d_pairs );
        cudaFree( d_pair_count );
        cudaFree( d_overflow_vec );
        cudaFree( d_degrees );
        return true;
    }

    thrust::sort( pairs_begin, pairs_end );
    auto unique_end = thrust::unique( pairs_begin, pairs_end );
    pair_count = static_cast<size_t>( unique_end - pairs_begin );

    thrust::device_vector<ROWTYPE> u_ai_dev( static_cast<size_t>( n ) + 1, ROWTYPE( 0 ) );
    thrust::device_vector<ROWTYPE> row_counts( n, ROWTYPE( 0 ) );

    auto row_it = thrust::make_transform_iterator( pairs_begin, PairSrc<COLTYPE>() );
    auto row_it_end = thrust::make_transform_iterator( pairs_begin + pair_count, PairSrc<COLTYPE>() );
    thrust::device_vector<COLTYPE> unique_rows( pair_count );
    thrust::device_vector<ROWTYPE> unique_counts( pair_count );

    auto reduce_end = thrust::reduce_by_key( row_it, row_it_end, cuda::make_constant_iterator( ROWTYPE{ 1 } ),
                                             unique_rows.begin(), unique_counts.begin() );
    size_t unique_size = static_cast<size_t>( reduce_end.first - unique_rows.begin() );
    unique_rows.resize( unique_size );
    unique_counts.resize( unique_size );

    thrust::scatter( unique_counts.begin(), unique_counts.end(), unique_rows.begin(), row_counts.begin() );
    thrust::inclusive_scan( row_counts.begin(), row_counts.end(), u_ai_dev.begin() + 1 );

    if ( base != 0 )
    {
        thrust::transform( u_ai_dev.begin() + 1, u_ai_dev.end(),
                           cuda::make_constant_iterator( ROWTYPE( base ) ), u_ai_dev.begin() + 1,
                           thrust::plus<ROWTYPE>() );
    }
    u_ai_dev[0] = ROWTYPE( base );

    cudaMemcpy( d_u_ai, thrust::raw_pointer_cast( u_ai_dev.data() ),
                ( static_cast<size_t>( n ) + 1 ) * sizeof( ROWTYPE ), cudaMemcpyDeviceToDevice );

    if ( pair_count > static_cast<size_t>( std::numeric_limits<ROWTYPE>::max() ) )
    {
        cudaFree( d_frontier );
        cudaFree( d_next_frontier );
        cudaFree( d_temp );
        cudaFree( d_hash );
        cudaFree( d_pairs );
        cudaFree( d_pair_count );
        cudaFree( d_overflow_vec );
        cudaFree( d_degrees );
        return false;
    }

    *u_nnz = static_cast<ROWTYPE>( pair_count );
    if ( *u_nnz > 0 )
    {
        cudaMalloc( d_u_aj, static_cast<size_t>( *u_nnz ) * sizeof( COLTYPE ) );
        auto u_aj_ptr = thrust::device_pointer_cast( *d_u_aj );
        thrust::transform( pairs_begin, pairs_begin + pair_count, u_aj_ptr, PairCurPlusBase<COLTYPE>( base ) );
    }
    else
    {
        *d_u_aj = nullptr;
    }

    cudaFree( d_frontier );
    cudaFree( d_next_frontier );
    cudaFree( d_temp );
    cudaFree( d_hash );
    cudaFree( d_pairs );
    cudaFree( d_pair_count );
    cudaFree( d_overflow_vec );
    cudaFree( d_degrees );

    std::cout << "[DEBUG] ILUSymbolicU_CUDA_Persistent completed successfully. u_nnz=" << *u_nnz << std::endl;

    return true;
}

template bool ILUSymbolicU_CUDA_Persistent<int, int>( int n,
                                                      const int* d_ai,
                                                      const int* d_aj,
                                                      int lvl,
                                                      int base,
                                                      bool keepdiag,
                                                      int* d_u_ai,
                                                      int** d_u_aj,
                                                      int* u_nnz );

template bool ILUSymbolicU_CUDA_Persistent<int64_t, int>( int n,
                                                          const int64_t* d_ai,
                                                          const int* d_aj,
                                                          int lvl,
                                                          int base,
                                                          bool keepdiag,
                                                          int64_t* d_u_ai,
                                                          int** d_u_aj,
                                                          int64_t* u_nnz );

} // namespace matrix_utils::sparse_cuda
