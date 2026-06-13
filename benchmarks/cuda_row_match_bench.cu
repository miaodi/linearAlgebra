#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
inline constexpr int kWarpSize = 32;
inline constexpr int kWarpsPerBlock = 8;
inline constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
inline constexpr int kRepeatWarps = 1 << 15;

enum class MatchKernel
{
    BinarySearch,
    MergeTiled
};

enum class RowPattern
{
    OneToOne,
    Subset,
    Mixed,
    InterleavedMiss,
    DisjointAfter
};

struct RowCase
{
    int curr_len = 0;
    int ref_len = 0;
    int match_percent = 0;
    RowPattern pattern = RowPattern::Mixed;
    const char* name = nullptr;
};

void checkCuda( const cudaError_t status, const char* message )
{
    if ( status != cudaSuccess )
    {
        throw std::runtime_error( std::string( message ) + ": " + cudaGetErrorString( status ) );
    }
}

template <typename T>
class DeviceBuffer
{
public:
    explicit DeviceBuffer( const std::size_t count ) : count_( count )
    {
        if ( count_ != 0 )
        {
            checkCuda( cudaMalloc( reinterpret_cast<void**>( &data_ ), count_ * sizeof( T ) ),
                       "allocate device buffer" );
        }
    }

    ~DeviceBuffer()
    {
        if ( data_ != nullptr )
        {
            cudaFree( data_ );
        }
    }

    DeviceBuffer( const DeviceBuffer& ) = delete;
    DeviceBuffer& operator=( const DeviceBuffer& ) = delete;

    T* data() { return data_; }
    const T* data() const { return data_; }

    void copyFromHost( const std::vector<T>& host )
    {
        if ( host.size() > count_ )
        {
            throw std::runtime_error( "host vector is larger than device buffer" );
        }
        if ( !host.empty() )
        {
            checkCuda( cudaMemcpy( data_, host.data(), host.size() * sizeof( T ), cudaMemcpyHostToDevice ),
                       "copy host data to device" );
        }
    }

    void copyToHost( std::vector<T>& host ) const
    {
        if ( host.size() > count_ )
        {
            throw std::runtime_error( "host vector is larger than device buffer" );
        }
        if ( !host.empty() )
        {
            checkCuda( cudaMemcpy( host.data(), data_, host.size() * sizeof( T ), cudaMemcpyDeviceToHost ),
                       "copy device data to host" );
        }
    }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

std::vector<int> makeCurrentRow( const int curr_len )
{
    std::vector<int> row( static_cast<std::size_t>( curr_len ) );
    for ( int i = 0; i < curr_len; ++i )
    {
        row[static_cast<std::size_t>( i )] = 2 * i;
    }
    return row;
}

void appendSubsetMatches( const std::vector<int>& curr, const int match_count, std::vector<int>& ref )
{
    if ( match_count <= 0 )
    {
        return;
    }
    if ( match_count > static_cast<int>( curr.size() ) )
    {
        throw std::runtime_error( "match_count exceeds current row length" );
    }

    for ( int m = 0; m < match_count; ++m )
    {
        const int curr_idx =
            ( match_count == 1 ) ? 0 : ( m * ( static_cast<int>( curr.size() ) - 1 ) ) / ( match_count - 1 );
        ref.push_back( curr[static_cast<std::size_t>( curr_idx )] );
    }
}

std::vector<int> makeReferenceRow( const RowCase& row_case, const std::vector<int>& curr )
{
    std::vector<int> ref;
    ref.reserve( static_cast<std::size_t>( row_case.ref_len ) );

    if ( row_case.pattern == RowPattern::OneToOne )
    {
        if ( row_case.ref_len != row_case.curr_len )
        {
            throw std::runtime_error( "one-to-one row match case requires curr_len == ref_len" );
        }
        return curr;
    }

    if ( row_case.pattern == RowPattern::DisjointAfter )
    {
        const int first = curr.back() + 2;
        for ( int i = 0; i < row_case.ref_len; ++i )
        {
            ref.push_back( first + 2 * i );
        }
        return ref;
    }

    if ( row_case.pattern == RowPattern::InterleavedMiss )
    {
        for ( int i = 0; i < row_case.ref_len; ++i )
        {
            ref.push_back( 2 * i + 1 );
        }
        return ref;
    }

    const int requested_matches = ( row_case.ref_len * row_case.match_percent ) / 100;
    const int match_count = std::min( requested_matches, static_cast<int>( curr.size() ) );
    appendSubsetMatches( curr, row_case.pattern == RowPattern::Subset ? row_case.ref_len : match_count, ref );

    int miss = 0;
    while ( static_cast<int>( ref.size() ) < row_case.ref_len )
    {
        ref.push_back( 2 * miss + 1 );
        ++miss;
    }

    std::sort( ref.begin(), ref.end() );
    ref.erase( std::unique( ref.begin(), ref.end() ), ref.end() );

    int tail = curr.empty() ? 1 : curr.back() + 1;
    while ( static_cast<int>( ref.size() ) < row_case.ref_len )
    {
        ref.push_back( tail );
        tail += 2;
    }
    std::sort( ref.begin(), ref.end() );
    return ref;
}

int countIntersectionHost( const std::vector<int>& curr, const std::vector<int>& ref )
{
    std::size_t curr_pos = 0;
    std::size_t ref_pos = 0;
    int count = 0;
    while ( curr_pos < curr.size() && ref_pos < ref.size() )
    {
        if ( curr[curr_pos] < ref[ref_pos] )
        {
            ++curr_pos;
        }
        else if ( ref[ref_pos] < curr[curr_pos] )
        {
            ++ref_pos;
        }
        else
        {
            ++count;
            ++curr_pos;
            ++ref_pos;
        }
    }
    return count;
}

template <typename T>
__device__ __forceinline__ T warpSum( T value )
{
    for ( int offset = kWarpSize / 2; offset > 0; offset /= 2 )
    {
        value += __shfl_down_sync( 0xffffffffu, value, offset );
    }
    return value;
}

__device__ __forceinline__ int binarySearchRow( const int target, const int* curr_cols, const int curr_len )
{
    int left = 0;
    int right = curr_len;
    while ( left < right )
    {
        const int mid = left + ( right - left ) / 2;
        const int col = curr_cols[mid];
        if ( col < target )
        {
            left = mid + 1;
        }
        else
        {
            right = mid;
        }
    }
    return ( left < curr_len && curr_cols[left] == target ) ? left : -1;
}

__global__ void rowMatchBinaryKernel( const int* curr_cols, int curr_len, const int* ref_cols, int ref_len, int repeat_warps, int* match_counts )
{
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp = global_thread / kWarpSize;
    const int lane = threadIdx.x & ( kWarpSize - 1 );
    if ( warp >= repeat_warps )
    {
        return;
    }

    int local_count = 0;
    for ( int ref_pos = lane; ref_pos < ref_len; ref_pos += kWarpSize )
    {
        local_count += binarySearchRow( ref_cols[ref_pos], curr_cols, curr_len ) >= 0 ? 1 : 0;
    }

    const int warp_count = warpSum( local_count );
    if ( lane == 0 )
    {
        match_counts[warp] = warp_count;
    }
}

__global__ void rowMatchMergeTiledKernel( const int* curr_cols,
                                          int curr_len,
                                          const int* ref_cols,
                                          int ref_len,
                                          int repeat_warps,
                                          int* match_counts )
{
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp = global_thread / kWarpSize;
    const int warp_in_block = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x & ( kWarpSize - 1 );
    if ( warp >= repeat_warps )
    {
        return;
    }

    extern __shared__ int shared_ref_cols[];
    int* ref_tile_cols = shared_ref_cols + warp_in_block * kWarpSize;
    int curr_head = 0;
    int ref_head = 0;
    int local_count = 0;

    while ( curr_head < curr_len && ref_head < ref_len )
    {
        const int curr_tile_end = min( curr_head + kWarpSize, curr_len );
        const int ref_tile_end = min( ref_head + kWarpSize, ref_len );
        const int ref_count = ref_tile_end - ref_head;

        const int ref_pos = ref_head + lane;
        if ( ref_pos < ref_tile_end )
        {
            ref_tile_cols[lane] = ref_cols[ref_pos];
        }
        __syncwarp();

        const int curr_pos = curr_head + lane;
        if ( curr_pos < curr_tile_end )
        {
            const int curr_col = curr_cols[curr_pos];
            for ( int ref_lane = 0; ref_lane < ref_count; ++ref_lane )
            {
                if ( curr_col == ref_tile_cols[ref_lane] )
                {
                    ++local_count;
                    break;
                }
            }
        }

        const int curr_last_col = curr_cols[curr_tile_end - 1];
        const int ref_last_col = ref_cols[ref_tile_end - 1];
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

    const int warp_count = warpSum( local_count );
    if ( lane == 0 )
    {
        match_counts[warp] = warp_count;
    }
}

void launchRowMatchKernel( const MatchKernel kernel,
                           const int* curr_cols,
                           const int curr_len,
                           const int* ref_cols,
                           const int ref_len,
                           const int repeat_warps,
                           int* match_counts )
{
    const int blocks = ( repeat_warps * kWarpSize + kThreadsPerBlock - 1 ) / kThreadsPerBlock;
    if ( kernel == MatchKernel::BinarySearch )
    {
        rowMatchBinaryKernel<<<blocks, kThreadsPerBlock>>>( curr_cols, curr_len, ref_cols, ref_len,
                                                            repeat_warps, match_counts );
    }
    else
    {
        const std::size_t shared_bytes =
            static_cast<std::size_t>( kWarpsPerBlock ) * kWarpSize * sizeof( int );
        rowMatchMergeTiledKernel<<<blocks, kThreadsPerBlock, shared_bytes>>>(
            curr_cols, curr_len, ref_cols, ref_len, repeat_warps, match_counts );
    }
    checkCuda( cudaGetLastError(), "launch row match kernel" );
}

void validateCounts( const DeviceBuffer<int>& counts, const int repeat_warps, const int expected_matches )
{
    std::vector<int> host_counts( static_cast<std::size_t>( repeat_warps ) );
    counts.copyToHost( host_counts );
    for ( int idx = 0; idx < repeat_warps; ++idx )
    {
        if ( host_counts[static_cast<std::size_t>( idx )] != expected_matches )
        {
            throw std::runtime_error( "row match kernel produced an unexpected count" );
        }
    }
}

void BM_RowMatch( benchmark::State& state, const RowCase row_case, const MatchKernel kernel )
{
    const std::vector<int> curr_cols = makeCurrentRow( row_case.curr_len );
    const std::vector<int> ref_cols = makeReferenceRow( row_case, curr_cols );
    const int expected_matches = countIntersectionHost( curr_cols, ref_cols );

    DeviceBuffer<int> d_curr( curr_cols.size() );
    DeviceBuffer<int> d_ref( ref_cols.size() );
    DeviceBuffer<int> d_counts( kRepeatWarps );
    d_curr.copyFromHost( curr_cols );
    d_ref.copyFromHost( ref_cols );

    launchRowMatchKernel( kernel, d_curr.data(), row_case.curr_len, d_ref.data(), row_case.ref_len,
                          kRepeatWarps, d_counts.data() );
    checkCuda( cudaDeviceSynchronize(), "synchronize row match warmup" );
    validateCounts( d_counts, kRepeatWarps, expected_matches );

    for ( auto _ : state )
    {
        launchRowMatchKernel( kernel, d_curr.data(), row_case.curr_len, d_ref.data(),
                              row_case.ref_len, kRepeatWarps, d_counts.data() );
        checkCuda( cudaDeviceSynchronize(), "synchronize row match benchmark" );
    }

    const double binary_work =
        static_cast<double>( row_case.ref_len ) * std::ceil( std::log2( row_case.curr_len ) );
    const double merge_work = static_cast<double>( row_case.curr_len + row_case.ref_len );
    state.counters["curr_len"] = row_case.curr_len;
    state.counters["ref_len"] = row_case.ref_len;
    state.counters["expected_matches"] = expected_matches;
    state.counters["repeat_warps"] = kRepeatWarps;
    state.counters["binary_cmp_est"] = binary_work;
    state.counters["merge_scan_est"] = merge_work;
    state.counters["row_pairs_per_second"] =
        benchmark::Counter( kRepeatWarps, benchmark::Counter::kIsIterationInvariantRate );
    state.SetItemsProcessed( state.iterations() * kRepeatWarps );
}

void registerCase( const RowCase row_case )
{
    const std::string base_name = std::string( row_case.name ) + "/curr_" +
                                  std::to_string( row_case.curr_len ) + "/ref_" +
                                  std::to_string( row_case.ref_len );
    const std::string binary_name = "RowMatch/binary_global/" + base_name;
    const std::string merge_name = "RowMatch/merge_tiled/" + base_name;

    benchmark::RegisterBenchmark( binary_name.c_str(), [row_case]( benchmark::State& state )
                                  { BM_RowMatch( state, row_case, MatchKernel::BinarySearch ); } )
        ->Unit( benchmark::kMicrosecond )
        ->UseRealTime();
    benchmark::RegisterBenchmark( merge_name.c_str(), [row_case]( benchmark::State& state )
                                  { BM_RowMatch( state, row_case, MatchKernel::MergeTiled ); } )
        ->Unit( benchmark::kMicrosecond )
        ->UseRealTime();
}

void registerBenchmarks()
{
    const RowCase cases[] = {
        // Tiny reference row, half hits. Binary search should benefit because merge still walks current-row tiles.
        { 64, 8, 50, RowPattern::Mixed, "tiny_ref_mixed_50" },
        // One-to-one identical rows. This is the best-case merge crossover test: every current column matches ref.
        { 64, 64, 100, RowPattern::OneToOne, "one_to_one" },
        // Equal-length rows with no matches but interleaved ranges. Merge cannot early-exit and must scan both rows.
        { 64, 64, 0, RowPattern::InterleavedMiss, "miss_interleaved" },

        // Tiny reference row against a larger current row. Tests the classic ref_len << curr_len binary advantage.
        { 256, 8, 50, RowPattern::Mixed, "tiny_ref_mixed_50" },
        // Short reference row mostly contained in current row. Tests whether merge can beat repeated searches when all ref entries hit.
        { 256, 64, 100, RowPattern::Subset, "short_ref_subset_100" },
        // Short reference row with sparse hits. Tests mixed hit/miss behavior with moderate current-row length.
        { 256, 64, 25, RowPattern::Mixed, "short_ref_mixed_25" },
        // One-to-one identical rows at 256 columns. If merge loses here, this tiled merge is not a good replacement.
        { 256, 256, 100, RowPattern::OneToOne, "one_to_one" },
        // Equal-length interleaved miss case. Tests worst-case merge scanning without useful updates.
        { 256, 256, 0, RowPattern::InterleavedMiss, "miss_interleaved" },

        // Very small reference row against a long current row. Binary should strongly favor logarithmic lookup.
        { 1024, 32, 25, RowPattern::Mixed, "tiny_ref_mixed_25" },
        // Medium reference subset of a long current row. Tests full-hit row matching without equal row lengths.
        { 1024, 256, 100, RowPattern::Subset, "medium_ref_subset_100" },
        // Medium reference row with sparse hits. Models ILU rows where only part of U(k,:) intersects row i.
        { 1024, 256, 25, RowPattern::Mixed, "medium_ref_mixed_25" },
        // One-to-one identical rows at 1024 columns. This is the main large-row merge best-case sanity check.
        { 1024, 1024, 100, RowPattern::OneToOne, "one_to_one" },
        // Equal-length interleaved miss case. Tests long merge scans with no arithmetic/update benefit.
        { 1024, 1024, 0, RowPattern::InterleavedMiss, "miss_interleaved" },
        // Reference row starts after current row. Merge should be able to advance current tiles and then stop.
        { 1024, 256, 0, RowPattern::DisjointAfter, "miss_disjoint_after" },

        // Very small reference row against a very long current row. Tests the strongest binary-search-favored shape.
        { 2048, 32, 25, RowPattern::Mixed, "tiny_ref_mixed_25" },
        // Medium reference subset of a very long current row. Tests all-hit matching with high current-row scan cost.
        { 2048, 512, 100, RowPattern::Subset, "medium_ref_subset_100" },
        // Medium reference row with sparse hits. Models larger ILU update rows with partial overlap.
        { 2048, 512, 25, RowPattern::Mixed, "medium_ref_mixed_25" },
        // One-to-one identical rows at 2048 columns. If merge loses here, binary is preferred for this implementation.
        { 2048, 2048, 100, RowPattern::OneToOne, "one_to_one" },
        // Equal-length interleaved miss case. Tests long rows where merge does maximum scan work and finds no updates.
        { 2048, 2048, 0, RowPattern::InterleavedMiss, "miss_interleaved" },
        // Reference row starts after current row. Tests merge early-exit behavior on disjoint sorted ranges.
        { 2048, 512, 0, RowPattern::DisjointAfter, "miss_disjoint_after" },
    };

    for ( const RowCase row_case : cases )
    {
        registerCase( row_case );
    }
}
} // namespace

int main( int argc, char** argv )
{
    registerBenchmarks();
    benchmark::Initialize( &argc, argv );
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
