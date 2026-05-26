#include "utils.h"
#include <algorithm>
#include <benchmark/benchmark.h>
#include <numeric>
#include <omp.h>
#include <vector>

namespace
{
// Measure utils::ParallelPrefixSum; d_first[0] is the base (0/1) per API contract.
void BM_ParallelPrefixSum( benchmark::State& state )
{
    const int size = static_cast<int>( state.range( 0 ) );
    const int base = static_cast<int>( state.range( 1 ) );
    const int nthreads = 8;

    std::vector<int> input( size, 1 );
    std::vector<int> output( static_cast<size_t>( size ) + 1 );

    for ( auto _ : state )
    {
        output[0] = base;
        benchmark::DoNotOptimize( utils::ParallelPrefixSum(
            nthreads, input.data(), input.data() + input.size(), output.data() ) );
        benchmark::ClobberMemory();
    }
}

// Measure std::inclusive_scan with the same base semantics (output[0] is base).
void BM_InclusiveScan( benchmark::State& state )
{
    const int size = static_cast<int>( state.range( 0 ) );
    const int base = static_cast<int>( state.range( 1 ) );

    std::vector<int> input( size, 1 );
    std::vector<int> output( static_cast<size_t>( size ) + 1 );

    for ( auto _ : state )
    {
        output[0] = base;
        std::inclusive_scan( input.begin(), input.end(), output.begin() + 1, std::plus<>{}, base );
        benchmark::ClobberMemory();
    }
}

// Measure in-place prefix sum (inclusive) on the same data pattern.
void BM_ParallelPrefixSumInplace( benchmark::State& state )
{
    const int size = static_cast<int>( state.range( 0 ) );
    const int nthreads = 8;

    std::vector<int> data( size, 1 );

    for ( auto _ : state )
    {
        std::fill( data.begin(), data.end(), 1 );
        benchmark::DoNotOptimize(
            utils::ParallelPrefixSumInplace( nthreads, data.data(), data.data() + data.size() ) );
        benchmark::ClobberMemory();
    }
}
} // namespace

BENCHMARK( BM_ParallelPrefixSum )
    ->Args( { 1'000, 0 } )
    ->Args( { 1'000, 1 } )
    ->Args( { 100'000, 0 } )
    ->Args( { 100'000, 1 } )
    ->Args( { 1'000'000, 0 } )
    ->Args( { 1'000'000, 1 } )
    ->Args( { 10'000'000, 0 } )
    ->Args( { 10'000'000, 1 } );

BENCHMARK( BM_InclusiveScan )
    ->Args( { 1'000, 0 } )
    ->Args( { 1'000, 1 } )
    ->Args( { 100'000, 0 } )
    ->Args( { 100'000, 1 } )
    ->Args( { 1'000'000, 0 } )
    ->Args( { 1'000'000, 1 } )
    ->Args( { 10'000'000, 0 } )
    ->Args( { 10'000'000, 1 } );

BENCHMARK( BM_ParallelPrefixSumInplace )->Arg( 1'000 )->Arg( 100'000 )->Arg( 1'000'000 )->Arg( 10'000'000 );

BENCHMARK_MAIN();
