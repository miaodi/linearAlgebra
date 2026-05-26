#include "permutation.hpp"
#include <algorithm>
#include <atomic>
#include <benchmark/benchmark.h>
#include <memory>
#include <numeric>
#include <omp.h>
#include <vector>

static std::map<int, std::vector<int>> perm_map;
class MyFixture : public benchmark::Fixture
{
public:
    MyFixture() {}
};

BENCHMARK_DEFINE_F( MyFixture, isPermutation )( benchmark::State& state )
{
    omp_set_num_threads( 8 );
    int size = state.range( 0 );
    auto perm_pair = perm_map.try_emplace( size, size );
    if ( perm_pair.second )
    {
        matrix_utils::randPerm( size, 0, perm_pair.first->second.data() );
    }
    for ( auto _ : state )
    {
        benchmark::DoNotOptimize( matrix_utils::isPermutation( size, 0, perm_pair.first->second.data() ) );
    }
}

BENCHMARK_REGISTER_F( MyFixture, isPermutation )
    ->Arg( 100 )
    ->Arg( 1000 )
    ->Arg( 10000 )
    ->Arg( 100000 )
    ->Arg( 1000000 )
    ->Arg( 10000000 );

BENCHMARK_DEFINE_F( MyFixture, isPermutation_Serial )( benchmark::State& state )
{
    int size = state.range( 0 );
    auto perm_pair = perm_map.try_emplace( size, size );
    if ( perm_pair.second )
    {
        matrix_utils::randPerm( size, 0, perm_pair.first->second.data() );
    }
    for ( auto _ : state )
    {
        benchmark::DoNotOptimize(
            matrix_utils::isPermutationSerial( size, 0, perm_pair.first->second.data() ) );
    }
}

BENCHMARK_REGISTER_F( MyFixture, isPermutation_Serial )
    ->Arg( 100 )
    ->Arg( 1000 )
    ->Arg( 10000 )
    ->Arg( 100000 )
    ->Arg( 1000000 )
    ->Arg( 10000000 );

BENCHMARK_MAIN();