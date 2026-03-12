#include "matrix_utils.hpp"
#include "sp_ops.hpp"
#include "utils.h"
#include <benchmark/benchmark.h>
#include <cxxopts.hpp>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

using CSRTYPE = matrix_utils::CSRMatrixVec<int32_t, int32_t, double>;

struct BenchInputs
{
    CSRTYPE A;
    int32_t size{0};
    int32_t base{0};
    int32_t nnz{0};
    std::vector<int32_t> ai_serial;
    std::vector<int32_t> aj_serial;
};

static BenchInputs loadMatrix( const std::string& path )
{
    BenchInputs bi;
    std::ifstream f( path );
    if ( !f.good() )
    {
        throw std::runtime_error( "Cannot open matrix file: " + path );
    }

    utils::read_matrix_market_csr( f, bi.A.ai, bi.A.aj, bi.A.av );
    bi.A.rows = static_cast<int32_t>( bi.A.ai.size() ) - 1;
    bi.A.cols = bi.A.rows;
    bi.size = bi.A.rows;
    bi.base = bi.A.ai.empty() ? 0 : bi.A.ai[0];
    bi.nnz = bi.A.ai.empty() ? 0 : bi.A.ai[bi.size] - bi.base;

    // Allocate buffers with a generous upper bound (2x nnz)
    bi.ai_serial.resize( bi.size + 1 );
    bi.aj_serial.resize( std::max<int32_t>( 1, 2 * bi.nnz ) );

    return bi;
}

static void BM_APlusATSerial( benchmark::State& state, BenchInputs* bi )
{
    for ( auto _ : state )
    {
        matrix_utils::APlusATSerial<int32_t, int32_t, true>(
            bi->size, bi->A.ai.data(), bi->A.aj.data(), bi->ai_serial.data(),
            bi->aj_serial.data() );
        benchmark::DoNotOptimize( bi->ai_serial );
        benchmark::DoNotOptimize( bi->aj_serial );
    }
    state.SetItemsProcessed( int64_t( state.iterations() ) * bi->nnz );
}

static void BM_APlusATStruct( benchmark::State& state, BenchInputs* bi )
{
    const int threads = static_cast<int>( state.range( 0 ) );
    matrix_utils::APlusATStruct<int32_t, int32_t, true> op( threads );
    std::vector<int32_t> ai_out( bi->size + 1 );
    std::vector<int32_t> aj_out( std::max<int32_t>( 1, 2 * bi->nnz ) );

    for ( auto _ : state )
    {
        op( bi->size, bi->A.ai.data(), bi->A.aj.data(), ai_out.data(),
            aj_out.data() );
        benchmark::DoNotOptimize( ai_out );
        benchmark::DoNotOptimize( aj_out );
    }
    state.SetItemsProcessed( int64_t( state.iterations() ) * bi->nnz );
}

static void BM_APlusATPrefix( benchmark::State& state, BenchInputs* bi )
{
    std::vector<int32_t> ai_out( bi->size + 1 );
    for ( auto _ : state )
    {
        matrix_utils::APlusATPrefix<int32_t, int32_t, true>(
            bi->size, bi->A.ai.data(), bi->A.aj.data(), ai_out.data() );
        benchmark::DoNotOptimize( ai_out );
    }
    state.SetItemsProcessed( int64_t( state.iterations() ) * bi->nnz );
}

static void BM_PrefixStruct( benchmark::State& state, BenchInputs* bi )
{
    const int threads = static_cast<int>( state.range( 0 ) );
    matrix_utils::APlusATStruct<int32_t, int32_t, true> op( threads );
    std::vector<int32_t> ai_out( bi->size + 1 );
    for ( auto _ : state )
    {
        op.prefixOnly( bi->size, bi->A.ai.data(), bi->A.aj.data(), ai_out.data() );
        benchmark::DoNotOptimize( ai_out );
    }
    state.SetItemsProcessed( int64_t( state.iterations() ) * bi->nnz );
}

static void BM_APlusATFill( benchmark::State& state,
                            BenchInputs* bi,
                            const std::vector<int32_t>& ai_prefix )
{
    std::vector<int32_t> aj_out( std::max<int32_t>( 1, 2 * bi->nnz ) );
    for ( auto _ : state )
    {
        matrix_utils::APlusATFill<int32_t, int32_t, true>(
            bi->size, bi->A.ai.data(), bi->A.aj.data(), ai_prefix.data(),
            aj_out.data() );
        benchmark::DoNotOptimize( aj_out );
    }
    state.SetItemsProcessed( int64_t( state.iterations() ) * bi->nnz );
}

static void BM_FillAndCompact( benchmark::State& state,
                               BenchInputs* bi,
                               matrix_utils::APlusATStruct<int32_t, int32_t, true>* op )
{
    const int threads = static_cast<int>( state.range( 0 ) );
    op->setNumThreads( threads );

    // Assumes prefixOnly was already called to initialize internal buffers
    std::vector<int32_t> ai_out( bi->size + 1 );
    std::vector<int32_t> aj_out( std::max<int32_t>( 1, 2 * bi->nnz ) );

    for ( auto _ : state )
    {
        op->fillAndCompactOnly( bi->size, bi->A.ai.data(), bi->A.aj.data(),
                                ai_out.data(), aj_out.data() );
        benchmark::DoNotOptimize( ai_out );
        benchmark::DoNotOptimize( aj_out );
    }
    state.SetItemsProcessed( int64_t( state.iterations() ) * bi->nnz );
}

int main( int argc, char** argv )
{
    cxxopts::Options options( "sp_ops_bench", "Benchmark A+A^T implementations" );
    options.allow_unrecognised_options().add_options()(
        "m,matrix", "Matrix Market file path",
        cxxopts::value<std::string>()->default_value( "data/mcfe.mtx" ) )(
        "h,help", "Print usage" );

    auto result = options.parse( argc, argv );
    if ( result.count( "help" ) )
    {
        std::cout << options.help() << std::endl;
        return 0;
    }

    BenchInputs bi = loadMatrix( result["m"].as<std::string>() );
    std::cout << "Matrix loaded: rows=" << bi.size << " nnz=" << bi.nnz
              << " base=" << bi.base << std::endl;

    const int thread_cases[] = { 1, 2, 4, 8 };

    // Precompute prefix for fill benchmarks
    std::vector<int32_t> ai_prefix( bi.size + 1 );
    matrix_utils::APlusATPrefix<int32_t, int32_t, true>( bi.size, bi.A.ai.data(),
                                                         bi.A.aj.data(), ai_prefix.data() );

    // Prepare struct with prefix pre-run
    matrix_utils::APlusATStruct<int32_t, int32_t, true> fill_struct( 1 );
    fill_struct.prefixOnly( bi.size, bi.A.ai.data(), bi.A.aj.data(), nullptr );

    // Benchmark 1: Full pipeline (serial vs struct)
    benchmark::RegisterBenchmark( "APlusATSerial", BM_APlusATSerial, &bi );
    for ( int t : thread_cases )
    {
        benchmark::RegisterBenchmark( ("APlusATStruct/threads_" + std::to_string( t )).c_str(),
                                      BM_APlusATStruct, &bi )
            ->Arg( t );
    }

    // Benchmark 2: Prefix only
    benchmark::RegisterBenchmark( "APlusATPrefix", BM_APlusATPrefix, &bi );
    for ( int t : thread_cases )
    {
        benchmark::RegisterBenchmark( ("PrefixStruct/threads_" + std::to_string( t )).c_str(),
                                      BM_PrefixStruct, &bi )
            ->Arg( t );
    }

    // Benchmark 3: Fill only (prefix precomputed)
    benchmark::RegisterBenchmark( "APlusATFill", BM_APlusATFill, &bi, ai_prefix );
    for ( int t : thread_cases )
    {
        benchmark::RegisterBenchmark( ("FillAndCompact/threads_" + std::to_string( t )).c_str(),
                                      BM_FillAndCompact, &bi, &fill_struct )
            ->Arg( t );
    }

    benchmark::Initialize( &argc, argv );
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
