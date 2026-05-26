#include "cuda_csr_utils.cuh"
#include "io.hpp"
#include "matrix_utils.hpp"
#include "utils.h"
#include <benchmark/benchmark.h>
#include <cxxopts.hpp>
#include <cuda_runtime.h>
#include <omp.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace matrix_utils;
namespace cuda_utils = matrix_utils::sparse_cuda;

// Global variables to hold the matrix data
static std::string g_matrix_file = "data/nv2.mtx";
static std::vector<int> g_ai, g_aj;
static std::vector<double> g_av;
static int g_rows = 0;
static int g_base = 0;
static int g_original_nnz = 0;

// Helper function to check CUDA errors
static void checkCudaError( cudaError_t error, const char* message )
{
    if ( error != cudaSuccess )
    {
        throw std::runtime_error( std::string( "CUDA error: " ) + message + " - " + cudaGetErrorString( error ) );
    }
}

template <typename T>
class CudaBuffer
{
public:
    explicit CudaBuffer( std::size_t size ) : size_( size )
    {
        if ( size_ > 0 )
        {
            checkCudaError( cudaMalloc( &data_, size_ * sizeof( T ) ), "cudaMalloc" );
        }
    }

    ~CudaBuffer()
    {
        if ( data_ != nullptr )
        {
            cudaFree( data_ );
        }
    }

    CudaBuffer( const CudaBuffer& ) = delete;
    CudaBuffer& operator=( const CudaBuffer& ) = delete;

    T* data() { return data_; }
    const T* data() const { return data_; }

    void copy_from( const std::vector<T>& source )
    {
        if ( source.size() > size_ )
        {
            throw std::runtime_error( "CUDA buffer copy source is larger than allocation" );
        }
        if ( !source.empty() )
        {
            checkCudaError( cudaMemcpy( data_, source.data(), source.size() * sizeof( T ), cudaMemcpyHostToDevice ),
                            "cudaMemcpy host to device" );
        }
    }

private:
    T* data_ = nullptr;
    std::size_t size_ = 0;
};

// CPU version benchmark
static void BM_DiagonalScaledPrune_CPU( benchmark::State& state )
{
    const double threshold = state.range( 0 ) / 1000.0; // Convert from integer millis to double

    // Make a copy for each iteration
    for ( auto _ : state )
    {
        std::vector<int> ai_cpu = g_ai;
        std::vector<int> aj_cpu = g_aj;
        std::vector<double> av_cpu = g_av;

        state.PauseTiming();
        // Ensure copy is complete before timing
        benchmark::DoNotOptimize( ai_cpu.data() );
        benchmark::DoNotOptimize( aj_cpu.data() );
        benchmark::DoNotOptimize( av_cpu.data() );
        state.ResumeTiming();

        int removed = DiagonalScaledPrune( g_rows, ai_cpu.data(), aj_cpu.data(), av_cpu.data(), threshold );

        benchmark::DoNotOptimize( removed );
        benchmark::ClobberMemory();
    }

    state.counters["Rows"] = g_rows;
    state.counters["OrigNNZ"] = g_original_nnz;
    state.counters["Threshold"] = threshold;
}

// GPU version benchmark
static void BM_DiagonalScaledPrune_GPU( benchmark::State& state )
{
    const double threshold = state.range( 0 ) / 1000.0; // Convert from integer millis to double

    // Allocate device memory once
    CudaBuffer<int> d_ai( g_ai.size() );
    CudaBuffer<int> d_aj( g_aj.size() );
    CudaBuffer<double> d_av( g_av.size() );
    CudaBuffer<int> d_mask( g_original_nnz );
    CudaBuffer<int> d_ai_out( g_rows + 1 );
    CudaBuffer<int> d_aj_out( g_original_nnz );
    CudaBuffer<double> d_av_out( g_original_nnz );
    d_ai.copy_from( g_ai );
    d_aj.copy_from( g_aj );
    d_av.copy_from( g_av );

    // Warm-up
    cuda_utils::CSRGenDiagScaledPruneMask( g_rows, d_ai.data(), d_aj.data(), d_av.data(), threshold,
                                           d_mask.data() );
    cudaDeviceSynchronize();

    for ( auto _ : state )
    {
        // Reset input data
        state.PauseTiming();
        d_ai.copy_from( g_ai );
        d_aj.copy_from( g_aj );
        d_av.copy_from( g_av );
        cudaDeviceSynchronize();
        state.ResumeTiming();

        // Step 1: Generate mask
        cuda_utils::CSRGenDiagScaledPruneMask( g_rows, d_ai.data(), d_aj.data(), d_av.data(),
                                               threshold, d_mask.data() );

        // Step 2: Apply mask
        int removed = cuda_utils::CSRSelectByMaskDevice( g_rows, d_ai.data(), d_aj.data(),
                                                         d_av.data(), d_mask.data(), d_ai_out.data(),
                                                         d_aj_out.data(), d_av_out.data() );

        cudaDeviceSynchronize();
        benchmark::DoNotOptimize( removed );
    }

    state.counters["Rows"] = g_rows;
    state.counters["OrigNNZ"] = g_original_nnz;
    state.counters["Threshold"] = threshold;
}

// GPU mask generation only benchmark
static void BM_DiagonalScaledPrune_GPU_MaskOnly( benchmark::State& state )
{
    const double threshold = state.range( 0 ) / 1000.0;

    CudaBuffer<int> d_ai( g_ai.size() );
    CudaBuffer<int> d_aj( g_aj.size() );
    CudaBuffer<double> d_av( g_av.size() );
    CudaBuffer<int> d_mask( g_original_nnz );
    d_ai.copy_from( g_ai );
    d_aj.copy_from( g_aj );
    d_av.copy_from( g_av );

    // Warm-up
    cuda_utils::CSRGenDiagScaledPruneMask( g_rows, d_ai.data(), d_aj.data(), d_av.data(), threshold,
                                           d_mask.data() );
    cudaDeviceSynchronize();

    for ( auto _ : state )
    {
        cuda_utils::CSRGenDiagScaledPruneMask( g_rows, d_ai.data(), d_aj.data(), d_av.data(),
                                               threshold, d_mask.data() );

        cudaDeviceSynchronize();
    }

    state.counters["Rows"] = g_rows;
    state.counters["OrigNNZ"] = g_original_nnz;
    state.counters["Threshold"] = threshold;
}

// GPU mask application only benchmark
static void BM_DiagonalScaledPrune_GPU_SelectOnly( benchmark::State& state )
{
    const double threshold = state.range( 0 ) / 1000.0;

    CudaBuffer<int> d_ai( g_ai.size() );
    CudaBuffer<int> d_aj( g_aj.size() );
    CudaBuffer<double> d_av( g_av.size() );
    CudaBuffer<int> d_mask( g_original_nnz );
    CudaBuffer<int> d_ai_out( g_rows + 1 );
    CudaBuffer<int> d_aj_out( g_original_nnz );
    CudaBuffer<double> d_av_out( g_original_nnz );
    d_ai.copy_from( g_ai );
    d_aj.copy_from( g_aj );
    d_av.copy_from( g_av );

    // Generate mask once
    cuda_utils::CSRGenDiagScaledPruneMask( g_rows, d_ai.data(), d_aj.data(), d_av.data(), threshold,
                                           d_mask.data() );
    cudaDeviceSynchronize();

    for ( auto _ : state )
    {
        int removed = cuda_utils::CSRSelectByMaskDevice( g_rows, d_ai.data(), d_aj.data(),
                                                         d_av.data(), d_mask.data(), d_ai_out.data(),
                                                         d_aj_out.data(), d_av_out.data() );

        cudaDeviceSynchronize();
        benchmark::DoNotOptimize( removed );
    }

    state.counters["Rows"] = g_rows;
    state.counters["OrigNNZ"] = g_original_nnz;
    state.counters["Threshold"] = threshold;
}

// Register benchmarks for different threshold values (in millis: 1, 5, 10, 50, 100 = 0.001, 0.005, 0.01, 0.05, 0.1)
BENCHMARK( BM_DiagonalScaledPrune_CPU )->Arg( 1 )->Arg( 5 )->Arg( 10 )->Arg( 50 )->Arg( 100 )->Unit( benchmark::kMillisecond );
BENCHMARK( BM_DiagonalScaledPrune_GPU )->Arg( 1 )->Arg( 5 )->Arg( 10 )->Arg( 50 )->Arg( 100 )->Unit( benchmark::kMillisecond );
BENCHMARK( BM_DiagonalScaledPrune_GPU_MaskOnly )
    ->Arg( 1 )
    ->Arg( 5 )
    ->Arg( 10 )
    ->Arg( 50 )
    ->Arg( 100 )
    ->Unit( benchmark::kMicrosecond );
BENCHMARK( BM_DiagonalScaledPrune_GPU_SelectOnly )
    ->Arg( 1 )
    ->Arg( 5 )
    ->Arg( 10 )
    ->Arg( 50 )
    ->Arg( 100 )
    ->Unit( benchmark::kMicrosecond );

int main( int argc, char** argv )
{
    // Parse command line arguments
    cxxopts::Options options( "cuda_diagonal_prune_bench",
                              "Benchmark diagonal scaled prune CPU vs GPU" );
    options.add_options()( "f,file", "Matrix file path (MTX format)",
                           cxxopts::value<std::string>()->default_value( "data/nv2.mtx" ) )(
        "t,threads", "Number of OpenMP threads for CPU",
        cxxopts::value<int>()->default_value( "8" ) )( "h,help", "Print usage" );

    auto result = options.parse( argc, argv );

    if ( result.count( "help" ) )
    {
        std::cout << options.help() << std::endl;
        return 0;
    }

    g_matrix_file = result["file"].as<std::string>();
    int num_threads = result["threads"].as<int>();

    // Set OpenMP thread count
    omp_set_num_threads( num_threads );
    std::cout << "Using " << num_threads << " OpenMP threads for CPU benchmarks" << std::endl;

    // Load matrix
    std::cout << "Loading matrix from: " << g_matrix_file << std::endl;
    std::ifstream f( g_matrix_file );
    if ( !f.is_open() )
    {
        std::cerr << "Error: Could not open matrix file: " << g_matrix_file << std::endl;
        return 1;
    }

    matrix_utils::readMatrixMarket( f, g_ai, g_aj, g_av );
    f.close();

    g_rows = g_ai.size() - 1;
    g_base = 0;
    g_original_nnz = g_ai[g_rows] - g_base;

    std::cout << "Matrix loaded successfully:" << std::endl;
    std::cout << "  Rows: " << g_rows << std::endl;
    std::cout << "  NNZ: " << g_original_nnz << std::endl;
    std::cout << "  Avg NNZ/row: " << (double)g_original_nnz / g_rows << std::endl;
    std::cout << std::endl;

    // Initialize Google Benchmark with remaining arguments
    // Filter out our custom arguments
    std::vector<char*> bench_argv;
    bench_argv.push_back( argv[0] );
    for ( int i = 1; i < argc; ++i )
    {
        std::string arg( argv[i] );
        if ( arg.find( "--file" ) == std::string::npos && arg.find( "-f" ) == std::string::npos &&
             arg.find( "--threads" ) == std::string::npos && arg.find( "-t" ) == std::string::npos )
        {
            bench_argv.push_back( argv[i] );
        }
    }
    int bench_argc = bench_argv.size();

    ::benchmark::Initialize( &bench_argc, bench_argv.data() );
    if ( ::benchmark::ReportUnrecognizedArguments( bench_argc, bench_argv.data() ) )
    {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    return 0;
}
