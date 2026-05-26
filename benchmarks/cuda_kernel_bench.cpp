#include "cuda_kernels.cuh"
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>

using namespace matrix_utils::sparse_cuda;

// Helper function to check CUDA errors
static void checkCudaError( cudaError_t error, const char* message )
{
    if ( error != cudaSuccess )
    {
        throw std::runtime_error( std::string( "CUDA error: " ) + message + " - " + cudaGetErrorString( error ) );
    }
}

// Benchmark template for different items_per_thread values
template <int items_per_thread>
static void BM_ElementwiseMultiply( benchmark::State& state )
{
    const size_t n = state.range( 0 );

    // Allocate and initialize host vectors
    std::vector<double> h_a( n );
    std::vector<double> h_b( n );
    std::vector<double> h_output( n );

    std::random_device rd;
    std::mt19937 gen( 42 ); // Fixed seed for reproducibility
    std::uniform_real_distribution<double> dis( -1.0, 1.0 );

    for ( size_t i = 0; i < n; ++i )
    {
        h_a[i] = dis( gen );
        h_b[i] = dis( gen );
    }

    // Allocate device memory
    double *d_a, *d_b, *d_output;
    checkCudaError( cudaMalloc( &d_a, n * sizeof( double ) ), "Failed to allocate d_a" );
    checkCudaError( cudaMalloc( &d_b, n * sizeof( double ) ), "Failed to allocate d_b" );
    checkCudaError( cudaMalloc( &d_output, n * sizeof( double ) ), "Failed to allocate d_output" );

    // Copy data to device
    checkCudaError( cudaMemcpy( d_a, h_a.data(), n * sizeof( double ), cudaMemcpyHostToDevice ),
                    "Failed to copy d_a to device" );
    checkCudaError( cudaMemcpy( d_b, h_b.data(), n * sizeof( double ), cudaMemcpyHostToDevice ),
                    "Failed to copy d_b to device" );

    // Warm-up
    elementwiseMultiply<items_per_thread>( d_a, d_b, d_output, n );
    cudaDeviceSynchronize();

    // Benchmark loop
    for ( auto _ : state )
    {
        elementwiseMultiply<items_per_thread>( d_a, d_b, d_output, n );
        cudaDeviceSynchronize();
    }

    // Set items processed
    state.SetItemsProcessed( state.iterations() * n );
    state.SetBytesProcessed( state.iterations() * n * 3 * sizeof( double ) ); // 2 reads + 1 write

    // Cleanup
    cudaFree( d_a );
    cudaFree( d_b );
    cudaFree( d_output );
}

// Register benchmarks: for each size, test all items_per_thread values from 1 to 16
// This creates output like:
// BM_ElementwiseMultiply<1>/1024
// BM_ElementwiseMultiply<2>/1024
// BM_ElementwiseMultiply<4>/1024
// ...
// BM_ElementwiseMultiply<1>/2048
// BM_ElementwiseMultiply<2>/2048
// ...

#define REGISTER_BENCHMARK_FOR_SIZE( size, unit )                               \
    BENCHMARK_TEMPLATE( BM_ElementwiseMultiply, 1 )->Arg( size )->Unit( unit ); \
    BENCHMARK_TEMPLATE( BM_ElementwiseMultiply, 2 )->Arg( size )->Unit( unit ); \
    BENCHMARK_TEMPLATE( BM_ElementwiseMultiply, 4 )->Arg( size )->Unit( unit ); \
    BENCHMARK_TEMPLATE( BM_ElementwiseMultiply, 8 )->Arg( size )->Unit( unit ); \
    BENCHMARK_TEMPLATE( BM_ElementwiseMultiply, 16 )->Arg( size )->Unit( unit );

// Small sizes (1K - 128K)
REGISTER_BENCHMARK_FOR_SIZE( 1 << 10, benchmark::kMicrosecond ); // 1K
REGISTER_BENCHMARK_FOR_SIZE( 1 << 11, benchmark::kMicrosecond ); // 2K
REGISTER_BENCHMARK_FOR_SIZE( 1 << 12, benchmark::kMicrosecond ); // 4K
REGISTER_BENCHMARK_FOR_SIZE( 1 << 13, benchmark::kMicrosecond ); // 8K
REGISTER_BENCHMARK_FOR_SIZE( 1 << 14, benchmark::kMicrosecond ); // 16K
REGISTER_BENCHMARK_FOR_SIZE( 1 << 15, benchmark::kMicrosecond ); // 32K
REGISTER_BENCHMARK_FOR_SIZE( 1 << 16, benchmark::kMicrosecond ); // 64K
REGISTER_BENCHMARK_FOR_SIZE( 1 << 17, benchmark::kMicrosecond ); // 128K

// Medium sizes (256K - 8M)
REGISTER_BENCHMARK_FOR_SIZE( 1 << 18, benchmark::kMicrosecond ); // 256K
REGISTER_BENCHMARK_FOR_SIZE( 1 << 19, benchmark::kMicrosecond ); // 512K
REGISTER_BENCHMARK_FOR_SIZE( 1 << 20, benchmark::kMicrosecond ); // 1M
REGISTER_BENCHMARK_FOR_SIZE( 1 << 21, benchmark::kMicrosecond ); // 2M
REGISTER_BENCHMARK_FOR_SIZE( 1 << 22, benchmark::kMicrosecond ); // 4M
REGISTER_BENCHMARK_FOR_SIZE( 1 << 23, benchmark::kMicrosecond ); // 8M

// Large sizes (16M - 128M)
REGISTER_BENCHMARK_FOR_SIZE( 1 << 24, benchmark::kMillisecond ); // 16M
REGISTER_BENCHMARK_FOR_SIZE( 1 << 25, benchmark::kMillisecond ); // 32M
REGISTER_BENCHMARK_FOR_SIZE( 1 << 26, benchmark::kMillisecond ); // 64M
REGISTER_BENCHMARK_FOR_SIZE( 1 << 27, benchmark::kMillisecond ); // 128M

BENCHMARK_MAIN();
