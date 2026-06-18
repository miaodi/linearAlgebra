#include "io.hpp"
#include "matrix_utils.hpp"
#include "spmv.hpp"
#include "utils.h"
#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <cxxopts.hpp>
#include <fstream>
#include <memory>
#include <new>
#include <numeric>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef USE_MKL
#include <mkl.h>
#endif

#ifdef USE_CUDA
#include "cuda_spmv.cuh"
#include <cuda_runtime.h>
#endif

#ifdef LINEAR_ALGEBRA_ENABLE_CSR5_REFERENCE_BENCH
#include "anonymouslib_avx2.h"
#endif

using CSRTYPE_DOUBLE = typename matrix_utils::CSRMatrixVec<int, int, double>;
using CSRTYPE_FLOAT = typename matrix_utils::CSRMatrixVec<int, int, float>;

template <typename CSRMatrixType>
int64_t csrSpmvBytesPerIteration( const CSRMatrixType& mat )
{
    using ROWTYPE = typename CSRMatrixType::ROWTYPE;
    using COLTYPE = typename CSRMatrixType::COLTYPE;
    using VALTYPE = typename CSRMatrixType::VALTYPE;

    const int64_t rows = int64_t( mat.rows );
    const int64_t nnz = int64_t( mat.NNZ() );
    return nnz * ( sizeof( VALTYPE ) + sizeof( COLTYPE ) + sizeof( VALTYPE ) ) +
           ( rows + 1 ) * sizeof( ROWTYPE ) + rows * sizeof( VALTYPE );
}

template <typename CSRMatrixType>
void setSpmvBytesProcessed( benchmark::State& state, const CSRMatrixType& mat, const int it )
{
    state.SetBytesProcessed( int64_t( state.iterations() ) * int64_t( it ) * csrSpmvBytesPerIteration( mat ) );
}

template <typename Fn, typename... Args>
void registerSpmvBenchmark( const char* name, Fn&& fn, Args&&... args )
{
    benchmark::RegisterBenchmark( name, std::forward<Fn>( fn ), std::forward<Args>( args )... )->UseRealTime();
}

template <typename VALTYPE>
auto Serial = []( benchmark::State& state, const auto& mat, const int threads, const int it )
{
    std::vector<VALTYPE> x( mat.rows, 0.0 );
    std::vector<VALTYPE> b( mat.rows, 1.0 );

    matrix_utils::SPMV<std::remove_cvref_t<decltype( mat )>, matrix_utils::SerialSPMV> spmv;
    spmv.setMatrix( &mat );
    spmv.preprocess();
    for ( auto _ : state )
    {
        for ( int i = 0; i < it; i++ )
        {
            spmv( b.data(), x.data() );
        }
    }
    setSpmvBytesProcessed( state, mat, it );
};

template <typename VALTYPE>
auto Parallel = []( benchmark::State& state, const auto& mat, const int threads, const int it )
{
    std::vector<VALTYPE> x( mat.rows, 0.0 );
    std::vector<VALTYPE> b( mat.rows, 1.0 );

    matrix_utils::SPMV<std::remove_cvref_t<decltype( mat )>, matrix_utils::ParallelSPMV<int, int, VALTYPE>> spmv;
    spmv.setMatrix( &mat );
    spmv._spmv.setNumThreads( threads );
    spmv.preprocess();
    for ( auto _ : state )
    {
        for ( int i = 0; i < it; i++ )
        {
            spmv( b.data(), x.data() );
        }
    }
    setSpmvBytesProcessed( state, mat, it );
};

template <typename VALTYPE>
auto RowBalanced = []( benchmark::State& state, const auto& mat, const int threads, const int it )
{
    std::vector<VALTYPE> x( mat.rows, 0.0 );
    std::vector<VALTYPE> b( mat.rows, 1.0 );

    matrix_utils::SPMV<std::remove_cvref_t<decltype( mat )>, matrix_utils::RowBalancedParallelSPMV<int, int, VALTYPE>> spmv;
    spmv.setMatrix( &mat );
    spmv._spmv.setNumThreads( threads );
    spmv.preprocess();
    for ( auto _ : state )
    {
        for ( int i = 0; i < it; i++ )
        {
            spmv( b.data(), x.data() );
        }
    }
    setSpmvBytesProcessed( state, mat, it );
};

template <typename VALTYPE>
auto RowBalancedSimd = []( benchmark::State& state, const auto& mat, const int threads, const int it )
{
    std::vector<VALTYPE> x( mat.rows, 0.0 );
    std::vector<VALTYPE> b( mat.rows, 1.0 );

    matrix_utils::SPMV<std::remove_cvref_t<decltype( mat )>, matrix_utils::RowBalancedParallelSPMV<int, int, VALTYPE, matrix_utils::RowDotKernel::Simd>> spmv;
    spmv.setMatrix( &mat );
    spmv._spmv.setNumThreads( threads );
    spmv.preprocess();
    for ( auto _ : state )
    {
        for ( int i = 0; i < it; i++ )
        {
            spmv( b.data(), x.data() );
        }
    }
    setSpmvBytesProcessed( state, mat, it );
};

template <typename VALTYPE>
auto ALBUSSum = []( benchmark::State& state, const auto& mat, const int threads, const int it )
{
    std::vector<VALTYPE> x( mat.rows, 0.0 );
    std::vector<VALTYPE> b( mat.rows, 1.0 );

    matrix_utils::SPMV<std::remove_cvref_t<decltype( mat )>, matrix_utils::ALBUSSPMV<int32_t, int32_t, VALTYPE>> spmv;
    spmv.setMatrix( &mat );
    spmv._spmv.setNumThreads( threads );
    spmv.preprocess();
    for ( auto _ : state )
    {
        for ( int i = 0; i < it; i++ )
        {
            spmv( b.data(), x.data() );
        }
    }
    setSpmvBytesProcessed( state, mat, it );
};

template <typename VALTYPE>
auto ALBUSSimd = []( benchmark::State& state, const auto& mat, const int threads, const int it )
{
    std::vector<VALTYPE> x( mat.rows, 0.0 );
    std::vector<VALTYPE> b( mat.rows, 1.0 );

    matrix_utils::SPMV<std::remove_cvref_t<decltype( mat )>, matrix_utils::ALBUSSPMV<int32_t, int32_t, VALTYPE, matrix_utils::RowDotKernel::Simd>> spmv;
    spmv.setMatrix( &mat );
    spmv._spmv.setNumThreads( threads );
    spmv.preprocess();
    for ( auto _ : state )
    {
        for ( int i = 0; i < it; i++ )
        {
            spmv( b.data(), x.data() );
        }
    }
    setSpmvBytesProcessed( state, mat, it );
};

template <typename VALTYPE>
auto CAMLBSum = []( benchmark::State& state, const auto& mat, const int threads, const int it )
{
    std::vector<VALTYPE> x( mat.rows, 0.0 );
    std::vector<VALTYPE> b( mat.rows, 1.0 );

    matrix_utils::SPMV<std::remove_cvref_t<decltype( mat )>,
                       matrix_utils::ALBUSSPMV<int32_t, int32_t, VALTYPE, matrix_utils::RowDotKernel::Scalar, matrix_utils::WorkloadMode::CAMLB>>
        spmv;
    spmv.setMatrix( &mat );
    spmv._spmv.setNumThreads( threads );
    spmv.preprocess();
    for ( auto _ : state )
    {
        for ( int i = 0; i < it; i++ )
        {
            spmv( b.data(), x.data() );
        }
    }
    setSpmvBytesProcessed( state, mat, it );
};

template <typename VALTYPE>
auto CAMLBSimd = []( benchmark::State& state, const auto& mat, const int threads, const int it )
{
    std::vector<VALTYPE> x( mat.rows, 0.0 );
    std::vector<VALTYPE> b( mat.rows, 1.0 );

    matrix_utils::SPMV<std::remove_cvref_t<decltype( mat )>,
                       matrix_utils::ALBUSSPMV<int32_t, int32_t, VALTYPE, matrix_utils::RowDotKernel::Simd, matrix_utils::WorkloadMode::CAMLB>>
        spmv;
    spmv.setMatrix( &mat );
    spmv._spmv.setNumThreads( threads );
    spmv.preprocess();
    for ( auto _ : state )
    {
        for ( int i = 0; i < it; i++ )
        {
            spmv( b.data(), x.data() );
        }
    }
    setSpmvBytesProcessed( state, mat, it );
};

auto CSR5Double_Bench = []( benchmark::State& state, const auto& mat, const int threads, const int it )
{
    std::vector<double> x( mat.rows, 0.0 );
    std::vector<double> b( mat.rows, 1.0 );

    matrix_utils::SPMV<std::remove_cvref_t<decltype( mat )>, matrix_utils::CSR5SPMV> spmv;
    spmv.setMatrix( &mat );
    spmv._spmv.setNumThreads( threads );
    try
    {
        spmv.preprocess();
    }
    catch ( const std::invalid_argument& e )
    {
        state.SkipWithError( e.what() );
        return;
    }

    for ( auto _ : state )
    {
        for ( int i = 0; i < it; i++ )
        {
            spmv( b.data(), x.data() );
        }
    }
    setSpmvBytesProcessed( state, mat, it );
};

#ifdef LINEAR_ALGEBRA_ENABLE_CSR5_REFERENCE_BENCH
struct ScopedOmpNumThreads
{
    explicit ScopedOmpNumThreads( const int threads ) : old_threads( omp_get_max_threads() )
    {
        omp_set_num_threads( threads );
    }

    ~ScopedOmpNumThreads() { omp_set_num_threads( old_threads ); }

    int old_threads;
};

template <typename T>
struct CSR5ReferenceAlignedFree
{
    void operator()( T* ptr ) const { _mm_free( ptr ); }
};

template <typename T>
using CSR5ReferenceAlignedPtr = std::unique_ptr<T, CSR5ReferenceAlignedFree<T>>;

template <typename T>
CSR5ReferenceAlignedPtr<T> makeCSR5ReferenceAlignedArray( const std::size_t count )
{
    auto* ptr = static_cast<T*>( _mm_malloc( count * sizeof( T ), ANONYMOUSLIB_X86_CACHELINE ) );
    if ( count != 0 && ptr == nullptr )
    {
        throw std::bad_alloc{};
    }
    return CSR5ReferenceAlignedPtr<T>{ ptr };
}

template <typename T>
CSR5ReferenceAlignedPtr<T> makeCSR5ReferenceAlignedCopy( const std::vector<T>& values )
{
    auto copy = makeCSR5ReferenceAlignedArray<T>( values.size() );
    std::copy( values.begin(), values.end(), copy.get() );
    return copy;
}

void skipCSR5ReferenceError( benchmark::State& state, const std::string_view stage, const int err )
{
    const std::string message =
        "reference CSR5 " + std::string( stage ) + " failed with error " + std::to_string( err );
    state.SkipWithError( message.c_str() );
}

auto CSR5ReferenceDouble_Bench =
    []( benchmark::State& state, const CSRTYPE_DOUBLE& mat, const int threads, const int it )
{
    ScopedOmpNumThreads scoped_threads( threads );

    const int base = mat.ai.empty() ? 0 : mat.ai.front();
    auto row_ptr = makeCSR5ReferenceAlignedArray<int>( mat.ai.size() );
    std::transform( mat.ai.begin(), mat.ai.end(), row_ptr.get(),
                    [base]( const int row_offset ) { return row_offset - base; } );

    auto col_idx = makeCSR5ReferenceAlignedArray<int>( mat.aj.size() );
    std::transform( mat.aj.begin(), mat.aj.end(), col_idx.get(),
                    [base]( const int col ) { return col - base; } );

    auto val = makeCSR5ReferenceAlignedCopy( mat.av );
    auto x = makeCSR5ReferenceAlignedArray<double>( static_cast<std::size_t>( mat.rows ) );
    auto y = makeCSR5ReferenceAlignedArray<double>( static_cast<std::size_t>( mat.rows ) );
    std::fill_n( x.get(), mat.rows, 1.0 );
    std::fill_n( y.get(), mat.rows, 0.0 );

    anonymouslibHandle<int, uint32_t, double> spmv( mat.rows, mat.rows );

    int err = spmv.inputCSR( mat.NNZ(), row_ptr.get(), col_idx.get(), val.get() );
    if ( err != ANONYMOUSLIB_SUCCESS )
    {
        skipCSR5ReferenceError( state, "inputCSR", err );
        return;
    }

    err = spmv.setX( x.get() );
    if ( err != ANONYMOUSLIB_SUCCESS )
    {
        skipCSR5ReferenceError( state, "setX", err );
        return;
    }

    spmv.setSigma( ANONYMOUSLIB_CSR5_SIGMA );
    err = spmv.asCSR5();
    if ( err != ANONYMOUSLIB_SUCCESS )
    {
        skipCSR5ReferenceError( state, "asCSR5", err );
        return;
    }

    bool spmv_failed = false;
    for ( auto _ : state )
    {
        for ( int i = 0; i < it; i++ )
        {
            err = spmv.spmv( 1.0, y.get() );
            if ( err != ANONYMOUSLIB_SUCCESS )
            {
                state.SkipWithError( "reference CSR5 spmv failed" );
                spmv_failed = true;
                break;
            }
        }
        if ( spmv_failed )
        {
            break;
        }
    }

    spmv.destroy();
    if ( spmv_failed )
    {
        return;
    }
    setSpmvBytesProcessed( state, mat, it );
};
#endif

#ifdef USE_MKL
template <typename VALTYPE>
auto MKLSPMV_Bench = []( benchmark::State& state, const auto& mat, const int threads, const int it )
{
    std::vector<VALTYPE> x( mat.rows, 0.0 );
    std::vector<VALTYPE> b( mat.rows, 1.0 );

    mkl_set_num_threads_local( threads );

    matrix_utils::SPMV<std::remove_cvref_t<decltype( mat )>, matrix_utils::MKLSPMV<MKL_INT, MKL_INT, VALTYPE>> spmv;
    spmv.setMatrix( &mat );
    spmv.preprocess();

    for ( auto _ : state )
    {
        for ( int i = 0; i < it; i++ )
        {
            spmv( b.data(), x.data() );
        }
    }
    setSpmvBytesProcessed( state, mat, it );

    mkl_set_num_threads_local( 0 );
};
#endif

#ifdef USE_CUDA
template <typename VALTYPE>
auto CuSparseSPMV_Bench = []( benchmark::State& state, const auto& mat, const int threads, const int it )
{
    // Allocate host memory
    std::vector<VALTYPE> x( mat.rows, 0.0 );
    std::vector<VALTYPE> b( mat.rows, 1.0 );

    // Allocate device memory for vectors
    VALTYPE* d_x;
    VALTYPE* d_y;
    cudaMalloc( &d_x, mat.rows * sizeof( VALTYPE ) );
    cudaMalloc( &d_y, mat.rows * sizeof( VALTYPE ) );

    // Copy CSR matrix to device
    int* d_ia = nullptr;
    int* d_ja = nullptr;
    VALTYPE* d_av = nullptr;
    const int base = mat.ai.empty() ? 0 : mat.ai[0];
    const int nnz = mat.ai.empty() ? 0 : ( mat.ai[mat.rows] - base );
    matrix_utils::sparse_cuda::copy_csr_host_to_device<int, int, VALTYPE>(
        mat.rows, mat.ai.data(), mat.aj.data(), mat.av.data(), &d_ia, &d_ja, &d_av );

    // Copy input vector to device
    cudaMemcpy( d_x, b.data(), mat.rows * sizeof( VALTYPE ), cudaMemcpyHostToDevice );
    cudaMemcpy( d_y, x.data(), mat.rows * sizeof( VALTYPE ), cudaMemcpyHostToDevice );

    // Create cuSPARSE handle and preprocess CUDA SpMV
    cusparseHandle_t handle;
    cusparseCreate( &handle );
    matrix_utils::sparse_cuda::CuSparseSPMV<int, int, VALTYPE> cuda_spmv( handle );
    cuda_spmv.preprocess( mat.rows, d_ia, d_ja, d_av, base, nnz );

    // Warm-up run
    cuda_spmv( d_x, d_y );
    cudaDeviceSynchronize();

    // Benchmark
    for ( auto _ : state )
    {
        for ( int i = 0; i < it; i++ )
        {
            cuda_spmv( d_x, d_y );
        }
        cudaDeviceSynchronize();
    }

    setSpmvBytesProcessed( state, mat, it );

    // Cleanup
    cusparseDestroy( handle );
    cudaFree( d_x );
    cudaFree( d_y );
    if ( d_ia )
        cudaFree( d_ia );
    if ( d_ja )
        cudaFree( d_ja );
    if ( d_av )
        cudaFree( d_av );
};

template <typename VALTYPE>
auto CSRScalarSPMV_Bench = []( benchmark::State& state, const auto& mat, const int threads, const int it )
{
    // Allocate host memory
    std::vector<VALTYPE> x( mat.rows, 0.0 );
    std::vector<VALTYPE> b( mat.rows, 1.0 );

    // Allocate device memory for vectors
    VALTYPE* d_x;
    VALTYPE* d_y;
    cudaMalloc( &d_x, mat.rows * sizeof( VALTYPE ) );
    cudaMalloc( &d_y, mat.rows * sizeof( VALTYPE ) );

    // Copy CSR matrix to device
    int* d_ia = nullptr;
    int* d_ja = nullptr;
    VALTYPE* d_av = nullptr;
    const int base = mat.ai.empty() ? 0 : mat.ai[0];
    const int nnz = mat.ai.empty() ? 0 : ( mat.ai[mat.rows] - base );
    matrix_utils::sparse_cuda::copy_csr_host_to_device<int, int, VALTYPE>(
        mat.rows, mat.ai.data(), mat.aj.data(), mat.av.data(), &d_ia, &d_ja, &d_av );

    // Copy input vector to device
    cudaMemcpy( d_x, b.data(), mat.rows * sizeof( VALTYPE ), cudaMemcpyHostToDevice );
    cudaMemcpy( d_y, x.data(), mat.rows * sizeof( VALTYPE ), cudaMemcpyHostToDevice );

    // Preprocess CSR scalar SpMV
    matrix_utils::sparse_cuda::CSRScalarSPMV<int, int, VALTYPE> cuda_spmv;
    cuda_spmv.preprocess( mat.rows, d_ia, d_ja, d_av, base, nnz );

    // Warm-up run
    cuda_spmv( d_x, d_y );
    cudaDeviceSynchronize();

    // Benchmark
    for ( auto _ : state )
    {
        for ( int i = 0; i < it; i++ )
        {
            cuda_spmv( d_x, d_y );
        }
        cudaDeviceSynchronize();
    }

    setSpmvBytesProcessed( state, mat, it );

    // Cleanup
    cudaFree( d_x );
    cudaFree( d_y );
    if ( d_ia )
        cudaFree( d_ia );
    if ( d_ja )
        cudaFree( d_ja );
    if ( d_av )
        cudaFree( d_av );
};

template <typename VALTYPE>
auto CSRVectorSPMV_Bench = []( benchmark::State& state, const auto& mat, const int threads, const int it )
{
    // Allocate host memory
    std::vector<VALTYPE> x( mat.rows, 0.0 );
    std::vector<VALTYPE> b( mat.rows, 1.0 );

    // Allocate device memory for vectors
    VALTYPE* d_x;
    VALTYPE* d_y;
    cudaMalloc( &d_x, mat.rows * sizeof( VALTYPE ) );
    cudaMalloc( &d_y, mat.rows * sizeof( VALTYPE ) );

    // Copy CSR matrix to device
    int* d_ia = nullptr;
    int* d_ja = nullptr;
    VALTYPE* d_av = nullptr;
    const int base = mat.ai.empty() ? 0 : mat.ai[0];
    const int nnz = mat.ai.empty() ? 0 : ( mat.ai[mat.rows] - base );
    matrix_utils::sparse_cuda::copy_csr_host_to_device<int, int, VALTYPE>(
        mat.rows, mat.ai.data(), mat.aj.data(), mat.av.data(), &d_ia, &d_ja, &d_av );

    // Copy input vector to device
    cudaMemcpy( d_x, b.data(), mat.rows * sizeof( VALTYPE ), cudaMemcpyHostToDevice );
    cudaMemcpy( d_y, x.data(), mat.rows * sizeof( VALTYPE ), cudaMemcpyHostToDevice );

    // Preprocess CSR vector SpMV
    matrix_utils::sparse_cuda::CSRVectorSPMV<int, int, VALTYPE> cuda_spmv;
    cuda_spmv.preprocess( mat.rows, d_ia, d_ja, d_av, base, nnz );

    // Warm-up run
    cuda_spmv( d_x, d_y );
    cudaDeviceSynchronize();

    // Benchmark
    for ( auto _ : state )
    {
        for ( int i = 0; i < it; i++ )
        {
            cuda_spmv( d_x, d_y );
        }
        cudaDeviceSynchronize();
    }

    setSpmvBytesProcessed( state, mat, it );

    // Cleanup
    cudaFree( d_x );
    cudaFree( d_y );
    if ( d_ia )
        cudaFree( d_ia );
    if ( d_ja )
        cudaFree( d_ja );
    if ( d_av )
        cudaFree( d_av );
};

template <typename VALTYPE>
auto CSRMergeSPMV_Bench = []( benchmark::State& state, const auto& mat, const int threads, const int it )
{
    // Allocate host memory
    std::vector<VALTYPE> x( mat.rows, 0.0 );
    std::vector<VALTYPE> b( mat.rows, 1.0 );

    // Allocate device memory for vectors
    VALTYPE* d_x;
    VALTYPE* d_y;
    cudaMalloc( &d_x, mat.rows * sizeof( VALTYPE ) );
    cudaMalloc( &d_y, mat.rows * sizeof( VALTYPE ) );

    // Copy CSR matrix to device
    int* d_ia = nullptr;
    int* d_ja = nullptr;
    VALTYPE* d_av = nullptr;
    const int base = mat.ai.empty() ? 0 : mat.ai[0];
    const int nnz = mat.ai.empty() ? 0 : ( mat.ai[mat.rows] - base );
    matrix_utils::sparse_cuda::copy_csr_host_to_device<int, int, VALTYPE>(
        mat.rows, mat.ai.data(), mat.aj.data(), mat.av.data(), &d_ia, &d_ja, &d_av );

    // Copy input vector to device
    cudaMemcpy( d_x, b.data(), mat.rows * sizeof( VALTYPE ), cudaMemcpyHostToDevice );
    cudaMemcpy( d_y, x.data(), mat.rows * sizeof( VALTYPE ), cudaMemcpyHostToDevice );

    // Preprocess CSR merge SpMV
    matrix_utils::sparse_cuda::CSRMergeSPMV<int, int, VALTYPE> cuda_spmv;
    cuda_spmv.preprocess( mat.rows, d_ia, d_ja, d_av, base, nnz );

    // Warm-up run
    cuda_spmv( d_x, d_y );
    cudaDeviceSynchronize();

    // Benchmark
    for ( auto _ : state )
    {
        for ( int i = 0; i < it; i++ )
        {
            cuda_spmv( d_x, d_y );
        }
        cudaDeviceSynchronize();
    }

    setSpmvBytesProcessed( state, mat, it );

    // Cleanup
    cudaFree( d_x );
    cudaFree( d_y );
    if ( d_ia )
        cudaFree( d_ia );
    if ( d_ja )
        cudaFree( d_ja );
    if ( d_av )
        cudaFree( d_av );
};
#endif

int main( int argc, char** argv )
{
    CSRTYPE_DOUBLE mat_double;
    CSRTYPE_FLOAT mat_float;
    int num_threads = 1;
    int iterations = 1;
    cxxopts::Options options( "SPMV benchmark", "Benchmark different types of SPMV" );
    options.allow_unrecognised_options().add_options()(
        "n,nt", "Number of threads", cxxopts::value<int>()->default_value( "1" ) )(
        "i,it", "Number of iterations", cxxopts::value<int>()->default_value( "100" ) )(
        "f,file", "Matrix location", cxxopts::value<std::string>()->default_value( "data/thermal2.mtx" ) )(
        "h,help", "Print usage" );

    auto result = options.parse( argc, argv );

    if ( result.count( "help" ) )
    {
        std::cout << options.help() << std::endl;
        benchmark::Initialize( &argc, argv );
        benchmark::Shutdown();
        exit( 0 );
    }
    num_threads = result["n"].as<int>();
    iterations = result["i"].as<int>();
    std::string file = result["f"].as<std::string>();

    // Read matrix as double
    {
        std::ifstream f( file );
        f.clear();
        f.seekg( 0, std::ios::beg );
        matrix_utils::readMatrixMarket( f, mat_double.ai, mat_double.aj, mat_double.av );
        mat_double.rows = mat_double.ai.size() - 1;
    }

    // Convert to float
    mat_float.rows = mat_double.rows;
    mat_float.ai = mat_double.ai;
    mat_float.aj = mat_double.aj;
    mat_float.av.resize( mat_double.av.size() );
    std::transform( mat_double.av.begin(), mat_double.av.end(), mat_float.av.begin(),
                    []( double d ) { return static_cast<float>( d ); } );

    // Print matrix statistics
    std::size_t nnz = mat_double.av.size();
    std::cout << "Matrix information:\n";
    std::cout << "  Size: " << mat_double.rows << " x " << mat_double.rows << "\n";
    std::cout << "  NNZ: " << nnz << "\n";
    std::cout << "  Avg NNZ/row: " << static_cast<double>( nnz ) / mat_double.rows << "\n";
    std::cout << "  Sparsity: "
              << ( 100.0 * nnz ) / ( static_cast<double>( mat_double.rows ) * mat_double.rows ) << "%\n";
    std::cout << "  Threads: " << num_threads << "\n";
    std::cout << "  Iterations: " << iterations << "\n";

    // Double precision benchmarks
    // registerSpmvBenchmark( "Serial_double", Serial<double>, mat_double, num_threads, iterations );
    registerSpmvBenchmark( "Parallel_double", Parallel<double>, mat_double, num_threads, iterations );
    registerSpmvBenchmark( "RowBalanced_double", RowBalanced<double>, mat_double, num_threads, iterations );
    registerSpmvBenchmark( "RowBalancedSimd_double", RowBalancedSimd<double>, mat_double, num_threads, iterations );
    registerSpmvBenchmark( "ALBUSSum_double", ALBUSSum<double>, mat_double, num_threads, iterations );
    registerSpmvBenchmark( "ALBUSSimd_double", ALBUSSimd<double>, mat_double, num_threads, iterations );
    registerSpmvBenchmark( "CAMLBSum_double", CAMLBSum<double>, mat_double, num_threads, iterations );
    registerSpmvBenchmark( "CAMLBSimd_double", CAMLBSimd<double>, mat_double, num_threads, iterations );
    registerSpmvBenchmark( "CSR5_double", CSR5Double_Bench, mat_double, num_threads, iterations );
#ifdef LINEAR_ALGEBRA_ENABLE_CSR5_REFERENCE_BENCH
    registerSpmvBenchmark( "CSR5Reference_double", CSR5ReferenceDouble_Bench, mat_double, num_threads, iterations );
#endif

#ifdef USE_MKL
    registerSpmvBenchmark( "MKLSPMV_double", MKLSPMV_Bench<double>, mat_double, num_threads, iterations );
#endif

#ifdef USE_CUDA
    registerSpmvBenchmark( "CuSparseSPMV_double", CuSparseSPMV_Bench<double>, mat_double, num_threads, iterations );
    registerSpmvBenchmark( "CSRScalarSPMV_double", CSRScalarSPMV_Bench<double>, mat_double,
                           num_threads, iterations );
    registerSpmvBenchmark( "CSRVectorSPMV_double", CSRVectorSPMV_Bench<double>, mat_double,
                           num_threads, iterations );
    registerSpmvBenchmark( "CSRMergeSPMV_double", CSRMergeSPMV_Bench<double>, mat_double, num_threads, iterations );
#endif

    // Float precision benchmarks
    // registerSpmvBenchmark( "Serial_float", Serial<float>, mat_float, num_threads, iterations );
    registerSpmvBenchmark( "Parallel_float", Parallel<float>, mat_float, num_threads, iterations );
    registerSpmvBenchmark( "RowBalanced_float", RowBalanced<float>, mat_float, num_threads, iterations );
    registerSpmvBenchmark( "RowBalancedSimd_float", RowBalancedSimd<float>, mat_float, num_threads, iterations );
    registerSpmvBenchmark( "ALBUSSum_float", ALBUSSum<float>, mat_float, num_threads, iterations );
    registerSpmvBenchmark( "ALBUSSimd_float", ALBUSSimd<float>, mat_float, num_threads, iterations );
    registerSpmvBenchmark( "CAMLBSum_float", CAMLBSum<float>, mat_float, num_threads, iterations );
    registerSpmvBenchmark( "CAMLBSimd_float", CAMLBSimd<float>, mat_float, num_threads, iterations );

#ifdef USE_MKL
    registerSpmvBenchmark( "MKLSPMV_float", MKLSPMV_Bench<float>, mat_float, num_threads, iterations );
#endif

#ifdef USE_CUDA
    registerSpmvBenchmark( "CuSparseSPMV_float", CuSparseSPMV_Bench<float>, mat_float, num_threads, iterations );
    registerSpmvBenchmark( "CSRScalarSPMV_float", CSRScalarSPMV_Bench<float>, mat_float, num_threads, iterations );
    registerSpmvBenchmark( "CSRVectorSPMV_float", CSRVectorSPMV_Bench<float>, mat_float, num_threads, iterations );
    registerSpmvBenchmark( "CSRMergeSPMV_float", CSRMergeSPMV_Bench<float>, mat_float, num_threads, iterations );
#endif

    benchmark::Initialize( &argc, argv );
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
