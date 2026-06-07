#include "cuda_ilu_base.cuh"
#include "cuda_memory.cuh"
#include "graph_algs.hpp"
#include "io.hpp"
#include "matrix_utils.hpp"
#include "precond.hpp"

#include <benchmark/benchmark.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <cxxopts.hpp>

#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuda_utils = matrix_utils::sparse_cuda;

namespace
{
using CSRMatrix = matrix_utils::CSRMatrix<int, int, double>;
using HostCSRMatrix = matrix_utils::CSRMatrixVec<int, int, double>;
using DeviceDoubleArray = cuda_utils::DeviceArray<double>;
using DeviceIntArray = cuda_utils::DeviceArray<int>;

void checkCuda( const cudaError_t status, const char* message )
{
    if ( status != cudaSuccess )
    {
        throw std::runtime_error( std::string( message ) + ": " + cudaGetErrorString( status ) );
    }
}

void checkCusparse( const cusparseStatus_t status, const char* message )
{
    if ( status != CUSPARSE_STATUS_SUCCESS )
    {
        throw std::runtime_error( std::string( message ) + ": " + cusparseGetErrorString( status ) );
    }
}

struct CusparseMatDescrGuard
{
    cusparseMatDescr_t descr = nullptr;

    CusparseMatDescrGuard()
    {
        checkCusparse( cusparseCreateMatDescr( &descr ), "create cuSPARSE matrix descriptor" );
        checkCusparse( cusparseSetMatType( descr, CUSPARSE_MATRIX_TYPE_GENERAL ),
                       "set cuSPARSE matrix type" );
    }

    ~CusparseMatDescrGuard()
    {
        if ( descr != nullptr )
        {
            cusparseDestroyMatDescr( descr );
        }
    }

    CusparseMatDescrGuard( const CusparseMatDescrGuard& ) = delete;
    CusparseMatDescrGuard& operator=( const CusparseMatDescrGuard& ) = delete;
};

struct CusparseCsriluInfoGuard
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    csrilu02Info_t info = nullptr;
#pragma GCC diagnostic pop

    CusparseCsriluInfoGuard()
    {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        checkCusparse( cusparseCreateCsrilu02Info( &info ), "create cuSPARSE csrilu02 info" );
#pragma GCC diagnostic pop
    }

    ~CusparseCsriluInfoGuard()
    {
        if ( info != nullptr )
        {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            cusparseDestroyCsrilu02Info( info );
#pragma GCC diagnostic pop
        }
    }

    CusparseCsriluInfoGuard( const CusparseCsriluInfoGuard& ) = delete;
    CusparseCsriluInfoGuard& operator=( const CusparseCsriluInfoGuard& ) = delete;
};

struct ILU0BenchmarkData
{
    int n = 0;
    int nnz_a = 0;
    int nnz_lu = 0;
    int base = 0;
    int levels = 0;

    std::vector<int> level_prefix;

    cudaStream_t stream = nullptr;
    cusparseHandle_t cusparse_handle = nullptr;
    CusparseMatDescrGuard cusparse_descr;
    CusparseCsriluInfoGuard cusparse_info;

    DeviceIntArray d_a_ai;
    DeviceIntArray d_a_aj;
    DeviceDoubleArray d_a_av;
    DeviceIntArray d_lu_ai;
    DeviceIntArray d_lu_aj;
    DeviceIntArray d_lu_diag;
    DeviceIntArray d_level_perm;
    DeviceIntArray d_status;
    DeviceDoubleArray d_lu_av_initial;
    DeviceDoubleArray d_our_lu_av;
    DeviceDoubleArray d_cusparse_lu_av;
    cuda_utils::DeviceArray<char> d_cusparse_buffer;

    ILU0BenchmarkData( const std::string& matrix_file, const int symbolic_threads )
    {
        int device_count = 0;
        checkCuda( cudaGetDeviceCount( &device_count ), "query CUDA device count" );
        if ( device_count <= 0 )
        {
            throw std::runtime_error( "no CUDA device is available" );
        }

        HostCSRMatrix matrix;
        matrix_utils::readMatrix( matrix_file, matrix );

        n = matrix.rows;
        if ( n <= 0 || matrix.cols <= 0 )
        {
            throw std::runtime_error( "matrix must have at least one row and column" );
        }
        if ( matrix.rows != matrix.cols )
        {
            throw std::runtime_error( "ILU0 benchmark requires a square matrix" );
        }
        base = matrix.Base();
        nnz_a = static_cast<int>( matrix.NNZ() );

        CSRMatrix lu_pattern;
        matrix_utils::ILULevelSymbolicParallel<CSRMatrix, enums::matrix_utils::LU, true> symbolic( symbolic_threads );
        if ( !symbolic( n, matrix.AI(), matrix.AJ(), 0, lu_pattern ) )
        {
            throw std::runtime_error( "failed to build CPU ILU(0) symbolic LU pattern" );
        }
        if ( lu_pattern.Diagonal() == nullptr )
        {
            throw std::runtime_error(
                "ILU(0) symbolic pattern did not produce diagonal positions" );
        }
        nnz_lu = static_cast<int>( lu_pattern.NNZ() );

        std::vector<int> level_perm( static_cast<std::size_t>( n ) );
        level_prefix.resize( static_cast<std::size_t>( n ) + 1 );
        graph::TopologicalSort2<int, int, matrix_utils::TriangularMatrix::LU> topological_sort;
        levels = topological_sort( n, lu_pattern.AI(), lu_pattern.AJ(), level_perm.data(),
                                   level_prefix.data() );

        checkCuda( cudaStreamCreate( &stream ), "create CUDA stream" );
        checkCusparse( cusparseCreate( &cusparse_handle ), "create cuSPARSE handle" );
        checkCusparse( cusparseSetStream( cusparse_handle, stream ), "set cuSPARSE stream" );
        checkCusparse( cusparseSetMatIndexBase( cusparse_descr.descr, base == 0 ? CUSPARSE_INDEX_BASE_ZERO
                                                                                : CUSPARSE_INDEX_BASE_ONE ),
                       "set cuSPARSE index base" );

        d_a_ai.copyFromHost( matrix.AI(), static_cast<std::size_t>( n ) + 1 );
        d_a_aj.copyFromHost( matrix.AJ(), static_cast<std::size_t>( nnz_a ) );
        d_a_av.copyFromHost( matrix.AV(), static_cast<std::size_t>( nnz_a ) );
        d_lu_ai.copyFromHost( lu_pattern.AI(), static_cast<std::size_t>( n ) + 1 );
        d_lu_aj.copyFromHost( lu_pattern.AJ(), static_cast<std::size_t>( nnz_lu ) );
        d_lu_diag.copyFromHost( lu_pattern.Diagonal(), static_cast<std::size_t>( n ) );
        d_level_perm.copyFromHost( level_perm.data(), static_cast<std::size_t>( n ) );
        d_status.resize( 1 );
        d_lu_av_initial.resize( static_cast<std::size_t>( nnz_lu ) );
        d_our_lu_av.resize( static_cast<std::size_t>( nnz_lu ) );
        d_cusparse_lu_av.resize( static_cast<std::size_t>( nnz_lu ) );

        checkCuda( cuda_utils::ILUEmbedAValuesToLUAsync<int, int, double>(
                       n, d_a_ai.data(), d_a_aj.data(), d_a_av.data(), d_lu_ai.data(),
                       d_lu_aj.data(), base, d_lu_av_initial.data(), stream ),
                   "embed A values into LU pattern" );
        checkCuda( cudaStreamSynchronize( stream ), "sync after value embedding" );

        resetOurValues();
        resetCusparseValues();
        prepareCusparseAnalysis();
        warmUpAndValidate();

        std::cout << "Loaded " << matrix_file << ": n=" << n << ", nnz(A)=" << nnz_a
                  << ", nnz(LU0 pattern)=" << nnz_lu << ", levels=" << levels << std::endl;
    }

    ~ILU0BenchmarkData()
    {
        if ( cusparse_handle != nullptr )
        {
            cusparseDestroy( cusparse_handle );
        }
        if ( stream != nullptr )
        {
            cudaStreamDestroy( stream );
        }
    }

    ILU0BenchmarkData( const ILU0BenchmarkData& ) = delete;
    ILU0BenchmarkData& operator=( const ILU0BenchmarkData& ) = delete;

    void resetOurValues()
    {
        checkCuda( cudaMemcpyAsync( d_our_lu_av.data(), d_lu_av_initial.data(),
                                    static_cast<std::size_t>( nnz_lu ) * sizeof( double ),
                                    cudaMemcpyDeviceToDevice, stream ),
                   "reset our LU values" );
        checkCuda( cudaStreamSynchronize( stream ), "sync after resetting our LU values" );
    }

    void resetCusparseValues()
    {
        checkCuda( cudaMemcpyAsync( d_cusparse_lu_av.data(), d_lu_av_initial.data(),
                                    static_cast<std::size_t>( nnz_lu ) * sizeof( double ),
                                    cudaMemcpyDeviceToDevice, stream ),
                   "reset cuSPARSE LU values" );
        checkCuda( cudaStreamSynchronize( stream ), "sync after resetting cuSPARSE LU values" );
    }

    void prepareCusparseAnalysis()
    {
        int buffer_size = 0;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        checkCusparse( cusparseDcsrilu02_bufferSize( cusparse_handle, n, nnz_lu, cusparse_descr.descr,
                                                     d_cusparse_lu_av.data(), d_lu_ai.data(),
                                                     d_lu_aj.data(), cusparse_info.info, &buffer_size ),
                       "query cuSPARSE csrilu02 buffer size" );
#pragma GCC diagnostic pop
        d_cusparse_buffer.resize( static_cast<std::size_t>( std::max( buffer_size, 0 ) ) );

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        checkCusparse( cusparseDcsrilu02_analysis(
                           cusparse_handle, n, nnz_lu, cusparse_descr.descr, d_cusparse_lu_av.data(),
                           d_lu_ai.data(), d_lu_aj.data(), cusparse_info.info,
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_cusparse_buffer.data() ),
                       "analyze cuSPARSE csrilu02" );
#pragma GCC diagnostic pop
        checkCuda( cudaStreamSynchronize( stream ), "sync after cuSPARSE analysis" );
        checkCusparseZeroPivot( "cuSPARSE ILU0 structural zero" );
    }

    void checkCusparseZeroPivot( const char* message )
    {
        int pivot = -1;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        const cusparseStatus_t status =
            cusparseXcsrilu02_zeroPivot( cusparse_handle, cusparse_info.info, &pivot );
#pragma GCC diagnostic pop
        if ( status == CUSPARSE_STATUS_ZERO_PIVOT )
        {
            throw std::runtime_error( std::string( message ) + " at row " + std::to_string( pivot ) );
        }
        checkCusparse( status, message );
    }

    void warmUpAndValidate()
    {
        resetOurValues();
        checkCuda( cuda_utils::ILUBaseNumericFactorizationAsync<int, int, double>(
                       n, d_lu_ai.data(), d_lu_aj.data(), d_lu_diag.data(), d_level_perm.data(),
                       level_prefix.data(), levels, base, d_our_lu_av.data(), d_status.data(),
                       cuda_utils::ILUNumericRowLookup::Shared, stream ),
                   "warm up shared ILU0 factorization" );
        int host_status = 1;
        checkCuda( cudaMemcpyAsync( &host_status, d_status.data(), sizeof( int ), cudaMemcpyDeviceToHost, stream ),
                   "copy shared ILU0 status" );
        checkCuda( cudaStreamSynchronize( stream ), "sync after shared ILU0 warmup" );
        if ( host_status != 0 )
        {
            throw std::runtime_error(
                "shared ILU0 factorization found a zero pivot during warmup" );
        }

        resetOurValues();
        checkCuda( cuda_utils::ILUBaseNumericFactorizationAsync<int, int, double>(
                       n, d_lu_ai.data(), d_lu_aj.data(), d_lu_diag.data(), d_level_perm.data(),
                       level_prefix.data(), levels, base, d_our_lu_av.data(), d_status.data(),
                       cuda_utils::ILUNumericRowLookup::Global, stream ),
                   "warm up global ILU0 factorization" );
        host_status = 1;
        checkCuda( cudaMemcpyAsync( &host_status, d_status.data(), sizeof( int ), cudaMemcpyDeviceToHost, stream ),
                   "copy global ILU0 status" );
        checkCuda( cudaStreamSynchronize( stream ), "sync after global ILU0 warmup" );
        if ( host_status != 0 )
        {
            throw std::runtime_error(
                "global ILU0 factorization found a zero pivot during warmup" );
        }

        resetCusparseValues();
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        checkCusparse( cusparseDcsrilu02( cusparse_handle, n, nnz_lu, cusparse_descr.descr,
                                          d_cusparse_lu_av.data(), d_lu_ai.data(), d_lu_aj.data(),
                                          cusparse_info.info, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                          d_cusparse_buffer.data() ),
                       "warm up cuSPARSE ILU0 factorization" );
#pragma GCC diagnostic pop
        checkCuda( cudaStreamSynchronize( stream ), "sync after cuSPARSE ILU0 warmup" );
        checkCusparseZeroPivot( "cuSPARSE ILU0 numerical zero" );

        resetOurValues();
        resetCusparseValues();
    }
};

void setCounters( benchmark::State& state, const ILU0BenchmarkData& data )
{
    state.counters["n"] = static_cast<double>( data.n );
    state.counters["nnz_A"] = static_cast<double>( data.nnz_a );
    state.counters["nnz_LU"] = static_cast<double>( data.nnz_lu );
    state.counters["levels"] = static_cast<double>( data.levels );
    state.SetItemsProcessed( state.iterations() * static_cast<int64_t>( data.nnz_lu ) );
}

void BM_OurILU0Numeric( benchmark::State& state, ILU0BenchmarkData& data, const cuda_utils::ILUNumericRowLookup row_lookup )
{
    for ( auto _ : state )
    {
        state.PauseTiming();
        data.resetOurValues();
        state.ResumeTiming();

        checkCuda( cuda_utils::ILUBaseNumericFactorizationAsync<int, int, double>(
                       data.n, data.d_lu_ai.data(), data.d_lu_aj.data(), data.d_lu_diag.data(),
                       data.d_level_perm.data(), data.level_prefix.data(), data.levels, data.base,
                       data.d_our_lu_av.data(), data.d_status.data(), row_lookup, data.stream ),
                   "run our ILU0 numeric factorization" );
        checkCuda( cudaStreamSynchronize( data.stream ),
                   "sync after our ILU0 numeric factorization" );
    }
    setCounters( state, data );
}

void BM_CuSparseILU0Numeric( benchmark::State& state, ILU0BenchmarkData& data )
{
    for ( auto _ : state )
    {
        state.PauseTiming();
        data.resetCusparseValues();
        state.ResumeTiming();

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        checkCusparse(
            cusparseDcsrilu02( data.cusparse_handle, data.n, data.nnz_lu, data.cusparse_descr.descr,
                               data.d_cusparse_lu_av.data(), data.d_lu_ai.data(),
                               data.d_lu_aj.data(), data.cusparse_info.info,
                               CUSPARSE_SOLVE_POLICY_USE_LEVEL, data.d_cusparse_buffer.data() ),
            "run cuSPARSE ILU0 numeric factorization" );
#pragma GCC diagnostic pop
        checkCuda( cudaStreamSynchronize( data.stream ),
                   "sync after cuSPARSE ILU0 numeric factorization" );
    }
    setCounters( state, data );
}

void printUsage()
{
    std::cout
        << "cuda_ilu0_bench - compare CUDA ILU(0) numeric factorization variants with cuSPARSE\n"
        << "\nCustom Options:\n"
        << "  -f, --file FILE              Matrix file path, .mtx or .bin (default: "
           "data/thermal2.mtx)\n"
        << "  -t, --symbolic-threads N    CPU threads for one-time ILU(0) symbolic setup (default: "
           "4)\n"
        << "\nGoogle Benchmark Options:\n"
        << "  --benchmark_filter=<regex>\n"
        << "  --benchmark_repetitions=N\n"
        << "  --benchmark_min_time=T\n";
}

} // namespace

int main( int argc, char** argv )
{
    cxxopts::Options options( "cuda_ilu0_bench", "CUDA ILU(0) numeric factorization benchmark" );
    options.allow_unrecognised_options().add_options()(
        "f,file", "Matrix file path (.mtx or .bin)",
        cxxopts::value<std::string>()->default_value( "data/thermal2.mtx" ) )(
        "t,symbolic-threads", "CPU threads for symbolic setup",
        cxxopts::value<int>()->default_value( "4" ) )( "h,help", "Print custom benchmark help" );

    std::vector<std::string> benchmark_arg_storage;
    benchmark_arg_storage.emplace_back( argv[0] );

    try
    {
        const auto result = options.parse( argc, argv );
        if ( result.count( "help" ) != 0 )
        {
            printUsage();
            return 0;
        }

        const std::string matrix_file = result["file"].as<std::string>();
        const int symbolic_threads = result["symbolic-threads"].as<int>();
        if ( symbolic_threads <= 0 )
        {
            throw std::runtime_error( "--symbolic-threads must be > 0" );
        }

        for ( const auto& arg : result.unmatched() )
        {
            benchmark_arg_storage.push_back( arg );
        }

        auto data = std::make_shared<ILU0BenchmarkData>( matrix_file, symbolic_threads );
        benchmark::RegisterBenchmark(
            "ILU0Numeric/ours_global", [data]( benchmark::State& state )
            { BM_OurILU0Numeric( state, *data, cuda_utils::ILUNumericRowLookup::Global ); } )
            ->Unit( benchmark::kMillisecond )
            ->UseRealTime();
        benchmark::RegisterBenchmark(
            "ILU0Numeric/ours_shared", [data]( benchmark::State& state )
            { BM_OurILU0Numeric( state, *data, cuda_utils::ILUNumericRowLookup::Shared ); } )
            ->Unit( benchmark::kMillisecond )
            ->UseRealTime();
        benchmark::RegisterBenchmark( "ILU0Numeric/cuSPARSE", [data]( benchmark::State& state )
                                      { BM_CuSparseILU0Numeric( state, *data ); } )
            ->Unit( benchmark::kMillisecond )
            ->UseRealTime();
    }
    catch ( const std::exception& e )
    {
        std::cerr << "Error: " << e.what() << std::endl;
        printUsage();
        return 1;
    }

    std::vector<char*> benchmark_argv;
    benchmark_argv.reserve( benchmark_arg_storage.size() );
    for ( std::string& arg : benchmark_arg_storage )
    {
        benchmark_argv.push_back( arg.data() );
    }

    int benchmark_argc = static_cast<int>( benchmark_argv.size() );
    benchmark::Initialize( &benchmark_argc, benchmark_argv.data() );
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
