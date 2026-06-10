#include "cuda_ilu_base.cuh"
#include "cuda_memory.cuh"
#include "graph_algs.hpp"
#include "io.hpp"
#include "matrix_utils.hpp"
#include "precond.hpp"

#include <benchmark/benchmark.h>
#include <cusparse.h>
#if defined( LINEAR_ALGEBRA_ENABLE_CUDA_PROFILER_RANGE )
#include <cuda_profiler_api.h>
#endif
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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
const char* cusparseSolvePolicyName( const cusparseSolvePolicy_t policy )
{
    switch ( policy )
    {
    case CUSPARSE_SOLVE_POLICY_NO_LEVEL:
        return "no_level";
    case CUSPARSE_SOLVE_POLICY_USE_LEVEL:
        return "use_level";
    }
    return "unknown";
}
#pragma GCC diagnostic pop

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

void startCudaProfilerRange( const char* message )
{
#if defined( LINEAR_ALGEBRA_ENABLE_CUDA_PROFILER_RANGE )
    checkCuda( cudaProfilerStart(), message );
#else
    (void)message;
#endif
}

void stopCudaProfilerRange( const char* message )
{
#if defined( LINEAR_ALGEBRA_ENABLE_CUDA_PROFILER_RANGE )
    checkCuda( cudaProfilerStop(), message );
#else
    (void)message;
#endif
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
    int cusparse_buffer_size = 0;

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
    DeviceIntArray d_next_row;
    DeviceIntArray d_row_done;
    DeviceDoubleArray d_lu_av_initial;
    DeviceDoubleArray d_our_lu_av;
    DeviceDoubleArray d_diag_inv;
    DeviceDoubleArray d_cusparse_lu_av;
    cuda_utils::DeviceArray<char> d_cusparse_buffer;
    cuda_utils::DeviceILUUpdateCache<int> update_cache;
    cuda_utils::ILULevelCtaSchedule<int, int> level_cta_schedule;
    cuda_utils::DeviceILULevelCtaSchedule<int, int> d_level_cta_schedule;
    cuda_utils::ILULevelCtaScratch level_cta_scratch;
    cuda_utils::ILULevelCtaLaunchConfig level_cta_launch;
    cuda_utils::ILUPersistentLaunchConfig persistent_launch;
    cuda_utils::ILUPersistentLaunchConfig persistent_cached_launch;

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
        d_next_row.resize( 1 );
        d_row_done.resize( static_cast<std::size_t>( n ) );
        d_lu_av_initial.resize( static_cast<std::size_t>( nnz_lu ) );
        d_our_lu_av.resize( static_cast<std::size_t>( nnz_lu ) );
        d_diag_inv.resize( static_cast<std::size_t>( n ) );
        d_cusparse_lu_av.resize( static_cast<std::size_t>( nnz_lu ) );

        checkCuda( cuda_utils::BuildILUUpdateCacheAsync<int, int>(
                       n, d_lu_ai.data(), d_lu_aj.data(), d_lu_diag.data(), base, update_cache, stream ),
                   "build device ILU update cache" );
        level_cta_schedule = cuda_utils::BuildILULevelCtaSchedule<int, int>(
            n, lu_pattern.AI(), lu_pattern.AJ(), lu_pattern.Diagonal(), level_perm.data(),
            level_prefix.data(), levels, base );
        checkCuda( cuda_utils::UploadILULevelCtaSchedule<int, int>( level_cta_schedule, d_level_cta_schedule ),
                   "upload level CTA schedule" );

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
                  << ", nnz(LU0 pattern)=" << nnz_lu << ", levels=" << levels
                  << ", strict lower nnz=" << update_cache.strict_lower_nnz
                  << ", cached updates=" << update_cache.total_updates
                  << ", cache bytes=" << update_cache.bytes() << ", cuSPARSE buffer bytes=" << cusparse_buffer_size
                  << ", level CTA dag edges=" << level_cta_schedule.cta_edge_count
                  << ", level CTA grid blocks=" << level_cta_launch.total_blocks
                  << ", level CTA hollow warps=" << level_cta_launch.hollow_warps
                  << ", persistent block size=" << persistent_launch.block_size
                  << ", persistent grid blocks=" << persistent_launch.grid_blocks
                  << ", persistent cached block size=" << persistent_cached_launch.block_size
                  << ", persistent cached grid blocks=" << persistent_cached_launch.grid_blocks
                  << std::endl;
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
        cusparse_buffer_size = buffer_size;
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
        for ( const auto row_lookup :
              { cuda_utils::ILUNumericRowLookup::Global, cuda_utils::ILUNumericRowLookup::Shared } )
        {
            for ( const auto row_update : { cuda_utils::ILUNumericRowUpdateStrategy::BinarySearch,
                                            cuda_utils::ILUNumericRowUpdateStrategy::Merge } )
            {
                resetOurValues();
                checkCuda( cuda_utils::ILUBaseNumericFactorizationAsync<int, int, double>(
                               n, d_lu_ai.data(), d_lu_aj.data(), d_lu_diag.data(),
                               d_level_perm.data(), level_prefix.data(), levels, base,
                               d_our_lu_av.data(), d_status.data(), row_lookup, row_update, stream ),
                           "warm up our ILU0 factorization" );
                int host_status = 1;
                checkCuda( cudaMemcpyAsync( &host_status, d_status.data(), sizeof( int ),
                                            cudaMemcpyDeviceToHost, stream ),
                           "copy our ILU0 status" );
                checkCuda( cudaStreamSynchronize( stream ), "sync after our ILU0 warmup" );
                if ( host_status != 0 )
                {
                    throw std::runtime_error(
                        "our ILU0 factorization found a zero pivot during warmup" );
                }
            }
        }

        resetOurValues();
        checkCuda( cuda_utils::ILUBaseNumericFactorizationLevelCtaAsync<int, int, double>(
                       d_level_cta_schedule, d_lu_ai.data(), d_lu_aj.data(), d_lu_diag.data(), base,
                       d_our_lu_av.data(), d_diag_inv.data(), d_status.data(),
                       cuda_utils::ILUNumericRowLookup::Shared, cuda_utils::ILUNumericRowUpdateStrategy::BinarySearch,
                       level_cta_scratch, stream, &level_cta_launch ),
                   "warm up level CTA ILU0 factorization" );
        int level_cta_host_status = 1;
        checkCuda( cudaMemcpyAsync( &level_cta_host_status, d_status.data(), sizeof( int ),
                                    cudaMemcpyDeviceToHost, stream ),
                   "copy level CTA ILU0 status" );
        checkCuda( cudaStreamSynchronize( stream ), "sync after level CTA ILU0 warmup" );
        if ( level_cta_host_status != 0 )
        {
            throw std::runtime_error(
                "level CTA ILU0 factorization found a zero pivot during warmup" );
        }

        resetOurValues();
        checkCuda( cuda_utils::ILUBaseNumericFactorizationPersistentAsync<int, int, double>(
                       n, d_lu_ai.data(), d_lu_aj.data(), d_lu_diag.data(), base,
                       d_our_lu_av.data(), d_diag_inv.data(), d_status.data(), d_next_row.data(),
                       d_row_done.data(), stream, &persistent_launch ),
                   "warm up persistent ILU0 factorization" );
        int persistent_host_status = 1;
        checkCuda( cudaMemcpyAsync( &persistent_host_status, d_status.data(), sizeof( int ),
                                    cudaMemcpyDeviceToHost, stream ),
                   "copy persistent ILU0 status" );
        checkCuda( cudaStreamSynchronize( stream ), "sync after persistent ILU0 warmup" );
        if ( persistent_host_status != 0 )
        {
            throw std::runtime_error(
                "persistent ILU0 factorization found a zero pivot during warmup" );
        }

        resetOurValues();
        checkCuda( cuda_utils::ILUBaseNumericFactorizationCachedAsync<int, int, double>(
                       n, d_lu_ai.data(), d_lu_aj.data(), d_lu_diag.data(),
                       update_cache.lower_row_ptr.data(), update_cache.update_ptr.data(),
                       update_cache.update_jpos.data(), update_cache.update_pos.data(), d_level_perm.data(),
                       level_prefix.data(), levels, base, d_our_lu_av.data(), d_status.data(), stream ),
                   "warm up cached ILU0 factorization" );
        int cached_host_status = 1;
        checkCuda( cudaMemcpyAsync( &cached_host_status, d_status.data(), sizeof( int ),
                                    cudaMemcpyDeviceToHost, stream ),
                   "copy cached ILU0 status" );
        checkCuda( cudaStreamSynchronize( stream ), "sync after cached ILU0 warmup" );
        if ( cached_host_status != 0 )
        {
            throw std::runtime_error(
                "cached ILU0 factorization found a zero pivot during warmup" );
        }

        resetOurValues();
        checkCuda( cuda_utils::ILUBaseNumericFactorizationPersistentCachedAsync<int, int, double>(
                       n, d_lu_ai.data(), d_lu_aj.data(), d_lu_diag.data(),
                       update_cache.lower_row_ptr.data(), update_cache.update_ptr.data(),
                       update_cache.update_jpos.data(), update_cache.update_pos.data(), base,
                       d_our_lu_av.data(), d_diag_inv.data(), d_status.data(), d_next_row.data(),
                       d_row_done.data(), stream, &persistent_cached_launch ),
                   "warm up persistent cached ILU0 factorization" );
        int persistent_cached_host_status = 1;
        checkCuda( cudaMemcpyAsync( &persistent_cached_host_status, d_status.data(), sizeof( int ),
                                    cudaMemcpyDeviceToHost, stream ),
                   "copy persistent cached ILU0 status" );
        checkCuda( cudaStreamSynchronize( stream ), "sync after persistent cached ILU0 warmup" );
        if ( persistent_cached_host_status != 0 )
        {
            throw std::runtime_error(
                "persistent cached ILU0 factorization found a zero pivot during warmup" );
        }

        for ( const auto policy : { CUSPARSE_SOLVE_POLICY_USE_LEVEL, CUSPARSE_SOLVE_POLICY_NO_LEVEL } )
        {
            resetCusparseValues();
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            checkCusparse( cusparseDcsrilu02( cusparse_handle, n, nnz_lu, cusparse_descr.descr,
                                              d_cusparse_lu_av.data(), d_lu_ai.data(), d_lu_aj.data(),
                                              cusparse_info.info, policy, d_cusparse_buffer.data() ),
                           "warm up cuSPARSE ILU0 factorization" );
#pragma GCC diagnostic pop
            checkCuda( cudaStreamSynchronize( stream ), "sync after cuSPARSE ILU0 warmup" );
            checkCusparseZeroPivot( "cuSPARSE ILU0 numerical zero" );
        }

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
    state.counters["strict_lower_nnz"] = static_cast<double>( data.update_cache.strict_lower_nnz );
    state.counters["cached_updates"] = static_cast<double>( data.update_cache.total_updates );
    state.counters["cache_MB"] = static_cast<double>( data.update_cache.bytes() ) / ( 1024.0 * 1024.0 );
    state.counters["cusparse_buffer_MB"] =
        static_cast<double>( data.cusparse_buffer_size ) / ( 1024.0 * 1024.0 );
    state.counters["level_cta_block_size"] = static_cast<double>( data.level_cta_launch.block_size );
    state.counters["level_cta_dag_edges"] = static_cast<double>( data.level_cta_schedule.cta_edge_count );
    state.counters["level_cta_hollow_warps"] = static_cast<double>( data.level_cta_launch.hollow_warps );
    state.counters["level_cta_schedule_MB"] =
        static_cast<double>( data.d_level_cta_schedule.bytes() ) / ( 1024.0 * 1024.0 );
    state.counters["level_cta_scratch_MB"] =
        static_cast<double>( data.level_cta_scratch.bytes() ) / ( 1024.0 * 1024.0 );
    state.counters["level_cta_total_blocks"] = static_cast<double>( data.level_cta_launch.total_blocks );
    state.counters["level_cta_warps_per_block"] = static_cast<double>( data.level_cta_launch.warps_per_block );
    state.counters["persistent_block_size"] = static_cast<double>( data.persistent_launch.block_size );
    state.counters["persistent_grid_blocks"] = static_cast<double>( data.persistent_launch.grid_blocks );
    state.counters["persistent_resident_warps"] = static_cast<double>( data.persistent_launch.resident_warps );
    state.counters["persistent_cached_block_size"] =
        static_cast<double>( data.persistent_cached_launch.block_size );
    state.counters["persistent_cached_grid_blocks"] =
        static_cast<double>( data.persistent_cached_launch.grid_blocks );
    state.counters["persistent_cached_resident_warps"] =
        static_cast<double>( data.persistent_cached_launch.resident_warps );
    state.SetItemsProcessed( state.iterations() * static_cast<int64_t>( data.nnz_lu ) );
}

void BM_OurILU0Numeric( benchmark::State& state,
                        ILU0BenchmarkData& data,
                        const cuda_utils::ILUNumericRowLookup row_lookup,
                        const cuda_utils::ILUNumericRowUpdateStrategy row_update )
{
    for ( auto _ : state )
    {
        state.PauseTiming();
        data.resetOurValues();
        startCudaProfilerRange( "start CUDA profiler for our ILU0 numeric factorization" );
        state.ResumeTiming();

        checkCuda( cuda_utils::ILUBaseNumericFactorizationAsync<int, int, double>(
                       data.n, data.d_lu_ai.data(), data.d_lu_aj.data(), data.d_lu_diag.data(),
                       data.d_level_perm.data(), data.level_prefix.data(), data.levels, data.base,
                       data.d_our_lu_av.data(), data.d_status.data(), row_lookup, row_update, data.stream ),
                   "run our ILU0 numeric factorization" );
        checkCuda( cudaStreamSynchronize( data.stream ),
                   "sync after our ILU0 numeric factorization" );
        state.PauseTiming();
        stopCudaProfilerRange( "stop CUDA profiler after our ILU0 numeric factorization" );
        state.ResumeTiming();
    }
    setCounters( state, data );
}

void BM_OurILU0NumericPersistent( benchmark::State& state, ILU0BenchmarkData& data )
{
    for ( auto _ : state )
    {
        state.PauseTiming();
        data.resetOurValues();
        startCudaProfilerRange( "start CUDA profiler for persistent ILU0 numeric factorization" );
        state.ResumeTiming();

        checkCuda( cuda_utils::ILUBaseNumericFactorizationPersistentAsync<int, int, double>(
                       data.n, data.d_lu_ai.data(), data.d_lu_aj.data(), data.d_lu_diag.data(),
                       data.base, data.d_our_lu_av.data(), data.d_diag_inv.data(), data.d_status.data(),
                       data.d_next_row.data(), data.d_row_done.data(), data.stream, &data.persistent_launch ),
                   "run persistent ILU0 numeric factorization" );
        checkCuda( cudaStreamSynchronize( data.stream ),
                   "sync after persistent ILU0 numeric factorization" );
        state.PauseTiming();
        stopCudaProfilerRange( "stop CUDA profiler after persistent ILU0 numeric factorization" );
        state.ResumeTiming();
    }
    setCounters( state, data );
}

void BM_OurILU0NumericCached( benchmark::State& state, ILU0BenchmarkData& data )
{
    for ( auto _ : state )
    {
        state.PauseTiming();
        data.resetOurValues();
        startCudaProfilerRange( "start CUDA profiler for cached ILU0 numeric factorization" );
        state.ResumeTiming();

        checkCuda( cuda_utils::ILUBaseNumericFactorizationCachedAsync<int, int, double>(
                       data.n, data.d_lu_ai.data(), data.d_lu_aj.data(), data.d_lu_diag.data(),
                       data.update_cache.lower_row_ptr.data(), data.update_cache.update_ptr.data(),
                       data.update_cache.update_jpos.data(), data.update_cache.update_pos.data(),
                       data.d_level_perm.data(), data.level_prefix.data(), data.levels, data.base,
                       data.d_our_lu_av.data(), data.d_status.data(), data.stream ),
                   "run cached ILU0 numeric factorization" );
        checkCuda( cudaStreamSynchronize( data.stream ),
                   "sync after cached ILU0 numeric factorization" );
        state.PauseTiming();
        stopCudaProfilerRange( "stop CUDA profiler after cached ILU0 numeric factorization" );
        state.ResumeTiming();
    }
    setCounters( state, data );
}

void BM_OurILU0NumericLevelCta( benchmark::State& state, ILU0BenchmarkData& data )
{
    for ( auto _ : state )
    {
        state.PauseTiming();
        data.resetOurValues();
        startCudaProfilerRange( "start CUDA profiler for level CTA ILU0 numeric factorization" );
        state.ResumeTiming();

        checkCuda( cuda_utils::ILUBaseNumericFactorizationLevelCtaAsync<int, int, double>(
                       data.d_level_cta_schedule, data.d_lu_ai.data(), data.d_lu_aj.data(),
                       data.d_lu_diag.data(), data.base, data.d_our_lu_av.data(), data.d_diag_inv.data(),
                       data.d_status.data(), cuda_utils::ILUNumericRowLookup::Shared,
                       cuda_utils::ILUNumericRowUpdateStrategy::BinarySearch,
                       data.level_cta_scratch, data.stream, &data.level_cta_launch ),
                   "run level CTA ILU0 numeric factorization" );
        checkCuda( cudaStreamSynchronize( data.stream ),
                   "sync after level CTA ILU0 numeric factorization" );
        state.PauseTiming();
        stopCudaProfilerRange( "stop CUDA profiler after level CTA ILU0 numeric factorization" );
        state.ResumeTiming();
    }
    setCounters( state, data );
}

void BM_OurILU0NumericPersistentCached( benchmark::State& state, ILU0BenchmarkData& data )
{
    for ( auto _ : state )
    {
        state.PauseTiming();
        data.resetOurValues();
        startCudaProfilerRange(
            "start CUDA profiler for persistent cached ILU0 numeric factorization" );
        state.ResumeTiming();

        checkCuda( cuda_utils::ILUBaseNumericFactorizationPersistentCachedAsync<int, int, double>(
                       data.n, data.d_lu_ai.data(), data.d_lu_aj.data(), data.d_lu_diag.data(),
                       data.update_cache.lower_row_ptr.data(), data.update_cache.update_ptr.data(),
                       data.update_cache.update_jpos.data(), data.update_cache.update_pos.data(),
                       data.base, data.d_our_lu_av.data(), data.d_diag_inv.data(), data.d_status.data(),
                       data.d_next_row.data(), data.d_row_done.data(), data.stream, &data.persistent_cached_launch ),
                   "run persistent cached ILU0 numeric factorization" );
        checkCuda( cudaStreamSynchronize( data.stream ),
                   "sync after persistent cached ILU0 numeric factorization" );
        state.PauseTiming();
        stopCudaProfilerRange(
            "stop CUDA profiler after persistent cached ILU0 numeric factorization" );
        state.ResumeTiming();
    }
    setCounters( state, data );
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
void BM_CuSparseILU0Numeric( benchmark::State& state, ILU0BenchmarkData& data, const cusparseSolvePolicy_t policy )
{
    for ( auto _ : state )
    {
        state.PauseTiming();
        data.resetCusparseValues();
        startCudaProfilerRange( "start CUDA profiler for cuSPARSE ILU0 numeric factorization" );
        state.ResumeTiming();

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        checkCusparse(
            cusparseDcsrilu02( data.cusparse_handle, data.n, data.nnz_lu, data.cusparse_descr.descr,
                               data.d_cusparse_lu_av.data(), data.d_lu_ai.data(), data.d_lu_aj.data(),
                               data.cusparse_info.info, policy, data.d_cusparse_buffer.data() ),
            "run cuSPARSE ILU0 numeric factorization" );
#pragma GCC diagnostic pop
        checkCuda( cudaStreamSynchronize( data.stream ),
                   "sync after cuSPARSE ILU0 numeric factorization" );
        state.PauseTiming();
        stopCudaProfilerRange( "stop CUDA profiler after cuSPARSE ILU0 numeric factorization" );
        state.ResumeTiming();
    }
    setCounters( state, data );
    state.counters["cusparse_policy"] = policy == CUSPARSE_SOLVE_POLICY_USE_LEVEL ? 1.0 : 0.0;
}
#pragma GCC diagnostic pop

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
            "ILU0Numeric/ours_binary_global",
            [data]( benchmark::State& state )
            {
                BM_OurILU0Numeric( state, *data, cuda_utils::ILUNumericRowLookup::Global,
                                   cuda_utils::ILUNumericRowUpdateStrategy::BinarySearch );
            } )
            ->Unit( benchmark::kMillisecond )
            ->UseRealTime();
        benchmark::RegisterBenchmark(
            "ILU0Numeric/ours_binary_shared",
            [data]( benchmark::State& state )
            {
                BM_OurILU0Numeric( state, *data, cuda_utils::ILUNumericRowLookup::Shared,
                                   cuda_utils::ILUNumericRowUpdateStrategy::BinarySearch );
            } )
            ->Unit( benchmark::kMillisecond )
            ->UseRealTime();
        benchmark::RegisterBenchmark( "ILU0Numeric/ours_merge_global",
                                      [data]( benchmark::State& state )
                                      {
                                          BM_OurILU0Numeric(
                                              state, *data, cuda_utils::ILUNumericRowLookup::Global,
                                              cuda_utils::ILUNumericRowUpdateStrategy::Merge );
                                      } )
            ->Unit( benchmark::kMillisecond )
            ->UseRealTime();
        benchmark::RegisterBenchmark( "ILU0Numeric/ours_merge_shared",
                                      [data]( benchmark::State& state )
                                      {
                                          BM_OurILU0Numeric(
                                              state, *data, cuda_utils::ILUNumericRowLookup::Shared,
                                              cuda_utils::ILUNumericRowUpdateStrategy::Merge );
                                      } )
            ->Unit( benchmark::kMillisecond )
            ->UseRealTime();
        benchmark::RegisterBenchmark( "ILU0Numeric/ours_cached", [data]( benchmark::State& state )
                                      { BM_OurILU0NumericCached( state, *data ); } )
            ->Unit( benchmark::kMillisecond )
            ->UseRealTime();
        benchmark::RegisterBenchmark( "ILU0Numeric/ours_level_cta", [data]( benchmark::State& state )
                                      { BM_OurILU0NumericLevelCta( state, *data ); } )
            ->Unit( benchmark::kMillisecond )
            ->UseRealTime();
        benchmark::RegisterBenchmark( "ILU0Numeric/ours_persistent_spin", [data]( benchmark::State& state )
                                      { BM_OurILU0NumericPersistent( state, *data ); } )
            ->Unit( benchmark::kMillisecond )
            ->UseRealTime();
        benchmark::RegisterBenchmark( "ILU0Numeric/ours_persistent_cached", [data]( benchmark::State& state )
                                      { BM_OurILU0NumericPersistentCached( state, *data ); } )
            ->Unit( benchmark::kMillisecond )
            ->UseRealTime();
        for ( const auto policy : { CUSPARSE_SOLVE_POLICY_USE_LEVEL, CUSPARSE_SOLVE_POLICY_NO_LEVEL } )
        {
            const std::string name = std::string( "ILU0Numeric/cuSPARSE_" ) + cusparseSolvePolicyName( policy );
            benchmark::RegisterBenchmark( name.c_str(), [data, policy]( benchmark::State& state )
                                          { BM_CuSparseILU0Numeric( state, *data, policy ); } )
                ->Unit( benchmark::kMillisecond )
                ->UseRealTime();
        }
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
