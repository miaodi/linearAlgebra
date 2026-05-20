#include "config.h"
#include "graph_algs.hpp"
#include "io.hpp"
#include "matrix_utils.hpp"
#include "permutation.hpp"
#include "precond.hpp"
#include "Reordering.h"
#include "sp_ops.hpp"

#ifdef USE_CUDA
#include "ilu/ilu_numeric.cuh"
#include "ilu/ilu_numeric_workqueue.cuh"
#include "ilu/ilu_update_cache.hpp"

#include <cuda_runtime.h>
#endif

#include <cxxopts.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using CSR = matrix_utils::CSRMatrix<int, int, double>;

#ifdef USE_CUDA
struct GpuNumericResult
{
    double elapsed_ms = 0.0;
    double max_abs_diff = 0.0;
    int mismatches = 0;
};

enum class GpuNumericMode
{
    Cached,
    Global,
    WorkQueue
};

GpuNumericMode parse_gpu_numeric_mode( const std::string& mode )
{
    if ( mode == "cached" )
    {
        return GpuNumericMode::Cached;
    }
    if ( mode == "global" )
    {
        return GpuNumericMode::Global;
    }
    if ( mode == "workqueue" )
    {
        return GpuNumericMode::WorkQueue;
    }
    throw std::invalid_argument( "GPU numeric mode must be one of: cached, global, workqueue" );
}

const char* gpu_numeric_mode_label( const GpuNumericMode mode )
{
    switch ( mode )
    {
    case GpuNumericMode::Cached:
        return "ILUBaseNumericFactorizationCachedAsync";
    case GpuNumericMode::Global:
        return "ILUBaseNumericFactorizationAsync";
    case GpuNumericMode::WorkQueue:
        return "ILUBaseNumericFactorizationWorkQueueAsync";
    }
    return "Unknown GPU numeric mode";
}

void check_cuda( const cudaError_t status, const char* operation )
{
    if ( status != cudaSuccess )
    {
        throw std::runtime_error( std::string( operation ) + ": " + cudaGetErrorString( status ) );
    }
}

template <typename T>
class DeviceBuffer
{
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer( const std::size_t count )
    {
        resize( count );
    }

    DeviceBuffer( const DeviceBuffer& ) = delete;
    DeviceBuffer& operator=( const DeviceBuffer& ) = delete;

    DeviceBuffer( DeviceBuffer&& other ) noexcept : data_( other.data_ )
    {
        other.data_ = nullptr;
    }

    DeviceBuffer& operator=( DeviceBuffer&& other ) noexcept
    {
        if ( this != &other )
        {
            release();
            data_ = other.data_;
            other.data_ = nullptr;
        }
        return *this;
    }

    ~DeviceBuffer()
    {
        release();
    }

    void resize( const std::size_t count )
    {
        release();
        check_cuda( cudaMalloc( reinterpret_cast<void**>( &data_ ), count * sizeof( T ) ), "cudaMalloc" );
    }

    T* get()
    {
        return data_;
    }

    const T* get() const
    {
        return data_;
    }

private:
    void release()
    {
        if ( data_ != nullptr )
        {
            cudaFree( data_ );
            data_ = nullptr;
        }
    }

    T* data_ = nullptr;
};

class CudaStream
{
public:
    CudaStream()
    {
        check_cuda( cudaStreamCreate( &stream_ ), "cudaStreamCreate" );
    }

    CudaStream( const CudaStream& ) = delete;
    CudaStream& operator=( const CudaStream& ) = delete;

    ~CudaStream()
    {
        if ( stream_ != nullptr )
        {
            cudaStreamDestroy( stream_ );
        }
    }

    cudaStream_t get() const
    {
        return stream_;
    }

private:
    cudaStream_t stream_ = nullptr;
};

class CudaEvent
{
public:
    CudaEvent()
    {
        check_cuda( cudaEventCreate( &event_ ), "cudaEventCreate" );
    }

    CudaEvent( const CudaEvent& ) = delete;
    CudaEvent& operator=( const CudaEvent& ) = delete;

    ~CudaEvent()
    {
        if ( event_ != nullptr )
        {
            cudaEventDestroy( event_ );
        }
    }

    cudaEvent_t get() const
    {
        return event_;
    }

private:
    cudaEvent_t event_ = nullptr;
};


GpuNumericResult run_gpu_numeric_factorization( const CSR& matrix,
                                                const CSR& lu,
                                                const std::vector<int>& permutation,
                                                const std::vector<int>& level_prefix,
                                                const int levels,
                                                const GpuNumericMode gpu_numeric_mode,
                                                const matrix_utils::sparse_cuda::ILUUpdateCache<int>* update_cache,
                                                const int workqueue_blocks_per_sm )
{
    int device_count = 0;
    check_cuda( cudaGetDeviceCount( &device_count ), "cudaGetDeviceCount" );
    if ( device_count == 0 )
    {
        throw std::runtime_error( "No CUDA device available" );
    }

    CudaStream stream;
    CudaEvent start;
    CudaEvent stop;

    DeviceBuffer<int> d_a_ai( static_cast<std::size_t>( matrix.rows + 1 ) );
    DeviceBuffer<int> d_a_aj( static_cast<std::size_t>( matrix.NNZ() ) );
    DeviceBuffer<double> d_a_av( static_cast<std::size_t>( matrix.NNZ() ) );
    DeviceBuffer<int> d_lu_ai( static_cast<std::size_t>( lu.rows + 1 ) );
    DeviceBuffer<int> d_lu_aj( static_cast<std::size_t>( lu.NNZ() ) );
    DeviceBuffer<int> d_lu_diag( static_cast<std::size_t>( lu.rows ) );
    DeviceBuffer<int> d_level_perm( static_cast<std::size_t>( permutation.size() ) );
    DeviceBuffer<double> d_lu_av( static_cast<std::size_t>( lu.NNZ() ) );
    DeviceBuffer<int> d_status( 1 );
    DeviceBuffer<int> d_update_ptr;
    DeviceBuffer<int> d_update_jpos;
    DeviceBuffer<int> d_update_pos;
    DeviceBuffer<int> d_level_row_counter;

    check_cuda( cudaMemcpyAsync( d_a_ai.get(), matrix.AI(), static_cast<std::size_t>( matrix.rows + 1 ) * sizeof( int ),
                                 cudaMemcpyHostToDevice, stream.get() ),
                "cudaMemcpyAsync d_a_ai" );
    check_cuda( cudaMemcpyAsync( d_a_aj.get(), matrix.AJ(), static_cast<std::size_t>( matrix.NNZ() ) * sizeof( int ),
                                 cudaMemcpyHostToDevice, stream.get() ),
                "cudaMemcpyAsync d_a_aj" );
    check_cuda( cudaMemcpyAsync( d_a_av.get(), matrix.AV(), static_cast<std::size_t>( matrix.NNZ() ) * sizeof( double ),
                                 cudaMemcpyHostToDevice, stream.get() ),
                "cudaMemcpyAsync d_a_av" );
    check_cuda( cudaMemcpyAsync( d_lu_ai.get(), lu.AI(), static_cast<std::size_t>( lu.rows + 1 ) * sizeof( int ),
                                 cudaMemcpyHostToDevice, stream.get() ),
                "cudaMemcpyAsync d_lu_ai" );
    check_cuda( cudaMemcpyAsync( d_lu_aj.get(), lu.AJ(), static_cast<std::size_t>( lu.NNZ() ) * sizeof( int ),
                                 cudaMemcpyHostToDevice, stream.get() ),
                "cudaMemcpyAsync d_lu_aj" );
    check_cuda( cudaMemcpyAsync( d_lu_diag.get(), lu.Diagonal(), static_cast<std::size_t>( lu.rows ) * sizeof( int ),
                                 cudaMemcpyHostToDevice, stream.get() ),
                "cudaMemcpyAsync d_lu_diag" );
    check_cuda( cudaMemcpyAsync( d_level_perm.get(), permutation.data(), permutation.size() * sizeof( int ),
                                 cudaMemcpyHostToDevice, stream.get() ),
                "cudaMemcpyAsync d_level_perm" );

    if ( gpu_numeric_mode == GpuNumericMode::Cached )
    {
        if ( update_cache == nullptr )
        {
            throw std::runtime_error( "Cached GPU numeric mode requires an update cache" );
        }
        d_update_ptr.resize( update_cache->update_ptr.size() );
        d_update_jpos.resize( std::max<std::size_t>( update_cache->update_jpos.size(), 1 ) );
        d_update_pos.resize( std::max<std::size_t>( update_cache->update_pos.size(), 1 ) );
        check_cuda( cudaMemcpyAsync( d_update_ptr.get(), update_cache->update_ptr.data(),
                                     update_cache->update_ptr.size() * sizeof( int ),
                                     cudaMemcpyHostToDevice, stream.get() ),
                    "cudaMemcpyAsync d_update_ptr" );
        if ( !update_cache->update_jpos.empty() )
        {
            check_cuda( cudaMemcpyAsync( d_update_jpos.get(), update_cache->update_jpos.data(),
                                         update_cache->update_jpos.size() * sizeof( int ),
                                         cudaMemcpyHostToDevice, stream.get() ),
                        "cudaMemcpyAsync d_update_jpos" );
            check_cuda( cudaMemcpyAsync( d_update_pos.get(), update_cache->update_pos.data(),
                                         update_cache->update_pos.size() * sizeof( int ),
                                         cudaMemcpyHostToDevice, stream.get() ),
                        "cudaMemcpyAsync d_update_pos" );
        }
    }
    else if ( gpu_numeric_mode == GpuNumericMode::WorkQueue )
    {
        d_level_row_counter.resize( 1 );
    }

    check_cuda( cudaEventRecord( start.get(), stream.get() ), "cudaEventRecord start" );
    if ( gpu_numeric_mode == GpuNumericMode::Cached )
    {
        check_cuda( matrix_utils::sparse_cuda::ILUBaseNumericFactorizationCachedAsync<int, int, double>(
                        matrix.rows, d_a_ai.get(), d_a_aj.get(), d_a_av.get(), d_lu_ai.get(), d_lu_aj.get(),
                        d_lu_diag.get(), d_update_ptr.get(), d_update_jpos.get(), d_update_pos.get(),
                        d_level_perm.get(), level_prefix.data(), levels, matrix.Base(),
                        d_lu_av.get(), d_status.get(), stream.get() ),
                    "ILUBaseNumericFactorizationCachedAsync" );
    }
    else if ( gpu_numeric_mode == GpuNumericMode::WorkQueue )
    {
        check_cuda( matrix_utils::sparse_cuda::ILUBaseNumericFactorizationWorkQueueAsync<int, int, double>(
                        matrix.rows, d_a_ai.get(), d_a_aj.get(), d_a_av.get(), d_lu_ai.get(), d_lu_aj.get(),
                        d_lu_diag.get(), d_level_perm.get(), level_prefix.data(), levels, matrix.Base(),
                        d_lu_av.get(), d_status.get(), d_level_row_counter.get(), workqueue_blocks_per_sm,
                        stream.get() ),
                    "ILUBaseNumericFactorizationWorkQueueAsync" );
    }
    else
    {
        check_cuda( matrix_utils::sparse_cuda::ILUBaseNumericFactorizationAsync<int, int, double>(
                        matrix.rows, d_a_ai.get(), d_a_aj.get(), d_a_av.get(), d_lu_ai.get(), d_lu_aj.get(),
                        d_lu_diag.get(), d_level_perm.get(), level_prefix.data(), levels, matrix.Base(),
                        d_lu_av.get(), d_status.get(), stream.get() ),
                    "ILUBaseNumericFactorizationAsync" );
    }
    check_cuda( cudaEventRecord( stop.get(), stream.get() ), "cudaEventRecord stop" );

    int h_status = 1;
    check_cuda( cudaMemcpyAsync( &h_status, d_status.get(), sizeof( int ), cudaMemcpyDeviceToHost, stream.get() ),
                "cudaMemcpyAsync d_status" );
    check_cuda( cudaStreamSynchronize( stream.get() ), "cudaStreamSynchronize" );
    if ( h_status != 0 )
    {
        throw std::runtime_error( "GPU ILU numeric factorization failed: zero pivot" );
    }

    float elapsed_ms = 0.0f;
    check_cuda( cudaEventElapsedTime( &elapsed_ms, start.get(), stop.get() ), "cudaEventElapsedTime" );

    std::vector<double> gpu_lu_values( static_cast<std::size_t>( lu.NNZ() ) );
    check_cuda( cudaMemcpyAsync( gpu_lu_values.data(), d_lu_av.get(), gpu_lu_values.size() * sizeof( double ),
                                 cudaMemcpyDeviceToHost, stream.get() ),
                "cudaMemcpyAsync d_lu_av" );
    check_cuda( cudaStreamSynchronize( stream.get() ), "cudaStreamSynchronize d_lu_av" );

    constexpr double abs_tolerance = 1.0e-10;
    constexpr double rel_tolerance = 1.0e-10;
    GpuNumericResult result;
    result.elapsed_ms = static_cast<double>( elapsed_ms );
    for ( int i = 0; i < lu.NNZ(); ++i )
    {
        const double cpu_value = lu.AV()[i];
        const double gpu_value = gpu_lu_values[static_cast<std::size_t>( i )];
        const double abs_diff = std::abs( gpu_value - cpu_value );
        const double allowed = abs_tolerance + rel_tolerance * std::abs( cpu_value );
        result.max_abs_diff = std::max( result.max_abs_diff, abs_diff );
        if ( abs_diff > allowed )
        {
            ++result.mismatches;
        }
    }
    return result;
}
#endif

std::vector<int> find_missing_diagonal_rows( const CSR& matrix )
{
    std::vector<int> missing;
    const int base = matrix.Base();
    for ( int row = 0; row < matrix.rows; ++row )
    {
        const int target = row + base;
        const auto row_begin = matrix.AJ() + matrix.AI()[row] - base;
        const auto row_end = matrix.AJ() + matrix.AI()[row + 1] - base;
        const auto found = std::binary_search( row_begin, row_end, target );
        if ( !found )
        {
            missing.push_back( target );
        }
    }
    return missing;
}

void print_missing_rows( const std::vector<int>& missing )
{
    std::cerr << "Missing diagonal entries in " << missing.size() << " row(s):";
    const std::size_t limit = std::min<std::size_t>( missing.size(), 32 );
    for ( std::size_t i = 0; i < limit; ++i )
    {
        std::cerr << ' ' << missing[i];
    }
    if ( missing.size() > limit )
    {
        std::cerr << " ...";
    }
    std::cerr << '\n';
}

CSR load_matrix_market( const std::string& file_path )
{
    std::ifstream input( file_path );
    if ( !input.is_open() )
    {
        throw std::runtime_error( "Cannot open MatrixMarket file: " + file_path );
    }

    CSR matrix;
    matrix_utils::readMatrixMarket( input, matrix );
    return matrix;
}

bool metis_reorder_matrix( const CSR& matrix, CSR& reordered, const int threads )
{
#ifndef USE_METIS_LIB
    std::cerr << "METIS support not enabled (USE_METIS_LIB=OFF).\n";
    return false;
#else
    if ( matrix.rows != matrix.cols )
    {
        std::cerr << "METIS nested dissection requires a square matrix\n";
        return false;
    }
    if ( matrix.Base() != 0 )
    {
        std::cerr << "METIS reorder path expects zero-based CSR input\n";
        return false;
    }

    std::vector<int> xadj( static_cast<std::size_t>( matrix.rows + 1 ) );
    matrix_utils::APlusATPrefix<int, int, false>( matrix.rows, matrix.AI(), matrix.AJ(), xadj.data() );
    const int actual_edges = xadj[matrix.rows] - xadj[0];
    std::vector<int> adjncy( static_cast<std::size_t>( actual_edges ) );
    matrix_utils::APlusATFill<int, int, false>( matrix.rows, matrix.AI(), matrix.AJ(), xadj.data(), adjncy.data() );

    std::vector<int> iperm( static_cast<std::size_t>( matrix.rows ) );
    std::vector<int> perm( static_cast<std::size_t>( matrix.rows ) );

    reordering::MetisNDOptions nd_opts;
    nd_opts.seed = 42;

    const auto reorder_start = std::chrono::steady_clock::now();
    const int rc = reordering::MetisND( matrix.rows, matrix.cols, xadj.data(), adjncy.data(),
                                        iperm.data(), perm.data(), nd_opts );
    const auto reorder_end = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> reorder_ms = reorder_end - reorder_start;
    std::cout << "METIS nested dissection time: " << reorder_ms.count() << " ms\n";
    if ( rc != 0 )
    {
        std::cerr << "METIS nested dissection failed with code " << rc << '\n';
        return false;
    }

    const bool perm_ok = matrix_utils::isPermutation<int>( matrix.rows, 0, perm.data(), threads );
    const bool iperm_ok = matrix_utils::isPermutation<int>( matrix.rows, 0, iperm.data(), threads );
    if ( !perm_ok || !iperm_ok )
    {
        std::cerr << "Invalid METIS permutation: perm_ok=" << ( perm_ok ? "true" : "false" )
                  << ", iperm_ok=" << ( iperm_ok ? "true" : "false" ) << '\n';
        return false;
    }

    reordered.rows = matrix.rows;
    reordered.cols = matrix.cols;
    reordered.ResizeAI( matrix.rows + 1 );
    reordered.ResizeAJ( matrix.NNZ() );
    reordered.ResizeAV( matrix.NNZ() );
    matrix_utils::permuteMat( matrix.rows, matrix.cols, perm.data(), iperm.data(),
                              matrix.AI(), matrix.AJ(), matrix.AV(),
                              reordered.AI(), reordered.AJ(), reordered.AV(), threads );

    return true;
#endif
}
} // namespace

int main( int argc, char** argv )
{
    cxxopts::Options options(
        "ilu_factorization_levels",
        "Build ILU symbolic/numeric factors and compute TopologicalSort2 dependency levels" );
    options.add_options()(
        "f,file", "Matrix Market file path",
        cxxopts::value<std::string>()->default_value( "../../tests/data/ex5.mtx" ) )(
        "l,level", "ILU level", cxxopts::value<int>()->default_value( "0" ) )(
        "n,threads", "Number of threads for ILU symbolic setup",
        cxxopts::value<int>()->default_value( "1" ) )(
        "disable-gpu-update-cache", "Use the original GPU numeric path without the precomputed update cache" )(
        "gpu-numeric-mode", "GPU numeric mode: cached, global, workqueue",
        cxxopts::value<std::string>()->default_value( "cached" ) )(
        "gpu-workqueue-blocks-per-sm", "Persistent workqueue blocks per SM",
        cxxopts::value<int>()->default_value( "4" ) )(
        "h,help", "Print usage" );

    const auto parsed = options.parse( argc, argv );
    if ( parsed.count( "help" ) )
    {
        std::cout << options.help() << '\n';
        return 0;
    }

    const std::string file_path = parsed["file"].as<std::string>();
    const int level = parsed["level"].as<int>();
    const int threads = parsed["threads"].as<int>();
#ifdef USE_CUDA
    std::string gpu_numeric_mode_name = parsed["gpu-numeric-mode"].as<std::string>();
    if ( parsed.count( "disable-gpu-update-cache" ) != 0 )
    {
        gpu_numeric_mode_name = "global";
    }

    GpuNumericMode gpu_numeric_mode = GpuNumericMode::Cached;
    try
    {
        gpu_numeric_mode = parse_gpu_numeric_mode( gpu_numeric_mode_name );
    }
    catch ( const std::exception& e )
    {
        std::cerr << e.what() << '\n';
        return 1;
    }

    const int gpu_workqueue_blocks_per_sm = parsed["gpu-workqueue-blocks-per-sm"].as<int>();
    if ( gpu_workqueue_blocks_per_sm <= 0 )
    {
        std::cerr << "GPU workqueue blocks per SM must be positive\n";
        return 1;
    }
#endif
    if ( level < 0 )
    {
        std::cerr << "ILU level must be non-negative\n";
        return 1;
    }
    if ( threads <= 0 )
    {
        std::cerr << "Thread count must be positive\n";
        return 1;
    }

    const CSR input_matrix = load_matrix_market( file_path );
    std::cout << "Loaded matrix: rows=" << input_matrix.rows << ", cols=" << input_matrix.cols
              << ", nnz=" << input_matrix.NNZ() << ", base=" << input_matrix.Base() << '\n';

    if ( input_matrix.rows != input_matrix.cols )
    {
        std::cerr << "ILU symbolic factorization requires a square matrix\n";
        return 1;
    }

    CSR matrix;
    if ( !metis_reorder_matrix( input_matrix, matrix, threads ) )
    {
        return 1;
    }
    std::cout << "METIS reordered matrix: rows=" << matrix.rows << ", cols=" << matrix.cols
              << ", nnz=" << matrix.NNZ() << ", base=" << matrix.Base() << '\n';

    std::vector<int> diagonal_positions( static_cast<std::size_t>( matrix.rows ) );
    std::vector<double> diagonal_values( static_cast<std::size_t>( matrix.rows ) );
    const bool has_full_diagonal =
        matrix_utils::Diagonal( matrix.rows, matrix.AI(), matrix.AJ(), matrix.AV(),
                                diagonal_positions.data(), diagonal_values.data() );
    if ( !has_full_diagonal )
    {
        print_missing_rows( find_missing_diagonal_rows( matrix ) );
        return 1;
    }
    std::cout << "All diagonal entries are present\n";

    matrix_utils::ILULevelSymbolicParallel<CSR, enums::matrix_utils::LU, true> symbolic( threads );
    CSR lu;
    const auto symbolic_start = std::chrono::steady_clock::now();
    if ( !symbolic( matrix.rows, matrix.AI(), matrix.AJ(), level, lu ) )
    {
        std::cerr << "ILULevelSymbolicParallel failed\n";
        return 1;
    }
    const auto symbolic_end = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> symbolic_ms = symbolic_end - symbolic_start;
    std::cout << "ILU(" << level << ") symbolic LU: rows=" << lu.rows << ", cols=" << lu.cols
              << ", nnz=" << lu.NNZ() << ", base=" << lu.Base() << '\n';
    std::cout << "ILULevelSymbolicParallel time: " << symbolic_ms.count() << " ms\n";

    std::vector<int> permutation( static_cast<std::size_t>( lu.rows ) );
    std::vector<int> level_prefix( static_cast<std::size_t>( lu.rows + 1 ) );
    graph::TopologicalSort2<int, int, matrix_utils::TriangularMatrix::LU> topological_sort;
    const int levels =
        topological_sort( lu.rows, lu.AI(), lu.AJ(), permutation.data(), level_prefix.data() );

    int max_level_width = 0;
    for ( int level_id = 0; level_id < levels; ++level_id )
    {
        max_level_width = std::max( max_level_width, level_prefix[level_id + 1] - level_prefix[level_id] );
    }

    std::cout << "TopologicalSort2 dependency levels: " << levels
              << ", max level width=" << max_level_width << '\n';

#ifdef USE_CUDA
    matrix_utils::sparse_cuda::ILUUpdateCache<int> update_cache;
    if ( gpu_numeric_mode == GpuNumericMode::Cached )
    {
        update_cache = matrix_utils::sparse_cuda::BuildILUUpdateCache<int, int>(
            lu.rows, lu.AI(), lu.AJ(), lu.Diagonal(), lu.Base(), threads );
        const double cache_mib = static_cast<double>( update_cache.bytes() ) / ( 1024.0 * 1024.0 );
        std::cout << "ILU update cache: entries=" << update_cache.update_jpos.size()
                  << ", memory=" << cache_mib << " MiB"
                  << ", build time=" << update_cache.build_ms << " ms\n";
    }
#endif

    const auto numeric_start = std::chrono::steady_clock::now();
    if ( !matrix_utils::ILULevelNumeric( matrix.rows, matrix.AI(), matrix.AJ(), matrix.AV(), level, lu ) )
    {
        std::cerr << "ILULevelNumeric failed\n";
        return 1;
    }
    const auto numeric_end = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> numeric_ms = numeric_end - numeric_start;
    std::cout << "ILU(" << level << ") numeric factorization complete\n";
    std::cout << "ILULevelNumeric time: " << numeric_ms.count() << " ms\n";

#ifdef USE_CUDA
    try
    {
        const GpuNumericResult gpu_result =
            run_gpu_numeric_factorization(
                matrix, lu, permutation, level_prefix, levels, gpu_numeric_mode,
                gpu_numeric_mode == GpuNumericMode::Cached ? &update_cache : nullptr,
                gpu_workqueue_blocks_per_sm );
        std::cout << "ILU(" << level << ") GPU numeric factorization complete\n";
        std::cout << gpu_numeric_mode_label( gpu_numeric_mode )
                  << " GPU event time: " << gpu_result.elapsed_ms << " ms\n";
        std::cout << "GPU vs CPU LU value check: mismatches=" << gpu_result.mismatches
                  << ", max abs diff=" << gpu_result.max_abs_diff << '\n';
        if ( gpu_result.mismatches != 0 )
        {
            return 1;
        }
    }
    catch ( const std::exception& e )
    {
        std::cerr << "GPU numeric factorization failed: " << e.what() << '\n';
        return 1;
    }
#else
    std::cout << "GPU numeric factorization skipped: USE_CUDA=OFF\n";
#endif

    std::cout << "ILU factorization level analysis complete\n";
    return 0;
}
