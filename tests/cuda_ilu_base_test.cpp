#include "cuda_ilu_base.cuh"
#include "graph_algs.hpp"
#include "io.hpp"
#include "matrix_utils.hpp"
#include "precond.hpp"
#include "utils.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

using matrix_utils::CSRMatrix;
using matrix_utils::ILULevelSymbolicParallel;
using matrix_utils::ILUNumeric;
using matrix_utils::TriangularMatrix;
using matrix_utils::sparse_cuda::BuildILUUpdateCache;
using matrix_utils::sparse_cuda::BuildILUUpdateCacheAsync;
using matrix_utils::sparse_cuda::DeviceILUUpdateCache;
using matrix_utils::sparse_cuda::ILUBaseNumericFactorizationAsync;
using matrix_utils::sparse_cuda::ILUBaseNumericFactorizationCachedAsync;
using matrix_utils::sparse_cuda::ILUBaseNumericFactorizationPersistentAsync;
using matrix_utils::sparse_cuda::ILUEmbedAValuesToLUAsync;
using matrix_utils::sparse_cuda::ILUNumericRowLookup;
using matrix_utils::sparse_cuda::ILUNumericRowUpdateStrategy;
using matrix_utils::sparse_cuda::ILUPersistentLaunchConfig;
using matrix_utils::sparse_cuda::ILUUpdateCache;

namespace
{
std::ifstream open_test_matrix( const std::string& name )
{
    for ( const std::string& prefix : { "data/", "release/tests/data/" } )
    {
        std::ifstream file( prefix + name );
        if ( file.is_open() )
        {
            return file;
        }
    }
    return {};
}

void expect_cache_matches_pattern( const ILUUpdateCache<int>& cache,
                                   int n,
                                   const int* lu_ai,
                                   const int* lu_aj,
                                   const int* lu_diag,
                                   int base )
{
    int strict_lower_nnz = 0;
    for ( int i = 0; i < n; ++i )
    {
        strict_lower_nnz += lu_diag[i] - lu_ai[i];
    }

    ASSERT_EQ( cache.strict_lower_nnz, strict_lower_nnz );
    ASSERT_EQ( cache.lower_row_ptr.size(), static_cast<std::size_t>( n + 1 ) );
    ASSERT_EQ( cache.update_ptr.size(), static_cast<std::size_t>( strict_lower_nnz + 1 ) );
    ASSERT_EQ( cache.update_jpos.size(), cache.update_pos.size() );
    ASSERT_EQ( cache.total_updates, static_cast<int>( cache.update_jpos.size() ) );

    for ( int i = 0; i < n; ++i )
    {
        const int row_begin = lu_ai[i] - base;
        const int row_end = lu_ai[i + 1] - base;
        const int lower_end = lu_diag[i] - base;
        ASSERT_EQ( cache.lower_row_ptr[i + 1] - cache.lower_row_ptr[i], lower_end - row_begin );

        for ( int k_pos = row_begin; k_pos < lower_end; ++k_pos )
        {
            const int lower_id = cache.lower_row_ptr[i] + ( k_pos - row_begin );
            ASSERT_GE( lower_id, 0 );
            ASSERT_LT( lower_id, strict_lower_nnz );
            ASSERT_LE( cache.update_ptr[lower_id], cache.update_ptr[lower_id + 1] );

            const int k = lu_aj[k_pos] - base;
            const int k_u_begin = ( lu_diag[k] - base ) + 1;
            const int k_u_end = lu_ai[k + 1] - base;
            for ( int update = cache.update_ptr[lower_id]; update < cache.update_ptr[lower_id + 1]; ++update )
            {
                const int src = cache.update_jpos[update];
                const int dst = cache.update_pos[update];
                EXPECT_GE( src, k_u_begin );
                EXPECT_LT( src, k_u_end );
                EXPECT_GT( dst, k_pos );
                EXPECT_LT( dst, row_end );
                EXPECT_EQ( lu_aj[src], lu_aj[dst] );
            }
        }
    }
}
} // namespace

#define ASSERT_CUDA_OK( ... )                                                       \
    do                                                                              \
    {                                                                               \
        const cudaError_t cuda_status = ( __VA_ARGS__ );                            \
        ASSERT_EQ( cuda_status, cudaSuccess ) << cudaGetErrorString( cuda_status ); \
    } while ( false )

TEST( CudaILUBase, MatchesCPULevel0Ex5 )
{
    std::vector<int> csr_rows;
    std::vector<int> csr_cols;
    std::vector<double> csr_vals;

    std::ifstream f = open_test_matrix( "spd/ex5.mtx" );
    ASSERT_TRUE( f.is_open() );
    matrix_utils::readMatrixMarket( f, csr_rows, csr_cols, csr_vals );

    const int n = static_cast<int>( csr_rows.size() ) - 1;
    const int nnz = static_cast<int>( csr_cols.size() );
    const int base = 0;

    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount( &device_count );
    if ( device_status != cudaSuccess || device_count == 0 )
    {
        GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString( device_status );
    }

    ILULevelSymbolicParallel<CSRMatrix<int, int, double>, enums::matrix_utils::LU, true> symbolic( 1 );
    CSRMatrix<int, int, double> lu_cpu;
    ASSERT_TRUE( symbolic( n, csr_rows.data(), csr_cols.data(), 0, lu_cpu ) );
    ASSERT_NE( lu_cpu.Diagonal(), nullptr );
    ASSERT_TRUE( ILUNumeric( n, csr_rows.data(), csr_cols.data(), csr_vals.data(), lu_cpu ) );
    const int nnz_lu = static_cast<int>( lu_cpu.NNZ() );

    std::vector<int> level_perm( n );
    std::vector<int> level_prefix( n + 1 );
    graph::TopologicalSort2<int, int, TriangularMatrix::LU> topological_sort;
    const int levels =
        topological_sort( n, lu_cpu.AI(), lu_cpu.AJ(), level_perm.data(), level_prefix.data() );

    const ILUUpdateCache<int> host_cache =
        BuildILUUpdateCache<int, int>( n, lu_cpu.AI(), lu_cpu.AJ(), lu_cpu.Diagonal(), base, 1 );
    expect_cache_matches_pattern( host_cache, n, lu_cpu.AI(), lu_cpu.AJ(), lu_cpu.Diagonal(), base );

    int* d_a_ai = nullptr;
    int* d_a_aj = nullptr;
    double* d_a_av = nullptr;
    int* d_lu_ai = nullptr;
    int* d_lu_aj = nullptr;
    int* d_lu_diag = nullptr;
    int* d_level_perm = nullptr;
    int* d_status = nullptr;
    int* d_next_row = nullptr;
    int* d_row_done = nullptr;
    double* d_diag_inv = nullptr;
    double* d_lu_initial = nullptr;
    double* d_lu_av = nullptr;

    ASSERT_CUDA_OK( cudaMalloc( &d_a_ai, static_cast<size_t>( n + 1 ) * sizeof( int ) ) );
    ASSERT_CUDA_OK( cudaMalloc( &d_a_aj, static_cast<size_t>( nnz ) * sizeof( int ) ) );
    ASSERT_CUDA_OK( cudaMalloc( &d_a_av, static_cast<size_t>( nnz ) * sizeof( double ) ) );
    ASSERT_CUDA_OK( cudaMalloc( &d_lu_ai, static_cast<size_t>( n + 1 ) * sizeof( int ) ) );
    ASSERT_CUDA_OK( cudaMalloc( &d_lu_aj, static_cast<size_t>( nnz_lu ) * sizeof( int ) ) );
    ASSERT_CUDA_OK( cudaMalloc( &d_lu_diag, static_cast<size_t>( n ) * sizeof( int ) ) );
    ASSERT_CUDA_OK( cudaMalloc( &d_level_perm, static_cast<size_t>( n ) * sizeof( int ) ) );
    ASSERT_CUDA_OK( cudaMalloc( &d_status, sizeof( int ) ) );
    ASSERT_CUDA_OK( cudaMalloc( &d_next_row, sizeof( int ) ) );
    ASSERT_CUDA_OK( cudaMalloc( &d_row_done, static_cast<size_t>( n ) * sizeof( int ) ) );
    ASSERT_CUDA_OK( cudaMalloc( &d_diag_inv, static_cast<size_t>( n ) * sizeof( double ) ) );
    ASSERT_CUDA_OK( cudaMalloc( &d_lu_initial, static_cast<size_t>( nnz_lu ) * sizeof( double ) ) );
    ASSERT_CUDA_OK( cudaMalloc( &d_lu_av, static_cast<size_t>( nnz_lu ) * sizeof( double ) ) );

    ASSERT_CUDA_OK( cudaMemcpy( d_a_ai, csr_rows.data(), static_cast<size_t>( n + 1 ) * sizeof( int ),
                                cudaMemcpyHostToDevice ) );
    ASSERT_CUDA_OK( cudaMemcpy( d_a_aj, csr_cols.data(), static_cast<size_t>( nnz ) * sizeof( int ),
                                cudaMemcpyHostToDevice ) );
    ASSERT_CUDA_OK( cudaMemcpy( d_a_av, csr_vals.data(), static_cast<size_t>( nnz ) * sizeof( double ),
                                cudaMemcpyHostToDevice ) );
    ASSERT_CUDA_OK( cudaMemcpy( d_lu_ai, lu_cpu.AI(), static_cast<size_t>( n + 1 ) * sizeof( int ),
                                cudaMemcpyHostToDevice ) );
    ASSERT_CUDA_OK( cudaMemcpy( d_lu_aj, lu_cpu.AJ(), static_cast<size_t>( nnz_lu ) * sizeof( int ),
                                cudaMemcpyHostToDevice ) );
    ASSERT_CUDA_OK( cudaMemcpy( d_lu_diag, lu_cpu.Diagonal(),
                                static_cast<size_t>( n ) * sizeof( int ), cudaMemcpyHostToDevice ) );
    ASSERT_CUDA_OK( cudaMemcpy( d_level_perm, level_perm.data(),
                                static_cast<size_t>( n ) * sizeof( int ), cudaMemcpyHostToDevice ) );

    cudaStream_t stream = nullptr;
    ASSERT_CUDA_OK( cudaStreamCreate( &stream ) );

    DeviceILUUpdateCache<int> device_cache;
    ASSERT_CUDA_OK( BuildILUUpdateCacheAsync<int, int>( n, d_lu_ai, d_lu_aj, d_lu_diag, base,
                                                        device_cache, stream ) );

    std::vector<int> device_lower_row_ptr( device_cache.lower_row_ptr.size() );
    std::vector<int> device_update_ptr( device_cache.update_ptr.size() );
    std::vector<int> device_update_jpos( device_cache.update_jpos.size() );
    std::vector<int> device_update_pos( device_cache.update_pos.size() );
    device_cache.lower_row_ptr.copyToHost( device_lower_row_ptr.data() );
    device_cache.update_ptr.copyToHost( device_update_ptr.data() );
    device_cache.update_jpos.copyToHost( device_update_jpos.data() );
    device_cache.update_pos.copyToHost( device_update_pos.data() );

    EXPECT_EQ( device_cache.strict_lower_nnz, host_cache.strict_lower_nnz );
    EXPECT_EQ( device_cache.total_updates, host_cache.total_updates );
    EXPECT_EQ( device_lower_row_ptr, host_cache.lower_row_ptr );
    EXPECT_EQ( device_update_ptr, host_cache.update_ptr );
    EXPECT_EQ( device_update_jpos, host_cache.update_jpos );
    EXPECT_EQ( device_update_pos, host_cache.update_pos );

    ASSERT_CUDA_OK( ILUEmbedAValuesToLUAsync<int, int, double>(
        n, d_a_ai, d_a_aj, d_a_av, d_lu_ai, d_lu_aj, base, d_lu_initial, stream ) );

    for ( const auto row_lookup : { ILUNumericRowLookup::Global, ILUNumericRowLookup::Shared } )
    {
        for ( const auto row_update :
              { ILUNumericRowUpdateStrategy::BinarySearch, ILUNumericRowUpdateStrategy::Merge } )
        {
            SCOPED_TRACE( "row_lookup=" + std::to_string( static_cast<int>( row_lookup ) ) +
                          " row_update=" + std::to_string( static_cast<int>( row_update ) ) );
            ASSERT_CUDA_OK( cudaMemcpyAsync( d_lu_av, d_lu_initial, static_cast<size_t>( nnz_lu ) * sizeof( double ),
                                             cudaMemcpyDeviceToDevice, stream ) );
            ASSERT_CUDA_OK( ILUBaseNumericFactorizationAsync<int, int, double>(
                n, d_lu_ai, d_lu_aj, d_lu_diag, d_level_perm, level_prefix.data(), levels, base,
                d_lu_av, d_status, row_lookup, row_update, stream ) );

            int h_status = 1;
            ASSERT_CUDA_OK( cudaMemcpyAsync( &h_status, d_status, sizeof( int ), cudaMemcpyDeviceToHost, stream ) );
            ASSERT_CUDA_OK( cudaStreamSynchronize( stream ) );
            ASSERT_EQ( h_status, 0 );

            std::vector<double> lu_gpu( static_cast<size_t>( nnz_lu ) );
            ASSERT_CUDA_OK( cudaMemcpy( lu_gpu.data(), d_lu_av, static_cast<size_t>( nnz_lu ) * sizeof( double ),
                                        cudaMemcpyDeviceToHost ) );

            for ( int i = 0; i < nnz_lu; ++i )
            {
                EXPECT_NEAR( lu_gpu[i], lu_cpu.AV()[i], 1e-10 ) << "Mismatch at LU value " << i;
            }
        }
    }

    ASSERT_CUDA_OK( cudaMemcpyAsync( d_lu_av, d_lu_initial, static_cast<size_t>( nnz_lu ) * sizeof( double ),
                                     cudaMemcpyDeviceToDevice, stream ) );
    ASSERT_CUDA_OK( ILUBaseNumericFactorizationCachedAsync<int, int, double>(
        n, d_lu_ai, d_lu_aj, d_lu_diag, device_cache.lower_row_ptr.data(),
        device_cache.update_ptr.data(), device_cache.update_jpos.data(), device_cache.update_pos.data(),
        d_level_perm, level_prefix.data(), levels, base, d_lu_av, d_status, stream ) );

    int h_status = 1;
    ASSERT_CUDA_OK( cudaMemcpyAsync( &h_status, d_status, sizeof( int ), cudaMemcpyDeviceToHost, stream ) );
    ASSERT_CUDA_OK( cudaStreamSynchronize( stream ) );
    ASSERT_EQ( h_status, 0 );

    std::vector<double> lu_cached_gpu( static_cast<size_t>( nnz_lu ) );
    ASSERT_CUDA_OK( cudaMemcpy( lu_cached_gpu.data(), d_lu_av,
                                static_cast<size_t>( nnz_lu ) * sizeof( double ), cudaMemcpyDeviceToHost ) );

    for ( int i = 0; i < nnz_lu; ++i )
    {
        EXPECT_NEAR( lu_cached_gpu[i], lu_cpu.AV()[i], 1e-10 ) << "Cached mismatch at LU value " << i;
    }

    ASSERT_CUDA_OK( cudaMemcpyAsync( d_lu_av, d_lu_initial, static_cast<size_t>( nnz_lu ) * sizeof( double ),
                                     cudaMemcpyDeviceToDevice, stream ) );
    ILUPersistentLaunchConfig persistent_launch;
    ASSERT_CUDA_OK( ILUBaseNumericFactorizationPersistentAsync<int, int, double>(
        n, d_lu_ai, d_lu_aj, d_lu_diag, base, d_lu_av, d_diag_inv, d_status, d_next_row, d_row_done,
        stream, &persistent_launch ) );

    h_status = 1;
    ASSERT_CUDA_OK( cudaMemcpyAsync( &h_status, d_status, sizeof( int ), cudaMemcpyDeviceToHost, stream ) );
    ASSERT_CUDA_OK( cudaStreamSynchronize( stream ) );
    ASSERT_EQ( h_status, 0 );
    EXPECT_GT( persistent_launch.block_size, 0 );
    EXPECT_GT( persistent_launch.grid_blocks, 0 );
    EXPECT_GT( persistent_launch.resident_warps, 0 );

    std::vector<double> lu_persistent_gpu( static_cast<size_t>( nnz_lu ) );
    ASSERT_CUDA_OK( cudaMemcpy( lu_persistent_gpu.data(), d_lu_av,
                                static_cast<size_t>( nnz_lu ) * sizeof( double ), cudaMemcpyDeviceToHost ) );

    for ( int i = 0; i < nnz_lu; ++i )
    {
        EXPECT_NEAR( lu_persistent_gpu[i], lu_cpu.AV()[i], 1e-8 )
            << "Persistent mismatch at LU value " << i;
    }

    ASSERT_CUDA_OK( cudaStreamDestroy( stream ) );

    cudaFree( d_a_ai );
    cudaFree( d_a_aj );
    cudaFree( d_a_av );
    cudaFree( d_lu_ai );
    cudaFree( d_lu_aj );
    cudaFree( d_lu_diag );
    cudaFree( d_level_perm );
    cudaFree( d_status );
    cudaFree( d_next_row );
    cudaFree( d_row_done );
    cudaFree( d_diag_inv );
    cudaFree( d_lu_initial );
    cudaFree( d_lu_av );
}
