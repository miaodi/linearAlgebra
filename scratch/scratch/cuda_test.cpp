#include "cuda_tiled_sparse_mat.cuh"
#include "io.hpp"
#include "matrix_utils.hpp"

#include <cuda_runtime.h>
#include <cxxopts.hpp>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
inline void check_cuda( cudaError_t status, const char* msg )
{
    if ( status != cudaSuccess )
    {
        throw std::runtime_error( std::string( msg ) + ": " + cudaGetErrorString( status ) );
    }
}

} // namespace

int main( int argc, char** argv )
{
    cxxopts::Options options( "cuda_test",
                              "Read MTX, convert CSR to tiled CSR on GPU, and print entries" );
    options.add_options()( "f,file", "Input MatrixMarket file path", cxxopts::value<std::string>() )(
        "k", "Tile exponent (tile size = 2^k)", cxxopts::value<int>()->default_value( "4" ) )(
        "h,help", "Show help" );

    const auto parsed = options.parse( argc, argv );
    if ( parsed.count( "help" ) || !parsed.count( "file" ) )
    {
        std::cout << options.help() << '\n';
        return parsed.count( "file" ) ? 0 : 1;
    }

    const std::string file_path = parsed["file"].as<std::string>();
    const int k = parsed["k"].as<int>();
    if ( k < 0 || k >= 63 )
    {
        throw std::invalid_argument( "k must satisfy 0 <= k < 63" );
    }

    using HostCSR = matrix_utils::CSRMatrixVec<int, int, double>;
    HostCSR h_csr;

    {
        std::ifstream in( file_path );
        if ( !in.is_open() )
        {
            throw std::runtime_error( "Cannot open MatrixMarket file: " + file_path );
        }
        matrix_utils::readMatrixMarket( in, h_csr );
    }

    const int rows = h_csr.rows;
    const int cols = h_csr.cols;
    const int nnz = static_cast<int>( h_csr.NNZ() );

    std::cout << "Loaded matrix: rows=" << rows << ", cols=" << cols << ", nnz=" << nnz
              << ", base=" << h_csr.Base() << '\n';

    int* d_ai = nullptr;
    int* d_aj = nullptr;
    double* d_av = nullptr;

    check_cuda( cudaMalloc( reinterpret_cast<void**>( &d_ai ), sizeof( int ) * static_cast<size_t>( rows + 1 ) ),
                "cudaMalloc d_ai" );
    check_cuda( cudaMalloc( reinterpret_cast<void**>( &d_aj ), sizeof( int ) * static_cast<size_t>( nnz ) ),
                "cudaMalloc d_aj" );
    check_cuda( cudaMalloc( reinterpret_cast<void**>( &d_av ), sizeof( double ) * static_cast<size_t>( nnz ) ),
                "cudaMalloc d_av" );

    check_cuda( cudaMemcpy( d_ai, h_csr.AI(), sizeof( int ) * static_cast<size_t>( rows + 1 ), cudaMemcpyHostToDevice ),
                "copy AI to device" );
    check_cuda( cudaMemcpy( d_aj, h_csr.AJ(), sizeof( int ) * static_cast<size_t>( nnz ), cudaMemcpyHostToDevice ),
                "copy AJ to device" );
    check_cuda( cudaMemcpy( d_av, h_csr.AV(), sizeof( double ) * static_cast<size_t>( nnz ), cudaMemcpyHostToDevice ),
                "copy AV to device" );

    matrix_utils::sparse_cuda::DeviceTileCOOMatrix<int, int, double> d_tiled;
    matrix_utils::sparse_cuda::CSRToTileCOO<int, int, double>( rows, cols, d_ai, d_aj, d_av, k, d_tiled, nullptr );
    check_cuda( cudaDeviceSynchronize(), "CSRToTileCOO synchronize" );

    std::vector<int> h_perm( static_cast<size_t>( nnz ) );
    std::vector<int> h_tile_nnz_prefix( static_cast<size_t>( d_tiled.n_tiles + 1 ) );
    std::vector<int> h_tile_rows( static_cast<size_t>( d_tiled.n_tiles ) );
    std::vector<int> h_tile_cols( static_cast<size_t>( d_tiled.n_tiles ) );
    std::vector<int> h_row_ind( static_cast<size_t>( nnz ) );
    std::vector<int> h_col_ind( static_cast<size_t>( nnz ) );
    std::vector<double> h_values( static_cast<size_t>( nnz ) );

    d_tiled.permutation.copyToHost( h_perm.data() );
    d_tiled.tile_nnz_prefix.copyToHost( h_tile_nnz_prefix.data() );
    d_tiled.tile_row_ind.copyToHost( h_tile_rows.data() );
    d_tiled.tile_col_ind.copyToHost( h_tile_cols.data() );
    d_tiled.row_ind.copyToHost( h_row_ind.data() );
    d_tiled.col_ind.copyToHost( h_col_ind.data() );
    d_tiled.values.copyToHost( h_values.data() );

    const std::uint64_t tile_size = std::uint64_t{ 1 } << k;
    std::cout << "Tiled metadata COO (tile_k=" << k << ", tile_size=" << tile_size
              << ", n_tiles=" << d_tiled.n_tiles << "):" << '\n';

    for ( int t = 0; t < d_tiled.n_tiles; ++t )
    {
        std::cout << "tile=" << t << " tile_row=" << h_tile_rows[static_cast<size_t>( t )]
                  << " tile_col=" << h_tile_cols[static_cast<size_t>( t )] << " tile_nnz="
                  << ( h_tile_nnz_prefix[static_cast<size_t>( t + 1 )] -
                       h_tile_nnz_prefix[static_cast<size_t>( t )] )
                  << '\n';
    }

    // std::cout << "\nValues/indices grouped by sorted tile order:" << '\n';

    // for (int i = 0; i < nnz; ++i)
    // {
    //     std::cout << "i=" << i << " perm=" << h_perm[static_cast<size_t>(i)]
    //               << " row=" << h_row_ind[static_cast<size_t>(i)]
    //               << " col=" << h_col_ind[static_cast<size_t>(i)]
    //               << " val=" << h_values[static_cast<size_t>(i)] << '\n';
    // }

    check_cuda( cudaFree( d_ai ), "cudaFree d_ai" );
    check_cuda( cudaFree( d_aj ), "cudaFree d_aj" );
    check_cuda( cudaFree( d_av ), "cudaFree d_av" );

    return 0;
}
