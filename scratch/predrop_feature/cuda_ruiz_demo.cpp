#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cxxopts.hpp>

#include "io.hpp"
#include "matrix_utils.hpp"
#include "cuda_ruiz_scale.cuh"

struct Options
{
    std::string filename;
    std::string output_file;
    int max_iters;
    std::string norm_type;
    bool verbose;
};

void printOptions( const Options& opts )
{
    std::cout << "Options:" << std::endl;
    std::cout << "  filename: " << opts.filename << std::endl;
    std::cout << "  output: " << opts.output_file << std::endl;
    std::cout << "  max_iters: " << opts.max_iters << std::endl;
    std::cout << "  norm_type: " << opts.norm_type << std::endl;
    std::cout << "  verbose: " << ( opts.verbose ? "true" : "false" ) << std::endl;
}

void cudaCheck( cudaError_t error, const char* message )
{
    if ( error != cudaSuccess )
    {
        std::cerr << "CUDA error: " << message << " - " << cudaGetErrorString( error ) << std::endl;
        exit( 1 );
    }
}

int main( int argc, char* argv[] )
{
    cxxopts::Options options( "cuda_ruiz_demo", "CUDA Ruiz scaling for matrix equilibration" );

    // clang-format off
    options.add_options()
        ("f,filename", "Matrix Market file to read", cxxopts::value<std::string>()->default_value("../../data/ex5.mtx"))
        ("o,output", "Output Matrix Market file path", cxxopts::value<std::string>()->default_value("ruiz_scaled.mtx"))
        ("i,iters", "Maximum iterations", cxxopts::value<int>()->default_value("20"))
        ("n,norm", "Norm type: maxnorm | l2norm", cxxopts::value<std::string>()->default_value("maxnorm"))
        ("v,verbose", "Verbose output", cxxopts::value<bool>()->default_value("false"))
        ("h,help", "Print usage");
    // clang-format on

    auto result = options.parse( argc, argv );

    if ( result.count( "help" ) )
    {
        std::cout << options.help() << std::endl;
        return 0;
    }

    Options opts;
    opts.filename = result["filename"].as<std::string>();
    opts.output_file = result["output"].as<std::string>();
    opts.max_iters = result["iters"].as<int>();
    opts.norm_type = result["norm"].as<std::string>();
    opts.verbose = result["verbose"].as<bool>();

    printOptions( opts );

    // Read matrix
    std::ifstream f( opts.filename );
    if ( !f.is_open() )
    {
        std::cerr << "Failed to open file: " << opts.filename << std::endl;
        return -1;
    }

    matrix_utils::CSRMatrix<int32_t, int32_t, double> matrix;
    matrix_utils::readMatrixMarket( f, matrix );
    f.close();

    std::cout << "\nOriginal matrix: " << matrix.rows << " x " << matrix.cols
              << ", NNZ: " << matrix.NNZ() << std::endl;

    // Allocate device memory
    std::cout << "\nAllocating device memory..." << std::endl;
    int32_t *d_ai, *d_aj;
    double *d_av, *d_dr, *d_dc;

    size_t rows_bytes = ( matrix.rows + 1 ) * sizeof( int32_t );
    size_t nnz_idx_bytes = matrix.NNZ() * sizeof( int32_t );
    size_t nnz_val_bytes = matrix.NNZ() * sizeof( double );
    size_t rows_vals_bytes = matrix.rows * sizeof( double );
    size_t cols_vals_bytes = matrix.cols * sizeof( double );

    cudaCheck( cudaMalloc( &d_ai, rows_bytes ), "Failed to allocate d_ai" );
    cudaCheck( cudaMalloc( &d_aj, nnz_idx_bytes ), "Failed to allocate d_aj" );
    cudaCheck( cudaMalloc( &d_av, nnz_val_bytes ), "Failed to allocate d_av" );
    cudaCheck( cudaMalloc( &d_dr, rows_vals_bytes ), "Failed to allocate d_dr" );
    cudaCheck( cudaMalloc( &d_dc, cols_vals_bytes ), "Failed to allocate d_dc" );

    // Copy matrix to device
    std::cout << "Copying matrix to device..." << std::endl;
    cudaCheck( cudaMemcpy( d_ai, matrix.AI(), rows_bytes, cudaMemcpyHostToDevice ),
               "Failed to copy d_ai" );
    cudaCheck( cudaMemcpy( d_aj, matrix.AJ(), nnz_idx_bytes, cudaMemcpyHostToDevice ),
               "Failed to copy d_aj" );
    cudaCheck( cudaMemcpy( d_av, matrix.AV(), nnz_val_bytes, cudaMemcpyHostToDevice ),
               "Failed to copy d_av" );

    // Perform Ruiz scaling
    std::cout << "\n=== Performing Ruiz scaling ===" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    bool converged = false;
    if ( opts.norm_type == "maxnorm" )
    {
        std::cout << "Using MaxNorm..." << std::endl;
        converged =
            matrix_utils::sparse_cuda::RuizScaleCuda<int32_t, int32_t, double, matrix_utils::sparse_cuda::CudaRuizScalingNormType::MaxNorm>(
                matrix.rows, matrix.cols, d_ai, d_aj, d_av, d_dr, d_dc, opts.max_iters );
    }
    else if ( opts.norm_type == "l2norm" )
    {
        std::cout << "Using L2Norm..." << std::endl;
        converged =
            matrix_utils::sparse_cuda::RuizScaleCuda<int32_t, int32_t, double, matrix_utils::sparse_cuda::CudaRuizScalingNormType::L2Norm>(
                matrix.rows, matrix.cols, d_ai, d_aj, d_av, d_dr, d_dc, opts.max_iters );
    }
    else
    {
        std::cerr << "Unknown norm type: " << opts.norm_type << std::endl;
        return -1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Ruiz scaling completed in " << duration.count() << " seconds" << std::endl;
    std::cout << "Converged: " << ( converged ? "yes" : "no" ) << std::endl;

    // Copy scaled matrix back to host
    std::cout << "\nCopying scaled matrix back to host..." << std::endl;
    cudaCheck( cudaMemcpy( matrix.AV(), d_av, nnz_val_bytes, cudaMemcpyDeviceToHost ),
               "Failed to copy scaled matrix values" );

    // Copy scaling factors to host for inspection
    std::vector<double> h_dr( matrix.rows );
    std::vector<double> h_dc( matrix.cols );
    cudaCheck( cudaMemcpy( h_dr.data(), d_dr, rows_vals_bytes, cudaMemcpyDeviceToHost ),
               "Failed to copy row scaling factors" );
    cudaCheck( cudaMemcpy( h_dc.data(), d_dc, cols_vals_bytes, cudaMemcpyDeviceToHost ),
               "Failed to copy column scaling factors" );

    // Print some statistics
    std::cout << "\n=== Scaling Statistics ===" << std::endl;
    double min_row_scale = h_dr[0], max_row_scale = h_dr[0];
    for ( int i = 0; i < matrix.rows; ++i )
    {
        min_row_scale = std::min( min_row_scale, h_dr[i] );
        max_row_scale = std::max( max_row_scale, h_dr[i] );
    }
    std::cout << "Row scaling factors range: [" << min_row_scale << ", " << max_row_scale << "]" << std::endl;

    double min_col_scale = h_dc[0], max_col_scale = h_dc[0];
    for ( int i = 0; i < matrix.cols; ++i )
    {
        min_col_scale = std::min( min_col_scale, h_dc[i] );
        max_col_scale = std::max( max_col_scale, h_dc[i] );
    }
    std::cout << "Column scaling factors range: [" << min_col_scale << ", " << max_col_scale << "]"
              << std::endl;

    // Write scaled matrix to MTX file
    std::cout << "\nWriting scaled matrix to: " << opts.output_file << std::endl;
    std::ofstream out( opts.output_file );
    if ( !out.is_open() )
    {
        std::cerr << "Failed to create output file: " << opts.output_file << std::endl;
        return -1;
    }

    matrix_utils::writeMatrixMarket( matrix, out );
    out.close();
    std::cout << "Scaled matrix written successfully" << std::endl;

    // Cleanup
    std::cout << "\nCleaning up..." << std::endl;
    cudaFree( d_ai );
    cudaFree( d_aj );
    cudaFree( d_av );
    cudaFree( d_dr );
    cudaFree( d_dc );
    cudaCheck( cudaDeviceSynchronize(), "Final synchronization failed" );

    std::cout << "Done!" << std::endl;
    return 0;
}
