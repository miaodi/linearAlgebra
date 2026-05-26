#include "io.hpp"
#include "matrix_utils.hpp"
#include "graph_algs.hpp"
#include <cxxopts.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <iomanip>

int main( int argc, char* argv[] )
{
    cxxopts::Options options(
        "jaccard_compare",
        "Compare Jaccard similarity between multiple matrices (optionally after drop)" );
    options.add_options()( "f,filename", "Matrix Market files to read (multiple allowed)",
                           cxxopts::value<std::vector<std::string>>() )(
        "t,threshold", "Pruning threshold (absolute value)",
        cxxopts::value<double>()->default_value( "0.0" ) )( "h,help", "Print usage" );
    auto result = options.parse( argc, argv );
    if ( result.count( "help" ) || !result.count( "filename" ) )
    {
        std::cout << options.help() << std::endl;
        return 0;
    }
    std::vector<std::string> filenames = result["filename"].as<std::vector<std::string>>();
    double threshold = result["threshold"].as<double>();
    size_t n = filenames.size();
    if ( n < 2 )
    {
        std::cerr << "Please provide at least two matrix files." << std::endl;
        return 1;
    }
    std::vector<matrix_utils::CSRMatrix<int, int, double>> matrices( n );
    for ( size_t i = 0; i < n; ++i )
    {
        std::ifstream f( filenames[i] );
        if ( !f.is_open() )
        {
            std::cerr << "Failed to open file: " << filenames[i] << std::endl;
            return 1;
        }
        matrix_utils::readMatrixMarket( f, matrices[i] );
        f.close();
    }
    // Check dimensions
    int rows = matrices[0].rows, cols = matrices[0].cols;
    for ( size_t i = 1; i < n; ++i )
    {
        if ( matrices[i].rows != rows || matrices[i].cols != cols )
        {
            std::cerr << "All matrices must have the same dimensions." << std::endl;
            return 1;
        }
    }
    // Apply drop (pruning) to each matrix
    std::vector<matrix_utils::CSRMatrix<int, int, double>> pruned( n );
    for ( size_t i = 0; i < n; ++i )
    {
        pruned[i].rows = matrices[i].rows;
        pruned[i].cols = matrices[i].cols;
        pruned[i].ResizeAI( matrices[i].rows + 1 );
        pruned[i].ResizeAJ( matrices[i].NNZ() );
        pruned[i].ResizeAV( matrices[i].NNZ() );
        std::memcpy( pruned[i].AI(), matrices[i].AI(), ( matrices[i].rows + 1 ) * sizeof( int ) );
        std::memcpy( pruned[i].AJ(), matrices[i].AJ(), matrices[i].NNZ() * sizeof( int ) );
        std::memcpy( pruned[i].AV(), matrices[i].AV(), matrices[i].NNZ() * sizeof( double ) );
        int original_nnz = pruned[i].NNZ();
        int removed = matrix_utils::DiagonalScaledPrune(
            pruned[i].rows, pruned[i].AI(), pruned[i].AJ(), pruned[i].AV(), threshold );
        int pruned_nnz = pruned[i].NNZ();
        std::cout << "Matrix " << i << " dropped: " << removed << " entries (" << original_nnz
                  << " -> " << pruned_nnz << ", " << std::fixed << std::setprecision( 2 )
                  << ( pruned_nnz * 100.0 / original_nnz ) << "% retained)" << std::endl;
    }
    // Compute Jaccard similarity matrix (after drop)
    std::vector<std::vector<double>> jaccard( n, std::vector<double>( n, 0.0 ) );
    for ( size_t i = 0; i < n; ++i )
    {
        for ( size_t j = 0; j < n; ++j )
        {
            jaccard[i][j] = graph::jaccardSimilarity( rows, cols, pruned[i].AI(), pruned[i].AJ(),
                                                      rows, cols, pruned[j].AI(), pruned[j].AJ() );
        }
    }
    // Print matrix
    std::cout << "Jaccard Similarity Matrix (after drop):" << std::endl;
    std::cout << std::fixed << std::setprecision( 4 );
    for ( size_t i = 0; i < n; ++i )
    {
        for ( size_t j = 0; j < n; ++j )
        {
            std::cout << jaccard[i][j] << ( j + 1 == n ? "\n" : " " );
        }
    }
    // Optionally print file names as header
    std::cout << "Files:" << std::endl;
    for ( size_t i = 0; i < n; ++i )
    {
        std::cout << i << ": " << filenames[i] << std::endl;
    }
    return 0;
}
