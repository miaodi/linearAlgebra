#include "Reordering.h"
#include "config.h"
#include "io.hpp"
#include "matrix_utils.hpp"
#include "permutation.hpp"
#include "sp_ops.hpp"

#include <chrono>
#include <cstdlib>
#include <cxxopts.hpp>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using CSR = matrix_utils::CSRMatrix<int, int, double>;

struct Options
{
    std::string input;
    std::string output;
    std::string sparsity_png;
    bool metis_reorder{ false };
    int threads{ 1 };
    int seed{ 42 };
    int sparsity_resolution{ 4096 };
};

CSR metisReorder( const CSR& matrix, const int threads, const int seed )
{
#ifndef USE_METIS_LIB
    (void)matrix;
    (void)threads;
    (void)seed;
    throw std::runtime_error( "METIS support is not enabled. Reconfigure with USE_METIS_LIB=ON." );
#else
    if ( matrix.rows != matrix.cols )
    {
        throw std::runtime_error( "METIS nested dissection requires a square matrix." );
    }
    if ( matrix.Base() != 0 )
    {
        throw std::runtime_error( "METIS reorder path expects zero-based CSR input." );
    }

    std::vector<int> xadj( static_cast<std::size_t>( matrix.rows + 1 ) );
    matrix_utils::APlusATPrefix<int, int, false>( matrix.rows, matrix.AI(), matrix.AJ(), xadj.data() );
    const int graph_edges = xadj[matrix.rows] - xadj[0];
    std::vector<int> adjncy( static_cast<std::size_t>( graph_edges ) );
    matrix_utils::APlusATFill<int, int, false>( matrix.rows, matrix.AI(), matrix.AJ(), xadj.data(),
                                                adjncy.data() );

    std::vector<int> iperm( static_cast<std::size_t>( matrix.rows ) );
    std::vector<int> perm( static_cast<std::size_t>( matrix.rows ) );

    reordering::MetisNDOptions nd_options;
    nd_options.seed = seed;

    const int rc = reordering::MetisND<int, int>(
        matrix.rows, matrix.cols, xadj.data(), adjncy.data(), iperm.data(), perm.data(), nd_options );
    if ( rc != 0 )
    {
        throw std::runtime_error( "METIS nested dissection failed with code " + std::to_string( rc ) );
    }

    const bool perm_ok = matrix_utils::isPermutation<int>( matrix.rows, 0, perm.data(), threads );
    const bool iperm_ok = matrix_utils::isPermutation<int>( matrix.rows, 0, iperm.data(), threads );
    if ( !perm_ok || !iperm_ok )
    {
        throw std::runtime_error( "METIS produced an invalid permutation." );
    }

    CSR reordered;
    reordered.rows = matrix.rows;
    reordered.cols = matrix.cols;
    reordered.ResizeAI( matrix.rows + 1 );
    reordered.ResizeAJ( matrix.NNZ() );
    reordered.ResizeAV( matrix.NNZ() );

    matrix_utils::permuteMat( matrix.rows, matrix.cols, perm.data(), iperm.data(), matrix.AI(),
                              matrix.AJ(), matrix.AV(), reordered.AI(), reordered.AJ(),
                              reordered.AV(), threads );
    return reordered;
#endif
}

Options parseOptions( const int argc, char** argv )
{
    cxxopts::Options options(
        "mtx_to_binary",
        "Convert a MatrixMarket text matrix to the project binary matrix format." );
    options.add_options()( "i,input", "Input MatrixMarket .mtx file", cxxopts::value<std::string>() )(
        "o,output", "Output binary matrix file", cxxopts::value<std::string>() )(
        "sparsity-png", "Write the output matrix sparsity pattern to a PNG file",
        cxxopts::value<std::string>() )( "sparsity-resolution",
                                         "Maximum sparsity PNG width/height in pixels",
                                         cxxopts::value<int>()->default_value( "4096" ) )(
        "metis", "Apply METIS nested-dissection symmetric reordering before writing" )(
        "t,threads", "Threads used for permutation work", cxxopts::value<int>()->default_value( "1" ) )(
        "seed", "METIS random seed", cxxopts::value<int>()->default_value( "42" ) )(
        "h,help", "Print usage" );

    const auto parsed = options.parse( argc, argv );
    if ( parsed.count( "help" ) )
    {
        std::cout << options.help() << '\n';
        std::exit( 0 );
    }
    if ( !parsed.count( "input" ) || !parsed.count( "output" ) )
    {
        throw std::invalid_argument( "Both --input and --output are required.\n" + options.help() );
    }

    Options result;
    result.input = parsed["input"].as<std::string>();
    result.output = parsed["output"].as<std::string>();
    if ( parsed.count( "sparsity-png" ) )
    {
        result.sparsity_png = parsed["sparsity-png"].as<std::string>();
    }
    result.metis_reorder = parsed.count( "metis" ) > 0;
    result.threads = parsed["threads"].as<int>();
    result.seed = parsed["seed"].as<int>();
    result.sparsity_resolution = parsed["sparsity-resolution"].as<int>();
    if ( result.threads <= 0 )
    {
        throw std::invalid_argument( "--threads must be positive." );
    }
    if ( result.sparsity_resolution <= 0 )
    {
        throw std::invalid_argument( "--sparsity-resolution must be positive." );
    }
    return result;
}
} // namespace

int main( int argc, char** argv )
{
    try
    {
        const Options options = parseOptions( argc, argv );

        const auto read_start = std::chrono::steady_clock::now();
        CSR matrix;
        matrix_utils::readMatrix( options.input, matrix, matrix_utils::MatrixDataType::MatrixMarket );
        const auto read_end = std::chrono::steady_clock::now();

        std::cout << "Loaded " << options.input << ": rows=" << matrix.rows << ", cols=" << matrix.cols
                  << ", nnz=" << matrix.NNZ() << ", base=" << matrix.Base() << '\n';

        if ( options.metis_reorder )
        {
            const auto reorder_start = std::chrono::steady_clock::now();
            matrix = metisReorder( matrix, options.threads, options.seed );
            const auto reorder_end = std::chrono::steady_clock::now();
            const std::chrono::duration<double> reorder_time = reorder_end - reorder_start;
            std::cout << "Applied METIS nested-dissection reordering in " << reorder_time.count() << " s\n";
        }

        if ( !options.sparsity_png.empty() )
        {
            const auto png_start = std::chrono::steady_clock::now();
            matrix_utils::writePNG( matrix, options.sparsity_png, options.sparsity_resolution );
            const auto png_end = std::chrono::steady_clock::now();
            const std::chrono::duration<double> png_time = png_end - png_start;
            std::cout << "Wrote sparsity PNG " << options.sparsity_png << " at up to "
                      << options.sparsity_resolution << "x" << options.sparsity_resolution
                      << " pixels in " << png_time.count() << " s\n";
        }

        const auto write_start = std::chrono::steady_clock::now();
        matrix_utils::writeMatrix( matrix, options.output, matrix_utils::MatrixDataType::Binary );
        const auto write_end = std::chrono::steady_clock::now();

        const std::chrono::duration<double> read_time = read_end - read_start;
        const std::chrono::duration<double> write_time = write_end - write_start;
        std::cout << "Wrote " << options.output << " in " << write_time.count() << " s\n";
        std::cout << "Read time: " << read_time.count() << " s\n";
        return 0;
    }
    catch ( const std::exception& e )
    {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
