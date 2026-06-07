#include "config.h"
#include "graph_algs.hpp"
#include "io.hpp"
#include "matrix_utils.hpp"
#include "permutation.hpp"
#include "precond.hpp"
#include "Reordering.h"
#include "sp_ops.hpp"

#include <cxxopts.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using CSR = matrix_utils::CSRMatrix<int, int, double>;

enum class SymbolicMode
{
    V1,
    V2
};

enum class ReorderMode
{
    None,
    Metis
};

SymbolicMode parse_symbolic_mode( const std::string& mode )
{
    if ( mode == "v1" )
    {
        return SymbolicMode::V1;
    }
    if ( mode == "v2" )
    {
        return SymbolicMode::V2;
    }
    throw std::invalid_argument( "Symbolic mode must be one of: v1, v2" );
}

ReorderMode parse_reorder_mode( const std::string& mode )
{
    if ( mode == "none" )
    {
        return ReorderMode::None;
    }
    if ( mode == "metis" )
    {
        return ReorderMode::Metis;
    }
    throw std::invalid_argument( "Reorder mode must be one of: none, metis" );
}

const char* symbolic_mode_label( const SymbolicMode mode )
{
    switch ( mode )
    {
    case SymbolicMode::V1:
        return "ILULevelSymbolicParallel";
    case SymbolicMode::V2:
        return "ILULevelSymbolicParallelV2";
    }
    return "Unknown symbolic mode";
}

const char* reorder_mode_label( const ReorderMode mode )
{
    switch ( mode )
    {
    case ReorderMode::None:
        return "none";
    case ReorderMode::Metis:
        return "metis";
    }
    return "unknown";
}

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
    matrix_utils::APlusATFill<int, int, false>( matrix.rows, matrix.AI(), matrix.AJ(), xadj.data(),
                                                adjncy.data() );

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
    matrix_utils::permuteMat( matrix.rows, matrix.cols, perm.data(), iperm.data(), matrix.AI(),
                              matrix.AJ(), matrix.AV(), reordered.AI(), reordered.AJ(),
                              reordered.AV(), threads );

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
        "n,threads", "Number of threads for ILU symbolic setup", cxxopts::value<int>()->default_value( "1" ) )(
        "symbolic-mode", "Symbolic setup mode: v1, v2", cxxopts::value<std::string>()->default_value( "v2" ) )(
        "reorder", "Matrix reordering mode before ILU: none, metis",
        cxxopts::value<std::string>()->default_value( "none" ) )( "h,help", "Print usage" );

    const auto parsed = options.parse( argc, argv );
    if ( parsed.count( "help" ) )
    {
        std::cout << options.help() << '\n';
        return 0;
    }

    const std::string file_path = parsed["file"].as<std::string>();
    const int level = parsed["level"].as<int>();
    const int threads = parsed["threads"].as<int>();
    SymbolicMode symbolic_mode = SymbolicMode::V2;
    ReorderMode reorder_mode = ReorderMode::Metis;
    try
    {
        symbolic_mode = parse_symbolic_mode( parsed["symbolic-mode"].as<std::string>() );
        reorder_mode = parse_reorder_mode( parsed["reorder"].as<std::string>() );
    }
    catch ( const std::exception& e )
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
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
    if ( reorder_mode == ReorderMode::Metis )
    {
        if ( !metis_reorder_matrix( input_matrix, matrix, threads ) )
        {
            return 1;
        }
    }
    else
    {
        matrix = input_matrix;
    }
    std::cout << "Reorder mode: " << reorder_mode_label( reorder_mode ) << '\n';
    std::cout << "ILU input matrix: rows=" << matrix.rows << ", cols=" << matrix.cols
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

    CSR lu;
    const auto symbolic_start = std::chrono::steady_clock::now();
    bool symbolic_success = false;
    if ( symbolic_mode == SymbolicMode::V1 )
    {
        matrix_utils::ILULevelSymbolicParallel<CSR, enums::matrix_utils::LU, true> symbolic( threads );
        symbolic_success = symbolic( matrix.rows, matrix.AI(), matrix.AJ(), level, lu );
    }
    else
    {
        matrix_utils::ILULevelSymbolicParallelV2<CSR, enums::matrix_utils::LU, true> symbolic( threads );
        symbolic_success = symbolic.apply( matrix.rows, matrix.AI(), matrix.AJ(), level, lu );
    }
    if ( !symbolic_success )
    {
        std::cerr << symbolic_mode_label( symbolic_mode ) << " failed\n";
        return 1;
    }
    const auto symbolic_end = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> symbolic_ms = symbolic_end - symbolic_start;
    std::cout << "ILU(" << level << ") symbolic LU: rows=" << lu.rows << ", cols=" << lu.cols
              << ", nnz=" << lu.NNZ() << ", base=" << lu.Base() << '\n';
    std::cout << symbolic_mode_label( symbolic_mode ) << " time: " << symbolic_ms.count() << " ms\n";

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

    // const auto numeric_start = std::chrono::steady_clock::now();
    // if ( !matrix_utils::ILUNumeric( matrix.rows, matrix.AI(), matrix.AJ(), matrix.AV(), lu ) )
    // {
    //     std::cerr << "ILUNumeric failed\n";
    //     return 1;
    // }
    // const auto numeric_end = std::chrono::steady_clock::now();
    // const std::chrono::duration<double, std::milli> numeric_ms = numeric_end - numeric_start;
    // std::cout << "ILU(" << level << ") numeric factorization complete\n";
    // std::cout << "ILUNumeric time: " << numeric_ms.count() << " ms\n";

    // std::cout << "ILU factorization level analysis complete\n";
    return 0;
}
