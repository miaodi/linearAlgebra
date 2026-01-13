#include "io.hpp"
#include "matrix_utils.hpp"
#include "precond.hpp"
#include <algorithm>
#include <cxxopts.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include <omp.h>
#include <string>
#include <vector>

using CSR = matrix_utils::CSRMatrix<int, int, double>;

static bool compare_csr_pattern( const CSR& a, const CSR& b )
{
    if ( a.rows != b.rows || a.cols != b.cols )
    {
        std::cerr << "Dimension mismatch: (" << a.rows << "x" << a.cols << ") vs ("
                  << b.rows << "x" << b.cols << ")\n";
        return false;
    }
    const auto a_base = a.AI()[0];
    const auto b_base = b.AI()[0];
    if ( a_base != b_base )
    {
        std::cerr << "Base mismatch: " << a_base << " vs " << b_base << "\n";
        return false;
    }
    const auto a_nnz = a.AI()[a.rows] - a_base;
    const auto b_nnz = b.AI()[b.rows] - b_base;
    if ( a_nnz != b_nnz )
    {
        std::cerr << "NNZ mismatch: " << a_nnz << " vs " << b_nnz << "\n";
        return false;
    }
    for ( int i = 0; i <= a.rows; ++i )
    {
        if ( a.AI()[i] != b.AI()[i] )
        {
            std::cerr << "AI mismatch at row " << i << ": " << a.AI()[i] << " vs "
                      << b.AI()[i] << "\n";
            return false;
        }
    }
    for ( int i = 0; i < a_nnz; ++i )
    {
        if ( a.AJ()[i] != b.AJ()[i] )
        {
            std::cerr << "AJ mismatch at idx " << i << ": " << a.AJ()[i] << " vs "
                      << b.AJ()[i] << "\n";
            return false;
        }
    }
    return true;
}

static bool has_duplicate_row_indices( const CSR& m )
{
    const auto base = m.AI()[0];
    std::vector<int> marker( m.cols, -1 );
    bool found = false;
    for ( int row = 0; row < m.rows; ++row )
    {
        const auto row_start = m.AI()[row] - base;
        const auto row_end = m.AI()[row + 1] - base;
        for ( int idx = row_start; idx < row_end; ++idx )
        {
            const int col = m.AJ()[idx] - base;
            if ( col < 0 || col >= m.cols )
            {
                std::cerr << "Invalid column in row " << row << ": " << col << "\n";
                found = true;
                continue;
            }
            if ( marker[col] == row )
            {
                std::cerr << "Duplicate column in row " << row << ": " << col << "\n";
                found = true;
            }
            else
            {
                marker[col] = row;
            }
        }
    }
    return found;
}

int main( int argc, char** argv )
{
    cxxopts::Options options( "precond_debug",
                              "Compare ILU(k) L patterns between serial and parallel symbolic factorization" );
    options.add_options()(
        "f,file", "Matrix Market file",
        cxxopts::value<std::string>()->default_value( "../tests/data/nos5.mtx" ) )(
        "l,level", "ILU level", cxxopts::value<int>()->default_value( "0" ) )(
        "t,threads", "OpenMP threads for parallel L",
        cxxopts::value<int>()->default_value( std::to_string( omp_get_max_threads() ) ) )(
        "o,out-prefix", "SVG output prefix",
        cxxopts::value<std::string>()->default_value( "ilu_l" ) )(
        "h,help", "Print help" );

    auto result = options.parse( argc, argv );
    if ( result.count( "help" ) )
    {
        std::cout << options.help() << std::endl;
        return 0;
    }

    const std::string file = result["file"].as<std::string>();
    const int level = result["level"].as<int>();
    int threads = result["threads"].as<int>();
    if ( threads <= 0 )
        threads = 1;
    const std::string out_prefix = result["out-prefix"].as<std::string>();

    std::ifstream f( file );
    if ( !f.good() )
    {
        std::cerr << "Cannot open file: " << file << "\n";
        return 1;
    }

    CSR A;
    matrix_utils::readMatrixMarket( f, A );
    if ( A.rows != A.cols )
    {
        std::cerr << "Matrix must be square for ILU: " << A.rows << "x" << A.cols << "\n";
        return 1;
    }

    auto ilu_serial = std::make_unique<CSR>();
    matrix_utils::ILULevelSymbolic<CSR> ilu_sym;
    if ( !ilu_sym( A.rows, A.AI(), A.AJ(), level, *ilu_serial ) )
    {
        std::cerr << "ILULevelSymbolic failed\n";
        return 1;
    }
    if ( ilu_serial->NNZ() > 0 )
        std::fill( ilu_serial->AV(), ilu_serial->AV() + ilu_serial->NNZ(), 1.0 );

    auto ilu_parallel = std::make_unique<CSR>();
    matrix_utils::ILULevelSymbolicParallel<CSR, enums::matrix_utils::LU, true>
        ilu_parallel_symbolic( threads );
    if ( !ilu_parallel_symbolic( A.rows, A.AI(), A.AJ(), level, *ilu_parallel ) )
    {
        std::cerr << "ILULevelSymbolicParallel failed\n";
        return 1;
    }
    if ( ilu_parallel->NNZ() > 0 )
        std::fill( ilu_parallel->AV(), ilu_parallel->AV() + ilu_parallel->NNZ(), 1.0 );

    // const bool parallel_has_dupes = has_duplicate_row_indices( *ilu_parallel );

    const std::string serial_svg = out_prefix + "_serial.svg";
    const std::string parallel_svg = out_prefix + "_parallel.svg";
    std::ofstream serial_out( serial_svg );
    if ( !serial_out.good() )
    {
        std::cerr << "Cannot write: " << serial_svg << "\n";
        return 1;
    }
    matrix_utils::writeSVG( ilu_serial->rows, ilu_serial->cols, ilu_serial->AI(),
                            ilu_serial->AJ(), serial_out );

    std::ofstream parallel_out( parallel_svg );
    if ( !parallel_out.good() )
    {
        std::cerr << "Cannot write: " << parallel_svg << "\n";
        return 1;
    }
    matrix_utils::writeSVG( ilu_parallel->rows, ilu_parallel->cols, ilu_parallel->AI(),
                            ilu_parallel->AJ(),
                            parallel_out );

    std::cout << "Wrote SVGs: " << serial_svg << ", " << parallel_svg << "\n";
    std::cout << "ILU serial nnz: " << ilu_serial->NNZ() << "\n";
    std::cout << "ILU parallel nnz: " << ilu_parallel->NNZ() << "\n";
    const bool same = compare_csr_pattern( *ilu_serial, *ilu_parallel );
    ilu_serial.reset();
    ilu_parallel.reset();
    // std::cout << "Parallel ILU duplicate indices: " << ( parallel_has_dupes ? "yes" : "no" )
    //           << "\n";
    std::cout << "CSR pattern match: " << ( same ? "yes" : "no" ) << "\n";

    return same ? 0 : 2;
}
