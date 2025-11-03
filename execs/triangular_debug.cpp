#include "matrix_utils.hpp"
#include "precond.hpp" // for ILULevelSymbolic / ILULevelNumeric / SplitLDU
#include "triangle_solve.hpp"
#include "utils.h"
#include <cxxopts.hpp>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <vector>

using namespace matrix_utils;

// Helper to load a Matrix Market file into CSRMatrix<int,int,double>
static CSRMatrix<int, int, double> loadMatrix( const std::string& path )
{
    std::ifstream f( path );
    if ( !f.good() )
        throw std::runtime_error( "Cannot open file: " + path );
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<double> vals;
    utils::read_matrix_market_csr( f, rows, cols, vals );
    CSRMatrix<int, int, double> M;
    M.rows = rows.size() - 1;
    M.cols = M.rows;
    M.ResizeAI( rows.size() );
    M.ResizeAJ( vals.size() );
    M.ResizeAV( vals.size() );
    std::copy( rows.begin(), rows.end(), M.AI() );
    std::copy( cols.begin(), cols.end(), M.AJ() );
    std::copy( vals.begin(), vals.end(), M.AV() );
    return M;
}

int main( int argc, char** argv )
{
    cxxopts::Options options( "triangular_debug",
                              "Debug triangular solves and level scheduling" );
    options.add_options()( "f,file", "MatrixMarket file",
                           cxxopts::value<std::string>()->default_value(
                               "../tests/data/nos5.mtx" ) )(
        "l,ilu-level", "ILU level (for building LDU)",
        cxxopts::value<int>()->default_value( "0" ) )(
        "u,upper", "Run upper triangular solve (default: lower)" )(
        "t,threads", "Number of OpenMP threads",
        cxxopts::value<int>()->default_value( std::to_string( omp_get_max_threads() ) ) )(
        "p,p2p", "Use P2PTriangularSubstitution (default)" )(
        "o,optimized", "Use OptimizedTriangularSolve instead of P2P" )(
        "x,max-task-size", "P2P maximum works per task",
        cxxopts::value<int>()->default_value( "128" ) )(
        "v,verify", "Compare against serial & BLAS" )( "h,help", "Print help" );
    auto result = options.parse( argc, argv );
    if ( result.count( "help" ) )
    {
        std::cout << options.help() << std::endl;
        return 0;
    }

    omp_set_num_threads( result["threads"].as<int>() );
    const std::string file = result["file"].as<std::string>();
    const int level = result["ilu-level"].as<int>();
    const bool runUpper = result.count( "upper" );
    const bool useP2P = !result.count( "optimized" );  // P2P is default, unless -o is specified
    const int maxTaskSize = result["max-task-size"].as<int>();
    const bool verify = result.count( "verify" );

    CSRMatrix<int, int, double> A = loadMatrix( file );

    // Build ILU (symbolic+numeric) for factor splitting
    CSRMatrix<int, int, double> ilu;
    ILULevelSymbolic<CSRMatrix<int, int, double>> iluSym;
    if ( !iluSym( A.rows, A.AI(), A.AJ(), level, ilu ) )
    {
        std::cerr << "Symbolic ILU failed\n";
        return 1;
    }
    if ( !ILULevelNumeric( A.rows, A.AI(), A.AJ(), A.AV(), level, ilu ) )
    {
        std::cerr << "Numeric ILU failed\n";
        return 1;
    }

    CSRMatrix<int, int, double> L, U;
    std::vector<double> D;
    SplitLDU( ilu.rows, ilu.Base(), ilu.AI(), ilu.AJ(), ilu.AV(), L, D, U );

    const int n = A.rows;
    std::vector<double> b( n, 1.0 ), x_solver( n, 0.0 ), x_serial( n, 0.0 );

    if ( !runUpper )
    {
        // Lower triangular solve (unit diagonal assumed)
        std::cout << "\n=== LOWER TRIANGULAR SOLVE ===\n";
        
        if ( useP2P )
        {
            std::cout << "\n--- P2PTriangularSubstitution ---\n";
            std::cout << "Parameters: " << result["threads"].as<int>() << " threads"
                      << ", max " << maxTaskSize << " works per task\n";
            P2PTriangularSubstitution<TriangularMatrix::L, int, int, double> p2pSolve(
                result["threads"].as<int>(), maxTaskSize );
            p2pSolve.analysis( L.rows, L.AI(), L.AJ(), L.AV(),
                              static_cast<const double*>( nullptr ) );
            p2pSolve( b.data(), x_solver.data() );
        }
        else
        {
            std::cout << "\n--- OptimizedTriangularSolve (Level-Scheduled) ---\n";
            OptimizedTriangularSolve<FBSubstitutionType::NoBarrierSuperNode, TriangularMatrix::L, int, int, double> levelSolve(
                result["threads"].as<int>() );
            levelSolve.analysis( L.rows, L.Base(), L.AI(), L.AJ(), L.AV(),
                                 static_cast<const double*>( nullptr ) );
            levelSolve( b.data(), x_solver.data() );
        }

        if ( verify )
        {
            std::cout << "\n--- Serial Reference ---\n";
            TriangularSolve<TriangularMatrix::L>(
                L.rows, L.AI(), L.AJ(), L.AV(),
                static_cast<const double*>( nullptr ), b.data(), x_serial.data() );
            
            // Compare solutions
            double maxdiff = 0;
            for ( int i = 0; i < n; i++ ) {
                maxdiff = std::max( maxdiff, std::abs( x_solver[i] - x_serial[i] ) );
            }
            std::cout << "\nVerification Results:\n";
            std::cout << "  Max |x_solver - x_serial| = " << maxdiff << "\n";
        }
    }
    else
    {
        // Upper triangular solve with diagonal
        std::cout << "\n=== UPPER TRIANGULAR SOLVE ===\n";
        
        if ( useP2P )
        {
            std::cout << "\n--- P2PTriangularSubstitution ---\n";
            std::cout << "Parameters: " << result["threads"].as<int>() << " threads"
                      << ", max " << maxTaskSize << " works per task\n";
            P2PTriangularSubstitution<TriangularMatrix::U, int, int, double> p2pSolve(
                result["threads"].as<int>(), maxTaskSize );
            p2pSolve.analysis( U.rows, U.AI(), U.AJ(), U.AV(), D.data() );
            p2pSolve( b.data(), x_solver.data() );
        }
        else
        {
            std::cout << "\n--- OptimizedTriangularSolve (Level-Scheduled) ---\n";
            OptimizedTriangularSolve<FBSubstitutionType::NoBarrierSuperNode, TriangularMatrix::U, int, int, double> levelSolve(
                result["threads"].as<int>() );
            levelSolve.analysis( U.rows, U.Base(), U.AI(), U.AJ(), U.AV(), D.data() );
            levelSolve( b.data(), x_solver.data() );
        }
        
        if ( verify )
        {
            std::cout << "\n--- Serial Reference ---\n";
            TriangularSolve<TriangularMatrix::U>(
                U.rows, U.AI(), U.AJ(), U.AV(), D.data(), b.data(), x_serial.data() );
            
            // Compare solutions
            double maxdiff = 0;
            for ( int i = 0; i < n; i++ ) {
                maxdiff = std::max( maxdiff, std::abs( x_solver[i] - x_serial[i] ) );
            }
            std::cout << "\nVerification Results:\n";
            std::cout << "  Max |x_solver - x_serial| = " << maxdiff << "\n";
        }
    }

    // Print a few sample entries
    std::cout << "\nSample Results (first 10 entries):\n";
    std::cout << "x_solver:";
    for ( int i = 0; i < std::min( 10, n ); ++i )
        std::cout << " " << x_solver[i];
    std::cout << "\n";

    return 0;
}
