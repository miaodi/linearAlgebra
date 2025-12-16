#include "cuda_gmres.h"
#include "cuda_preconditioner.h"
#include "io.hpp"
#include "matrix_utils.hpp"
#include "precond.hpp"
#include "sparse_mat_traits.hpp"
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cxxopts.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace cuda_iterative_solver;

/**
 * @brief CUDA GMRES solver with Matrix Market file support
 *
 * This example shows how to:
 * 1. Read a Matrix Market file into CSR format
 * 2. Transfer data to GPU
 * 3. Configure and run the CUDA GMRES solver with various options
 * 4. Retrieve and verify the solution
 */
int main( int argc, char** argv )
{
    // Parse command-line arguments
    cxxopts::Options options(
        "CUDA GMRES Test",
        "CUDA GMRES solver with Matrix Market file support" );
    options.add_options()(
        "f,filename", "Matrix Market file to read",
        cxxopts::value<std::string>()->default_value( "../data/ex5.mtx" ) )(
        "l,level", "ILU level", cxxopts::value<int>()->default_value( "0" ) )(
        "r,restart", "GMRES restart parameter",
        cxxopts::value<int>()->default_value( "60" ) )(
        "m,maxiter", "Maximum number of GMRES iterations",
        cxxopts::value<int>()->default_value( "1000" ) )(
        "t,reltol", "Relative tolerance for GMRES convergence",
        cxxopts::value<double>()->default_value( "1e-8" ) )(
        "a,abstol", "Absolute tolerance for GMRES convergence",
        cxxopts::value<double>()->default_value( "1e-12" ) )(
        "p,precond", "Preconditioner type: none, left, right",
        cxxopts::value<std::string>()->default_value( "none" ) )(
        "print-lu", "Write L and U factors to SVG files for visualization",
        cxxopts::value<bool>()->default_value( "false" ) )(
        "rhs-file", "Text file containing RHS vector data (one value per line)",
        cxxopts::value<std::string>()->default_value( "" ) )(
        "L-file",
        "Matrix Market file containing L factor for ILU preconditioner",
        cxxopts::value<std::string>()->default_value( "" ) )(
        "U-file",
        "Matrix Market file containing U factor for ILU preconditioner",
        cxxopts::value<std::string>()->default_value( "" ) )(
        "b,batch-ortho", "Enable batch orthogonalization in GMRES",
        cxxopts::value<bool>()->default_value( "false" ) )( "h,help",
                                                              "Print usage" );

    auto result = options.parse( argc, argv );

    if ( result.count( "help" ) )
    {
        std::cout << options.help() << std::endl;
        return 0;
    }

    std::string filename = result["filename"].as<std::string>();
    int level = result["level"].as<int>();
    int restart = result["restart"].as<int>();
    int maxiter = result["maxiter"].as<int>();
    double reltol = result["reltol"].as<double>();
    double abstol = result["abstol"].as<double>();
    std::string precond_str = result["precond"].as<std::string>();
    bool print_lu = result["print-lu"].as<bool>();
    std::string rhs_file = result["rhs-file"].as<std::string>();
    std::string l_file = result["L-file"].as<std::string>();
    std::string u_file = result["U-file"].as<std::string>();
    bool batch_ortho = result["batch-ortho"].as<bool>();

    // Validate file options for L and U matrices
    bool has_lu_files = !l_file.empty() && !u_file.empty();
    bool has_partial_lu_files = !l_file.empty() || !u_file.empty();

    if ( has_partial_lu_files && !has_lu_files )
    {
        std::cerr << "Error: Both L file (--L-file) and U file (--U-file) must "
                     "be specified together"
                  << std::endl;
        return 1;
    }

    // Parse preconditioner type
    PreconditionerType precond_type;
    if ( precond_str == "none" )
    {
        precond_type = PreconditionerType::NONE;
    }
    else if ( precond_str == "left" )
    {
        precond_type = PreconditionerType::LEFT;
    }
    else if ( precond_str == "right" )
    {
        precond_type = PreconditionerType::RIGHT;
    }
    else
    {
        std::cerr << "Invalid preconditioner type: " << precond_str
                  << ". Valid options are: none, left, right" << std::endl;
        return 1;
    }

    // Print configuration
    std::cout << "CUDA GMRES Configuration:" << std::endl;
    std::cout << "  Matrix file: " << filename << std::endl;
    std::cout << "  ILU level: " << level << std::endl;
    std::cout << "  Restart: " << restart << std::endl;
    std::cout << "  Max iterations: " << maxiter << std::endl;
    std::cout << "  Relative tolerance: " << std::scientific << reltol << std::endl;
    std::cout << "  Absolute tolerance: " << std::scientific << abstol << std::endl;
    std::cout << "  Preconditioner: " << precond_str << std::endl;
    std::cout << "  Print LU factors: " << ( print_lu ? "yes" : "no" ) << std::endl;
    std::cout << "  Batch orthogonalization: " << ( batch_ortho ? "enabled" : "disabled" ) << std::endl;
    std::cout << "  RHS file: " << ( rhs_file.empty() ? "(none - using all ones)" : rhs_file )
              << std::endl;
    if ( has_lu_files )
    {
        std::cout << "  L matrix file: " << l_file << std::endl;
        std::cout << "  U matrix file: " << u_file << std::endl;
    }
    std::cout << std::endl;
    try
    {
        // Read Matrix Market file
        std::cout << "Reading matrix from file: " << filename << std::endl;
        std::ifstream file( filename );
        if ( !file.is_open() )
        {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return 1;
        }

        matrix_utils::CSRMatrix<int, int, double> csr_matrix;
        matrix_utils::readMatrixMarket( file, csr_matrix );
        file.close();

        const size_t n = csr_matrix.rows;

        std::cout << "Matrix loaded successfully:" << std::endl;
        std::cout << "  Size: " << n << " x " << csr_matrix.cols << std::endl;
        std::cout << "  Non-zeros: " << csr_matrix.NNZ() << std::endl;
        std::cout << "  Density: " << std::fixed << std::setprecision( 4 )
                  << ( 100.0 * csr_matrix.NNZ() ) / ( n * n ) << "%" << std::endl;

        // Perform ILU factorization if preconditioner is requested
        matrix_utils::CSRMatrix<int, int, double> ilu_matrix;
        matrix_utils::CSRMatrix<int, int, double> L_matrix, U_matrix;
        bool has_preconditioner = ( precond_type != PreconditionerType::NONE );

        if ( has_preconditioner )
        {
            if ( has_lu_files )
            {
                // Read L and U matrices from files
                std::cout << "\nReading L and U matrices from files..." << std::endl;

                // Read L matrix
                std::cout << "Reading L matrix from file: " << l_file << std::endl;
                std::ifstream l_stream( l_file );
                if ( !l_stream.is_open() )
                {
                    std::cerr << "Error: Cannot open L matrix file " << l_file << std::endl;
                    return 1;
                }
                matrix_utils::readMatrixMarket( l_stream, L_matrix );
                l_stream.close();

                // Read U matrix
                std::cout << "Reading U matrix from file: " << u_file << std::endl;
                std::ifstream u_stream( u_file );
                if ( !u_stream.is_open() )
                {
                    std::cerr << "Error: Cannot open U matrix file " << u_file << std::endl;
                    return 1;
                }
                matrix_utils::readMatrixMarket( u_stream, U_matrix );
                u_stream.close();

                // Validate matrix dimensions
                if ( L_matrix.rows != n || L_matrix.cols != n )
                {
                    std::cerr << "Error: L matrix dimensions (" << L_matrix.rows
                              << "x" << L_matrix.cols << ") do not match problem size ("
                              << n << "x" << n << ")" << std::endl;
                    return 1;
                }
                if ( U_matrix.rows != n || U_matrix.cols != n )
                {
                    std::cerr << "Error: U matrix dimensions (" << U_matrix.rows
                              << "x" << U_matrix.cols << ") do not match problem size ("
                              << n << "x" << n << ")" << std::endl;
                    return 1;
                }

                std::cout << "L matrix loaded: " << L_matrix.rows << "x"
                          << L_matrix.cols << ", nnz: " << L_matrix.NNZ() << std::endl;
                std::cout << "U matrix loaded: " << U_matrix.rows << "x"
                          << U_matrix.cols << ", nnz: " << U_matrix.NNZ() << std::endl;
            }
            else
            {
                // Compute ILU factorization
                std::cout << "\nPerforming ILU(" << level
                          << ") factorization..." << std::endl;

                // Symbolic ILU factorization
                std::cout << "Symbolic ILU factorization..." << std::endl;
                matrix_utils::ILULevelSymbolic<decltype( ilu_matrix )> ilu;
                auto t1 = std::chrono::high_resolution_clock::now();
                bool success = ilu( csr_matrix.rows, csr_matrix.AI(),
                                    csr_matrix.AJ(), level, ilu_matrix );
                auto t2 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = t2 - t1;
                std::cout << "Symbolic ILU factorization time: " << elapsed.count()
                          << " s" << std::endl;

                if ( !success )
                {
                    std::cerr << "Symbolic ILU factorization failed." << std::endl;
                    return 1;
                }
                std::cout << "Symbolic ILU factorization done. nnz: "
                          << ilu_matrix.NNZ() << std::endl;

                // Numeric ILU factorization
                std::cout << "Numeric ILU factorization..." << std::endl;
                auto t3 = std::chrono::high_resolution_clock::now();
                success = matrix_utils::ILULevelNumeric(
                    csr_matrix.rows, csr_matrix.AI(), csr_matrix.AJ(),
                    csr_matrix.AV(), level, ilu_matrix );
                auto t4 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_numeric = t4 - t3;
                std::cout << "Numeric ILU factorization time: "
                          << elapsed_numeric.count() << " s" << std::endl;

                if ( !success )
                {
                    std::cerr << "Numeric ILU factorization failed." << std::endl;
                    return 1;
                }
                std::cout << "ILU factorization completed successfully." << std::endl;

                // Split ILU matrix into L (unit diagonal) and U (with diagonal) components
                std::cout << "Splitting ILU matrix into L, U factors..." << std::endl;
                matrix_utils::SplitLU<matrix_utils::CSRMatrix<int, int, double>> splitLU;
                splitLU( n, ilu_matrix.AI(), ilu_matrix.Diagonal(),
                         ilu_matrix.AJ(), ilu_matrix.AV(), L_matrix, U_matrix );

                std::cout << "L factor nnz: " << L_matrix.NNZ() << std::endl;
                std::cout << "U factor nnz: " << U_matrix.NNZ() << std::endl;
            }

            // Write L and U factors to SVG files for visualization (if requested)
            if ( print_lu )
            {
                std::cout << "Writing L and U factors to SVG files..." << std::endl;
                {
                    std::ofstream L_svg( "L_factor.svg" );
                    if ( L_svg.is_open() )
                    {
                        matrix_utils::writeSVG( L_matrix.rows, L_matrix.cols,
                                                L_matrix.AI(), L_matrix.AJ(), L_svg );
                        L_svg.close();
                        std::cout << "L factor written to L_factor.svg" << std::endl;
                    }
                    else
                    {
                        std::cerr << "Warning: Could not create L_factor.svg" << std::endl;
                    }
                }
                {
                    std::ofstream U_svg( "U_factor.svg" );
                    if ( U_svg.is_open() )
                    {
                        matrix_utils::writeSVG( U_matrix.rows, U_matrix.cols,
                                                U_matrix.AI(), U_matrix.AJ(), U_svg );
                        U_svg.close();
                        std::cout << "U factor written to U_factor.svg" << std::endl;
                    }
                    else
                    {
                        std::cerr << "Warning: Could not create U_factor.svg" << std::endl;
                    }
                }
            }
        }

        // Generate or read right-hand side vector
        std::vector<double> b_host( n );
        if ( !rhs_file.empty() )
        {
            std::cout << "Reading RHS vector from file: " << rhs_file << std::endl;
            std::ifstream rhs_stream( rhs_file );
            if ( !rhs_stream.is_open() )
            {
                std::cerr << "Error: Cannot open RHS file " << rhs_file << std::endl;
                return 1;
            }

            std::vector<double> rhs_data;
            std::string line;
            while ( std::getline( rhs_stream, line ) )
            {
                // Skip empty lines and comments
                if ( line.empty() || line[0] == '#' || line[0] == '%' )
                {
                    continue;
                }
                try
                {
                    double value = std::stod( line );
                    rhs_data.push_back( value );
                }
                catch ( const std::exception& e )
                {
                    std::cerr << "Error: Invalid number in RHS file at line: " << line
                              << std::endl;
                    return 1;
                }
            }
            rhs_stream.close();

            // Validate RHS vector size
            if ( rhs_data.size() != n )
            {
                std::cerr << "Error: RHS vector size (" << rhs_data.size()
                          << ") does not match matrix dimension (" << n << ")"
                          << std::endl;
                return 1;
            }

            // Copy RHS data
            std::copy( rhs_data.begin(), rhs_data.end(), b_host.begin() );
            std::cout << "RHS vector loaded successfully from file" << std::endl;
        }
        else
        {
            // Generate right-hand side vector (all ones) - default behavior
            std::fill( b_host.begin(), b_host.end(), 1.0 );
            std::cout << "Generated RHS vector (all ones)" << std::endl;
        }

        // Generate random initial guess
        std::vector<double> x_host( n, 0.0 );
        // std::random_device rd;
        // std::mt19937 gen( rd() );
        // std::uniform_real_distribution<double> dis( -0.1, 0.1 );
        // for ( size_t i = 0; i < n; ++i )
        // {
        //     x_host[i] = dis( gen );
        // }

        // std::cout << "Generated random initial guess" << std::endl;

        // Create and configure GMRES solver
        CudaGMRES solver;
        solver.setMaxIter( maxiter );
        solver.setRelTol( reltol );
        solver.setAbsTol( abstol );
        solver.setRestart( restart );
        solver.setPreconditionerType( precond_type );
        solver.setUseBatchOrthogonalization( batch_ortho );

        std::cout << "\nStarting CUDA GMRES solver..." << std::endl;

        State result;

        // Setup matrix operator
        std::cout << "Setting up matrix operator..." << std::endl;
        solver.setupOperator( n, csr_matrix.AI(), csr_matrix.AJ(), csr_matrix.AV() );

        // Setup ILU preconditioner if needed
        ILUPreconditioner precond;
        if ( has_preconditioner )
        {
            std::cout << "Setting up ILU preconditioner..." << std::endl;
            precond.setup( solver.getCusparseHandle(), n,
                          L_matrix.AI(), L_matrix.AJ(), L_matrix.AV(), // L factor
                          U_matrix.AI(), U_matrix.AJ(), U_matrix.AV() ); // U factor
            solver.setPreconditioner( &precond );
        }
        auto start_time = std::chrono::high_resolution_clock::now();
        // Solve the system
        std::cout << "Solving linear system..." << std::endl;
        result = solver.solve( b_host.data(), x_host.data() );

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>( end_time - start_time );

        std::cout << "Solver completed in " << duration.count() << " ms" << std::endl;

        // Solution is already in host memory (x_host)

        // Print results
        std::cout << "\nSolver finished with state: ";
        switch ( result )
        {
        case State::CONVERGED:
            std::cout << "CONVERGED" << std::endl;
            break;
        case State::MAX_ITER_REACHED:
            std::cout << "MAX_ITER_REACHED" << std::endl;
            break;
        case State::FAILED:
            std::cout << "FAILED" << std::endl;
            break;
        default:
            std::cout << "UNKNOWN" << std::endl;
            break;
        }

        // Compute and display solution statistics
        if ( n <= 10 )
        {
            std::cout << "Solution: [";
            for ( size_t i = 0; i < n; ++i )
            {
                std::cout << std::scientific << std::setprecision( 6 ) << x_host[i];
                if ( i < n - 1 )
                    std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        else
        {
            std::cout << "Solution (first 5 elements): [";
            for ( size_t i = 0; i < 5; ++i )
            {
                std::cout << std::scientific << std::setprecision( 6 ) << x_host[i];
                if ( i < 4 )
                    std::cout << ", ";
            }
            std::cout << ", ...]" << std::endl;
        }

        // Compute residual: r = A*x - b
        std::vector<double> residual( n, 0.0 );
        for ( size_t i = 0; i < n; ++i )
        {
            double ax_i = 0.0;
            for ( int j = csr_matrix.AI()[i]; j < csr_matrix.AI()[i + 1]; ++j )
            {
                ax_i += csr_matrix.AV()[j] * x_host[csr_matrix.AJ()[j]];
            }
            residual[i] = ax_i - b_host[i];
        }

        // Compute residual norms
        double residual_norm = 0.0;
        double b_norm = 0.0;
        for ( size_t i = 0; i < n; ++i )
        {
            residual_norm += residual[i] * residual[i];
            b_norm += b_host[i] * b_host[i];
        }
        residual_norm = std::sqrt( residual_norm );
        b_norm = std::sqrt( b_norm );
        double relative_residual = ( b_norm > 0.0 ) ? residual_norm / b_norm : residual_norm;

        std::cout << "\nResidual Analysis:" << std::endl;
        std::cout << "  Absolute L2 norm: " << std::scientific
                  << std::setprecision( 6 ) << residual_norm << std::endl;
        std::cout << "  Relative L2 norm: " << std::scientific
                  << std::setprecision( 6 ) << relative_residual << std::endl;
        std::cout << "  RHS L2 norm:      " << std::scientific
                  << std::setprecision( 6 ) << b_norm << std::endl;

        if ( n <= 10 )
        {
            std::cout << "Residual (A*x - b): [";
            for ( size_t i = 0; i < n; ++i )
            {
                std::cout << std::scientific << std::setprecision( 6 ) << residual[i];
                if ( i < n - 1 )
                    std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }

        return ( result == State::CONVERGED ) ? 0 : 1;
    }
    catch ( const std::exception& e )
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}