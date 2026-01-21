#include "cuda_gmres.cuh"
#include "cuda_bicgstab.cuh"
#include "cuda_preconditioner.cuh"
#include "cuda_spmv.cuh"
#include "io.hpp"
#include "matrix_utils.hpp"
#include "precond.hpp"
#include "sparse_mat_traits.hpp"
#include "Reordering.h"
#include "Transformation.hpp"
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cxxopts.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include <nvtx3/nvToolsExt.h>
#include "spadd.hpp"
#include "sp_ops.hpp"
using namespace cuda_iterative_solver;
using namespace matrix_utils;

enum class PreconditionerImpl {
    NONE,
    ILU,
    GPU_ILU0,
    JACOBI
};

/**
 * @brief Create and setup a preconditioner based on the specified type
 * 
 * @param precond_impl The preconditioner implementation type
 * @param cusparse_handle cuSPARSE handle from the solver
 * @param n Matrix dimension
 * @param csr_matrix The original CSR matrix
 * @param options Parsed command-line options containing ILU level, file paths, etc.
 * @return unique_ptr to the setup preconditioner, or nullptr if none
 */
std::unique_ptr<Preconditioner> setupPreconditioner(
    PreconditionerImpl precond_impl,
    cusparseHandle_t cusparse_handle,
    size_t n,
    const matrix_utils::CSRMatrix<int, int, double>& csr_matrix,
    const cxxopts::ParseResult& options)
{
    // Extract parameters from options
    int level = options["level"].as<int>();
    std::string l_file = options["L-file"].as<std::string>();
    std::string u_file = options["U-file"].as<std::string>();
    bool print_lu = options["print-lu"].as<bool>();
    bool has_lu_files = !l_file.empty() && !u_file.empty();
    int nthreads = options["nthreads"].as<int>();

    // Perform ILU factorization if ILU-based preconditioner is requested
    matrix_utils::CSRMatrix<int, int, double> ilu_matrix;
    matrix_utils::CSRMatrix<int, int, double> L_matrix, U_matrix;

    if (precond_impl == PreconditionerImpl::ILU || precond_impl == PreconditionerImpl::GPU_ILU0)
    {
        if (has_lu_files)
        {
            // Read L and U matrices from files
            std::cout << "\nReading L and U matrices from files..." << std::endl;

            // Read L matrix
            std::cout << "Reading L matrix from file: " << l_file << std::endl;
            std::ifstream l_stream(l_file);
            if (!l_stream.is_open())
            {
                std::cerr << "Error: Cannot open L matrix file " << l_file << std::endl;
                throw std::runtime_error("Cannot open L matrix file");
            }
            matrix_utils::readMatrixMarket(l_stream, L_matrix);
            l_stream.close();

            // Read U matrix
            std::cout << "Reading U matrix from file: " << u_file << std::endl;
            std::ifstream u_stream(u_file);
            if (!u_stream.is_open())
            {
                std::cerr << "Error: Cannot open U matrix file " << u_file << std::endl;
                throw std::runtime_error("Cannot open U matrix file");
            }
            matrix_utils::readMatrixMarket(u_stream, U_matrix);
            u_stream.close();

            // Validate matrix dimensions
            if (L_matrix.rows != n || L_matrix.cols != n)
            {
                std::cerr << "Error: L matrix dimensions (" << L_matrix.rows
                          << "x" << L_matrix.cols << ") do not match problem size ("
                          << n << "x" << n << ")" << std::endl;
                throw std::runtime_error("L matrix dimension mismatch");
            }
            if (U_matrix.rows != n || U_matrix.cols != n)
            {
                std::cerr << "Error: U matrix dimensions (" << U_matrix.rows
                          << "x" << U_matrix.cols << ") do not match problem size ("
                          << n << "x" << n << ")" << std::endl;
                throw std::runtime_error("U matrix dimension mismatch");
            }

            std::cout << "L matrix loaded: " << L_matrix.rows << "x"
                      << L_matrix.cols << ", nnz: " << L_matrix.NNZ() << std::endl;
            std::cout << "U matrix loaded: " << U_matrix.rows << "x"
                      << U_matrix.cols << ", nnz: " << U_matrix.NNZ() << std::endl;
        }
        else
        {
            // Compute ILU factorization
            std::cout << "\nPerforming ILU(" << level << ") factorization..." << std::endl;

            // Symbolic ILU factorization
            std::cout << "Symbolic ILU factorization..." << std::endl;
            auto ilu =
                std::make_unique<matrix_utils::ILULevelSymbolicParallel<decltype(ilu_matrix), enums::matrix_utils::LU, true>>(
                    nthreads);
            // auto ilu = std::make_unique<matrix_utils::ILULevelSymbolic<decltype(ilu_matrix)>>();
            auto t1 = std::chrono::high_resolution_clock::now();
            bool success = (*ilu)(csr_matrix.rows, csr_matrix.AI(),
                                  csr_matrix.AJ(), level, ilu_matrix);
            auto t2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = t2 - t1;
            std::cout << "Symbolic ILU factorization time: " << elapsed.count()
                      << " s" << std::endl;
            {
                bool diag_ok = true;
                const auto *diag = ilu_matrix.Diagonal();
                if (diag == nullptr)
                {
                    diag_ok = false;
                }
                else
                {
                    const auto *ai = ilu_matrix.AI();
                    const auto *aj = ilu_matrix.AJ();
                    const auto base = ai[0];
                    for (int row = 0; row < ilu_matrix.rows; ++row)
                    {
                        const auto diag_idx = diag[row] - base;
                        const auto row_start = ai[row] - base;
                        const auto row_end = ai[row + 1] - base;
                        if (diag_idx < row_start || diag_idx >= row_end ||
                            aj[diag_idx] != row + base)
                        {
                            std::cerr << "Temporary check: invalid diagonal at row "
                                      << row << std::endl;
                            diag_ok = false;
                            break;
                        }
                    }
                }
                if (diag_ok)
                {
                    std::cout
                        << "Temporary check: diagonal positions are valid."
                        << std::endl;
                }
            }
            if (!success)
            {
                ilu.reset();
                std::cerr << "Symbolic ILU factorization failed." << std::endl;
                throw std::runtime_error("Symbolic ILU factorization failed");
            }
            ilu.reset();
            std::cout << "Symbolic ILU factorization done. nnz: "
                      << ilu_matrix.NNZ() << std::endl;

            // Numeric ILU factorization
            std::cout << "Numeric ILU factorization..." << std::endl;
            auto t3 = std::chrono::high_resolution_clock::now();
            success = matrix_utils::ILULevelNumeric(
                csr_matrix.rows, csr_matrix.AI(), csr_matrix.AJ(),
                csr_matrix.AV(), level, ilu_matrix);
            auto t4 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_numeric = t4 - t3;
            std::cout << "Numeric ILU factorization time: "
                      << elapsed_numeric.count() << " s" << std::endl;

            if (!success)
            {
                std::cerr << "Numeric ILU factorization failed." << std::endl;
                throw std::runtime_error("Numeric ILU factorization failed");
            }
            std::cout << "ILU factorization completed successfully." << std::endl;

            // Split ILU matrix into L (unit diagonal) and U (with diagonal) components
            std::cout << "Splitting ILU matrix into L, U factors..." << std::endl;
            matrix_utils::SplitLU<matrix_utils::CSRMatrix<int, int, double>> splitLU;
            splitLU(n, ilu_matrix.AI(), ilu_matrix.Diagonal(),
                   ilu_matrix.AJ(), ilu_matrix.AV(), L_matrix, U_matrix);

            std::cout << "L factor nnz: " << L_matrix.NNZ() << std::endl;
            std::cout << "U factor nnz: " << U_matrix.NNZ() << std::endl;
        }

        // Write L and U factors to SVG files for visualization (if requested)
        if (print_lu)
        {
            std::cout << "Writing L and U factors to SVG files..." << std::endl;
            {
                std::ofstream L_svg("L_factor.svg");
                if (L_svg.is_open())
                {
                    matrix_utils::writeSVG(L_matrix.rows, L_matrix.cols,
                                          L_matrix.AI(), L_matrix.AJ(), L_svg);
                    L_svg.close();
                    std::cout << "L factor written to L_factor.svg" << std::endl;
                }
                else
                {
                    std::cerr << "Warning: Could not create L_factor.svg" << std::endl;
                }
            }
            {
                std::ofstream U_svg("U_factor.svg");
                if (U_svg.is_open())
                {
                    matrix_utils::writeSVG(U_matrix.rows, U_matrix.cols,
                                          U_matrix.AI(), U_matrix.AJ(), U_svg);
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

    // Now create and setup the appropriate preconditioner
    if (precond_impl == PreconditionerImpl::ILU)
    {
        std::cout << "Setting up ILU preconditioner..." << std::endl;
        auto precond = std::make_unique<CuSparseILUPrec>(cusparse_handle);
        precond->setup(n,
                      L_matrix.AI(), L_matrix.AJ(), L_matrix.AV(), // L factor
                      U_matrix.AI(), U_matrix.AJ(), U_matrix.AV()); // U factor
        return precond;
    }
    else if (precond_impl == PreconditionerImpl::GPU_ILU0)
    {
        std::cout << "Setting up GPU ILU0 preconditioner..." << std::endl;
        auto precond = std::make_unique<CuSparseILU0Prec>(cusparse_handle);
        precond->setup(n,
                      csr_matrix.AI(), csr_matrix.AJ(), csr_matrix.AV());
        return precond;
    }
    else if (precond_impl == PreconditionerImpl::JACOBI)
    {
        std::cout << "Setting up Jacobi preconditioner..." << std::endl;
        auto precond = std::make_unique<JacobiPreconditioner>();
        precond->setupFromMatrix(n,
                                csr_matrix.AI(), csr_matrix.AJ(), csr_matrix.AV());
        return precond;
    }
    
    // Default: no preconditioning
    return std::make_unique<NoPreconditioner>();
}

/**
 * @brief Reorder matrix and RHS using specified algorithm
 * 
 * @param algorithm Reordering algorithm: "none", "rcm", "rcm-par", "nd"
 * @param matrix Input matrix (will NOT be modified)
 * @param rhs Input RHS vector (will NOT be modified)
 * @param permuted_matrix Output permuted matrix
 * @param permuted_rhs Output permuted RHS
 * @param nthreads Number of threads for parallel operations
 * @return unique_ptr to RowColPermutation transformation, or nullptr for "none"
 */
std::unique_ptr<solver::RowColPermutation<matrix_utils::CSRMatrix<int, int, double>>>
reorderMatrixAndRHS(const std::string& algorithm,
                    const matrix_utils::CSRMatrix<int, int, double>& matrix,
                    const std::vector<double>& rhs,
                    matrix_utils::CSRMatrix<int, int, double>& permuted_matrix,
                    std::vector<double>& permuted_rhs,
                    int nthreads = 1)
{
    if (algorithm == "none") {
        std::cout << "No reordering applied" << std::endl;
        // Copy original data to output
        permuted_matrix = matrix;
        permuted_rhs = rhs;
        return nullptr;
    }

    const size_t n = matrix.rows;
    std::cout << "\n=== Matrix Reordering ===" << std::endl;
    std::cout << "Algorithm: " << algorithm << std::endl;
    std::cout << "Threads: " << nthreads << std::endl;

    // Compute A+A^T (symmetric adjacency graph without diagonal)
    std::cout << "Computing adjacency graph (A+A^T)..." << std::endl;
    std::vector<int> xadj(n + 1);
    auto graph_start = std::chrono::high_resolution_clock::now();
    
    matrix_utils::APlusATPrefix<int, int, false>(
        n, matrix.AI(), matrix.AJ(), xadj.data());
    
    int actual_edges = xadj[n] - xadj[0];
    std::vector<int> adjncy(actual_edges);
    matrix_utils::APlusATFill<int, int, false>(
        n, matrix.AI(), matrix.AJ(), xadj.data(), adjncy.data());
    
    auto graph_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> graph_time = graph_end - graph_start;
    std::cout << "  Vertices: " << n << ", Edges: " << actual_edges 
              << ", Time: " << graph_time.count() << " s" << std::endl;

    // Compute permutation
    static std::vector<int> perm(n);
    static std::vector<int> iperm(n);
    
    auto order_start = std::chrono::high_resolution_clock::now();
    
    if (algorithm == "rcm") {
        std::cout << "Computing RCM ordering (Traditional)..." << std::endl;
        reordering::RCM_MultiComponent<reordering::RCMKernel::Traditional, int, int>(
            static_cast<int>(n), xadj.data(), adjncy.data(), perm.data(), iperm.data(), nthreads);
    } else if (algorithm == "rcm-par") {
        std::cout << "Computing RCM ordering (ParallelSort)..." << std::endl;
        reordering::RCM_MultiComponent<reordering::RCMKernel::ParallelSort, int, int>(
            static_cast<int>(n), xadj.data(), adjncy.data(), perm.data(), iperm.data(), nthreads);
    } else if (algorithm == "nd") {
#ifdef USE_METIS_LIB
        std::cout << "Computing METIS ND ordering..." << std::endl;
        int rc = reordering::MetisND<int, int>(
            n, n, xadj.data(), adjncy.data(), iperm.data(), perm.data());
        if (rc != 0) {
            throw std::runtime_error("METIS ND failed with code " + std::to_string(rc));
        }
#else
        throw std::runtime_error("METIS support not enabled (USE_METIS_LIB=OFF)");
#endif
    } else {
        throw std::runtime_error("Unknown reordering algorithm: " + algorithm);
    }
    
    auto order_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> order_time = order_end - order_start;
    std::cout << "  Ordering time: " << order_time.count() << " s" << std::endl;

    // Setup output matrix for permuted result
    permuted_matrix.rows = n;
    permuted_matrix.cols = n;
    permuted_matrix.ResizeAI(n + 1);
    permuted_matrix.ResizeAJ(matrix.NNZ());
    permuted_matrix.ResizeAV(matrix.NNZ());
    
    // Permute matrix: P * A * P^T (symmetric permutation)
    std::cout << "Permuting matrix..." << std::endl;
    auto perm_start = std::chrono::high_resolution_clock::now();
    matrix_utils::permuteMat<int, int, double>(
                            static_cast<int>(n), static_cast<int>(n),
                            const_cast<const int*>(perm.data()), 
                            const_cast<const int*>(iperm.data()),
                            matrix.AI(), matrix.AJ(), matrix.AV(),
                            permuted_matrix.AI(), permuted_matrix.AJ(), permuted_matrix.AV(),
                            nthreads);
    auto perm_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> perm_time = perm_end - perm_start;
    std::cout << "  Permutation time: " << perm_time.count() << " s" << std::endl;
    
    // Permute RHS: b_new = P * b
    permuted_rhs.resize(n);
    matrix_utils::permVec(static_cast<int>(n), 0, rhs.data(), perm.data(), 
                         permuted_rhs.data(), nthreads);
    
    std::cout << "Reordering complete" << std::endl;
    std::cout << "  Total time: " << (graph_time.count() + order_time.count() + perm_time.count()) 
              << " s" << std::endl;

    // Create and return RowColPermutation for solution recovery
    return std::make_unique<solver::RowColPermutation<matrix_utils::CSRMatrix<int, int, double>>>(
        std::span<const int>(perm.data(), perm.size()),
        std::span<const int>(perm.data(), perm.size()),
        0);
}

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
        "precond-impl", "Preconditioner implementation: none, ilu, gpuilu0, jacobi",
        cxxopts::value<std::string>()->default_value( "ilu" ) )(
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
        cxxopts::value<bool>()->default_value( "false" ) )(
        "reorder", "Matrix reordering algorithm: none, rcm, rcm-par, nd",
        cxxopts::value<std::string>()->default_value( "none" ) )(
        "n,nthreads", "Number of threads for parallel operations",
        cxxopts::value<int>()->default_value( "1" ) )(
        "method", "Solver method: gmres, bicgstab",
        cxxopts::value<std::string>()->default_value( "gmres" ) )( "h,help",
                                                        "Print usage" );

    auto parsed_options = options.parse( argc, argv );

    if ( parsed_options.count( "help" ) )
    {
        std::cout << options.help() << std::endl;
        return 0;
    }

    std::string filename = parsed_options["filename"].as<std::string>();
    std::string rhs_file = parsed_options["rhs-file"].as<std::string>();
    int level = parsed_options["level"].as<int>();
    int restart = parsed_options["restart"].as<int>();
    int maxiter = parsed_options["maxiter"].as<int>();
    double reltol = parsed_options["reltol"].as<double>();
    double abstol = parsed_options["abstol"].as<double>();
    std::string precond_str = parsed_options["precond"].as<std::string>();
    std::string precond_impl_str = parsed_options["precond-impl"].as<std::string>();
    bool print_lu = parsed_options["print-lu"].as<bool>();
    std::string l_file = parsed_options["L-file"].as<std::string>();
    std::string u_file = parsed_options["U-file"].as<std::string>();
    bool batch_ortho = parsed_options["batch-ortho"].as<bool>();
    std::string reorder_alg = parsed_options["reorder"].as<std::string>();
    int nthreads = parsed_options["nthreads"].as<int>();
    std::string method_str = parsed_options["method"].as<std::string>();

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

    // Parse preconditioner implementation
    PreconditionerImpl precond_impl;
    if ( precond_impl_str == "none" )
    {
        precond_impl = PreconditionerImpl::NONE;
    }
    else if ( precond_impl_str == "ilu" )
    {
        precond_impl = PreconditionerImpl::ILU;
    }
    else if ( precond_impl_str == "gpuilu0" )
    {
        precond_impl = PreconditionerImpl::GPU_ILU0;
    }
    else if ( precond_impl_str == "jacobi" )
    {
        precond_impl = PreconditionerImpl::JACOBI;
    }
    else
    {
        std::cerr << "Invalid preconditioner implementation: " << precond_impl_str
                  << ". Valid options are: none, ilu, gpuilu0, jacobi" << std::endl;
        return 1;
    }

    // Print configuration
    std::cout << "CUDA GMRES Configuration:" << std::endl;
    std::cout << "  Matrix file: " << filename << std::endl;
    std::cout << "  RHS file: " << ( rhs_file.empty() ? "(none - using all ones)" : rhs_file )
              << std::endl;
    std::cout << "  ILU level: " << level << std::endl;
    std::cout << "  Restart: " << restart << std::endl;
    std::cout << "  Max iterations: " << maxiter << std::endl;
    std::cout << "  Relative tolerance: " << std::scientific << reltol << std::endl;
    std::cout << "  Absolute tolerance: " << std::scientific << abstol << std::endl;
    std::cout << "  Preconditioner: " << precond_str << std::endl;
    std::cout << "  Preconditioner implementation: " << precond_impl_str << std::endl;
    std::cout << "  Print LU factors: " << ( print_lu ? "yes" : "no" ) << std::endl;
    std::cout << "  Batch orthogonalization: " << ( batch_ortho ? "enabled" : "disabled" ) << std::endl;
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

        // Prune zero entries from the matrix
        {
            const int nnz_before = csr_matrix.NNZ();
            double const* row_thresholds = nullptr;
            const int num_pruned = matrix_utils::Prune(
                csr_matrix.rows,
                csr_matrix.AI(),
                csr_matrix.AJ(),
                csr_matrix.AV(),
                0.0,
                row_thresholds);
            
            std::cout << "\nMatrix pruning:" << std::endl;
            std::cout << "  Entries pruned: " << num_pruned << std::endl;
            std::cout << "  NNZ after pruning: " << csr_matrix.NNZ() << std::endl;
            std::cout << "  Pruning ratio: " << std::fixed << std::setprecision( 2 )
                      << ( 100.0 * num_pruned ) / nnz_before << "%" << std::endl;
        }

        // Generate or read right-hand side vector
        std::vector<double> b_host;
        if ( !rhs_file.empty() )
        {
            std::cout << "Reading RHS vector from file: " << rhs_file << std::endl;
            std::ifstream rhs_stream( rhs_file );
            if ( !rhs_stream.is_open() )
            {
                std::cerr << "Error: Cannot open RHS file " << rhs_file << std::endl;
                return 1;
            }

            matrix_utils::readMatrixMarketVec( rhs_stream, b_host );
            rhs_stream.close();

            // Validate RHS vector size
            if ( b_host.size() != n )
            {
                std::cerr << "Error: RHS vector size (" << b_host.size()
                          << ") does not match matrix dimension (" << n << ")"
                          << std::endl;
                return 1;
            }

            std::cout << "RHS vector loaded successfully from file" << std::endl;
        }
        else
        {
            // Generate right-hand side vector (all ones) - default behavior
            b_host.resize( n );
            std::fill( b_host.begin(), b_host.end(), 1.0 );
            std::cout << "Generated RHS vector (all ones)" << std::endl;
        }

        // Apply matrix reordering if requested
        std::unique_ptr<solver::RowColPermutation<matrix_utils::CSRMatrix<int, int, double>>> perm;
        matrix_utils::CSRMatrix<int, int, double> working_matrix;
        std::vector<double> working_rhs;
        
        if (reorder_alg != "none")
        {
            std::cout << "\nApplying matrix reordering (" << reorder_alg << ")..." << std::endl;
            perm = reorderMatrixAndRHS(reorder_alg, csr_matrix, b_host, 
                                       working_matrix, working_rhs, nthreads);
            std::cout << "Matrix reordering completed" << std::endl;
        }
        else
        {
            // No reordering - use original matrix and RHS
            working_matrix = csr_matrix;
            working_rhs = b_host;
        }

        int* d_ia = nullptr;
        int* d_ja = nullptr;
        double* d_av = nullptr;
        const int nrows = static_cast<int>(n);
        const int base = working_matrix.AI()[0];
        const int nnz = working_matrix.AI()[nrows] - base;
        matrix_utils::copy_csr_host_to_device<int, int, double>(
            nrows, working_matrix.AI(), working_matrix.AJ(), working_matrix.AV(),
            &d_ia, &d_ja, &d_av);

        // Initialize solution vector with zeros
        std::vector<double> x_host( n, 0.0 );

        // Determine solver method
        bool use_bicgstab = (method_str == "bicgstab" || method_str == "BiCGSTAB");
        std::string solver_name = use_bicgstab ? "BiCGSTAB" : "GMRES";
        
        std::cout << "\nStarting CUDA " << solver_name << " solver..." << std::endl;

        State result;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Create CUDA events for profiling
        cudaEvent_t solve_start, solve_stop;
        cudaEventCreate(&solve_start);
        cudaEventCreate(&solve_stop);
        
        if (use_bicgstab) {
            // Create and configure BiCGSTAB solver
            CudaBiCGSTAB solver;
            solver.setMaxIter( maxiter );
            solver.setRelTol( reltol );
            solver.setAbsTol( abstol );
            solver.setPreconditionerType( precond_type );
            
            // Create and setup SpMV operator using solver's cuSPARSE handle
            std::cout << "Setting up matrix operator..." << std::endl;
            CuSparseSPMV<int, int, double> spmv_operator(solver.getCusparseHandle());
            spmv_operator.preprocess(nrows, d_ia, d_ja, d_av, base, nnz);
            
            // Setup solver with SpMV operator (size is obtained from operator)
            solver.setupOperator(&spmv_operator);

            // Setup preconditioner (use working_matrix which may be reordered)
            std::unique_ptr<Preconditioner> precond = setupPreconditioner(
                precond_impl,
                solver.getCusparseHandle(),
                n,
                working_matrix,
                parsed_options);
            
            solver.setPreconditioner( precond.get() );
            
            // Solve the system (use working_rhs which may be reordered)
            std::cout << "Solving linear system..." << std::endl;
            nvtxRangePush("BiCGSTAB_Solve");
            cudaEventRecord(solve_start);
            result = solver.solve<true>( working_rhs.data(), x_host.data() );
            cudaEventRecord(solve_stop);
            cudaEventSynchronize(solve_stop);
            nvtxRangePop();
        } else {
            // Create and configure GMRES solver
            CudaGMRES solver;
            solver.setMaxIter( maxiter );
            solver.setRelTol( reltol );
            solver.setAbsTol( abstol );
            solver.setRestart( restart );
            solver.setPreconditionerType( precond_type );
            solver.setUseBatchOrthogonalization( batch_ortho );
            
            // Create and setup SpMV operator using solver's cuSPARSE handle
            std::cout << "Setting up matrix operator..." << std::endl;
            CuSparseSPMV<int, int, double> spmv_operator(solver.getCusparseHandle());
            spmv_operator.preprocess(nrows, d_ia, d_ja, d_av, base, nnz);
            
            // Setup solver with SpMV operator (size is obtained from operator)
            solver.setupOperator(&spmv_operator);

            // Setup preconditioner (use working_matrix which may be reordered)
            std::unique_ptr<Preconditioner> precond = setupPreconditioner(
                precond_impl,
                solver.getCusparseHandle(),
                n,
                working_matrix,
                parsed_options);
            
            solver.setPreconditioner( precond.get() );
            
            // Solve the system (use working_rhs which may be reordered)
            std::cout << "Solving linear system..." << std::endl;
            nvtxRangePush("GMRES_Solve");
            cudaEventRecord(solve_start);
            result = solver.solve<true>( working_rhs.data(), x_host.data() );
            cudaEventRecord(solve_stop);
            cudaEventSynchronize(solve_stop);
            nvtxRangePop();
        }
        
        // Calculate solve time
        float solve_time_ms = 0;
        cudaEventElapsedTime(&solve_time_ms, solve_start, solve_stop);

        if (d_ia) cudaFree(d_ia);
        if (d_ja) cudaFree(d_ja);
        if (d_av) cudaFree(d_av);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>( end_time - start_time );

        std::cout << "Solver completed in " << duration.count() << " ms (total), "
                  << solve_time_ms << " ms (GPU solve time)" << std::endl;
        
        // Clean up CUDA events
        cudaEventDestroy(solve_start);
        cudaEventDestroy(solve_stop);

        // Apply inverse permutation to solution if reordering was used
        if (perm)
        {
            std::cout << "Applying inverse permutation to solution..." << std::endl;
            std::vector<double> x_temp(n);
            perm->applyInverseToX(x_host, x_temp, nthreads);
            std::swap(x_host, x_temp);
        }

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
