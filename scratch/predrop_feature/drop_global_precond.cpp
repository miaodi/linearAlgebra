#include "io.hpp"
#include "iterative_solver.hpp"
#include "matrix_utils.hpp"
#include "permutation.hpp"
#include "precond.hpp"
#include "ruiz_scale.hpp"
#include "sp_ops.hpp"
#include "spadd.hpp"
#include "sparse_mat_traits.hpp"
#include "spmv.hpp"
#include "triangle_solve.hpp"
#ifdef USE_METIS_LIB
#include "Reordering.h"
#endif
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cxxopts.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

enum class Factorization
{
    ILUK,
    ILUT
};

struct Options
{
    std::string filename;
    std::string output_file;
    int max_display_size;
    double threshold;
    int level;
    int threads;
    Factorization factorization;
    double droptol;
    iterative_solver::PreconditionerType precond_type;
    int restart;
    int maxiter;
    double reltol;
    bool solve;
};

// Forward declarations
void printOptions(const Options& opts, const std::string& factorization_str, const std::string& precond_str);

template <matrix_utils::ResizableDiagonal CSRMatrixType>
class ILUPrec;

iterative_solver::State solveWithPreconditioner(const Options& opts,
                                                const matrix_utils::CSRMatrix<int, int, double>& csr_matrix,
                                                const matrix_utils::CSRMatrix<int, int, double>& pruned);

void precondExplore(const Options& opts, const matrix_utils::CSRMatrix<int, int, double>& pruned);

#ifdef USE_METIS_LIB
matrix_utils::CSRMatrix<int, int, double> permuteMatrixWithMetisND(
    const matrix_utils::CSRMatrix<int, int, double>& matrix);
#endif

int main(int argc, char* argv[])
{
    cxxopts::Options options("drop_global_precond",
                             "Test GMRES with global preconditioner dropping");

    // clang-format off
    options.add_options()
        ( "f,filename", "Matrix Market file to read", cxxopts::value<std::string>()->default_value( "../../tests/data/ex5.mtx" ) )
        ( "o,output", "Output SVG file path", cxxopts::value<std::string>()->default_value( "matrix.svg" ) )
        ( "s,size", "Maximum display size (pixels)", cxxopts::value<int>()->default_value( "2000" ) )
        ( "t,threshold", "Pruning threshold (absolute value)", cxxopts::value<double>()->default_value( "0.0" ) )
        ( "l,level", "ILU level", cxxopts::value<int>()->default_value( "0" ) )
        ( "n,threads", "Number of threads", cxxopts::value<int>()->default_value( "8" ) )
        ( "F,factorization", "ILU variant: iluk or ilut", cxxopts::value<std::string>()->default_value( "iluk" ) )
        ( "d,droptol", "ILUT drop tolerance", cxxopts::value<double>()->default_value( "1e-3" ) )
        ( "p,precond", "Preconditioner type: none (no preconditioning), left (M^-1 A x = M^-1 b), right (A M^-1 y = b)",
          cxxopts::value<std::string>()->default_value( "left" ) )
        ( "r,restart", "GMRES restart parameter", cxxopts::value<int>()->default_value( "60" ) )
        ( "m,maxiter", "Maximum number of GMRES iterations", cxxopts::value<int>()->default_value( "1000" ) )
        ( "reltol", "Relative tolerance for GMRES convergence", cxxopts::value<double>()->default_value( "1e-10" ) )
        ( "solve", "Execute solve routine", cxxopts::value<bool>()->default_value( "false" ) )
        ( "h,help", "Print usage" );
    // clang-format on

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        return 0;
    }

    // Parse all options into Options struct
    Options opts;
    opts.filename = result["filename"].as<std::string>();
    opts.output_file = result["output"].as<std::string>();
    opts.max_display_size = result["size"].as<int>();
    opts.threshold = result["threshold"].as<double>();
    opts.level = result["level"].as<int>();
    opts.threads = result["threads"].as<int>();
    opts.droptol = result["droptol"].as<double>();
    opts.restart = result["restart"].as<int>();
    opts.maxiter = result["maxiter"].as<int>();
    opts.reltol = result["reltol"].as<double>();
    opts.solve = result["solve"].as<bool>();

    std::string precond_type_str = result["precond"].as<std::string>();
    std::string factorization_str = result["factorization"].as<std::string>();

    // Parse factorization type
    std::string factorization_lower = factorization_str;
    std::transform(factorization_lower.begin(), factorization_lower.end(),
                   factorization_lower.begin(), [](unsigned char c) { return std::tolower(c); });

    if (factorization_lower == "iluk")
    {
        opts.factorization = Factorization::ILUK;
    }
    else if (factorization_lower == "ilut")
    {
        opts.factorization = Factorization::ILUT;
    }
    else
    {
        std::cerr << "Invalid factorization type: " << factorization_str
                  << ". Valid options are: iluk, ilut" << std::endl;
        return -1;
    }

    // Parse preconditioner type
    if (precond_type_str == "none")
    {
        opts.precond_type = iterative_solver::PreconditionerType::NONE;
    }
    else if (precond_type_str == "left")
    {
        opts.precond_type = iterative_solver::PreconditionerType::LEFT;
    }
    else if (precond_type_str == "right")
    {
        opts.precond_type = iterative_solver::PreconditionerType::RIGHT;
    }
    else
    {
        std::cerr << "Invalid preconditioner type: " << precond_type_str
                  << ". Valid options are: none, left, right" << std::endl;
        return -1;
    }

    // Validate parameters
    if (opts.max_display_size <= 0)
    {
        std::cerr << "Invalid max_display_size: " << opts.max_display_size
                  << ". Must be a positive integer." << std::endl;
        return -1;
    }

    if (opts.restart <= 0)
    {
        std::cerr << "Invalid restart parameter: " << opts.restart
                  << ". Must be a positive integer." << std::endl;
        return -1;
    }

    printOptions(opts, factorization_str, precond_type_str);

    omp_set_num_threads(8);
    std::ifstream f(opts.filename);
    if (!f.is_open())
    {
        std::cerr << "Failed to open file: " << opts.filename << std::endl;
        return -1;
    }

    f.clear();
    f.seekg(0, std::ios::beg);
    matrix_utils::CSRMatrix<int, int, double> csr_matrix;
    matrix_utils::readMatrixMarket(f, csr_matrix);
    f.close();

    std::cout << "Matrix: " << csr_matrix.rows << " x " << csr_matrix.cols
              << ", NNZ: " << csr_matrix.NNZ() << std::endl;

    // Apply Ruiz scaling to the matrix
    std::cout << "\nApplying Ruiz scaling..." << std::endl;
    std::vector<double> dr(csr_matrix.rows);
    std::vector<double> dc(csr_matrix.cols);
    scaling::RuizScaleSerial<int, int, double, scaling::RuizScalingNormType::MaxNorm>(
        csr_matrix.rows, csr_matrix.cols, csr_matrix.AI(), csr_matrix.AJ(),
        csr_matrix.AV(), dr.data(), dc.data(), 20, 1e-2);
    std::cout << "Ruiz scaling completed" << std::endl;

    // Deep copy matrix for pruning
    matrix_utils::CSRMatrix<int, int, double> pruned;
    pruned.rows = csr_matrix.rows;
    pruned.cols = csr_matrix.cols;
    pruned.ResizeAI(csr_matrix.rows + 1);
    pruned.ResizeAJ(csr_matrix.NNZ());
    pruned.ResizeAV(csr_matrix.NNZ());
    std::memcpy(pruned.AI(), csr_matrix.AI(), (csr_matrix.rows + 1) * sizeof(int));
    std::memcpy(pruned.AJ(), csr_matrix.AJ(), csr_matrix.NNZ() * sizeof(int));
    std::memcpy(pruned.AV(), csr_matrix.AV(), csr_matrix.NNZ() * sizeof(double));

    std::cout << "\nPruning matrix with thresholds (a_ii * a_jj * " << opts.threshold << ")..." << std::endl;
    int original_nnz = pruned.NNZ();
    auto removed = matrix_utils::DiagonalScaledPrune(pruned.rows, pruned.AI(), pruned.AJ(),
                                                     pruned.AV(), opts.threshold);
    int pruned_nnz = pruned.NNZ();
    std::cout << "Original NNZ: " << original_nnz << std::endl;
    std::cout << "Pruned NNZ: " << pruned_nnz << std::endl;
    std::cout << "Removed entries: " << removed << std::endl;
    if (original_nnz > 0)
    {
        std::cout << "Retention rate: " << (static_cast<double>(pruned_nnz) / original_nnz * 100.0)
                  << "%" << std::endl;
    }

    // Write pruned matrix SVG to file
    std::cout << "\nWriting pruned matrix to SVG..." << std::endl;
    std::ofstream out(opts.output_file);
    if (!out.is_open())
    {
        std::cerr << "Failed to create output file: " << opts.output_file << std::endl;
        return -1;
    }

    matrix_utils::writeSVG(pruned.rows, pruned.cols, pruned.AI(), pruned.AJ(), out, opts.max_display_size);
    out.close();

    std::cout << "SVG written to: " << opts.output_file << std::endl;

#ifdef USE_METIS_LIB
    // Permute original matrix with MetisND
    auto permuted_original = permuteMatrixWithMetisND(csr_matrix);
    std::ofstream out_perm_orig("permuted_original_matrix.svg");
    if (out_perm_orig.is_open())
    {
        std::cout << "\nWriting permuted original matrix to SVG..." << std::endl;
        matrix_utils::writeSVG(permuted_original.rows, permuted_original.cols, 
                              permuted_original.AI(), permuted_original.AJ(), 
                              out_perm_orig, opts.max_display_size);
        out_perm_orig.close();
        std::cout << "Permuted original matrix written to permuted_original_matrix.svg" << std::endl;
    }

    // Permute pruned matrix with MetisND
    auto permuted_pruned = permuteMatrixWithMetisND(pruned);
    std::ofstream out_perm_pruned("permuted_pruned_matrix.svg");
    if (out_perm_pruned.is_open())
    {
        std::cout << "\nWriting permuted pruned matrix to SVG..." << std::endl;
        matrix_utils::writeSVG(permuted_pruned.rows, permuted_pruned.cols, 
                              permuted_pruned.AI(), permuted_pruned.AJ(), 
                              out_perm_pruned, opts.max_display_size);
        out_perm_pruned.close();
        std::cout << "Permuted pruned matrix written to permuted_pruned_matrix.svg" << std::endl;
    }
#endif

    precondExplore(opts, pruned);

    // Solve with preconditioner
    iterative_solver::State state = iterative_solver::State::CONVERGED;
    if (opts.solve)
    {
        state = solveWithPreconditioner(opts, csr_matrix, pruned);
    }
    else
    {
        std::cout << "\nSkipping solve routine (--solve not specified)" << std::endl;
    }

    return (state == iterative_solver::State::CONVERGED) ? 0 : -1;
}

// ============================================================================
// Implementation
// ============================================================================

void printOptions(const Options& opts, const std::string& factorization_str, const std::string& precond_str)
{
    std::cout << "Options:" << std::endl;
    std::cout << "  filename: " << opts.filename << std::endl;
    std::cout << "  output: " << opts.output_file << std::endl;
    std::cout << "  max_display_size: " << opts.max_display_size << std::endl;
    std::cout << "  threshold: " << opts.threshold << std::endl;
    std::cout << "  level: " << opts.level << std::endl;
    std::cout << "  threads: " << opts.threads << std::endl;
    std::cout << "  factorization: " << factorization_str << std::endl;
    if (opts.factorization == Factorization::ILUT)
    {
        std::cout << "  droptol: " << opts.droptol << std::endl;
    }
    std::cout << "  precond: " << precond_str << std::endl;
    std::cout << "  restart: " << opts.restart << std::endl;
    std::cout << "  maxiter: " << opts.maxiter << std::endl;
    std::cout << "  reltol: " << opts.reltol << std::endl;
    std::cout << "  solve: " << (opts.solve ? "true" : "false") << std::endl;
}

template <matrix_utils::ResizableDiagonal CSRMatrixType>
class ILUPrec
{
public:
    using ROWTYPE = typename CSRMatrixType::ROWTYPE;
    using COLTYPE = typename CSRMatrixType::COLTYPE;
    using VALTYPE = typename CSRMatrixType::VALTYPE;
    ILUPrec(const COLTYPE size, const CSRMatrixType& ilu) : _size(size), _ilu(ilu), tmp(size)
    {
        matrix_utils::SplitLDU(_size, _ilu.AI()[0], _ilu.AI(), _ilu.AJ(), _ilu.AV(), L, D, U);
    }

    COLTYPE size() const { return _size; }

    bool operator()(VALTYPE const* const b, VALTYPE* const x) const
    {
        matrix_utils::TriangularSolve<matrix_utils::TriangularMatrix::L>(
            _size, L.AI(), L.AJ(), L.AV(), null_diag, b, tmp.data());
        matrix_utils::TriangularSolve<matrix_utils::TriangularMatrix::U>(
            _size, U.AI(), U.AJ(), U.AV(), D.data(), tmp.data(), x);
        return true;
    }

    COLTYPE _size;
    const CSRMatrixType& _ilu;
    CSRMatrixType L;
    CSRMatrixType U;
    std::vector<typename CSRMatrixType::VALTYPE> D;
    mutable std::vector<VALTYPE> tmp;
    static constexpr VALTYPE* null_diag = nullptr;
};

iterative_solver::State solveWithPreconditioner(const Options& opts,
                                                const matrix_utils::CSRMatrix<int, int, double>& csr_matrix,
                                                const matrix_utils::CSRMatrix<int, int, double>& pruned)
{
    // Build ILU preconditioner from pruned matrix
    std::cout << "\nBuilding ILU preconditioner from pruned matrix..." << std::endl;
    matrix_utils::CSRMatrix<int, int, double> ilu_matrix;
    bool success = false;

    if (opts.factorization == Factorization::ILUK)
    {
        std::cout << "Symbolic ILU(k) factorization..." << std::endl;
        matrix_utils::ILULevelSymbolic<decltype(ilu_matrix)> ilu;
        auto t1 = std::chrono::high_resolution_clock::now();
        success = ilu(pruned.rows, pruned.AI(), pruned.AJ(), opts.level, ilu_matrix);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = t2 - t1;
        std::cout << "Symbolic ILU factorization time: " << elapsed.count() << " s" << std::endl;
        if (!success)
        {
            std::cout << "Symbolic ILU factorization failed." << std::endl;
            return iterative_solver::State::FAILED;
        }
        std::cout << "Symbolic ILU factorization done. nnz: " << ilu_matrix.NNZ() << std::endl;

        // Write serial ILU(k) sparsity pattern to SVG
        std::ofstream out_serial_iluk("serial_iluk_pattern.svg");
        std::cout << "Writing serial ILU(k) sparsity pattern to SVG..." << std::endl;
        matrix_utils::writeSVG(ilu_matrix.rows, ilu_matrix.cols, ilu_matrix.AI(), ilu_matrix.AJ(), 
                              out_serial_iluk, opts.max_display_size);
        out_serial_iluk.close();
        std::cout << "Serial ILU(k) sparsity pattern written to serial_iluk_pattern.svg" << std::endl;

        std::cout << "Numeric ILU factorization..." << std::endl;
        auto t3 = std::chrono::high_resolution_clock::now();
        success = matrix_utils::ILULevelNumeric(pruned.rows, pruned.AI(), pruned.AJ(), pruned.AV(),
                                                opts.level, ilu_matrix);
        auto t4 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_numeric = t4 - t3;
        std::cout << "Numeric ILU factorization time: " << elapsed_numeric.count() << " s" << std::endl;
    }
    else
    {
        std::cout << "Using ILUTNumeric with droptol = " << opts.droptol << std::endl;
        auto t3 = std::chrono::high_resolution_clock::now();
        success = matrix_utils::ILUTNumeric(pruned.rows, pruned.AI(), pruned.AJ(), pruned.AV(),
                                            opts.droptol, ilu_matrix);
        auto t4 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_numeric = t4 - t3;
        std::cout << "Numeric ILU factorization time: " << elapsed_numeric.count() << " s" << std::endl;
    }

    if (!success)
    {
        std::cout << "Numeric ILU factorization failed." << std::endl;
        return iterative_solver::State::FAILED;
    }
    std::cout << "ILU factorization done. nnz: " << ilu_matrix.NNZ() << std::endl;
    if (opts.factorization == Factorization::ILUT)
    {
        std::cout << "  Fill ratio: " << static_cast<double>(ilu_matrix.NNZ()) / pruned.NNZ() << std::endl;
    }

    // Setup SPMV operator on original matrix
    std::cout << "\nSetting up SPMV operator on original matrix..." << std::endl;
    using CSRTYPE = typename matrix_utils::CSRMatrix<int, int, double>;
    matrix_utils::SPMV<CSRTYPE, matrix_utils::SerialSPMV> spmv;
    spmv.setMatrix(&csr_matrix);
    spmv.preprocess();
    std::cout << "SPMV operator done." << std::endl;

    // Setup preconditioner operator
    std::cout << "Setting up preconditioner operator..." << std::endl;
    ILUPrec<decltype(ilu_matrix)> ilu_prec(csr_matrix.rows, ilu_matrix);
    std::cout << "Preconditioner operator done." << std::endl;

    // Setup RHS and initial guess
    std::vector<double> b(csr_matrix.rows, 1.0);
    std::vector<double> x(csr_matrix.rows, 0.0);

    std::cout << "\nRunning GMRES..." << std::endl;
    iterative_solver::GMRES<double> gmres_solver;
    gmres_solver.setMaxIter(opts.maxiter);
    gmres_solver.setRelTol(opts.reltol);
    gmres_solver.setRestart(opts.restart);
    gmres_solver.setPreconditionerType(opts.precond_type);

    auto solve_start = std::chrono::high_resolution_clock::now();
    auto state = gmres_solver(&spmv, &ilu_prec, b.data(), x.data());
    auto solve_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = solve_end - solve_start;

    std::cout << "GMRES done. Final state: ";
    switch (state)
    {
    case iterative_solver::State::CONVERGED:
        std::cout << "CONVERGED" << std::endl;
        break;
    case iterative_solver::State::MAX_ITER_REACHED:
        std::cout << "MAX_ITER_REACHED" << std::endl;
        break;
    case iterative_solver::State::FAILED:
        std::cout << "FAILED" << std::endl;
        break;
    default:
        std::cout << "UNKNOWN" << std::endl;
        break;
    }

    std::cout << "GMRES solve time: " << solve_time.count() << " s" << std::endl;

    // Compute final residual: r = Ax - b
    std::vector<double> residual(csr_matrix.rows);
    std::copy(b.begin(), b.end(), residual.begin()); // residual = b
    spmv(x.data(), residual.data(), 1.0, -1.0);      // residual = Ax - b

    // Compute L2 norm of residual (absolute)
    double residual_norm = 0.0;
    for (size_t i = 0; i < residual.size(); ++i)
    {
        residual_norm += residual[i] * residual[i];
    }
    residual_norm = std::sqrt(residual_norm);

    // Compute L2 norm of RHS vector b
    double b_norm = 0.0;
    for (size_t i = 0; i < b.size(); ++i)
    {
        b_norm += b[i] * b[i];
    }
    b_norm = std::sqrt(b_norm);

    // Compute relative residual norm
    double relative_residual_norm = (b_norm > 0.0) ? residual_norm / b_norm : residual_norm;

    std::cout << "\nFinal residual norms:" << std::endl;
    std::cout << "  Absolute L2 norm: " << std::scientific << std::setprecision(6) << residual_norm
              << std::endl;
    std::cout << "  Relative L2 norm: " << std::scientific << std::setprecision(6)
              << relative_residual_norm << std::endl;
    std::cout << "  RHS L2 norm:      " << std::scientific << std::setprecision(6) << b_norm << std::endl;

    return state;
}

void precondExplore(const Options& opts, const matrix_utils::CSRMatrix<int, int, double>& pruned)
{
    auto t_ilu_start = std::chrono::high_resolution_clock::now();
    matrix_utils::CSRMatrix<int, int, double> pruned_T;
    pruned_T.rows = pruned.cols;
    pruned_T.cols = pruned.rows;
    pruned_T.ResizeAI(pruned_T.rows + 1);
    pruned_T.ResizeAJ(pruned.NNZ());
    pruned_T.ResizeAV(pruned.NNZ());
    matrix_utils::ParallelTranspose2(pruned.rows, pruned.cols, pruned.AI(), pruned.AJ(),
                                     pruned.AV(), pruned_T.AI(), pruned_T.AJ(), pruned_T.AV());
    matrix_utils::CSRMatrix<int, int, double> L, U;
    matrix_utils::ILULevelSymbolicParallelU<matrix_utils::CSRMatrix<int, int, double>, true> ilu_sym(opts.threads);

    ilu_sym(pruned.rows, pruned.AI(), pruned.AJ(), opts.level, U);
    ilu_sym(pruned_T.rows, pruned_T.AI(), pruned_T.AJ(), opts.level, L);
    auto t_ilu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_ilu = t_ilu_end - t_ilu_start;
    std::cout << "\nParallel ILU(k) time: " << elapsed_ilu.count() << " s" << std::endl;

    // Transpose L from CSC to CSR format
    matrix_utils::CSRMatrix<int, int, double> L_csr;
    L_csr.rows = L.cols;
    L_csr.cols = L.rows;
    L_csr.ResizeAI(L_csr.rows + 1);
    L_csr.ResizeAJ(L.NNZ());
    L_csr.ResizeAV(L.NNZ());
    matrix_utils::ParallelTranspose2(L.rows, L.cols, L.AI(), L.AJ(),
                                     L.AV(), L_csr.AI(), L_csr.AJ(), L_csr.AV());

    // Sum L and U sparsity patterns
    matrix_utils::CSRMatrix<int, int, double> LplusU;
    matrix_utils::SpADD<matrix_utils::CSRMatrix<int, int, double>> spadd_op(opts.threads);
    
    // Analysis phase: determine sparsity pattern
    spadd_op.analysis(L_csr.rows, L_csr.cols, L_csr.AI(), L_csr.AJ(),
                      U.rows, U.cols, U.AI(), U.AJ(),
                      LplusU);
    
    // Numerical phase: compute sum (using alpha=1.0, beta=1.0 for simple addition)
    spadd_op(L_csr.rows, L_csr.cols, L_csr.AI(), L_csr.AJ(), L_csr.AV(), 1.0,
             U.rows, U.cols, U.AI(), U.AJ(), U.AV(), 1.0,
             LplusU);
    
    // Write L+U sparsity pattern to SVG
    std::ofstream out_lpu("L_plus_U_pattern.svg");
    std::cout << "\nWriting L+U sparsity pattern to SVG..." << std::endl;
    std::cout << "L+U dimensions: " << LplusU.rows << " x " << LplusU.cols << std::endl;
    std::cout << "L+U NNZ: " << LplusU.NNZ() << std::endl;
    matrix_utils::writeSVG(LplusU.rows, LplusU.cols, LplusU.AI(), LplusU.AJ(), out_lpu, opts.max_display_size);
    out_lpu.close();
    std::cout << "L+U sparsity pattern written to L_plus_U_pattern.svg" << std::endl;
}

#ifdef USE_METIS_LIB
matrix_utils::CSRMatrix<int, int, double> permuteMatrixWithMetisND(
    const matrix_utils::CSRMatrix<int, int, double>& matrix)
{
    std::cout << "\n=== Permuting matrix with MetisND ===" << std::endl;
    
    // Compute symmetric adjacency graph using A+A^T (without diagonal)
    // This is more efficient than CSRToMetisGraph and naturally handles symmetrization
    std::vector<int> xadj(matrix.rows + 1);
    matrix_utils::APlusATPrefix<int, int, false>(
        matrix.rows, matrix.AI(), matrix.AJ(), xadj.data());
    
    // Allocate and fill adjacency list
    int actual_edges = xadj[matrix.rows] - xadj[0];
    std::vector<int> adjncy(actual_edges);
    matrix_utils::APlusATFill<int, int, false>(
        matrix.rows, matrix.AI(), matrix.AJ(), xadj.data(), adjncy.data());
    
    std::cout << "Adjacency graph created (A+A^T without diagonal): " 
              << matrix.rows << " vertices, " << actual_edges << " edges" << std::endl;
    
    // Compute METIS nested dissection ordering
    std::vector<int> iperm(matrix.rows);
    std::vector<int> perm(matrix.rows);
    
    reordering::MetisNDOptions opts;
    opts.seed = 42;  // For reproducibility
    
    std::cout << "Computing MetisND ordering..." << std::endl;
    auto metis_start = std::chrono::high_resolution_clock::now();
    int result = reordering::MetisND(matrix.rows, matrix.cols,
                                     xadj.data(), adjncy.data(),
                                     iperm.data(), perm.data(), opts);
    auto metis_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> metis_time = metis_end - metis_start;
    std::cout << "MetisND ordering time: " << metis_time.count() << " s" << std::endl;

    // Print first 10 entries of iperm and perm for inspection
    std::cout << "\nFirst 10 entries of permutation vectors:" << std::endl;
    std::cout << "iperm: ";
    for (int i = 0; i < std::min(10, static_cast<int>(iperm.size())); ++i)
    {
        std::cout << iperm[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "perm:  ";
    for (int i = 0; i < std::min(10, static_cast<int>(perm.size())); ++i)
    {
        std::cout << perm[i] << " ";
    }
    std::cout << std::endl;

    if (result != 0) {
        std::cerr << "MetisND failed with error code: " << result << std::endl;
        return matrix;  // Return original matrix on failure
    }
    
    std::cout << "MetisND ordering computed successfully" << std::endl;
    
    // Allocate permuted matrix
    matrix_utils::CSRMatrix<int, int, double> permuted;
    permuted.rows = matrix.rows;
    permuted.cols = matrix.cols;
    permuted.ResizeAI(matrix.rows + 1);
    permuted.ResizeAJ(matrix.NNZ());
    permuted.ResizeAV(matrix.NNZ());
    
    // Permute the matrix: permuted = P * matrix * P^T (symmetric permutation)
    std::cout << "Permuting matrix structure and values..." << std::endl;
    matrix_utils::permuteMat(matrix.rows, matrix.cols,
                            perm.data(), iperm.data(),  // Use same perm for rows and cols (symmetric)
                            matrix.AI(), matrix.AJ(), matrix.AV(),
                            permuted.AI(), permuted.AJ(), permuted.AV());
    
    std::cout << "Matrix permuted successfully" << std::endl;
    std::cout << "Permuted matrix: " << permuted.rows << " x " << permuted.cols 
              << ", NNZ: " << permuted.NNZ() << std::endl;
    
    return permuted;
}
#endif

