#include "io.hpp"
#include "iterative_solver.hpp"
#include "matrix_utils.hpp"
#include "precond.hpp"
#include "sparse_mat_traits.hpp"
#include "spmv.hpp"
#include "triangle_solve.hpp"
#include <algorithm>
#include <cmath>
#include <cctype>
#include <cxxopts.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <omp.h>
#include <random>
#include <string>
#include <vector>

template <matrix_utils::ResizableDiagonalType CSRMatrixType>
class ILUPrec
{
public:
    using ROWTYPE = typename CSRMatrixType::ROWTYPE;
    using COLTYPE = typename CSRMatrixType::COLTYPE;
    using VALTYPE = typename CSRMatrixType::VALTYPE;
    ILUPrec(const COLTYPE size, const CSRMatrixType& ilu)
        : _size(size), _ilu(ilu), tmp(size)
    {
        matrix_utils::SplitLDU(_size, _ilu.AI()[0], _ilu.AI(), _ilu.AJ(),
                               _ilu.AV(), L, D, U);
    }

    COLTYPE size() const
    {
        return _size;
    }

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

int main(int argc, char* argv[])
{
    cxxopts::Options options("drop_global_precond", "Test GMRES with global preconditioner dropping");

    // clang-format off
    options.add_options()
        ( "f,filename", "Matrix Market file to read", cxxopts::value<std::string>()->default_value( "../../tests/data/ex5.mtx" ) )
        ( "o,output", "Output SVG file path", cxxopts::value<std::string>()->default_value( "matrix.svg" ) )
        ( "s,size", "Maximum display size (pixels)", cxxopts::value<int>()->default_value( "2000" ) )
        ( "t,threshold", "Pruning threshold (absolute value)", cxxopts::value<double>()->default_value( "0.0" ) )
        ( "l,level", "ILU level", cxxopts::value<int>()->default_value( "0" ) )
        ( "F,factorization", "ILU variant: iluk or ilut", cxxopts::value<std::string>()->default_value( "iluk" ) )
        ( "d,droptol", "ILUT drop tolerance", cxxopts::value<double>()->default_value( "1e-3" ) )
        ( "p,precond", "Preconditioner type: none (no preconditioning), left (M^-1 A x = M^-1 b), right (A M^-1 y = b)",
          cxxopts::value<std::string>()->default_value( "left" ) )
        ( "r,restart", "GMRES restart parameter", cxxopts::value<int>()->default_value( "60" ) )
        ( "m,maxiter", "Maximum number of GMRES iterations", cxxopts::value<int>()->default_value( "1000" ) )
        ( "reltol", "Relative tolerance for GMRES convergence", cxxopts::value<double>()->default_value( "1e-10" ) )
        ( "h,help", "Print usage" );
    // clang-format on

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        return 0;
    }

    std::string filename = result["filename"].as<std::string>();
    std::string output_file = result["output"].as<std::string>();
    int max_display_size = result["size"].as<int>();
    double threshold = result["threshold"].as<double>();
    int level = result["level"].as<int>();
    std::string precond_type_str = result["precond"].as<std::string>();
    std::string factorization_str = result["factorization"].as<std::string>();
    double droptol = result["droptol"].as<double>();
    int restart = result["restart"].as<int>();
    int maxiter = result["maxiter"].as<int>();
    double reltol = result["reltol"].as<double>();

    std::string factorization_lower = factorization_str;
    std::transform(factorization_lower.begin(), factorization_lower.end(),
                   factorization_lower.begin(), [](unsigned char c) { return std::tolower(c); });
    enum class Factorization
    {
        ILUK,
        ILUT
    };
    Factorization factorization;
    if (factorization_lower == "iluk")
    {
        factorization = Factorization::ILUK;
    }
    else if (factorization_lower == "ilut")
    {
        factorization = Factorization::ILUT;
    }
    else
    {
        std::cerr << "Invalid factorization type: " << factorization_str
                  << ". Valid options are: iluk, ilut" << std::endl;
        return -1;
    }

    std::cout << "Options:" << std::endl;
    std::cout << "  filename: " << filename << std::endl;
    std::cout << "  output: " << output_file << std::endl;
    std::cout << "  max_display_size: " << max_display_size << std::endl;
    std::cout << "  threshold: " << threshold << std::endl;
    std::cout << "  level: " << level << std::endl;
    std::cout << "  factorization: " << factorization_str << std::endl;
    if (factorization == Factorization::ILUT)
    {
        std::cout << "  droptol: " << droptol << std::endl;
    }
    std::cout << "  precond: " << precond_type_str << std::endl;
    std::cout << "  restart: " << restart << std::endl;
    std::cout << "  maxiter: " << maxiter << std::endl;
    std::cout << "  reltol: " << reltol << std::endl;

    // Validate max_display_size
    if (max_display_size <= 0)
    {
        std::cerr << "Invalid max_display_size: " << max_display_size
                  << ". Must be a positive integer." << std::endl;
        return -1;
    }

    // Validate restart parameter
    if (restart <= 0)
    {
        std::cerr << "Invalid restart parameter: " << restart
                  << ". Must be a positive integer." << std::endl;
        return -1;
    }

    // Parse preconditioner type
    iterative_solver::PreconditionerType precond_type;
    if (precond_type_str == "none")
    {
        precond_type = iterative_solver::PreconditionerType::NONE;
    }
    else if (precond_type_str == "left")
    {
        precond_type = iterative_solver::PreconditionerType::LEFT;
    }
    else if (precond_type_str == "right")
    {
        precond_type = iterative_solver::PreconditionerType::RIGHT;
    }
    else
    {
        std::cerr << "Invalid preconditioner type: " << precond_type_str
                  << ". Valid options are: none, left, right" << std::endl;
        return -1;
    }

    omp_set_num_threads(8);
    std::ifstream f(filename);
    if (!f.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return -1;
    }

    f.clear();
    f.seekg(0, std::ios::beg);
    matrix_utils::CSRMatrix<int, int, double> csr_matrix;
    matrix_utils::readMatrixMarket(f, csr_matrix);
    f.close();

    std::cout << "Matrix: " << csr_matrix.rows << " x " << csr_matrix.cols
              << ", NNZ: " << csr_matrix.NNZ() << std::endl;

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

    std::cout << "\nPruning matrix with thresholds (a_ii * a_jj * " << threshold << ")..." << std::endl;
    int original_nnz = pruned.NNZ();
    auto removed = matrix_utils::DiagonalScaledPrune(pruned.rows, pruned.AI(),
                                                     pruned.AJ(), pruned.AV(), threshold);
    int pruned_nnz = pruned.NNZ();
    std::cout << "Original NNZ: " << original_nnz << std::endl;
    std::cout << "Pruned NNZ: " << pruned_nnz << std::endl;
    std::cout << "Removed entries: " << removed << std::endl;
    if (original_nnz > 0)
    {
        std::cout << "Retention rate: " << (static_cast<double>(pruned_nnz) / original_nnz * 100.0)
                  << "%" << std::endl;
    }

    // Build ILU preconditioner from pruned matrix
    std::cout << "\nBuilding ILU preconditioner from pruned matrix..." << std::endl;
    matrix_utils::CSRMatrix<int, int, double> ilu_matrix;
    bool success = false;

    if (factorization == Factorization::ILUK)
    {
        std::cout << "Symbolic ILU(k) factorization..." << std::endl;
        matrix_utils::ILULevelSymbolic<decltype(ilu_matrix)> ilu;
        auto t1 = std::chrono::high_resolution_clock::now();
        success = ilu(pruned.rows, pruned.AI(), pruned.AJ(), level, ilu_matrix);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = t2 - t1;
        std::cout << "Symbolic ILU factorization time: " << elapsed.count() << " s" << std::endl;
        if (!success)
        {
            std::cout << "Symbolic ILU factorization failed." << std::endl;
            return -1;
        }
        std::cout << "Symbolic ILU factorization done. nnz: " << ilu_matrix.NNZ() << std::endl;

        std::cout << "Numeric ILU factorization..." << std::endl;
        auto t3 = std::chrono::high_resolution_clock::now();
        success = matrix_utils::ILULevelNumeric(pruned.rows, pruned.AI(), pruned.AJ(), pruned.AV(),
                                                level, ilu_matrix);
        auto t4 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_numeric = t4 - t3;
        std::cout << "Numeric ILU factorization time: " << elapsed_numeric.count() << " s" << std::endl;
    }
    else
    {
        std::cout << "Using ILUTNumeric with droptol = " << droptol << std::endl;
        auto t3 = std::chrono::high_resolution_clock::now();
        success = matrix_utils::ILUTNumeric(pruned.rows, pruned.AI(), pruned.AJ(), pruned.AV(),
                                            droptol, ilu_matrix);
        auto t4 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_numeric = t4 - t3;
        std::cout << "Numeric ILU factorization time: " << elapsed_numeric.count() << " s" << std::endl;
    }

    if (!success)
    {
        std::cout << "Numeric ILU factorization failed." << std::endl;
        return -1;
    }
    std::cout << "ILU factorization done. nnz: " << ilu_matrix.NNZ() << std::endl;
    if (factorization == Factorization::ILUT)
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
    gmres_solver.setMaxIter(maxiter);
    gmres_solver.setRelTol(reltol);
    gmres_solver.setRestart(restart);
    gmres_solver.setPreconditionerType(precond_type);
    
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
    std::cout << "  Absolute L2 norm: " << std::scientific << std::setprecision(6) << residual_norm << std::endl;
    std::cout << "  Relative L2 norm: " << std::scientific << std::setprecision(6) << relative_residual_norm << std::endl;
    std::cout << "  RHS L2 norm:      " << std::scientific << std::setprecision(6) << b_norm << std::endl;

    // Write pruned matrix SVG to file
    std::cout << "\nWriting pruned matrix to SVG..." << std::endl;
    std::ofstream out(output_file);
    if (!out.is_open())
    {
        std::cerr << "Failed to create output file: " << output_file << std::endl;
        return -1;
    }

    matrix_utils::writeSVG(pruned.rows, pruned.cols, pruned.AI(), pruned.AJ(), out, max_display_size);
    out.close();

    std::cout << "SVG written to: " << output_file << std::endl;

    return (state == iterative_solver::State::CONVERGED) ? 0 : -1;
}
