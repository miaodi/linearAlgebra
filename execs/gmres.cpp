#include "io.hpp"
#include "iterative_solver.hpp"
#include "matrix_utils.hpp"
#include "precond.hpp"
#include "sparse_mat_traits.hpp"
#include "spmv.hpp"
#include "triangle_solve.hpp"
#include <cmath>
#include <cxxopts.hpp>
#include <fstream>
#include <iomanip>
#include <mkl.h>
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
    ILUPrec( const COLTYPE size, const CSRMatrixType& ilu )
        : _size( size ), _ilu( ilu ), tmp( size )
    {
        matrix_utils::SplitLDU( _size, _ilu.AI()[0], _ilu.AI(), _ilu.AJ(),
                                _ilu.AV(), L, D, U );
    }

    COLTYPE size() const
    {
        return _size;
    }

    bool operator()( VALTYPE const* const b, VALTYPE* const x ) const
    {
        matrix_utils::TriangularSolve<matrix_utils::TriangularMatrix::L>(
            _size, L.AI(), L.AJ(), L.AV(), null_diag, b, tmp.data() );
        matrix_utils::TriangularSolve<matrix_utils::TriangularMatrix::U>(
            _size, U.AI(), U.AJ(), U.AV(), D.data(), tmp.data(), x );
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

int main( int argc, char** argv )
{
    cxxopts::Options options( "GMRES Example",
                              "Example of using GMRES with a CSR matrix" );
    options.add_options()( "f,filename", "Matrix Market file to read",
                           cxxopts::value<std::string>()->default_value(
                               "../tests/data/ex5.mtx" ) )(
        "l,level", "ILU level", cxxopts::value<int>()->default_value( "0" ) )(
        "p,precond",
        "Preconditioner type: none (no preconditioning), left (M^-1 A x = M^-1 "
        "b), right (A M^-1 y = b)",
        cxxopts::value<std::string>()->default_value( "none" ) )(
        "r,restart", "GMRES restart parameter",
        cxxopts::value<int>()->default_value( "20" ) )(
        "m,maxiter", "Maximum number of GMRES iterations",
        cxxopts::value<int>()->default_value( "10" ) )(
    "t,reltol", "Relative tolerance for GMRES convergence",
    cxxopts::value<double>()->default_value( "1e-10" ) )( "h,help",
                                  "Print usage" );
    auto result = options.parse( argc, argv );
    if ( result.count( "help" ) )
    {
        std::cout << options.help() << std::endl;
        return 0;
    }
    std::string filename = result["filename"].as<std::string>();
    int level = result["level"].as<int>();
    std::string precond_type_str = result["precond"].as<std::string>();
    int restart = result["restart"].as<int>();
    int maxiter = result["maxiter"].as<int>();
    double reltol = result["reltol"].as<double>();

    std::cout << "Options:" << std::endl;
    std::cout << "  filename: " << filename << std::endl;
    std::cout << "  level: " << level << std::endl;
    std::cout << "  precond: " << precond_type_str << std::endl;
    std::cout << "  restart: " << restart << std::endl;
    std::cout << "  maxiter: " << maxiter << std::endl;
    std::cout << "  reltol: " << reltol << std::endl;

    // Validate restart parameter
    if ( restart <= 0 )
    {
        std::cerr << "Invalid restart parameter: " << restart
                  << ". Must be a positive integer." << std::endl;
        return -1;
    }

    // Parse preconditioner type
    iterative_solver::PreconditionerType precond_type;
    if ( precond_type_str == "none" )
    {
        precond_type = iterative_solver::PreconditionerType::NONE;
    }
    else if ( precond_type_str == "left" )
    {
        precond_type = iterative_solver::PreconditionerType::LEFT;
    }
    else if ( precond_type_str == "right" )
    {
        precond_type = iterative_solver::PreconditionerType::RIGHT;
    }
    else
    {
        std::cerr << "Invalid preconditioner type: " << precond_type_str
                  << ". Valid options are: none, left, right" << std::endl;
        return -1;
    }

    std::cout << "Using preconditioner type: " << precond_type_str << std::endl;
    std::cout << "Using restart parameter: " << restart << std::endl;

    std::ifstream f( filename );
    f.clear();
    f.seekg( 0, std::ios::beg );
    matrix_utils::CSRMatrix<int, int, double> csr_matrix, ilu_matrix;
    matrix_utils::readMatrixMarket( f, csr_matrix );
    std::cout << "size: " << csr_matrix.rows << " nnz: " << csr_matrix.NNZ() << std::endl;
    // std::ofstream out0( "mat_csr.svg" );
    // matrix_utils::writeSVG( csr_matrix.rows, csr_matrix.cols, csr_matrix.AI(),
    //                         csr_matrix.AJ(), out0 );
    // out0.close();
    bool success = false;
    std::cout << "Symbolic ILU factorization..." << std::endl;
    matrix_utils::ILULevelSymbolic<decltype( ilu_matrix )> ilu;
    auto t1 = std::chrono::high_resolution_clock::now();
    success = ilu( csr_matrix.rows, csr_matrix.AI(), csr_matrix.AJ(), level, ilu_matrix );
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t2 - t1;
    std::cout << "Symbolic ILU factorization time: " << elapsed.count() << " s"
              << std::endl;
    if ( !success )
    {
        std::cout << "Symbolic ILU factorization failed." << std::endl;
        return -1;
    }
    std::cout << "Symbolic ILU factorization done. nnz: " << ilu_matrix.NNZ() << std::endl;
    std::cout << "Numeric ILU factorization..." << std::endl;
    auto t3 = std::chrono::high_resolution_clock::now();
    // Add option to select original/optimized ILULevelNumeric
    std::cout << "Using ILULevelNumeric." << std::endl;
    success = matrix_utils::ILULevelNumeric( csr_matrix.rows, csr_matrix.AI(),
                                             csr_matrix.AJ(), csr_matrix.AV(),
                                             level, ilu_matrix );
    auto t4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_numeric = t4 - t3;
    std::cout << "Numeric ILU factorization time: " << elapsed_numeric.count()
              << " s" << std::endl;
    if ( !success )
    {
        std::cout << "Numeric ILU factorization failed." << std::endl;
        return -1;
    }
    std::cout << "ILU factorization done." << std::endl;
    // std::ofstream out1( "ilu_csr.svg" );
    // matrix_utils::writeSVG( ilu_matrix.rows, ilu_matrix.cols, ilu_matrix.AI(),
    //                         ilu_matrix.AJ(), out1 );
    // out1.close();

    // spmv operator
    std::cout << "spmv operator..." << std::endl;
    using CSRTYPE = typename matrix_utils::CSRMatrix<int, int, double>;
    matrix_utils::SPMV<CSRTYPE, matrix_utils::SerialSPMV> spmv;
    spmv.setMatrix( &csr_matrix );
    spmv.preprocess();
    std::cout << "spmv operator done." << std::endl;

    // precond operator
    std::cout << "precond operator..." << std::endl;
    ILUPrec<decltype( ilu_matrix )> ilu_prec( csr_matrix.rows, ilu_matrix );
    std::cout << "precond operator done." << std::endl;

    std::vector<double> b( csr_matrix.rows, 1.0 );
    std::vector<double> x( csr_matrix.rows, 0.0 );

    // // Randomly generate b and x vectors
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_real_distribution<double> dis(-1.0, 1.0);
    // // Generate random b vector
    // for (size_t i = 0; i < b.size(); ++i) {
    //   b[i] = dis(gen);
    // }

    // // Generate random initial guess for x
    // for (size_t i = 0; i < x.size(); ++i) {
    //   x[i] = dis(gen);
    // }

    std::cout
        << "Generated random b and x vectors (uniform distribution [-1, 1])"
        << std::endl;
    std::cout << "GMRES..." << std::endl;
    iterative_solver::GMRES<double> gmres_solver;
    gmres_solver.setMaxIter( maxiter );
    gmres_solver.setRelTol( reltol );
    gmres_solver.setRestart( restart );
    gmres_solver.setPreconditionerType( precond_type );
    auto state = gmres_solver( &spmv, &ilu_prec, b.data(), x.data() );

    std::cout << "GMRES done. Final state: ";
    switch ( state )
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

    // Compute final residual: r = Ax - b
    std::vector<double> residual( csr_matrix.rows );
    std::copy( b.begin(), b.end(), residual.begin() ); // residual = b
    spmv( x.data(), residual.data(), 1.0, -1.0 );      // residual = Ax - b

    // Compute L2 norm of residual (absolute)
    double residual_norm = 0.0;
    for ( size_t i = 0; i < residual.size(); ++i )
    {
        residual_norm += residual[i] * residual[i];
    }
    residual_norm = std::sqrt( residual_norm );

    // Compute L2 norm of RHS vector b
    double b_norm = 0.0;
    for ( size_t i = 0; i < b.size(); ++i )
    {
        b_norm += b[i] * b[i];
    }
    b_norm = std::sqrt( b_norm );

    // Compute relative residual norm
    double relative_residual_norm = ( b_norm > 0.0 ) ? residual_norm / b_norm : residual_norm;

    std::cout << "Final residual norms:" << std::endl;
    std::cout << "  Absolute L2 norm: " << std::scientific
              << std::setprecision( 6 ) << residual_norm << std::endl;
    std::cout << "  Relative L2 norm: " << std::scientific
              << std::setprecision( 6 ) << relative_residual_norm << std::endl;
    std::cout << "  RHS L2 norm:      " << std::scientific
              << std::setprecision( 6 ) << b_norm << std::endl;

    return ( state == iterative_solver::State::CONVERGED ) ? 0 : -1;
}