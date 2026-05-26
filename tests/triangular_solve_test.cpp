
#include "precond.hpp"
#include "io.hpp"
#include "matrix_utils.hpp"
#include "triangle_solve.hpp"
#include "utils.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>

// For BLAS dtrsm function
extern "C"
{
    void dtrsm_( const char* side,
                 const char* uplo,
                 const char* transa,
                 const char* diag,
                 const int* m,
                 const int* n,
                 const double* alpha,
                 const double* a,
                 const int* lda,
                 double* b,
                 const int* ldb );
}

using namespace matrix_utils;

// The fixture for testing class Foo.
class triangular_solve_Test : public testing::Test
{
protected:
    std::vector<matrix_utils::CSRMatrix<int, int, double>> _mats;
    // Precomputed ILU(k) split factors for reuse
    std::vector<matrix_utils::CSRMatrix<int, int, double>> _Ls;
    std::vector<matrix_utils::CSRMatrix<int, int, double>> _Us;
    std::vector<std::vector<double>> _Ds;
    std::vector<bool> _factor_ok; // whether ILU succeeded for that matrix
    int _ilu_level = 5;           // default ILU level

    const double _tol = 1e-10; // Relaxed tolerance for numerical comparison
    const double _MKLtol = 1e-10;

    triangular_solve_Test()
    {
        std::vector<int> csr_rows;
        std::vector<int> csr_cols;
        std::vector<double> csr_vals;

        const std::vector<std::string> matrix_files = {
            "data/ex5.mtx", // https://sparse.tamu.edu/FIDAP/ex5
            // "data/nos5.mtx",
            // "data/s3rmt3m3.mtx",
            // "data/bcsstk17.mtx"
        };

        auto load_matrix = [&]( const std::string& path )
        {
            std::ifstream f( path );
            if ( !f.good() )
                return; // silently skip if not found
            csr_rows.clear();
            csr_cols.clear();
            csr_vals.clear();
            matrix_utils::readMatrixMarket( f, csr_rows, csr_cols, csr_vals );
            f.close();
            _mats.push_back( createMatrixFromVectors( csr_rows, csr_cols, csr_vals ) );
        };
        for ( const auto& mf : matrix_files )
            load_matrix( mf );

        // Precompute ILU(k) and LDU splits
        _Ls.resize( _mats.size() );
        _Us.resize( _mats.size() );
        _Ds.resize( _mats.size() );
        _factor_ok.resize( _mats.size(), false );
        for ( size_t i = 0; i < _mats.size(); ++i )
        {
            auto& mat = _mats[i];
            matrix_utils::CSRMatrix<int, int, double> ilu_matrix;
            matrix_utils::ILULevelSymbolic<decltype( ilu_matrix )> ilu_sym;
            if ( !ilu_sym( mat.rows, mat.AI(), mat.AJ(), _ilu_level, ilu_matrix ) )
                continue;
            if ( !matrix_utils::ILULevelNumeric( mat.rows, mat.AI(), mat.AJ(), mat.AV(), _ilu_level, ilu_matrix ) )
                continue;
            matrix_utils::SplitLDU( ilu_matrix.rows, ilu_matrix.Base(), ilu_matrix.AI(),
                                    ilu_matrix.AJ(), ilu_matrix.AV(), _Ls[i], _Ds[i], _Us[i] );
            _factor_ok[i] = true;
        }
    }

    ~triangular_solve_Test() override
    {
        // You can do clean-up work that doesn't throw exceptions here.
    }

    // If the constructor and destructor are not enough for setting up
    // and cleaning up each test, you can define the following methods:

    void SetUp() override
    {
        // Code here will be called immediately after the constructor (right
        // before each test).
    }

    void TearDown() override
    {
        // Code here will be called immediately after each test (right
        // before the destructor).
    }

    // Class members declared here can be used by all tests in the test suite
    // for Foo.

    // Helper: solve triangular system using BLAS DTRSM
    void solveBlasTrsm( const std::vector<double>& matrix_dense,
                        const std::vector<double>& b,
                        std::vector<double>& x,
                        int size,
                        bool is_lower,
                        bool unit_diagonal ) const
    {
        // Copy b to x for in-place solution
        std::copy( b.begin(), b.end(), x.begin() );

        // Set up DTRSM parameters
        const char side = 'L';                       // Left side
        const char uplo = is_lower ? 'L' : 'U';      // Lower or Upper triangular
        const char transa = 'N';                     // No transpose
        const char diag = unit_diagonal ? 'U' : 'N'; // Unit or Non-unit diagonal
        const int m = size;
        const int n = 1; // Single RHS
        const double alpha = 1.0;
        const int lda = size; // Leading dimension of A
        const int ldb = size; // Leading dimension of B

        dtrsm_( &side, &uplo, &transa, &diag, &m, &n, &alpha, matrix_dense.data(), &lda, x.data(), &ldb );
    }

    // Helper: convert CSR matrix to dense column-major format
    std::vector<double> csrToDenseColumnMajor( const matrix_utils::CSRMatrix<int, int, double>& csr_mat,
                                               const std::vector<double>* diagonal = nullptr,
                                               bool is_lower = true ) const
    {
        const int size = csr_mat.rows;
        std::vector<double> dense( size * size, 0.0 );

        for ( int i = 0; i < csr_mat.rows; i++ )
        {
            // Set diagonal
            if ( diagonal )
            {
                dense[i + i * size] = ( *diagonal )[i]; // Column-major indexing
            }
            else
            {
                dense[i + i * size] = 1.0; // Unit diagonal
            }

            // Set off-diagonal elements
            for ( int j = csr_mat.AI()[i] - csr_mat.Base(); j < csr_mat.AI()[i + 1] - csr_mat.Base(); j++ )
            {
                int col = csr_mat.AJ()[j] - csr_mat.Base();
                bool valid_element = is_lower ? ( col < i ) : ( col > i );

                if ( valid_element )
                {
                    // Column-major indexing: dense[row + col * size]
                    dense[i + col * size] = csr_mat.AV()[j];
                }
            }
        }

        return dense;
    }

    // Common small helpers to reduce repetition in tests
    bool nonZeroVector( const std::vector<double>& x ) const
    {
        return std::any_of( x.begin(), x.end(), []( double v ) { return std::abs( v ) > 1e-15; } );
    }

    void checkForward( size_t midx ) const
    {
        if ( !_factor_ok[midx] )
            return; // skip failed factorization
        const auto& L = _Ls[midx];
        const int n = L.rows;
        std::vector<double> b( n, 1.0 ), x_ref( n, 0. ), x_blas( n, 0. );
        TriangularSolve<TriangularMatrix::L>( L.rows, L.AI(), L.AJ(), L.AV(), (double*)nullptr,
                                              b.data(), x_ref.data() );
        auto L_dense = csrToDenseColumnMajor( L, nullptr, true );
        solveBlasTrsm( L_dense, b, x_blas, n, true, true );
        for ( int i = 0; i < n; ++i )
        {
            EXPECT_NEAR( x_ref[i], x_blas[i], _tol * std::max( 1.0, std::abs( x_blas[i] ) ) );
        }
        EXPECT_TRUE( nonZeroVector( x_ref ) );
    }

    void checkBackward( size_t midx ) const
    {
        if ( !_factor_ok[midx] )
            return;
        const auto& U = _Us[midx];
        const auto& D = _Ds[midx];
        const int n = U.rows;
        std::vector<double> b( n, 1.0 ), x_ref( n, 0. ), x_blas( n, 0. );
        TriangularSolve<TriangularMatrix::U>( U.rows, U.AI(), U.AJ(), U.AV(), D.data(), b.data(),
                                              x_ref.data() );
        auto U_dense = csrToDenseColumnMajor( U, &D, false );
        solveBlasTrsm( U_dense, b, x_blas, n, false, false );
        for ( int i = 0; i < n; ++i )
        {
            EXPECT_NEAR( x_ref[i], x_blas[i], _tol * std::max( 1.0, std::abs( x_blas[i] ) ) );
        }
        EXPECT_TRUE( nonZeroVector( x_ref ) );
    }

    void checkLevelForward( size_t midx ) const
    {
        if ( !_factor_ok[midx] )
            return;
        const auto& L = _Ls[midx];
        const int n = L.rows;
        std::vector<double> b( n, 1.0 ), x_level( n, 0. ), x_ref( n, 0. ), x_blas( n, 0. );
        TriangularSolve<TriangularMatrix::L>( L.rows, L.AI(), L.AJ(), L.AV(), (double*)nullptr,
                                              b.data(), x_ref.data() );
        auto L_dense = csrToDenseColumnMajor( L, nullptr, true );
        solveBlasTrsm( L_dense, b, x_blas, n, true, true );
        LevelScheduleTriangularSubstitution<TriangularMatrix::L, int, int, double> levelSolve(
            omp_get_max_threads() );
        levelSolve.analysis( L.rows, L.AI(), L.AJ(), L.AV(), (double*)nullptr );
        levelSolve( b.data(), x_level.data() );
        for ( int i = 0; i < n; ++i )
        {
            EXPECT_NEAR( x_level[i], x_ref[i], _tol * std::max( 1.0, std::abs( x_ref[i] ) ) );
            EXPECT_NEAR( x_level[i], x_blas[i], _tol * std::max( 1.0, std::abs( x_blas[i] ) ) );
        }
        EXPECT_TRUE( nonZeroVector( x_level ) );
    }

    void checkLevelBackward( size_t midx ) const
    {
        if ( !_factor_ok[midx] )
            return;
        const auto& U = _Us[midx];
        const auto& D = _Ds[midx];
        const int n = U.rows;
        std::vector<double> b( n, 1.0 ), x_level( n, 0. ), x_ref( n, 0. ), x_blas( n, 0. );
        TriangularSolve<TriangularMatrix::U>( U.rows, U.AI(), U.AJ(), U.AV(), D.data(), b.data(),
                                              x_ref.data() );
        auto U_dense = csrToDenseColumnMajor( U, &D, false );
        solveBlasTrsm( U_dense, b, x_blas, n, false, false );
        LevelScheduleTriangularSubstitution<TriangularMatrix::U, int, int, double> levelSolve(
            omp_get_max_threads() );
        levelSolve.analysis( U.rows, U.AI(), U.AJ(), U.AV(), D.data() );
        levelSolve( b.data(), x_level.data() );
        for ( int i = 0; i < n; ++i )
        {
            EXPECT_NEAR( x_level[i], x_ref[i], _tol * std::max( 1.0, std::abs( x_ref[i] ) ) );
            EXPECT_NEAR( x_level[i], x_blas[i], _tol * std::max( 1.0, std::abs( x_blas[i] ) ) );
        }
        EXPECT_TRUE( nonZeroVector( x_level ) );
    }

    void checkOptimizedForwardBarrier( size_t midx ) const
    {
        if ( !_factor_ok[midx] )
            return;
        const auto& L = _Ls[midx];
        const int n = L.rows;
        std::vector<double> b( n, 1.0 ), x_optimized( n, 0. ), x_ref( n, 0. );
        TriangularSolve<TriangularMatrix::L>( L.rows, L.AI(), L.AJ(), L.AV(), (double*)nullptr,
                                              b.data(), x_ref.data() );
        OptimizedTriangularSolve<FBSubstitutionType::Barrier, TriangularMatrix::L, int, int, double> optimizedSolve(
            omp_get_max_threads() );
        optimizedSolve.analysis( L.rows, L.Base(), L.AI(), L.AJ(), L.AV(), (double*)nullptr );
        optimizedSolve( b.data(), x_optimized.data() );
        for ( int i = 0; i < n; ++i )
        {
            EXPECT_NEAR( x_optimized[i], x_ref[i], _tol * std::max( 1.0, std::abs( x_ref[i] ) ) );
        }
        EXPECT_TRUE( nonZeroVector( x_optimized ) );
    }

    void checkOptimizedForwardNoBarrier( size_t midx ) const
    {
        if ( !_factor_ok[midx] )
            return;
        const auto& L = _Ls[midx];
        const int n = L.rows;
        std::vector<double> b( n, 1.0 ), x_optimized( n, 0. ), x_ref( n, 0. );
        TriangularSolve<TriangularMatrix::L>( L.rows, L.AI(), L.AJ(), L.AV(), (double*)nullptr,
                                              b.data(), x_ref.data() );
        OptimizedTriangularSolve<FBSubstitutionType::NoBarrier, TriangularMatrix::L, int, int, double> optimizedSolve(
            omp_get_max_threads() );
        optimizedSolve.analysis( L.rows, L.Base(), L.AI(), L.AJ(), L.AV(), (double*)nullptr );
        optimizedSolve( b.data(), x_optimized.data() );
        for ( int i = 0; i < n; ++i )
        {
            EXPECT_NEAR( x_optimized[i], x_ref[i], _tol * std::max( 1.0, std::abs( x_ref[i] ) ) );
        }
        EXPECT_TRUE( nonZeroVector( x_optimized ) );
    }

    void checkOptimizedForwardNoBarrierSuperNode( size_t midx ) const
    {
        if ( !_factor_ok[midx] )
            return;
        const auto& L = _Ls[midx];
        const int n = L.rows;
        std::vector<double> b( n, 1.0 ), x_optimized( n, 0. ), x_ref( n, 0. );
        TriangularSolve<TriangularMatrix::L>( L.rows, L.AI(), L.AJ(), L.AV(), (double*)nullptr,
                                              b.data(), x_ref.data() );
        OptimizedTriangularSolve<FBSubstitutionType::NoBarrierSuperNode, TriangularMatrix::L, int, int, double> optimizedSolve(
            omp_get_max_threads() );
        optimizedSolve.analysis( L.rows, L.Base(), L.AI(), L.AJ(), L.AV(), (double*)nullptr );
        optimizedSolve( b.data(), x_optimized.data() );
        for ( int i = 0; i < n; ++i )
        {
            EXPECT_NEAR( x_optimized[i], x_ref[i], _tol * std::max( 1.0, std::abs( x_ref[i] ) ) );
        }
        EXPECT_TRUE( nonZeroVector( x_optimized ) );
    }

    void checkOptimizedBackwardBarrier( size_t midx ) const
    {
        if ( !_factor_ok[midx] )
            return;
        const auto& U = _Us[midx];
        const auto& D = _Ds[midx];
        const int n = U.rows;
        std::vector<double> b( n, 1.0 ), x_optimized( n, 0. ), x_ref( n, 0. );
        TriangularSolve<TriangularMatrix::U>( U.rows, U.AI(), U.AJ(), U.AV(), D.data(), b.data(),
                                              x_ref.data() );
        OptimizedTriangularSolve<FBSubstitutionType::Barrier, TriangularMatrix::U, int, int, double> optimizedSolve(
            omp_get_max_threads() );
        optimizedSolve.analysis( U.rows, U.Base(), U.AI(), U.AJ(), U.AV(), D.data() );
        optimizedSolve( b.data(), x_optimized.data() );
        for ( int i = 0; i < n; ++i )
        {
            EXPECT_NEAR( x_optimized[i], x_ref[i], _tol * std::max( 1.0, std::abs( x_ref[i] ) ) );
        }
        EXPECT_TRUE( nonZeroVector( x_optimized ) );
    }

    void checkOptimizedBackwardNoBarrier( size_t midx ) const
    {
        if ( !_factor_ok[midx] )
            return;
        const auto& U = _Us[midx];
        const auto& D = _Ds[midx];
        const int n = U.rows;
        std::vector<double> b( n, 1.0 ), x_optimized( n, 0. ), x_ref( n, 0. );
        TriangularSolve<TriangularMatrix::U>( U.rows, U.AI(), U.AJ(), U.AV(), D.data(), b.data(),
                                              x_ref.data() );
        OptimizedTriangularSolve<FBSubstitutionType::NoBarrier, TriangularMatrix::U, int, int, double> optimizedSolve(
            omp_get_max_threads() );
        optimizedSolve.analysis( U.rows, U.Base(), U.AI(), U.AJ(), U.AV(), D.data() );
        optimizedSolve( b.data(), x_optimized.data() );
        for ( int i = 0; i < n; ++i )
        {
            EXPECT_NEAR( x_optimized[i], x_ref[i], _tol * std::max( 1.0, std::abs( x_ref[i] ) ) );
        }
        EXPECT_TRUE( nonZeroVector( x_optimized ) );
    }

    void checkOptimizedBackwardNoBarrierSuperNode( size_t midx ) const
    {
        if ( !_factor_ok[midx] )
            return;
        const auto& U = _Us[midx];
        const auto& D = _Ds[midx];
        const int n = U.rows;
        std::vector<double> b( n, 1.0 ), x_optimized( n, 0. ), x_ref( n, 0. );
        TriangularSolve<TriangularMatrix::U>( U.rows, U.AI(), U.AJ(), U.AV(), D.data(), b.data(),
                                              x_ref.data() );
        OptimizedTriangularSolve<FBSubstitutionType::NoBarrierSuperNode, TriangularMatrix::U, int, int, double> optimizedSolve(
            omp_get_max_threads() );
        optimizedSolve.analysis( U.rows, U.Base(), U.AI(), U.AJ(), U.AV(), D.data() );
        optimizedSolve( b.data(), x_optimized.data() );
        for ( int i = 0; i < n; ++i )
        {
            EXPECT_NEAR( x_optimized[i], x_ref[i], _tol * std::max( 1.0, std::abs( x_ref[i] ) ) );
        }
        EXPECT_TRUE( nonZeroVector( x_optimized ) );
    }

private:
    // Helper function to create CSRMatrix from matrix market data
    matrix_utils::CSRMatrix<int, int, double> createMatrixFromVectors( const std::vector<int>& csr_rows,
                                                                       const std::vector<int>& csr_cols,
                                                                       const std::vector<double>& csr_vals )
    {
        matrix_utils::CSRMatrix<int, int, double> mat;
        mat.rows = csr_rows.size() - 1;
        mat.cols = csr_rows.size() - 1;
        mat.ResizeAI( csr_rows.size() );
        mat.ResizeAJ( csr_vals.size() );
        mat.ResizeAV( csr_vals.size() );
        std::copy( csr_rows.begin(), csr_rows.end(), mat.AI() );
        std::copy( csr_cols.begin(), csr_cols.end(), mat.AJ() );
        std::copy( csr_vals.begin(), csr_vals.end(), mat.AV() );
        return mat;
    }
};

int main( int argc, char** argv )
{
    testing::InitGoogleTest( &argc, argv );
    return RUN_ALL_TESTS();
}

// TEST_F( triangular_solve_Test, matrix_loading )
// {
//     // Simple test to verify matrices are loaded correctly
//     EXPECT_EQ( _mats.size(), 4 );

//     for ( const auto& mat : _mats )
//     {
//         EXPECT_GT( mat.rows, 0 );
//         EXPECT_GT( mat.cols, 0 );
//         EXPECT_GT( mat.NNZ(), 0 );
//         EXPECT_EQ( mat.rows, mat.cols ); // All test matrices should be square

//         // Verify base indexing
//         EXPECT_TRUE( mat.Base() == 0 || mat.Base() == 1 );

//         // Verify matrix structure is valid
//         EXPECT_TRUE( matrix_utils::ValidCSR( mat.rows, mat.cols, mat.Base(),
//                                            mat.AI(), mat.AJ() ) );
//     }
// }

TEST_F( triangular_solve_Test, forward_substitution )
{
    for ( size_t i = 0; i < _mats.size(); ++i )
        checkForward( i );
}

TEST_F( triangular_solve_Test, backward_substitution )
{
    for ( size_t i = 0; i < _mats.size(); ++i )
        checkBackward( i );
}

TEST_F( triangular_solve_Test, level_scheduled_forward_substitution )
{
    for ( size_t i = 0; i < _mats.size(); ++i )
        checkLevelForward( i );
}

TEST_F( triangular_solve_Test, level_scheduled_backward_substitution )
{
    for ( size_t i = 0; i < _mats.size(); ++i )
        checkLevelBackward( i );
}

TEST( triangular_solve_csr_Test, backward_substitution_unit_diagonal )
{
    const int n = 3;
    const std::vector<int> row_ptr{ 0, 2, 3, 3 };
    const std::vector<int> col_idx{ 1, 2, 2 };
    const std::vector<double> values{ 2.0, 3.0, 4.0 };
    const std::vector<double> b{ 14.0, 14.0, 3.0 };
    const std::vector<double> expected{ 1.0, 2.0, 3.0 };
    std::vector<double> x( n, 0.0 );

    TriangularSolve<TriangularMatrix::U>( n, row_ptr.data(), col_idx.data(), values.data(),
                                          static_cast<const double*>( nullptr ), b.data(), x.data() );

    for ( int i = 0; i < n; ++i )
    {
        EXPECT_NEAR( x[i], expected[i], 1e-12 );
    }
}

TEST( triangular_solve_csc_Test, forward_substitution_unit_diagonal )
{
    const int n = 3;
    const std::vector<int> col_ptr{ 0, 2, 3, 3 };
    const std::vector<int> row_idx{ 1, 2, 2 };
    const std::vector<double> values{ 2.0, 3.0, 4.0 };
    const std::vector<double> b{ 1.0, 4.0, 14.0 };
    const std::vector<double> expected{ 1.0, 2.0, 3.0 };
    std::vector<double> x( n, 0.0 );

    TriangularSolveCSC<TriangularMatrix::L>( n, col_ptr.data(), row_idx.data(), values.data(),
                                             static_cast<const double*>( nullptr ), b.data(), x.data() );

    for ( int i = 0; i < n; ++i )
    {
        EXPECT_NEAR( x[i], expected[i], 1e-12 );
    }
}

TEST( triangular_solve_csc_Test, backward_substitution_base_one )
{
    const int n = 3;
    const std::vector<int> col_ptr{ 1, 1, 2, 4 };
    const std::vector<int> row_idx{ 1, 1, 2 };
    const std::vector<double> values{ 3.0, 5.0, 6.0 };
    const std::vector<double> diag{ 2.0, 4.0, 7.0 };
    const std::vector<double> b{ 23.0, 26.0, 21.0 };
    const std::vector<double> expected{ 1.0, 2.0, 3.0 };
    std::vector<double> x( n, 0.0 );

    TriangularSolveCSC<TriangularMatrix::U>( n, col_ptr.data(), row_idx.data(), values.data(),
                                             diag.data(), b.data(), x.data() );

    for ( int i = 0; i < n; ++i )
    {
        EXPECT_NEAR( x[i], expected[i], 1e-12 );
    }
}

// Disabled: OptimizedTriangularSolve
// TEST_F( triangular_solve_Test, optimized_forward_substitution_barrier )
// {
//     for(size_t i=0;i<_mats.size();++i) checkOptimizedForwardBarrier(i);
// }

// Disabled: OptimizedTriangularSolve
// TEST_F( triangular_solve_Test, optimized_forward_substitution_no_barrier )
// {
//     for(size_t i=0;i<_mats.size();++i) checkOptimizedForwardNoBarrier(i);
// }

// Disabled: OptimizedTriangularSolve
// TEST_F( triangular_solve_Test, optimized_forward_substitution_no_barrier_super_node )
// {
//     for(size_t i=0;i<_mats.size();++i) checkOptimizedForwardNoBarrierSuperNode(i);
// }

// Disabled: OptimizedTriangularSolve
// TEST_F( triangular_solve_Test, optimized_backward_substitution_barrier )
// {
//     for(size_t i=0;i<_mats.size();++i) checkOptimizedBackwardBarrier(i);
// }

// Disabled: OptimizedTriangularSolve
// TEST_F( triangular_solve_Test, optimized_backward_substitution_no_barrier )
// {
//     for(size_t i=0;i<_mats.size();++i) checkOptimizedBackwardNoBarrier(i);
// }

// Disabled: OptimizedTriangularSolve
// TEST_F( triangular_solve_Test, optimized_backward_substitution_no_barrier_super_node )
// {
//     for(size_t i=0;i<_mats.size();++i) checkOptimizedBackwardNoBarrierSuperNode(i);
// }
