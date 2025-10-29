
#include "precond.hpp"
#include "matrix_utils.hpp"
#include "triangle_solve.hpp"
#include "utils.h"
#include <algorithm>
#include <fstream>
#include <gtest/gtest.h>

// For BLAS dtrsm function
extern "C" {
    void dtrsm_(const char* side, const char* uplo, const char* transa, const char* diag,
                const int* m, const int* n, const double* alpha, const double* a, const int* lda,
                double* b, const int* ldb);
}

using namespace matrix_utils;

// The fixture for testing class Foo.
class triangular_solve_Test : public testing::Test
{
protected:
    std::vector<matrix_utils::CSRMatrix<int, int, double>> _mats;

    const double _tol = 1e-12;  // Relaxed tolerance for numerical comparison
    const double _MKLtol = 1e-10;

    triangular_solve_Test()
    {
        std::vector<int> csr_rows;
        std::vector<int> csr_cols;
        std::vector<double> csr_vals;

        std::ifstream f( "data/ex5.mtx" ); // https://sparse.tamu.edu/FIDAP/ex5
        utils::read_matrix_market_csr( f, csr_rows, csr_cols, csr_vals );
        f.close();
        _mats.push_back(createMatrixFromVectors(csr_rows, csr_cols, csr_vals));

        f.open( "data/nos5.mtx" );
        utils::read_matrix_market_csr( f, csr_rows, csr_cols, csr_vals );
        f.close();
        _mats.push_back(createMatrixFromVectors(csr_rows, csr_cols, csr_vals));

        f.open( "data/s3rmt3m3.mtx" );
        utils::read_matrix_market_csr( f, csr_rows, csr_cols, csr_vals );
        f.close();
        _mats.push_back(createMatrixFromVectors(csr_rows, csr_cols, csr_vals));

        f.open( "data/bcsstk17.mtx" );
        utils::read_matrix_market_csr( f, csr_rows, csr_cols, csr_vals );
        f.close();
        _mats.push_back(createMatrixFromVectors(csr_rows, csr_cols, csr_vals));
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

    // Helper function to solve triangular system using BLAS DTRSM
    void solveBlasTrsm(const std::vector<double>& matrix_dense, 
                       const std::vector<double>& b,
                       std::vector<double>& x,
                       int size,
                       bool is_lower,
                       bool unit_diagonal) const
    {
        // Copy b to x for in-place solution
        std::copy(b.begin(), b.end(), x.begin());
        
        // Set up DTRSM parameters
        const char side = 'L';     // Left side
        const char uplo = is_lower ? 'L' : 'U';  // Lower or Upper triangular
        const char transa = 'N';   // No transpose
        const char diag = unit_diagonal ? 'U' : 'N';  // Unit or Non-unit diagonal
        const int m = size;
        const int n = 1;           // Single RHS
        const double alpha = 1.0;
        const int lda = size;      // Leading dimension of A
        const int ldb = size;      // Leading dimension of B
        
        dtrsm_(&side, &uplo, &transa, &diag, &m, &n, &alpha, 
               matrix_dense.data(), &lda, x.data(), &ldb);
    }

    // Helper function to convert CSR matrix to dense column-major format
    std::vector<double> csrToDenseColumnMajor(const matrix_utils::CSRMatrix<int, int, double>& csr_mat,
                                              const std::vector<double>* diagonal = nullptr,
                                              bool is_lower = true) const
    {
        const int size = csr_mat.rows;
        std::vector<double> dense(size * size, 0.0);
        
        for (int i = 0; i < csr_mat.rows; i++)
        {
            // Set diagonal
            if (diagonal)
            {
                dense[i + i * size] = (*diagonal)[i];  // Column-major indexing
            }
            else
            {
                dense[i + i * size] = 1.0;  // Unit diagonal
            }
            
            // Set off-diagonal elements
            for (int j = csr_mat.AI()[i] - csr_mat.Base(); j < csr_mat.AI()[i + 1] - csr_mat.Base(); j++)
            {
                int col = csr_mat.AJ()[j] - csr_mat.Base();
                bool valid_element = is_lower ? (col < i) : (col > i);
                
                if (valid_element)
                {
                    // Column-major indexing: dense[row + col * size]
                    dense[i + col * size] = csr_mat.AV()[j];
                }
            }
        }
        
        return dense;
    }

private:
    // Helper function to create CSRMatrix from matrix market data
    matrix_utils::CSRMatrix<int, int, double> createMatrixFromVectors(
        const std::vector<int>& csr_rows,
        const std::vector<int>& csr_cols, 
        const std::vector<double>& csr_vals)
    {
        matrix_utils::CSRMatrix<int, int, double> mat;
        mat.rows = csr_rows.size() - 1;
        mat.cols = csr_rows.size() - 1;
        mat.ResizeAI(csr_rows.size());
        mat.ResizeAJ(csr_vals.size());
        mat.ResizeAV(csr_vals.size());
        std::copy(csr_rows.begin(), csr_rows.end(), mat.AI());
        std::copy(csr_cols.begin(), csr_cols.end(), mat.AJ());
        std::copy(csr_vals.begin(), csr_vals.end(), mat.AV());
        return mat;
    }
};

int main( int argc, char** argv )
{
    testing::InitGoogleTest( &argc, argv );
    return RUN_ALL_TESTS();
}

TEST_F( triangular_solve_Test, matrix_loading )
{
    // Simple test to verify matrices are loaded correctly
    EXPECT_EQ( _mats.size(), 4 );
    
    for ( const auto& mat : _mats )
    {
        EXPECT_GT( mat.rows, 0 );
        EXPECT_GT( mat.cols, 0 );
        EXPECT_GT( mat.NNZ(), 0 );
        EXPECT_EQ( mat.rows, mat.cols ); // All test matrices should be square
        
        // Verify base indexing
        EXPECT_TRUE( mat.Base() == 0 || mat.Base() == 1 );
        
        // Verify matrix structure is valid
        EXPECT_TRUE( matrix_utils::ValidCSR( mat.rows, mat.cols, mat.Base(), 
                                           mat.AI(), mat.AJ() ) );
    }
}

TEST_F( triangular_solve_Test, forward_substitution )
{
    for ( auto& mat : _mats )
    {
        const int size = mat.rows;
        
        // Create ILU preconditioner using the same approach as gmres.cpp
        matrix_utils::CSRMatrix<int, int, double> ilu_matrix;
        
        matrix_utils::ILULevelSymbolic<decltype( ilu_matrix )> ilu;
        int level = 5;
        bool success = ilu( mat.rows, mat.AI(), mat.AJ(), level, ilu_matrix );
        
        if ( !success )
        {
            continue;
        }
        
        success = matrix_utils::ILULevelNumeric( mat.rows, mat.AI(), mat.AJ(), 
                                                mat.AV(), level, ilu_matrix );
        
        if ( !success )
        {
            continue;
        }

        std::vector<double> b( size, 1.0 );
        std::vector<double> x_serial( size, 0.0 );
        std::vector<double> x_par( size, 0.0 );

        // Split ILU matrix into L, D, U components
        matrix_utils::CSRMatrix<int, int, double> L, U;
        std::vector<double> D;
        matrix_utils::SplitLDU( ilu_matrix.rows, ilu_matrix.Base(), ilu_matrix.AI(),
                               ilu_matrix.AJ(), ilu_matrix.AV(), L, D, U );

        // Test forward substitution with L (unit diagonal)
        matrix_utils::TriangularSolve<matrix_utils::TriangularMatrix::L>(
            L.rows, L.AI(), L.AJ(), L.AV(), (double*)( nullptr ), b.data(),
            x_serial.data() );

        // Compare with BLAS TRSM for forward substitution
        std::vector<double> x_blas( size, 0.0 );
        
        // Convert CSR L matrix to dense format and solve with BLAS
        auto L_dense = csrToDenseColumnMajor(L, nullptr, true);  // Lower triangular, unit diagonal
        solveBlasTrsm(L_dense, b, x_blas, size, true, true);     // Lower triangular, unit diagonal
        
        // Compare results
        for ( int i = 0; i < size; i++ )
        {
            EXPECT_NEAR( x_serial[i], x_blas[i], _tol * std::max( 1.0, std::abs( x_blas[i] ) ) );
        }

        // Basic sanity check - solution should not be all zeros for non-trivial b
        bool non_zero_solution = false;
        for ( double val : x_serial )
        {
            if ( std::abs(val) > 1e-15 )
            {
                non_zero_solution = true;
                break;
            }
        }
        EXPECT_TRUE( non_zero_solution );
    }
}

TEST_F( triangular_solve_Test, backward_substitution )
{
    for ( auto& mat : _mats )
    {
        const int size = mat.rows;
        
        // Create ILU preconditioner using the same approach as gmres.cpp
        matrix_utils::CSRMatrix<int, int, double> ilu_matrix;
        
        matrix_utils::ILULevelSymbolic<decltype( ilu_matrix )> ilu;
        int level = 5;
        bool success = ilu( mat.rows, mat.AI(), mat.AJ(), level, ilu_matrix );
        
        if ( !success )
        {
            continue;
        }
        
        success = matrix_utils::ILULevelNumeric( mat.rows, mat.AI(), mat.AJ(), 
                                                mat.AV(), level, ilu_matrix );
        
        if ( !success )
        {
            continue;
        }

        std::vector<double> b( size, 1.0 );
        std::vector<double> x_serial( size, 0.0 );

        // Split ILU matrix into L, D, U components
        matrix_utils::CSRMatrix<int, int, double> L, U;
        std::vector<double> D;
        matrix_utils::SplitLDU( ilu_matrix.rows, ilu_matrix.Base(), ilu_matrix.AI(),
                               ilu_matrix.AJ(), ilu_matrix.AV(), L, D, U );

        // Test backward substitution with U (diagonal included)
        matrix_utils::TriangularSolve<matrix_utils::TriangularMatrix::U>( 
            U.rows, U.AI(), U.AJ(), U.AV(), D.data(), b.data(), x_serial.data() );
        
        // Compare with BLAS TRSM for backward substitution
        std::vector<double> x_blas( size, 0.0 );
        
        // Convert CSR U matrix to dense format and solve with BLAS
        auto U_dense = csrToDenseColumnMajor(U, &D, false);     // Upper triangular, non-unit diagonal
        solveBlasTrsm(U_dense, b, x_blas, size, false, false);  // Upper triangular, non-unit diagonal
        
        // Compare results
        for ( int i = 0; i < size; i++ )
        {
            EXPECT_NEAR( x_serial[i], x_blas[i], _tol * std::max( 1.0, std::abs( x_blas[i] ) ) );
        }
        
        // Basic sanity check - solution should not be all zeros for non-trivial b
        bool non_zero_solution = false;
        for ( double val : x_serial )
        {
            if ( std::abs(val) > 1e-15 )
            {
                non_zero_solution = true;
                break;
            }
        }
        EXPECT_TRUE( non_zero_solution );
    }
}

// TEST_F( triangular_solve_Test, forward_substitution_optimized )
// {
//     omp_set_num_threads( 5 );
//     for ( auto mat : _mats )
//     {
//         const MKL_INT size = mat.rows();
//         mat.to_zero_based();
//         mkl_wrapper::incomplete_lu_k prec;
//         prec.set_level( 5 );
//         prec.symbolic_factorize( &mat );
//         prec.numeric_factorize( &mat );

//         // std::ofstream myfile;
//         // myfile.open("prec.svg");
//         // prec.print_svg(myfile);
//         // myfile.close();

//         std::vector<double> b( mat.rows() );
//         std::fill( std::begin( b ), std::end( b ), 1. );
//         std::vector<double> x( mat.rows(), 0.0 );
//         std::vector<double> x_serial( mat.rows(), 0.0 );

//         matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> L, U;
//         std::vector<double> D;

//         matrix_utils::SplitLDU( prec.rows(), (int)prec.mkl_base(), prec.get_ai().get(),
//                                 prec.get_aj().get(), prec.get_av().get(), L, D, U );

//         matrix_utils::ForwardSubstitution( L.rows, L.Base(), L.ai.get(), L.aj.get(),
//                                            L.av.get(), b.data(), x_serial.data() );

//         matrix_utils::OptimizedTriangularSolve<matrix_utils::FBSubstitutionType::Barrier,
//                                                matrix_utils::TriangularMatrix::L, int, int, double>
//             forwardsweep_barrier;
//         forwardsweep_barrier.analysis( L.rows, L.Base(), L.ai.get(), L.aj.get(),
//                                        L.av.get() );
//         for ( int i = 0; i < 100; i++ )
//         {
//             forwardsweep_barrier( b.data(), x.data() );
//             for ( int i = 0; i < x.size(); i++ )
//             {
//                 EXPECT_NEAR( x[i], x_serial[i], _tol * std::abs( x_serial[i] ) );
//             }
//         }

//         matrix_utils::OptimizedTriangularSolve<matrix_utils::FBSubstitutionType::NoBarrier,
//                                                matrix_utils::TriangularMatrix::L, int, int, double>
//             forwardsweep_nobarrier;
//         forwardsweep_nobarrier.analysis( L.rows, L.Base(), L.ai.get(),
//                                          L.aj.get(), L.av.get() );
//         for ( int i = 0; i < 100; i++ )
//         {
//             forwardsweep_nobarrier( b.data(), x.data() );
//             for ( int i = 0; i < x.size(); i++ )
//             {
//                 EXPECT_NEAR( x[i], x_serial[i], _tol * std::abs( x_serial[i] ) );
//             }
//         }

//         matrix_utils::OptimizedTriangularSolve<matrix_utils::FBSubstitutionType::NoBarrierSuperNode,
//                                                matrix_utils::TriangularMatrix::L, int, int, double>
//             forwardsweep_nobarrier_sn;
//         forwardsweep_nobarrier_sn.analysis( L.rows, L.Base(), L.ai.get(),
//                                             L.aj.get(), L.av.get() );
//         for ( int i = 0; i < 100; i++ )
//         {
//             forwardsweep_nobarrier_sn( b.data(), x.data() );
//             for ( int i = 0; i < x.size(); i++ )
//             {
//                 EXPECT_NEAR( x[i], x_serial[i], _tol * std::abs( x_serial[i] ) );
//             }
//         }
//     }
// }

// TEST_F( triangular_solve_Test, backward_substitution_optimized )
// {
//     omp_set_num_threads( 5 );
//     for ( auto mat : _mats )
//     {
//         const MKL_INT size = mat.rows();
//         mat.to_zero_based();
//         mkl_wrapper::incomplete_lu_k prec;
//         prec.set_level( 5 );
//         prec.symbolic_factorize( &mat );
//         prec.numeric_factorize( &mat );

//         // std::ofstream myfile;
//         // myfile.open("prec.svg");
//         // prec.print_svg(myfile);
//         // myfile.close();

//         std::vector<double> b( mat.rows() );
//         std::fill( std::begin( b ), std::end( b ), 1. );
//         std::vector<double> x( mat.rows(), 0.0 );
//         std::vector<double> x_serial( mat.rows(), 0.0 );

//         matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> L, U;
//         std::vector<double> D;

//         matrix_utils::SplitLDU( prec.rows(), (int)prec.mkl_base(), prec.get_ai().get(),
//                                 prec.get_aj().get(), prec.get_av().get(), L, D, U );

//         matrix_utils::BackwardSubstitution( U.rows, U.Base(), U.ai.get(),
//                                             U.aj.get(), U.av.get(), D.data(),
//                                             b.data(), x_serial.data() );

//         matrix_utils::OptimizedTriangularSolve<matrix_utils::FBSubstitutionType::Barrier,
//                                                matrix_utils::TriangularMatrix::U, int, int, double>
//             forwardsweep_barrier;
//         forwardsweep_barrier.analysis( U.rows, U.Base(), U.ai.get(), U.aj.get(),
//                                        U.av.get(), D.data() );
//         for ( int i = 0; i < 100; i++ )
//         {
//             forwardsweep_barrier( b.data(), x.data() );
//             for ( int i = 0; i < x.size(); i++ )
//             {
//                 EXPECT_NEAR( x[i], x_serial[i], _tol * std::abs( x_serial[i] ) );
//             }
//         }

//         matrix_utils::OptimizedTriangularSolve<matrix_utils::FBSubstitutionType::NoBarrier,
//                                                matrix_utils::TriangularMatrix::U, int, int, double>
//             forwardsweep_nobarrier;
//         forwardsweep_nobarrier.analysis( U.rows, U.Base(), U.ai.get(),
//                                          U.aj.get(), U.av.get(), D.data() );
//         for ( int i = 0; i < 100; i++ )
//         {
//             forwardsweep_nobarrier( b.data(), x.data() );
//             for ( int i = 0; i < x.size(); i++ )
//             {
//                 EXPECT_NEAR( x[i], x_serial[i], _tol * std::abs( x_serial[i] ) );
//             }
//         }

//         matrix_utils::OptimizedTriangularSolve<matrix_utils::FBSubstitutionType::NoBarrierSuperNode,
//                                                matrix_utils::TriangularMatrix::U, int, int, double>
//             forwardsweep_nobarrier_sn;
//         forwardsweep_nobarrier_sn.analysis( U.rows, U.Base(), U.ai.get(),
//                                             U.aj.get(), U.av.get(), D.data() );
//         for ( int i = 0; i < 100; i++ )
//         {
//             forwardsweep_nobarrier_sn( b.data(), x.data() );
//             for ( int i = 0; i < x.size(); i++ )
//             {
//                 EXPECT_NEAR( x[i], x_serial[i], _tol * std::abs( x_serial[i] ) );
//             }
//         }
//     }
// }