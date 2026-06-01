#include "cholesky_multifrontal.hpp"
#include "cholesky_symbolic.hpp"
#include "io.hpp"
#include "matrix_utils.hpp"
#include "tree.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

using CSRMatrixType = matrix_utils::CSRMatrix<int, int, double>;

namespace
{

CSRMatrixType makeFullSymmetricMatrix( const int base )
{
    CSRMatrixType matrix;
    matrix.rows = 5;
    matrix.cols = 5;
    matrix.ResizeAI( 6 );
    matrix.ResizeAJ( 17 );
    matrix.ResizeAV( 17 );

    const std::vector<int> ai{ base, base + 3, base + 6, base + 10, base + 14, base + 17 };
    const std::vector<int> aj{ base,     base + 1, base + 2, base,     base + 1, base + 3,
                               base,     base + 2, base + 3, base + 4, base + 1, base + 2,
                               base + 3, base + 4, base + 2, base + 3, base + 4 };
    const std::vector<double> av{ 6.0, 0.2, 0.1, 0.2, 7.0, 0.3, 0.1, 8.0, 0.4,
                                  0.2, 0.3, 0.4, 9.0, 0.5, 0.2, 0.5, 10.0 };

    std::copy( ai.begin(), ai.end(), matrix.AI() );
    std::copy( aj.begin(), aj.end(), matrix.AJ() );
    std::copy( av.begin(), av.end(), matrix.AV() );
    return matrix;
}

CSRMatrixType makeDenseSpdMatrix( const int base )
{
    CSRMatrixType matrix;
    matrix.rows = 4;
    matrix.cols = 4;
    matrix.ResizeAI( 5 );
    matrix.ResizeAJ( 16 );
    matrix.ResizeAV( 16 );

    const std::vector<int> ai{ base, base + 4, base + 8, base + 12, base + 16 };
    const std::vector<int> aj{ base,     base + 1, base + 2, base + 3, base,     base + 1,
                               base + 2, base + 3, base,     base + 1, base + 2, base + 3,
                               base,     base + 1, base + 2, base + 3 };
    const std::vector<double> av{ 10.0, 1.0, 2.0, 0.5, 1.0, 9.0,  1.5, 0.25,
                                  2.0,  1.5, 8.0, 1.0, 0.5, 0.25, 1.0, 7.0 };

    std::copy( ai.begin(), ai.end(), matrix.AI() );
    std::copy( aj.begin(), aj.end(), matrix.AJ() );
    std::copy( av.begin(), av.end(), matrix.AV() );
    return matrix;
}

CSRMatrixType makeUpperMatrixFromFull( const CSRMatrixType& full )
{
    const int base = full.Base();
    CSRMatrixType upper;
    upper.rows = full.rows;
    upper.cols = full.cols;

    std::vector<int> ai( static_cast<std::size_t>( full.rows ) + 1, base );
    std::vector<int> aj;
    std::vector<double> av;
    for ( int row = 0; row < full.rows; row++ )
    {
        for ( int pos = full.AI()[row] - base; pos < full.AI()[row + 1] - base; pos++ )
        {
            if ( full.AJ()[pos] >= row + base )
            {
                aj.push_back( full.AJ()[pos] );
                av.push_back( full.AV()[pos] );
            }
        }
        ai[static_cast<std::size_t>( row ) + 1] = static_cast<int>( aj.size() ) + base;
    }

    upper.ResizeAI( ai.size() );
    upper.ResizeAJ( aj.size() );
    upper.ResizeAV( av.size() );
    std::copy( ai.begin(), ai.end(), upper.AI() );
    std::copy( aj.begin(), aj.end(), upper.AJ() );
    std::copy( av.begin(), av.end(), upper.AV() );
    return upper;
}

Eigen::MatrixXd toDenseSymmetric( const CSRMatrixType& matrix )
{
    const int base = matrix.Base();
    Eigen::MatrixXd dense = Eigen::MatrixXd::Zero( matrix.rows, matrix.cols );
    for ( int row = 0; row < matrix.rows; row++ )
    {
        for ( int pos = matrix.AI()[row] - base; pos < matrix.AI()[row + 1] - base; pos++ )
        {
            const int col = matrix.AJ()[pos] - base;
            dense( row, col ) = matrix.AV()[pos];
            dense( col, row ) = matrix.AV()[pos];
        }
    }
    return dense;
}

Eigen::MatrixXd toDenseLowerFactor( const CSRMatrixType& L )
{
    const int base = L.Base();
    Eigen::MatrixXd dense = Eigen::MatrixXd::Zero( L.rows, L.cols );
    for ( int col = 0; col < L.cols; col++ )
    {
        for ( int pos = L.AI()[col] - base; pos < L.AI()[col + 1] - base; pos++ )
        {
            dense( L.AJ()[pos] - base, col ) = L.AV()[pos];
        }
    }
    return dense;
}

void expectFactorReconstructsMatrix( const CSRMatrixType& A, const CSRMatrixType& L )
{
    const Eigen::MatrixXd dense_a = toDenseSymmetric( A );
    const Eigen::MatrixXd dense_l = toDenseLowerFactor( L );
    const Eigen::MatrixXd reconstructed = dense_l * dense_l.transpose();
    ASSERT_EQ( reconstructed.rows(), dense_a.rows() );
    ASSERT_EQ( reconstructed.cols(), dense_a.cols() );
    for ( int row = 0; row < dense_a.rows(); row++ )
    {
        for ( int col = 0; col < dense_a.cols(); col++ )
        {
            EXPECT_NEAR( reconstructed( row, col ), dense_a( row, col ), 1e-12 )
                << "entry (" << row << ", " << col << ")";
        }
    }
}

Eigen::SparseMatrix<double, Eigen::RowMajor, int> toEigenRowMajor( const CSRMatrixType& A )
{
    Eigen::SparseMatrix<double, Eigen::RowMajor, int> sparse( A.rows, A.cols );
    sparse.resizeNonZeros( A.NNZ() );
    std::copy( A.AI(), A.AI() + A.rows + 1, sparse.outerIndexPtr() );
    std::copy( A.AJ(), A.AJ() + A.NNZ(), sparse.innerIndexPtr() );
    std::copy( A.AV(), A.AV() + A.NNZ(), sparse.valuePtr() );
    sparse.makeCompressed();
    return sparse;
}

CSRMatrixType copySymbolicPattern( const CSRMatrixType& pattern )
{
    CSRMatrixType copy;
    copy.rows = pattern.rows;
    copy.cols = pattern.cols;
    copy.ResizeAI( static_cast<std::size_t>( pattern.rows ) + 1 );
    copy.ResizeAJ( pattern.NNZ() );
    std::copy( pattern.AI(), pattern.AI() + pattern.rows + 1, copy.AI() );
    std::copy( pattern.AJ(), pattern.AJ() + pattern.NNZ(), copy.AJ() );
    return copy;
}

void compareWithEigenLower( const CSRMatrixType& L,
                            const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& expected,
                            const std::string& label )
{
    ASSERT_EQ( L.rows, expected.rows() );
    ASSERT_EQ( L.cols, expected.cols() );
    ASSERT_EQ( L.Base(), 0 );
    ASSERT_EQ( L.NNZ(), expected.nonZeros() );

    for ( int i = 0; i <= L.cols; i++ )
    {
        ASSERT_EQ( L.AI()[i], expected.outerIndexPtr()[i] ) << "outer " << i;
    }
    for ( int pos = 0; pos < L.NNZ(); pos++ )
    {
        ASSERT_EQ( L.AJ()[pos], expected.innerIndexPtr()[pos] ) << "pos " << pos;
    }

    double max_abs_diff = 0.0;
    int max_pos = 0;
    for ( int pos = 0; pos < L.NNZ(); pos++ )
    {
        const double diff = std::abs( L.AV()[pos] - expected.valuePtr()[pos] );
        if ( diff > max_abs_diff )
        {
            max_abs_diff = diff;
            max_pos = pos;
        }
    }
    EXPECT_LE( max_abs_diff, 1e-8 ) << label << " max pos " << max_pos;
}

void factorAndCompareWithEigen( const CSRMatrixType& A )
{
    ASSERT_EQ( A.rows, A.cols );
    ASSERT_EQ( A.Base(), 0 );

    const auto sparse = toEigenRowMajor( A );
    Eigen::SimplicialLLT<decltype( sparse ), Eigen::Lower, Eigen::NaturalOrdering<int>> llt;
    llt.compute( sparse );
    ASSERT_EQ( llt.info(), Eigen::Success );
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> expected_l = llt.matrixL();
    expected_l.makeCompressed();

    std::vector<int> parent( A.rows );
    std::vector<int> ancestor( A.rows );
    graph::eliminationTree( A.rows, A.AI(), A.AJ(), parent.data(), ancestor.data() );

    factorization::SymbolicCholeskyColV3<CSRMatrixType> symbolic( 1 );
    CSRMatrixType symbolic_l;
    ASSERT_TRUE( symbolic.apply( A.rows, A.AI(), A.AJ(), parent.data(), symbolic_l ) );

    std::vector<int> diagpos( A.rows );
    ASSERT_TRUE( matrix_utils::Diagonal( A.rows, A.AI(), A.AJ(), A.AV(), diagpos.data(),
                                         static_cast<double*>( nullptr ) ) );

    CSRMatrixType scalar_l = copySymbolicPattern( symbolic_l );
    factorization::MultifrontalCholesky<CSRMatrixType> scalar_numeric;
    ASSERT_TRUE( scalar_numeric.apply( A.rows, diagpos.data(), A.AI() + 1, A.AJ(), A.AV(),
                                       symbolic.eliminationTree(), scalar_l ) );
    compareWithEigenLower( scalar_l, expected_l, "scalar multifrontal" );

    CSRMatrixType supernodal_l = copySymbolicPattern( symbolic_l );
    factorization::MultifrontalCholeskySuperNodal<CSRMatrixType> supernodal_numeric;
    ASSERT_TRUE( supernodal_numeric.apply( A.rows, diagpos.data(), A.AI() + 1, A.AJ(), A.AV(),
                                           symbolic.eliminationTree(), supernodal_l ) );
    compareWithEigenLower( supernodal_l, expected_l, "supernodal multifrontal" );
}

} // namespace

TEST( MultifrontalCholesky, ReusesSymbolicV3ForFullCsr )
{
    const auto matrix = makeFullSymmetricMatrix( 0 );
    std::vector<int> parent( matrix.rows );
    std::vector<int> ancestor( matrix.rows );
    graph::eliminationTree( matrix.rows, matrix.AI(), matrix.AJ(), parent.data(), ancestor.data() );

    factorization::SymbolicCholeskyColV3<CSRMatrixType> symbolic( 1 );
    CSRMatrixType L;
    ASSERT_TRUE( symbolic.apply( matrix.rows, matrix.AI(), matrix.AJ(), parent.data(), L ) );

    std::vector<int> diagpos( matrix.rows );
    ASSERT_TRUE( matrix_utils::Diagonal( matrix.rows, matrix.AI(), matrix.AJ(), matrix.AV(),
                                         diagpos.data(), static_cast<double*>( nullptr ) ) );

    factorization::MultifrontalCholesky<CSRMatrixType> numeric;
    ASSERT_TRUE( numeric.apply( matrix.rows, diagpos.data(), matrix.AI() + 1, matrix.AJ(),
                                matrix.AV(), symbolic.eliminationTree(), L ) );

    expectFactorReconstructsMatrix( matrix, L );
}

TEST( MultifrontalCholesky, ReusesSymbolicV3ForUpperCsr )
{
    const auto full = makeFullSymmetricMatrix( 0 );
    const auto upper = makeUpperMatrixFromFull( full );
    std::vector<int> parent( full.rows );
    std::vector<int> ancestor( full.rows );
    graph::eliminationTree( full.rows, full.AI(), full.AJ(), parent.data(), ancestor.data() );

    factorization::SymbolicCholeskyColV3<CSRMatrixType> symbolic( 1 );
    CSRMatrixType L;
    ASSERT_TRUE( symbolic.apply( upper.rows, upper.AI(), upper.AJ(), parent.data(), L ) );

    factorization::MultifrontalCholesky<CSRMatrixType> numeric;
    ASSERT_TRUE( numeric.apply( upper.rows, upper.AI(), upper.AI() + 1, upper.AJ(), upper.AV(),
                                symbolic.eliminationTree(), L ) );

    expectFactorReconstructsMatrix( upper, L );
}

TEST( MultifrontalCholesky, AcceptsExplicitBeginPointers )
{
    const auto matrix = makeFullSymmetricMatrix( 1 );
    std::vector<int> parent( matrix.rows );
    std::vector<int> ancestor( matrix.rows );
    graph::eliminationTree( matrix.rows, matrix.AI(), matrix.AJ(), parent.data(), ancestor.data() );

    factorization::SymbolicCholeskyColV3<CSRMatrixType> symbolic( 1 );
    CSRMatrixType L;
    ASSERT_TRUE( symbolic.apply( matrix.rows, matrix.AI(), matrix.AJ(), parent.data(), L ) );

    std::vector<int> diagpos( matrix.rows );
    ASSERT_TRUE( matrix_utils::Diagonal( matrix.rows, matrix.AI(), matrix.AJ(), matrix.AV(),
                                         diagpos.data(), static_cast<double*>( nullptr ) ) );

    factorization::MultifrontalCholesky<CSRMatrixType> numeric;
    ASSERT_TRUE( numeric.apply( matrix.rows, diagpos.data(), matrix.AI() + 1, matrix.AJ(),
                                matrix.AV(), symbolic.eliminationTree(), L ) );

    expectFactorReconstructsMatrix( matrix, L );
}

TEST( MultifrontalCholeskySuperNodal, DetectsTheorem413DenseSupernode )
{
    const auto matrix = makeDenseSpdMatrix( 0 );
    std::vector<int> parent( matrix.rows );
    std::vector<int> ancestor( matrix.rows );
    graph::eliminationTree( matrix.rows, matrix.AI(), matrix.AJ(), parent.data(), ancestor.data() );

    factorization::SymbolicCholeskyColV3<CSRMatrixType> symbolic( 1 );
    CSRMatrixType L;
    ASSERT_TRUE( symbolic.apply( matrix.rows, matrix.AI(), matrix.AJ(), parent.data(), L ) );

    factorization::MultifrontalCholeskySuperNodal<CSRMatrixType> numeric;
    ASSERT_TRUE( numeric.analyzeSupernodes( matrix.rows, symbolic.eliminationTree(), L ) );

    EXPECT_EQ( numeric.supernodePrefix(), std::vector<int>( { 0, 4 } ) );
    EXPECT_EQ( numeric.columnToSupernode(), std::vector<int>( { 0, 0, 0, 0 } ) );
    EXPECT_EQ( numeric.assemblyTree().nnodes(), 1 );
    EXPECT_EQ( numeric.assemblyTree().parent()[0], 0 );
}

TEST( MultifrontalCholeskySuperNodal, FactorsDenseSupernode )
{
    const auto matrix = makeDenseSpdMatrix( 0 );
    std::vector<int> parent( matrix.rows );
    std::vector<int> ancestor( matrix.rows );
    graph::eliminationTree( matrix.rows, matrix.AI(), matrix.AJ(), parent.data(), ancestor.data() );

    factorization::SymbolicCholeskyColV3<CSRMatrixType> symbolic( 1 );
    CSRMatrixType L;
    ASSERT_TRUE( symbolic.apply( matrix.rows, matrix.AI(), matrix.AJ(), parent.data(), L ) );

    std::vector<int> diagpos( matrix.rows );
    ASSERT_TRUE( matrix_utils::Diagonal( matrix.rows, matrix.AI(), matrix.AJ(), matrix.AV(),
                                         diagpos.data(), static_cast<double*>( nullptr ) ) );

    factorization::MultifrontalCholeskySuperNodal<CSRMatrixType> numeric;
    ASSERT_TRUE( numeric.apply( matrix.rows, diagpos.data(), matrix.AI() + 1, matrix.AJ(),
                                matrix.AV(), symbolic.eliminationTree(), L ) );

    expectFactorReconstructsMatrix( matrix, L );
}

TEST( MultifrontalCholeskySuperNodal, ReusesSymbolicV3ForFullCsr )
{
    const auto matrix = makeFullSymmetricMatrix( 0 );
    std::vector<int> parent( matrix.rows );
    std::vector<int> ancestor( matrix.rows );
    graph::eliminationTree( matrix.rows, matrix.AI(), matrix.AJ(), parent.data(), ancestor.data() );

    factorization::SymbolicCholeskyColV3<CSRMatrixType> symbolic( 1 );
    CSRMatrixType L;
    ASSERT_TRUE( symbolic.apply( matrix.rows, matrix.AI(), matrix.AJ(), parent.data(), L ) );

    std::vector<int> diagpos( matrix.rows );
    ASSERT_TRUE( matrix_utils::Diagonal( matrix.rows, matrix.AI(), matrix.AJ(), matrix.AV(),
                                         diagpos.data(), static_cast<double*>( nullptr ) ) );

    factorization::MultifrontalCholeskySuperNodal<CSRMatrixType> numeric;
    ASSERT_TRUE( numeric.apply( matrix.rows, diagpos.data(), matrix.AI() + 1, matrix.AJ(),
                                matrix.AV(), symbolic.eliminationTree(), L ) );

    expectFactorReconstructsMatrix( matrix, L );
}

TEST( MultifrontalCholeskySuperNodal, ReusesSymbolicV3ForUpperCsr )
{
    const auto full = makeFullSymmetricMatrix( 0 );
    const auto upper = makeUpperMatrixFromFull( full );
    std::vector<int> parent( full.rows );
    std::vector<int> ancestor( full.rows );
    graph::eliminationTree( full.rows, full.AI(), full.AJ(), parent.data(), ancestor.data() );

    factorization::SymbolicCholeskyColV3<CSRMatrixType> symbolic( 1 );
    CSRMatrixType L;
    ASSERT_TRUE( symbolic.apply( upper.rows, upper.AI(), upper.AJ(), parent.data(), L ) );

    factorization::MultifrontalCholeskySuperNodal<CSRMatrixType> numeric;
    ASSERT_TRUE( numeric.apply( upper.rows, upper.AI(), upper.AI() + 1, upper.AJ(), upper.AV(),
                                symbolic.eliminationTree(), L ) );

    expectFactorReconstructsMatrix( upper, L );
}

TEST( MultifrontalCholeskySuperNodal, AcceptsExplicitBeginPointersBaseOne )
{
    const auto matrix = makeFullSymmetricMatrix( 1 );
    std::vector<int> parent( matrix.rows );
    std::vector<int> ancestor( matrix.rows );
    graph::eliminationTree( matrix.rows, matrix.AI(), matrix.AJ(), parent.data(), ancestor.data() );

    factorization::SymbolicCholeskyColV3<CSRMatrixType> symbolic( 1 );
    CSRMatrixType L;
    ASSERT_TRUE( symbolic.apply( matrix.rows, matrix.AI(), matrix.AJ(), parent.data(), L ) );

    std::vector<int> diagpos( matrix.rows );
    ASSERT_TRUE( matrix_utils::Diagonal( matrix.rows, matrix.AI(), matrix.AJ(), matrix.AV(),
                                         diagpos.data(), static_cast<double*>( nullptr ) ) );

    factorization::MultifrontalCholeskySuperNodal<CSRMatrixType> numeric;
    ASSERT_TRUE( numeric.apply( matrix.rows, diagpos.data(), matrix.AI() + 1, matrix.AJ(),
                                matrix.AV(), symbolic.eliminationTree(), L ) );

    expectFactorReconstructsMatrix( matrix, L );
}

TEST( MultifrontalCholesky, MatchesEigenSimplicialLLTOnCMakeSpdMatrices )
{
    const std::vector<std::string> spd_matrices = {
        "spd/bcsstk17.mtx",
        "spd/s3rmt3m3.mtx",
        "spd/ex5.mtx",
        "spd/nos5.mtx",
    };

    for ( const auto& matrix_file : spd_matrices )
    {
        SCOPED_TRACE( matrix_file );

        std::ifstream f( "data/" + matrix_file );
        ASSERT_TRUE( f.good() ) << "tests/CMakeLists.txt should provide data/" << matrix_file
                                << " in the test build directory";

        CSRMatrixType A;
        matrix_utils::readMatrixMarket( f, A );
        ASSERT_GT( A.rows, 0 );

        factorAndCompareWithEigen( A );
    }
}
