#include "cholesky_symbolic.hpp"
#include "matrix_utils.hpp"
#include "tree.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

using CSRMatrixType = matrix_utils::CSRMatrix<int, int, double>;

namespace
{
CSRMatrixType makeSymmetricMatrix( const int base )
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

    std::copy( ai.begin(), ai.end(), matrix.AI() );
    std::copy( aj.begin(), aj.end(), matrix.AJ() );
    std::fill( matrix.AV(), matrix.AV() + matrix.NNZ(), 1.0 );
    return matrix;
}

void expectSamePattern( const CSRMatrixType& expected, const CSRMatrixType& actual )
{
    ASSERT_EQ( expected.rows, actual.rows );
    ASSERT_EQ( expected.cols, actual.cols );
    ASSERT_EQ( expected.NNZ(), actual.NNZ() );

    const std::vector<int> expected_ai( expected.AI(), expected.AI() + expected.rows + 1 );
    const std::vector<int> actual_ai( actual.AI(), actual.AI() + actual.rows + 1 );
    EXPECT_EQ( expected_ai, actual_ai );

    const std::vector<int> expected_aj( expected.AJ(), expected.AJ() + expected.NNZ() );
    const std::vector<int> actual_aj( actual.AJ(), actual.AJ() + actual.NNZ() );
    EXPECT_EQ( expected_aj, actual_aj );
}
} // namespace

TEST( SymbolicCholeskyColVariants, MatchOnSmallSymmetricMatrix )
{
    for ( int base = 0; base <= 1; base++ )
    {
        const auto matrix = makeSymmetricMatrix( base );
        std::vector<int> parent( matrix.rows );
        std::vector<int> ancestor( matrix.rows );
        graph::eliminationTree( matrix.rows, matrix.AI(), matrix.AJ(), parent.data(), ancestor.data() );

        factorization::SymbolicCholeskyCol<CSRMatrixType> symbolic_v1( 2 );
        CSRMatrixType expected;
        ASSERT_TRUE( symbolic_v1.apply( matrix.rows, matrix.AI(), matrix.AJ(), parent.data(), expected ) );

        for ( const int nthreads : { 1, 2, 4 } )
        {
            factorization::SymbolicCholeskyColV2<CSRMatrixType> symbolic_v2( nthreads );
            CSRMatrixType actual_v2;
            ASSERT_TRUE( symbolic_v2.apply( matrix.rows, matrix.AI(), matrix.AJ(), parent.data(), actual_v2 ) );
            expectSamePattern( expected, actual_v2 );

            factorization::SymbolicCholeskyColV3<CSRMatrixType> symbolic_v3( nthreads );
            CSRMatrixType actual_v3;
            ASSERT_TRUE( symbolic_v3.apply( matrix.rows, matrix.AI(), matrix.AJ(), parent.data(), actual_v3 ) );
            expectSamePattern( expected, actual_v3 );
        }
    }
}
