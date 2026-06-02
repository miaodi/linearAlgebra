#include "cholesky_symbolic.hpp"
#include "io.hpp"
#include "lu_symbolic.hpp"
#include "tree.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

namespace
{
using CSRMatrixType = matrix_utils::CSRMatrixVec<int, int, double>;
using CholeskyCSRMatrixType = matrix_utils::CSRMatrix<int, int, double>;
using Edags = factorization::SymbolicLUEdags<CSRMatrixType>;
using GraphType = Edags::GraphType;

struct Pattern
{
    std::vector<int> ai;
    std::vector<int> aj;
};

std::vector<int> addBase( std::initializer_list<int> values, const int base )
{
    std::vector<int> out;
    out.reserve( values.size() );
    for ( const int value : values )
    {
        out.push_back( value + base );
    }
    return out;
}

Pattern makeMatrixPattern( const int base )
{
    return { addBase( { 0, 1, 2, 3, 4, 7 }, base ), addBase( { 0, 1, 2, 3, 0, 2, 4 }, base ) };
}

Pattern extractLowerPattern( const CSRMatrixType& lu )
{
    const int base = lu.Base();
    Pattern lower;
    lower.ai.resize( static_cast<std::size_t>( lu.rows ) + 1 );
    lower.ai[0] = base;

    for ( int row = 0; row < lu.rows; ++row )
    {
        const int row_label = row + base;
        const int row_begin = lu.AI()[row] - base;
        const int row_end = lu.AI()[row + 1] - base;
        auto out_begin = lower.aj.size();
        for ( int p = row_begin; p < row_end; ++p )
        {
            if ( lu.AJ()[p] <= row_label )
            {
                lower.aj.push_back( lu.AJ()[p] );
            }
        }
        std::sort( lower.aj.begin() + static_cast<std::ptrdiff_t>( out_begin ), lower.aj.end() );
        lower.ai[static_cast<std::size_t>( row ) + 1] = static_cast<int>( lower.aj.size() ) + base;
    }

    return lower;
}

void expectSamePatternRows( const int rows,
                            const int base,
                            const int* expected_ai,
                            const int* expected_aj,
                            const std::vector<int>& actual_ai,
                            const std::vector<int>& actual_aj )
{
    ASSERT_EQ( static_cast<std::size_t>( rows ) + 1, actual_ai.size() );
    ASSERT_EQ( base, actual_ai.front() );

    for ( int row = 0; row < rows; ++row )
    {
        std::vector<int> expected_row( expected_aj + expected_ai[row] - base,
                                       expected_aj + expected_ai[row + 1] - base );
        std::vector<int> actual_row( actual_aj.begin() + actual_ai[row] - base,
                                     actual_aj.begin() + actual_ai[row + 1] - base );
        std::sort( expected_row.begin(), expected_row.end() );
        std::sort( actual_row.begin(), actual_row.end() );
        EXPECT_EQ( expected_row, actual_row ) << "row " << row;
    }
}

CholeskyCSRMatrixType computeCholeskyPattern( const CholeskyCSRMatrixType& matrix, std::vector<int>& parent )
{
    std::vector<int> ancestor( matrix.rows );
    parent.resize( matrix.rows );
    graph::eliminationTree( matrix.rows, matrix.AI(), matrix.AJ(), parent.data(), ancestor.data() );

    CholeskyCSRMatrixType L;
    L.rows = matrix.rows;
    L.cols = matrix.cols;
    L.ResizeAI( static_cast<std::size_t>( matrix.rows ) + 1 );
    L.AI()[0] = matrix.Base();

    const int base = matrix.Base();
    std::vector<int> mark( matrix.rows, -1 );
    std::vector<int> aj;
    for ( int row = 0; row < matrix.rows; ++row )
    {
        const auto row_begin = aj.size();
        for ( int p = matrix.AI()[row] - base; p < matrix.AI()[row + 1] - base; ++p )
        {
            int node = matrix.AJ()[p] - base;
            if ( node >= row )
            {
                break;
            }

            while ( node < row && mark[node] != row )
            {
                mark[node] = row;
                aj.push_back( node + base );
                node = parent[node] - base;
            }
        }
        std::sort( aj.begin() + static_cast<std::ptrdiff_t>( row_begin ), aj.end() );
        aj.push_back( row + base );
        L.AI()[row + 1] = static_cast<int>( aj.size() ) + base;
    }

    L.ResizeAJ( aj.size() );
    L.ResizeAV( aj.size() );
    std::copy( aj.begin(), aj.end(), L.AJ() );
    std::fill( L.AV(), L.AV() + L.NNZ(), 1.0 );
    return L;
}

CholeskyCSRMatrixType makeFullSymmetricPattern( const CholeskyCSRMatrixType& matrix )
{
    const int base = matrix.Base();
    std::vector<std::vector<int>> rows( matrix.rows );
    for ( int row = 0; row < matrix.rows; ++row )
    {
        for ( int p = matrix.AI()[row] - base; p < matrix.AI()[row + 1] - base; ++p )
        {
            const int col = matrix.AJ()[p] - base;
            rows[row].push_back( col + base );
            rows[col].push_back( row + base );
        }
    }

    CholeskyCSRMatrixType full;
    full.rows = matrix.rows;
    full.cols = matrix.cols;
    full.ResizeAI( static_cast<std::size_t>( matrix.rows ) + 1 );
    full.AI()[0] = base;

    std::vector<int> aj;
    for ( int row = 0; row < matrix.rows; ++row )
    {
        auto& cols = rows[row];
        std::sort( cols.begin(), cols.end() );
        cols.erase( std::unique( cols.begin(), cols.end() ), cols.end() );
        aj.insert( aj.end(), cols.begin(), cols.end() );
        full.AI()[row + 1] = static_cast<int>( aj.size() ) + base;
    }

    full.ResizeAJ( aj.size() );
    full.ResizeAV( aj.size() );
    std::copy( aj.begin(), aj.end(), full.AJ() );
    std::fill( full.AV(), full.AV() + full.NNZ(), 1.0 );
    return full;
}

GraphType makeUpperEtreeGraph( const int rows, const int base, const std::vector<int>& parent )
{
    GraphType graph;
    graph.rows = rows;
    graph.cols = rows;
    graph.ai.assign( static_cast<std::size_t>( rows ) + 1, base );

    for ( int node = 0; node < rows; ++node )
    {
        if ( parent[node] != node + base )
        {
            graph.aj.push_back( parent[node] );
        }
        graph.ai[static_cast<std::size_t>( node ) + 1] = static_cast<int>( graph.aj.size() ) + base;
    }

    return graph;
}

GraphType makeLowerEtreeGraph( const int rows, const int base, const std::vector<int>& parent )
{
    std::vector<std::vector<int>> children( rows );
    for ( int node = 0; node < rows; ++node )
    {
        if ( parent[node] != node + base )
        {
            children[parent[node] - base].push_back( node + base );
        }
    }

    GraphType graph;
    graph.rows = rows;
    graph.cols = rows;
    graph.ai.assign( static_cast<std::size_t>( rows ) + 1, base );
    for ( int node = 0; node < rows; ++node )
    {
        auto& row = children[node];
        std::sort( row.begin(), row.end() );
        graph.aj.insert( graph.aj.end(), row.begin(), row.end() );
        graph.ai[static_cast<std::size_t>( node ) + 1] = static_cast<int>( graph.aj.size() ) + base;
    }

    return graph;
}

void expectSameGraphRows( const GraphType& expected, const GraphType& actual )
{
    ASSERT_EQ( expected.rows, actual.rows );
    ASSERT_EQ( expected.cols, actual.cols );
    ASSERT_EQ( expected.Base(), actual.Base() );

    const int base = expected.Base();
    for ( int row = 0; row < expected.rows; ++row )
    {
        std::vector<int> expected_row( expected.AJ() + expected.AI()[row] - base,
                                       expected.AJ() + expected.AI()[row + 1] - base );
        std::vector<int> actual_row( actual.AJ() + actual.AI()[row] - base,
                                     actual.AJ() + actual.AI()[row + 1] - base );
        std::sort( expected_row.begin(), expected_row.end() );
        std::sort( actual_row.begin(), actual_row.end() );
        EXPECT_EQ( expected_row, actual_row ) << "row " << row;
    }
}

} // namespace

TEST( SymbolicLUEdags, ApplyBuildsCombinedPatternThroughPublicApi )
{
    for ( const int base : { 0, 1 } )
    {
        const auto matrix = makeMatrixPattern( base );
        Edags edags;
        CSRMatrixType lu;

        ASSERT_TRUE( edags.apply( 5, matrix.ai.data(), matrix.aj.data(), lu ) );
        EXPECT_EQ( 5, lu.rows );
        EXPECT_EQ( 5, lu.cols );
        EXPECT_EQ( static_cast<std::size_t>( lu.NNZ() ), lu.aj.size() );
        EXPECT_EQ( lu.aj.size(), lu.av.size() );
        EXPECT_EQ( base, lu.AI()[0] );
        EXPECT_EQ( base + static_cast<int>( lu.aj.size() ), lu.AI()[5] );
    }
}

TEST( SymbolicLUEdags, SpdMatricesMatchCholeskyPatternAndEtree )
{
    const std::vector<std::string> spd_matrices = { "spd/ex5.mtx", "spd/nos5.mtx",
                                                    "spd/s3rmt3m3.mtx", "spd/bcsstk17.mtx" };

    for ( const auto& matrix_file : spd_matrices )
    {
        SCOPED_TRACE( matrix_file );

        std::ifstream f( "data/" + matrix_file );
        ASSERT_TRUE( f.good() ) << "tests/CMakeLists.txt should provide data/" << matrix_file
                                << " in the test build directory";

        CholeskyCSRMatrixType matrix;
        matrix_utils::readMatrixMarket( f, matrix );
        ASSERT_GT( matrix.rows, 0 );
        ASSERT_EQ( matrix.rows, matrix.cols );

        const auto full = makeFullSymmetricPattern( matrix );
        std::vector<int> parent;
        const auto cholesky_l = computeCholeskyPattern( full, parent );

        Edags edags;
        CSRMatrixType lu;
        ASSERT_TRUE( edags.apply( full.rows, full.AI(), full.AJ(), lu ) );

        const auto lu_l = extractLowerPattern( lu );
        expectSamePatternRows( full.rows, full.Base(), cholesky_l.AI(), cholesky_l.AJ(), lu_l.ai, lu_l.aj );

        const auto expected_upper_edag = makeUpperEtreeGraph( full.rows, full.Base(), parent );
        const auto expected_lower_edag = makeLowerEtreeGraph( full.rows, full.Base(), parent );
        expectSameGraphRows( expected_upper_edag, edags.upperEdag() );
        expectSameGraphRows( expected_lower_edag, edags.lowerEdag() );
    }
}
