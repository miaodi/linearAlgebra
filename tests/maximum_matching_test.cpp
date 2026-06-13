#include "MaximumMatching.hpp"
#include "io.hpp"
#include "matrix_utils.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

namespace
{
void ExpectValidMatching( const int rows,
                          const std::vector<int>& ai,
                          const std::vector<int>& aj,
                          const std::vector<int>& matching_row,
                          const std::vector<int>& matching_col,
                          const int expected_count )
{
    const int base = ai[0];
    const int invalid = std::numeric_limits<int>::max();
    int row_count = 0;
    int col_count = 0;

    for ( int row = 0; row < rows; ++row )
    {
        if ( matching_row[row] == invalid )
        {
            continue;
        }

        ++row_count;
        const int col = matching_row[row] - base;
        ASSERT_GE( col, 0 );
        ASSERT_LT( col, rows );
        EXPECT_EQ( matching_col[col], row + base );

        const auto row_begin = aj.begin() + ( ai[row] - base );
        const auto row_end = aj.begin() + ( ai[row + 1] - base );
        EXPECT_NE( std::find( row_begin, row_end, col + base ), row_end );
    }

    for ( int col = 0; col < rows; ++col )
    {
        if ( matching_col[col] == invalid )
        {
            continue;
        }

        ++col_count;
        const int row = matching_col[col] - base;
        ASSERT_GE( row, 0 );
        ASSERT_LT( row, rows );
        EXPECT_EQ( matching_row[row], col + base );
    }

    EXPECT_EQ( row_count, expected_count );
    EXPECT_EQ( col_count, expected_count );
}

int CountDiagonalEntries( const matrix_utils::CSRMatrix<int, int, double>& matrix )
{
    int diagonal_count = 0;
    const int base = matrix.Base();
    for ( int row = 0; row < matrix.rows; ++row )
    {
        for ( int offset = matrix.AI()[row] - base; offset < matrix.AI()[row + 1] - base; ++offset )
        {
            if ( matrix.AJ()[offset] - base == row )
            {
                ++diagonal_count;
                break;
            }
        }
    }
    return diagonal_count;
}

void ExpectFullPermutationMatching( const matrix_utils::CSRMatrix<int, int, double>& matrix,
                                    const std::vector<int>& matching_row,
                                    const std::vector<int>& matching_col,
                                    const int count )
{
    const int rows = matrix.rows;
    const int base = matrix.Base();
    const int invalid = std::numeric_limits<int>::max();
    std::vector<char> seen_matching_row( rows, false );
    std::vector<char> seen_matching_col( rows, false );

    ASSERT_EQ( matrix.rows, matrix.cols );
    ASSERT_EQ( count, rows );

    for ( int row = 0; row < rows; ++row )
    {
        ASSERT_NE( matching_row[row], invalid );
        const int col = matching_row[row] - base;
        ASSERT_GE( col, 0 );
        ASSERT_LT( col, rows );
        ASSERT_FALSE( seen_matching_row[col] );
        seen_matching_row[col] = true;
        EXPECT_EQ( matching_col[col], row + base );

        bool found_edge = false;
        for ( int offset = matrix.AI()[row] - base; offset < matrix.AI()[row + 1] - base; ++offset )
        {
            if ( matrix.AJ()[offset] - base == col )
            {
                found_edge = true;
                break;
            }
        }
        EXPECT_TRUE( found_edge );
    }

    for ( int col = 0; col < rows; ++col )
    {
        ASSERT_NE( matching_col[col], invalid );
        const int row = matching_col[col] - base;
        ASSERT_GE( row, 0 );
        ASSERT_LT( row, rows );
        ASSERT_FALSE( seen_matching_col[row] );
        seen_matching_col[row] = true;
        EXPECT_EQ( matching_row[row], col + base );
    }
}
} // namespace

TEST( MaximumMatching, perfect_diagonal_base0 )
{
    const int rows = 3;
    const std::vector<int> ai = { 0, 1, 2, 3 };
    const std::vector<int> aj = { 0, 1, 2 };
    std::vector<int> matching_row( rows );
    std::vector<int> matching_col( rows );

    const int count = reordering::MaximumMatching( rows, ai.data(), aj.data(), matching_row.data(),
                                                   matching_col.data() );

    EXPECT_EQ( count, rows );
    EXPECT_EQ( matching_row, ( std::vector<int>{ 0, 1, 2 } ) );
    EXPECT_EQ( matching_col, ( std::vector<int>{ 0, 1, 2 } ) );
    ExpectValidMatching( rows, ai, aj, matching_row, matching_col, rows );
}

TEST( MaximumMatching, perfect_off_diagonal_base0 )
{
    const int rows = 3;
    const std::vector<int> ai = { 0, 1, 2, 3 };
    const std::vector<int> aj = { 1, 2, 0 };
    std::vector<int> matching_row( rows );
    std::vector<int> matching_col( rows );

    const int count = reordering::MaximumMatching( rows, ai.data(), aj.data(), matching_row.data(),
                                                   matching_col.data() );

    EXPECT_EQ( count, rows );
    EXPECT_EQ( matching_row, ( std::vector<int>{ 1, 2, 0 } ) );
    EXPECT_EQ( matching_col, ( std::vector<int>{ 2, 0, 1 } ) );
    ExpectValidMatching( rows, ai, aj, matching_row, matching_col, rows );
}

TEST( MaximumMatching, augmenting_path_repairs_existing_match )
{
    const int rows = 2;
    const std::vector<int> ai = { 0, 2, 3 };
    const std::vector<int> aj = { 0, 1, 0 };
    std::vector<int> matching_row( rows );
    std::vector<int> matching_col( rows );

    const int count = reordering::MaximumMatching( rows, ai.data(), aj.data(), matching_row.data(),
                                                   matching_col.data() );

    EXPECT_EQ( count, rows );
    EXPECT_EQ( matching_row, ( std::vector<int>{ 1, 0 } ) );
    EXPECT_EQ( matching_col, ( std::vector<int>{ 1, 0 } ) );
    ExpectValidMatching( rows, ai, aj, matching_row, matching_col, rows );
}

TEST( MaximumMatching, no_perfect_matching_leaves_unmatched_entries_invalid )
{
    const int rows = 2;
    const int invalid = std::numeric_limits<int>::max();
    const std::vector<int> ai = { 0, 1, 2 };
    const std::vector<int> aj = { 0, 0 };
    std::vector<int> matching_row( rows );
    std::vector<int> matching_col( rows );

    const int count = reordering::MaximumMatching( rows, ai.data(), aj.data(), matching_row.data(),
                                                   matching_col.data() );

    EXPECT_EQ( count, 1 );
    EXPECT_EQ( matching_row[1], invalid );
    EXPECT_EQ( matching_col[1], invalid );
    ExpectValidMatching( rows, ai, aj, matching_row, matching_col, count );
}

TEST( MaximumMatching, shared_single_column_remains_consistent_on_failure )
{
    const int rows = 2;
    const int invalid = std::numeric_limits<int>::max();
    const std::vector<int> ai = { 0, 1, 2 };
    const std::vector<int> aj = { 1, 1 };
    std::vector<int> matching_row( rows );
    std::vector<int> matching_col( rows );

    const int count = reordering::MaximumMatching( rows, ai.data(), aj.data(), matching_row.data(),
                                                   matching_col.data() );

    EXPECT_EQ( count, 1 );
    EXPECT_EQ( matching_col[0], invalid );
    EXPECT_NE( matching_col[1], invalid );
    EXPECT_TRUE( matching_row[0] == invalid || matching_row[1] == invalid );
    ExpectValidMatching( rows, ai, aj, matching_row, matching_col, count );
}

TEST( MaximumMatching, perfect_matching_base1 )
{
    const int rows = 3;
    const std::vector<int> ai = { 1, 2, 3, 4 };
    const std::vector<int> aj = { 2, 3, 1 };
    std::vector<int> matching_row( rows );
    std::vector<int> matching_col( rows );

    const int count = reordering::MaximumMatching( rows, ai.data(), aj.data(), matching_row.data(),
                                                   matching_col.data() );

    EXPECT_EQ( count, rows );
    EXPECT_EQ( matching_row, ( std::vector<int>{ 2, 3, 1 } ) );
    EXPECT_EQ( matching_col, ( std::vector<int>{ 3, 1, 2 } ) );
    ExpectValidMatching( rows, ai, aj, matching_row, matching_col, rows );
}

TEST( MaximumMatching, perfect_matching_int64 )
{
    const std::int64_t rows = 3;
    const std::vector<std::int64_t> ai = { 0, 1, 2, 3 };
    const std::vector<std::int64_t> aj = { 1, 2, 0 };
    std::vector<std::int64_t> matching_row( rows );
    std::vector<std::int64_t> matching_col( rows );

    const std::int64_t count = reordering::MaximumMatching(
        rows, ai.data(), aj.data(), matching_row.data(), matching_col.data() );

    EXPECT_EQ( count, rows );
    EXPECT_EQ( matching_row, ( std::vector<std::int64_t>{ 1, 2, 0 } ) );
    EXPECT_EQ( matching_col, ( std::vector<std::int64_t>{ 2, 0, 1 } ) );
}

TEST( MaximumMatching, suitesparse_no_full_diagonal_full_structural_rank )
{
    const std::vector<std::string> matrix_files = {
        "d_ss.mtx",   "d_dyn.mtx",    "spaceStation_1.mtx", "robot.mtx",
        "rotor2.mtx", "west0067.mtx", "west0132.mtx" };

    for ( const std::string& matrix_file : matrix_files )
    {
        SCOPED_TRACE( matrix_file );
        const std::filesystem::path path = std::filesystem::path( "data/no_full_diag" ) / matrix_file;
        std::ifstream input( path );
        ASSERT_TRUE( input ) << "failed to open " << path;

        matrix_utils::CSRMatrix<int, int, double> matrix;
        matrix_utils::readMatrixMarket( input, matrix );

        ASSERT_EQ( matrix.rows, matrix.cols ) << matrix_file;
        ASSERT_LT( CountDiagonalEntries( matrix ), matrix.rows ) << matrix_file;

        std::vector<int> matching_row( matrix.rows );
        std::vector<int> matching_col( matrix.rows );
        const int count = reordering::MaximumMatching( matrix.rows, matrix.AI(), matrix.AJ(),
                                                       matching_row.data(), matching_col.data() );

        ExpectFullPermutationMatching( matrix, matching_row, matching_col, count );
    }
}
