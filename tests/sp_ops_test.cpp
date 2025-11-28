#include "matrix_utils.hpp"
#include "sp_ops.hpp"
#include "spadd.hpp"
#include "graph_algs.hpp"
#include "utils.h"
#include "mkl_sparse_mat.h"
#include <algorithm>
#include <array>
#include <fstream>
#include <gtest/gtest.h>
#include <set>
#include <omp.h>

using namespace matrix_utils;

class SymmetricOpsTest : public testing::Test
{
protected:
    const double _tol = 1e-10;

    // Helper function to verify CSR structure validity
    template <typename ROWTYPE, typename COLTYPE>
    void verifyCsrStructure( const COLTYPE size,
                             const ROWTYPE* ai,
                             const COLTYPE* aj,
                             const ROWTYPE base = 0 )
    {
        ASSERT_EQ( ai[0], base ) << "Row pointer should start with base";

        for ( COLTYPE i = 0; i < size; i++ )
        {
            ASSERT_LE( ai[i], ai[i + 1] )
                << "Row pointers must be non-decreasing";

            // Check column indices are sorted
            for ( ROWTYPE j = ai[i] - base; j < ai[i + 1] - base - 1; j++ )
            {
                ASSERT_LT( aj[j], aj[j + 1] )
                    << "Column indices must be sorted and unique";
            }
        }
    }

    // Helper function to compute A+A^T naively for verification
    template <typename ROWTYPE, typename COLTYPE, bool KEEPDIAG>
    CSRStructVec<ROWTYPE, COLTYPE> computeAPlusATNaive( const COLTYPE size,
                                                        const ROWTYPE* ai,
                                                        const COLTYPE* aj )
    {
        const ROWTYPE base = ai[0];

        // Use set to automatically handle sorting and uniqueness
        std::vector<std::set<COLTYPE>> rows( size );

        // Add entries from A
        for ( COLTYPE i = 0; i < size; i++ )
        {
            for ( ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; j++ )
            {
                COLTYPE col = aj[j] - base;
                if ( col == i )
                {
                    if constexpr ( KEEPDIAG )
                    {
                        rows[i].insert( col + base );
                    }
                }
                else
                {
                    rows[i].insert( col + base );
                }
            }
        }

        // Add entries from A^T
        for ( COLTYPE i = 0; i < size; i++ )
        {
            for ( ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; j++ )
            {
                COLTYPE col = aj[j] - base;
                if ( col != i )
                {
                    rows[col].insert( i + base );
                }
            }
        }

        // Convert to CSR format
        CSRStructVec<ROWTYPE, COLTYPE> result;
        result.ai.resize( size + 1 );
        result.ai[0] = base;

        ROWTYPE nnz = 0;
        for ( COLTYPE i = 0; i < size; i++ )
        {
            nnz += rows[i].size();
            result.ai[i + 1] = result.ai[i] + rows[i].size();
        }

        result.aj.reserve( nnz );
        for ( COLTYPE i = 0; i < size; i++ )
        {
            for ( COLTYPE col : rows[i] )
            {
                result.aj.push_back( col );
            }
        }

        return result;
    }

    // Helper to compare two CSR structures
    template <typename ROWTYPE, typename COLTYPE>
    void compareCsrStructures( const COLTYPE size,
                               const ROWTYPE* ai1,
                               const COLTYPE* aj1,
                               const ROWTYPE* ai2,
                               const COLTYPE* aj2,
                               const std::string& msg = "" )
    {
        const ROWTYPE base = ai1[0];
        ASSERT_EQ( ai2[0], base ) << msg << ": Base mismatch";

        for ( COLTYPE i = 0; i <= size; i++ )
        {
            ASSERT_EQ( ai1[i], ai2[i] ) << msg << ": Row pointer mismatch at row " << i;
        }

        const ROWTYPE nnz = ai1[size] - base;
        for ( ROWTYPE j = 0; j < nnz; j++ )
        {
            ASSERT_EQ( aj1[j], aj2[j] )
                << msg << ": Column index mismatch at position " << j;
        }
    }
};

TEST( PartitionOpsTest, Partition1xNZeroBased )
{
    const int32_t rows = 3;
    const int32_t cols = 6;
    const int32_t base = 0;

    // CSR for:
    // row0: (0,10) (2,20) (4,30)
    // row1: (1,40) (2,50) (3,60)
    // row2: (0,70) (5,80)
    std::vector<int32_t> ai = { 0, 3, 6, 8 };
    std::vector<int32_t> aj = { 0, 2, 4, 1, 2, 3, 0, 5 };
    std::vector<double>   av = { 10, 20, 30, 40, 50, 60, 70, 80 };

    // Split columns into [0,2), [2,4), [4,6)
    constexpr int N = 3;
    std::array<int32_t, N + 1> col_splits{ 0, 2, 4, 6 };

    CSRMatrixVec<int32_t, int32_t, double> blocks[N];
    partitionCSR1xN( rows, cols, ai.data(), aj.data(), av.data(), N,
                     col_splits.data(), base, blocks, /*nthreads=*/1 );

    auto expect_block = [&]( const CSRMatrixVec<int32_t, int32_t, double>& blk,
                             int32_t expected_cols,
                             std::vector<int32_t> expected_ai,
                             std::vector<int32_t> expected_aj,
                             std::vector<double> expected_av )
    {
        ASSERT_EQ( blk.rows, rows );
        ASSERT_EQ( blk.cols, expected_cols );
        ASSERT_EQ( blk.ai.size(), expected_ai.size() );
        ASSERT_EQ( blk.aj.size(), expected_aj.size() );
        ASSERT_EQ( blk.av.size(), expected_av.size() );

        EXPECT_TRUE( std::equal( blk.ai.begin(), blk.ai.end(), expected_ai.begin() ) );
        EXPECT_TRUE( std::equal( blk.aj.begin(), blk.aj.end(), expected_aj.begin() ) );
        EXPECT_TRUE( std::equal( blk.av.begin(), blk.av.end(), expected_av.begin() ) );
    };

    // Block 0: columns [0,2)
    expect_block( blocks[0], 2,
                  { 0, 1, 2, 3 },
                  { 0, 1, 0 },
                  { 10, 40, 70 } );

    // Block 1: columns [2,4) shifted to 0-based
    expect_block( blocks[1], 2,
                  { 0, 1, 3, 3 },
                  { 0, 0, 1 },
                  { 20, 50, 60 } );

    // Block 2: columns [4,6) shifted to 0-based
    expect_block( blocks[2], 2,
                  { 0, 1, 1, 2 },
                  { 0, 1 },
                  { 30, 80 } );
}

TEST( PartitionOpsTest, Partition1xNOneBased )
{
    const int32_t rows = 3;
    const int32_t cols = 6;
    const int32_t base = 1;

    // Same pattern as zero-based test but shifted to base=1
    std::vector<int32_t> ai = { 1, 4, 7, 9 };
    std::vector<int32_t> aj = { 1, 3, 5, 2, 3, 4, 1, 6 };
    std::vector<double>   av = { 10, 20, 30, 40, 50, 60, 70, 80 };

    constexpr int N = 3;
    std::array<int32_t, N + 1> col_splits{ 1, 3, 5, 7 }; // includes start/end with base

    CSRMatrixVec<int32_t, int32_t, double> blocks[N];
    partitionCSR1xN( rows, cols, ai.data(), aj.data(), av.data(), N,
                     col_splits.data(), base, blocks, /*nthreads=*/1 );

    auto expect_block = [&]( const CSRMatrixVec<int32_t, int32_t, double>& blk,
                             int32_t expected_cols,
                             std::vector<int32_t> expected_ai,
                             std::vector<int32_t> expected_aj,
                             std::vector<double> expected_av )
    {
        ASSERT_EQ( blk.rows, rows );
        ASSERT_EQ( blk.cols, expected_cols );
        ASSERT_EQ( blk.ai.size(), expected_ai.size() );
        ASSERT_EQ( blk.aj.size(), expected_aj.size() );
        ASSERT_EQ( blk.av.size(), expected_av.size() );

        EXPECT_TRUE( std::equal( blk.ai.begin(), blk.ai.end(), expected_ai.begin() ) );
        EXPECT_TRUE( std::equal( blk.aj.begin(), blk.aj.end(), expected_aj.begin() ) );
        EXPECT_TRUE( std::equal( blk.av.begin(), blk.av.end(), expected_av.begin() ) );
    };

    // Block 0: columns [1,3)
    expect_block( blocks[0], 2,
                  { 1, 2, 3, 4 },
                  { 1, 2, 1 },
                  { 10, 40, 70 } );

    // Block 1: columns [3,5) shifted to base=1
    expect_block( blocks[1], 2,
                  { 1, 2, 4, 4 },
                  { 1, 1, 2 },
                  { 20, 50, 60 } );

    // Block 2: columns [5,7) shifted to base=1
    expect_block( blocks[2], 2,
                  { 1, 2, 2, 3 },
                  { 1, 2 },
                  { 30, 80 } );
}

TEST( PartitionOpsTest, Partition1xNLargeRecompose )
{
    const int32_t rows = 5;
    const int32_t cols = 8;
    const int32_t base = 0;

    // Construct a moderately sized, sorted CSR
    std::vector<int32_t> ai = { 0, 3, 5, 9, 11, 14 };
    std::vector<int32_t> aj = { 0, 3, 7, 1, 2, 0, 2, 4, 6, 1, 5, 3, 4, 7 };
    std::vector<double>   av = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };

    constexpr int N = 4;
    std::array<int32_t, N + 1> col_splits{ 0, 2, 4, 6, 8 };

    CSRMatrixVec<int32_t, int32_t, double> blocks[N];
    partitionCSR1xN( rows, cols, ai.data(), aj.data(), av.data(), N,
                     col_splits.data(), base, blocks, /*nthreads=*/1 );

    // Recompose the original matrix from blocks and ensure it matches input
    std::vector<int32_t> rec_ai( rows + 1, 0 );
    std::vector<int32_t> rec_aj;
    std::vector<double>  rec_av;
    rec_ai[0] = base;

    for ( int32_t r = 0; r < rows; ++r )
    {
        for ( int b = 0; b < N; ++b )
        {
            const auto col_shift = col_splits[b] - base;
            auto* ai_b = blocks[b].AI();
            auto* aj_b = blocks[b].AJ();
            auto* av_b = blocks[b].AV();

            const int32_t start = ai_b[r] - base;
            const int32_t end   = ai_b[r + 1] - base;
            for ( int32_t k = start; k < end; ++k )
            {
                rec_aj.push_back( aj_b[k] + col_shift );
                rec_av.push_back( av_b[k] );
            }
        }
        rec_ai[r + 1] = static_cast<int32_t>( rec_aj.size() ) + base;
    }

    ASSERT_EQ( rec_ai, ai );
    ASSERT_EQ( rec_aj, aj );
    ASSERT_EQ( rec_av, av );
}

TEST( PartitionOpsTest, Partition1xNParallelRecompose )
{
    const int32_t rows = 5;
    const int32_t cols = 8;
    const int32_t base = 0;

    std::vector<int32_t> ai = { 0, 3, 5, 9, 11, 14 };
    std::vector<int32_t> aj = { 0, 3, 7, 1, 2, 0, 2, 4, 6, 1, 5, 3, 4, 7 };
    std::vector<double>   av = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };

    constexpr int N = 4;
    std::array<int32_t, N + 1> col_splits{ 0, 2, 4, 6, 8 };
    const int max_threads = std::max( 2, std::min( 8, omp_get_max_threads() ) );
    for ( int nthreads = 2; nthreads <= max_threads; ++nthreads )
    {
        CSRMatrixVec<int32_t, int32_t, double> blocks[N];
        partitionCSR1xN( rows, cols, ai.data(), aj.data(), av.data(), N,
                         col_splits.data(), base, blocks, nthreads );

        std::vector<int32_t> rec_ai( rows + 1, base );
        std::vector<int32_t> rec_aj;
        std::vector<double>  rec_av;

        for ( int32_t r = 0; r < rows; ++r )
        {
            for ( int b = 0; b < N; ++b )
            {
                const auto col_shift = col_splits[b] - base;
                auto* ai_b = blocks[b].AI();
                auto* aj_b = blocks[b].AJ();
                auto* av_b = blocks[b].AV();

                const int32_t start = ai_b[r] - base;
                const int32_t end   = ai_b[r + 1] - base;
                for ( int32_t k = start; k < end; ++k )
                {
                    rec_aj.push_back( aj_b[k] + col_shift );
                    rec_av.push_back( av_b[k] );
                }
            }
            rec_ai[r + 1] = static_cast<int32_t>( rec_aj.size() ) + base;
        }

        ASSERT_EQ( rec_ai, ai );
        ASSERT_EQ( rec_aj, aj );
        ASSERT_EQ( rec_av, av );
    }
}

TEST( PartitionOpsTest, Partition1xNBoundaryCases )
{
    const int32_t rows = 3;
    const int32_t cols = 4;
    const int32_t base = 0;

    // CSR for:
    // row0: (0,1) (2,2)
    // row1: (1,3)
    // row2: (3,4)
    std::vector<int32_t> ai = { 0, 2, 3, 4 };
    std::vector<int32_t> aj = { 0, 2, 1, 3 };
    std::vector<double>   av = { 1, 2, 3, 4 };

    // N=1 should return the same matrix; test threads 1..8
    for ( int nthreads = 1; nthreads <= 8; ++nthreads )
    {
        constexpr int N = 1;
        std::array<int32_t, N + 1> col_splits{ base, cols + base };
        CSRMatrixVec<int32_t, int32_t, double> blocks[N];
        partitionCSR1xN( rows, cols, ai.data(), aj.data(), av.data(), N,
                         col_splits.data(), base, blocks, nthreads );
        EXPECT_EQ( blocks[0].rows, rows );
        EXPECT_EQ( blocks[0].cols, cols );
        EXPECT_EQ( blocks[0].ai, ai );
        EXPECT_EQ( blocks[0].aj, aj );
        EXPECT_EQ( blocks[0].av, av );
    }

    // N = number of columns: each column becomes its own block (may be empty); test threads 1..8
    for ( int nthreads = 1; nthreads <= 8; ++nthreads )
    {
        const int N = cols;
        std::vector<int32_t> col_splits( static_cast<size_t>( N + 1 ) );
        for ( int i = 0; i <= N; ++i ) col_splits[i] = base + i;

        std::vector<CSRMatrixVec<int32_t, int32_t, double>> blocks( static_cast<size_t>( N ) );
        partitionCSR1xN( rows, cols, ai.data(), aj.data(), av.data(), N,
                         col_splits.data(), base, blocks.data(), nthreads );

        // Recompose and compare
        std::vector<int32_t> rec_ai( rows + 1, base );
        std::vector<int32_t> rec_aj;
        std::vector<double>  rec_av;

        for ( int32_t r = 0; r < rows; ++r )
        {
            for ( int c = 0; c < N; ++c )
            {
                const int col_shift = col_splits[c] - base;
                auto& blk = blocks[c];
                auto* ai_b = blk.AI();
                auto* aj_b = blk.AJ();
                auto* av_b = blk.AV();

                const int start = ai_b[r] - base;
                const int end   = ai_b[r + 1] - base;
                for ( int k = start; k < end; ++k )
                {
                    rec_aj.push_back( aj_b[k] + col_shift );
                    rec_av.push_back( av_b[k] );
                }
            }
            rec_ai[r + 1] = static_cast<int32_t>( rec_aj.size() ) + base;
        }

        EXPECT_EQ( rec_ai, ai );
        EXPECT_EQ( rec_aj, aj );
        EXPECT_EQ( rec_av, av );
    }
}

// Test basic functionality with small matrix
TEST_F( SymmetricOpsTest, SmallMatrix_KeepDiag )
{
    // 3x3 matrix:
    // [1 2 0]
    // [0 3 4]
    // [0 0 5]
    const int32_t size = 3;
    const int32_t base = 0;
    std::vector<int32_t> ai = { 0, 2, 4, 5 };
    std::vector<int32_t> aj = { 0, 1, 1, 2, 2 };

    // Expected A+A^T (with diagonal):
    // [1 2 0]
    // [2 3 4]
    // [0 4 5]
    std::vector<int32_t> expected_ai = { 0, 2, 5, 7 };
    std::vector<int32_t> expected_aj = { 0, 1, 0, 1, 2, 1, 2 };

    APlusATStruct<int32_t, int32_t, true> aplusatOp( 1 );

    std::vector<int32_t> result_ai( size + 1 );
    std::vector<int32_t> result_aj( expected_ai[size] );

    aplusatOp( size, ai.data(), aj.data(), result_ai.data(), result_aj.data() );

    verifyCsrStructure( size, result_ai.data(), result_aj.data(), base );
    compareCsrStructures( size, expected_ai.data(), expected_aj.data(),
                          result_ai.data(), result_aj.data(),
                          "SmallMatrix_KeepDiag" );
}

// Test without keeping diagonal
TEST_F( SymmetricOpsTest, SmallMatrix_NoDiag )
{
    // Same 3x3 matrix
    const int32_t size = 3;
    const int32_t base = 0;
    std::vector<int32_t> ai = { 0, 2, 4, 5 };
    std::vector<int32_t> aj = { 0, 1, 1, 2, 2 };

    // Expected A+A^T (without diagonal):
    // [0 2 0]
    // [2 0 4]
    // [0 4 0]
    std::vector<int32_t> expected_ai = { 0, 1, 3, 4 };
    std::vector<int32_t> expected_aj = { 1, 0, 2, 1 };

    APlusATStruct<int32_t, int32_t, false> aplusatOp( 1 );

    std::vector<int32_t> result_ai( size + 1 );
    std::vector<int32_t> result_aj( expected_ai[size] );

    aplusatOp( size, ai.data(), aj.data(), result_ai.data(), result_aj.data() );

    verifyCsrStructure( size, result_ai.data(), result_aj.data(), base );
    compareCsrStructures( size, expected_ai.data(), expected_aj.data(),
                          result_ai.data(), result_aj.data(),
                          "SmallMatrix_NoDiag" );
}

// Test with 1-based indexing
TEST_F( SymmetricOpsTest, OneBased_KeepDiag )
{
    // 3x3 matrix with 1-based indexing
    const int32_t size = 3;
    const int32_t base = 1;
    std::vector<int32_t> ai = { 1, 3, 5, 6 };
    std::vector<int32_t> aj = { 1, 2, 2, 3, 3 };

    // Expected A+A^T (with diagonal, 1-based):
    std::vector<int32_t> expected_ai = { 1, 3, 6, 8 };
    std::vector<int32_t> expected_aj = { 1, 2, 1, 2, 3, 2, 3 };

    APlusATStruct<int32_t, int32_t, true> aplusatOp( 1 );

    std::vector<int32_t> result_ai( size + 1 );
    std::vector<int32_t> result_aj( expected_ai[size] - base );

    aplusatOp( size, ai.data(), aj.data(), result_ai.data(), result_aj.data() );

    verifyCsrStructure( size, result_ai.data(), result_aj.data(), base );
    compareCsrStructures( size, expected_ai.data(), expected_aj.data(),
                          result_ai.data(), result_aj.data(),
                          "OneBased_KeepDiag" );
}

// Test memory reuse across multiple calls
TEST_F( SymmetricOpsTest, MemoryReuse )
{
    const int32_t size = 3;
    const int32_t base = 0;
    std::vector<int32_t> ai = { 0, 2, 4, 5 };
    std::vector<int32_t> aj = { 0, 1, 1, 2, 2 };

    APlusATStruct<int32_t, int32_t, true> aplusatOp( 1 );

    // First call
    std::vector<int32_t> result_ai1( size + 1 );
    std::vector<int32_t> result_aj1( 10 ); // Over-allocate
    aplusatOp( size, ai.data(), aj.data(), result_ai1.data(), result_aj1.data() );

    // Second call with same struct - should reuse memory
    std::vector<int32_t> result_ai2( size + 1 );
    std::vector<int32_t> result_aj2( 10 );
    aplusatOp( size, ai.data(), aj.data(), result_ai2.data(), result_aj2.data() );

    // Results should be identical
    compareCsrStructures( size, result_ai1.data(), result_aj1.data(),
                          result_ai2.data(), result_aj2.data(), "MemoryReuse" );
}

// Test with diagonal-only matrix
TEST_F( SymmetricOpsTest, DiagonalMatrix_KeepDiag )
{
    const int32_t size = 4;
    const int32_t base = 0;
    std::vector<int32_t> ai = { 0, 1, 2, 3, 4 };
    std::vector<int32_t> aj = { 0, 1, 2, 3 };

    // A+A^T should equal A for diagonal matrix with KEEPDIAG
    APlusATStruct<int32_t, int32_t, true> aplusatOp( 1 );

    std::vector<int32_t> result_ai( size + 1 );
    std::vector<int32_t> result_aj( ai[size] );

    aplusatOp( size, ai.data(), aj.data(), result_ai.data(), result_aj.data() );

    verifyCsrStructure( size, result_ai.data(), result_aj.data(), base );
    compareCsrStructures( size, ai.data(), aj.data(), result_ai.data(),
                          result_aj.data(), "DiagonalMatrix_KeepDiag" );
}

// Test with diagonal-only matrix without keeping diagonal
TEST_F( SymmetricOpsTest, DiagonalMatrix_NoDiag )
{
    const int32_t size = 4;
    const int32_t base = 0;
    std::vector<int32_t> ai = { 0, 1, 2, 3, 4 };
    std::vector<int32_t> aj = { 0, 1, 2, 3 };

    // A+A^T should be empty (all zeros) without diagonal
    std::vector<int32_t> expected_ai = { 0, 0, 0, 0, 0 };

    APlusATStruct<int32_t, int32_t, false> aplusatOp( 1 );

    std::vector<int32_t> result_ai( size + 1 );
    std::vector<int32_t> result_aj( 1 ); // Minimal allocation

    aplusatOp( size, ai.data(), aj.data(), result_ai.data(), result_aj.data() );

    ASSERT_EQ( result_ai[size], base ) << "Should have zero non-zeros";
    for ( int32_t i = 0; i <= size; i++ )
    {
        ASSERT_EQ( result_ai[i], base ) << "All row pointers should be base";
    }
}

// Test thread count setting
/**
 * @test
 * @brief Tests the thread count setting for the APlusATStruct operation.
 *
 * This test verifies that the APlusATStruct operator works correctly with different numbers of threads.
 * It constructs a symmetric matrix from the given CSR structure and checks the result for thread counts of 1, 2, 4, and 8.
 *
 * The input matrix is represented in CSR format by:
 *   ai = { 0, 2, 4, 5 }
 *   aj = { 0, 1, 1, 2, 2 }
 *
 * This corresponds to the following 3x3 matrix:
 *   [ 1 1 0 ]
 *   [ 0 1 1 ]
 *   [ 0 0 1 ]
 *
 * The test ensures that the operator produces the correct CSR structure for A + Aᵗ (the sum of the matrix and its transpose)
 * regardless of the number of threads used.
 *
 * The result is validated using verifyCsrStructure.
 *
 * Expected result for A + Aᵗ (with diagonal, CSR format):
 *   expected_ai = { 0, 2, 5, 7 }
 *   expected_aj = { 0, 1, 0, 1, 2, 1, 2 }
 *
 * Which corresponds to the matrix:
 *   [ 1 1 0 ]
 *   [ 1 1 1 ]
 *   [ 0 1 1 ]
 */
TEST_F( SymmetricOpsTest, ThreadCountSetting )
{
    const int32_t size = 3;
    const int32_t base = 0;
    std::vector<int32_t> ai = { 0, 2, 4, 5 };
    std::vector<int32_t> aj = { 0, 1, 1, 2, 2 };

    // Test with different thread counts
    for ( int nthreads : { 1, 2, 4, 8 } )
    {
        APlusATStruct<int32_t, int32_t, true> aplusatOp( nthreads );

        std::vector<int32_t> result_ai( size + 1 );
        std::vector<int32_t> result_aj( 10 );

        ASSERT_NO_THROW( aplusatOp( size, ai.data(), aj.data(),
                                    result_ai.data(), result_aj.data() ) )
            << "Should work with " << nthreads << " threads";

        verifyCsrStructure( size, result_ai.data(), result_aj.data(), base );
    }
}

// Test setNumThreads
TEST_F( SymmetricOpsTest, SetNumThreads )
{
    const int32_t size = 3;
    const int32_t base = 0;
    std::vector<int32_t> ai = { 0, 2, 4, 5 };
    std::vector<int32_t> aj = { 0, 1, 1, 2, 2 };

    APlusATStruct<int32_t, int32_t, true> aplusatOp( 2 );

    std::vector<int32_t> result_ai1( size + 1 );
    std::vector<int32_t> result_aj1( 10 );
    aplusatOp( size, ai.data(), aj.data(), result_ai1.data(), result_aj1.data() );

    // Change thread count
    aplusatOp.setNumThreads( 8 );

    std::vector<int32_t> result_ai2( size + 1 );
    std::vector<int32_t> result_aj2( 10 );
    aplusatOp( size, ai.data(), aj.data(), result_ai2.data(), result_aj2.data() );

    // Results should be the same
    compareCsrStructures( size, result_ai1.data(), result_aj1.data(), result_ai2.data(),
                          result_aj2.data(), "SetNumThreads" );
}

// Test with int64_t types
TEST_F( SymmetricOpsTest, Int64Types )
{
    const int64_t size = 3;
    const int64_t base = 0;
    std::vector<int64_t> ai = { 0, 2, 4, 5 };
    std::vector<int64_t> aj = { 0, 1, 1, 2, 2 };

    std::vector<int64_t> expected_ai = { 0, 2, 5, 7 };
    std::vector<int64_t> expected_aj = { 0, 1, 0, 1, 2, 1, 2 };

    APlusATStruct<int64_t, int64_t, true> aplusatOp( 1 );

    std::vector<int64_t> result_ai( size + 1 );
    std::vector<int64_t> result_aj( expected_ai[size] );

    aplusatOp( size, ai.data(), aj.data(), result_ai.data(), result_aj.data() );

    verifyCsrStructure( size, result_ai.data(), result_aj.data(), base );
    compareCsrStructures( size, expected_ai.data(), expected_aj.data(),
                          result_ai.data(), result_aj.data(), "Int64Types" );
}

// Test with larger matrix from file
TEST_F( SymmetricOpsTest, LargerMatrix_FromFile )
{
    // Try to load a test matrix
    std::vector<int32_t> ai, aj;
    std::vector<double> av;

    std::ifstream f( "data/ex5.mtx" );
    if ( !f.good() )
    {
        GTEST_SKIP() << "Test matrix data/ex5.mtx not found";
    }

    utils::read_matrix_market_csr( f, ai, aj, av );
    f.close();

    if ( ai.size() == 0 )
    {
        GTEST_SKIP() << "Could not read matrix";
    }

    const int32_t size = ai.size() - 1;

    // Test with KEEPDIAG=true
    APlusATStruct<int32_t, int32_t, true> aplusatOp( 2 );

    std::vector<int32_t> result_ai( size + 1 );

    const int32_t base = ai[0];
    const int32_t original_nnz = ai[size] - base;
    // Allocate enough space: worst case is 2x the original NNZ (A + A^T without duplicates)
    std::vector<int32_t> result_aj( 2 * original_nnz );

    // Single call to compute A+A^T
    aplusatOp( size, ai.data(), aj.data(), result_ai.data(), result_aj.data() );

    verifyCsrStructure( size, result_ai.data(), result_aj.data(), base );

    // Verify against naive implementation
    auto expected =
        computeAPlusATNaive<int32_t, int32_t, true>( size, ai.data(), aj.data() );

    compareCsrStructures( size, expected.ai.data(), expected.aj.data(),
                          result_ai.data(), result_aj.data(),
                          "LargerMatrix_FromFile" );
}

// Test correctness against naive implementation
TEST_F( SymmetricOpsTest, CorrectnessCheck_Various )
{
    struct TestCase
    {
        std::string name;
        std::vector<int32_t> ai;
        std::vector<int32_t> aj;
        int32_t size;
    };

    std::vector<TestCase> cases = {
        { "Empty", { 0 }, {}, 0 },
        { "SingleElement", { 0, 1 }, { 0 }, 1 },
        { "UpperTriangular", { 0, 3, 5, 6 }, { 0, 1, 2, 1, 2, 2 }, 3 },
        { "LowerTriangular", { 0, 1, 3, 6 }, { 0, 1, 0, 2, 1, 2 }, 3 },
        // Matrix with both A[i,j] and A[j,i] - creates duplicates in A+A^T
        // Row 0: cols [1]
        // Row 1: cols [0, 2]
        // Row 2: cols [1]
        // So A[0,1] and A[1,0] both exist, A[1,2] and A[2,1] both exist
        { "WithDuplicates", { 0, 1, 3, 4 }, { 1, 0, 2, 1 }, 3 },
    };

    for ( const auto& tc : cases )
    {
        if ( tc.size == 0 )
            continue;

        // Test with KEEPDIAG=true
        {
            APlusATStruct<int32_t, int32_t, true> aplusatOp( 1 );
            auto expected = computeAPlusATNaive<int32_t, int32_t, true>(
                tc.size, tc.ai.data(), tc.aj.data() );

            std::vector<int32_t> result_ai( tc.size + 1 );
            std::vector<int32_t> result_aj( expected.aj.size() );

            aplusatOp( tc.size, tc.ai.data(), tc.aj.data(), result_ai.data(),
                       result_aj.data() );

            compareCsrStructures( tc.size, expected.ai.data(),
                                  expected.aj.data(), result_ai.data(),
                                  result_aj.data(), tc.name + "_KeepDiag" );
        }

        // Test with KEEPDIAG=false
        {
            APlusATStruct<int32_t, int32_t, false> aplusatOp( 1 );
            auto expected = computeAPlusATNaive<int32_t, int32_t, false>(
                tc.size, tc.ai.data(), tc.aj.data() );

            std::vector<int32_t> result_ai( tc.size + 1 );
            std::vector<int32_t> result_aj( expected.aj.size() );

            aplusatOp( tc.size, tc.ai.data(), tc.aj.data(), result_ai.data(),
                       result_aj.data() );

            compareCsrStructures( tc.size, expected.ai.data(),
                                  expected.aj.data(), result_ai.data(),
                                  result_aj.data(), tc.name + "_NoDiag" );
        }
    }
}

// Validate APlusATPrefix matches A+A^T row pointers
TEST_F( SymmetricOpsTest, APlusATPrefix_AsAPlusATPrefix )
{
    const int32_t size = 3;
    const int32_t base = 0;
    // row0: 0,2 ; row1: 1 ; row2: 0,2 (not strictly triangular, exercising A+A^T view)
    std::vector<int32_t> ai = { 0, 2, 3, 5 };
    std::vector<int32_t> aj = { 0, 2, 1, 0, 2 };

    auto expected_keep = computeAPlusATNaive<int32_t, int32_t, true>(
        size, ai.data(), aj.data() );
    auto expected_nodiag = computeAPlusATNaive<int32_t, int32_t, false>(
        size, ai.data(), aj.data() );

    std::vector<int32_t> ai_AAT( size + 1 );

    APlusATPrefix<int32_t, int32_t, true>( size, ai.data(), aj.data(),
                                           ai_AAT.data() );
    ASSERT_EQ( ai_AAT, expected_keep.ai );

    APlusATPrefix<int32_t, int32_t, false>( size, ai.data(), aj.data(),
                                            ai_AAT.data() );
    ASSERT_EQ( ai_AAT, expected_nodiag.ai );
}

// Validate APlusATFill builds column indices for A+A^T
TEST_F( SymmetricOpsTest, APlusATFill_AsAPlusATColumns )
{
    const int32_t size = 3;
    const int32_t base = 0;
    std::vector<int32_t> ai = { 0, 2, 3, 5 };
    std::vector<int32_t> aj = { 0, 2, 1, 0, 2 };

    auto expected_keep = computeAPlusATNaive<int32_t, int32_t, true>(
        size, ai.data(), aj.data() );
    auto expected_nodiag = computeAPlusATNaive<int32_t, int32_t, false>(
        size, ai.data(), aj.data() );

    // KEEPDIAG = true
    std::vector<int32_t> ai_AAT( size + 1 );
    APlusATPrefix<int32_t, int32_t, true>( size, ai.data(), aj.data(),
                                           ai_AAT.data() );
    ASSERT_EQ( ai_AAT, expected_keep.ai );

    std::vector<int32_t> aj_AAT( expected_keep.aj.size() );
    APlusATFill<int32_t, int32_t, true>( size, ai.data(), aj.data(),
                                         ai_AAT.data(), aj_AAT.data() );
    verifyCsrStructure( size, ai_AAT.data(), aj_AAT.data(), base );
    compareCsrStructures( size, expected_keep.ai.data(), expected_keep.aj.data(),
                          ai_AAT.data(), aj_AAT.data(),
                          "APlusATFill_KeepDiag" );

    // KEEPDIAG = false
    APlusATPrefix<int32_t, int32_t, false>( size, ai.data(), aj.data(),
                                            ai_AAT.data() );
    ASSERT_EQ( ai_AAT, expected_nodiag.ai );

    aj_AAT.assign( expected_nodiag.aj.size(), 0 );
    APlusATFill<int32_t, int32_t, false>( size, ai.data(), aj.data(),
                                          ai_AAT.data(), aj_AAT.data() );
    verifyCsrStructure( size, ai_AAT.data(), aj_AAT.data(), base );
    compareCsrStructures( size, expected_nodiag.ai.data(), expected_nodiag.aj.data(),
                          ai_AAT.data(), aj_AAT.data(),
                          "APlusATFill_NoDiag" );
}

// Validate combined APlusATSerial helper produces A+A^T structure
TEST_F( SymmetricOpsTest, APlusATSerial_AsAPlusAT )
{
    const int32_t size = 3;
    const int32_t base = 0;
    std::vector<int32_t> ai = { 0, 2, 3, 5 };
    std::vector<int32_t> aj = { 0, 2, 1, 0, 2 };

    // KEEPDIAG = true
    auto expected_keep = computeAPlusATNaive<int32_t, int32_t, true>(
        size, ai.data(), aj.data() );
    std::vector<int32_t> ai_out( size + 1 );
    std::vector<int32_t> aj_out( expected_keep.aj.size() );

    APlusATSerial<int32_t, int32_t, true>( size, ai.data(), aj.data(),
                                           ai_out.data(), aj_out.data() );
    compareCsrStructures( size, expected_keep.ai.data(), expected_keep.aj.data(),
                          ai_out.data(), aj_out.data(), "APlusATSerial_KeepDiag" );

    // KEEPDIAG = false
    auto expected_nodiag = computeAPlusATNaive<int32_t, int32_t, false>(
        size, ai.data(), aj.data() );
    ai_out.assign( size + 1, 0 );
    aj_out.assign( expected_nodiag.aj.size(), 0 );

    APlusATSerial<int32_t, int32_t, false>( size, ai.data(), aj.data(),
                                            ai_out.data(), aj_out.data() );
    compareCsrStructures( size, expected_nodiag.ai.data(), expected_nodiag.aj.data(),
                          ai_out.data(), aj_out.data(), "APlusATSerial_NoDiag" );
}

// Test all matrices from CMakeLists.txt with different thread counts
TEST_F( SymmetricOpsTest, AllCMakeMatrices )
{
    std::vector<std::string> matrix_names = {
        "bcsstk17", "s3rmt3m3", "ex5", "jgl009", "rdist1", "nos5"
    };

    std::vector<int> thread_counts = { 1, 2, 3, 4, 5, 6, 7, 8 };

    for ( const auto& name : matrix_names )
    {
        std::string filepath = "data/" + name + ".mtx";
        std::cout << "\n=== Testing matrix: " << name << " ===" << std::endl;

        // Try to load the matrix
        std::ifstream f( filepath );
        if ( !f.good() )
        {
            std::cout << "Skipping matrix " << name << " (file not found)" << std::endl;
            GTEST_SKIP() << "Matrix file not found: " << filepath;
            continue;
        }

        std::vector<int32_t> ai, aj;
        std::vector<double> av;
        utils::read_matrix_market_csr( f, ai, aj, av );
        f.close();

        if ( ai.size() == 0 )
        {
            std::cout << "Skipping matrix " << name << " (could not read)" << std::endl;
            continue;
        }

        const int32_t size = ai.size() - 1;
        const int32_t base = ai[0];
        const int32_t nnz = ai[size] - base;

        std::cout << "Matrix " << name << ": size=" << size << ", nnz=" << nnz
                  << ", base=" << base << std::endl;

        auto run_all_checks = [&]( std::vector<int32_t>& ai_cur,
                                   std::vector<int32_t>& aj_cur,
                                   const std::string& suffix ) {
            const int32_t cur_base = ai_cur[0];

            // Compute expected result once (using naive method)
            auto expected_keepdiag = computeAPlusATNaive<int32_t, int32_t, true>(
                size, ai_cur.data(), aj_cur.data() );
            auto expected_nodiag = computeAPlusATNaive<int32_t, int32_t, false>(
                size, ai_cur.data(), aj_cur.data() );

            // Also validate the serial helper once per matrix
            {
                std::vector<int32_t> ai_serial( size + 1 );
                std::vector<int32_t> aj_serial( expected_keepdiag.aj.size() );
                APlusATSerial<int32_t, int32_t, true>( size, ai_cur.data(), aj_cur.data(),
                                                       ai_serial.data(), aj_serial.data() );
                compareCsrStructures( size, expected_keepdiag.ai.data(),
                                      expected_keepdiag.aj.data(), ai_serial.data(),
                                      aj_serial.data(), name + "_Serial_KeepDiag" + suffix );

                ai_serial.assign( size + 1, 0 );
                aj_serial.assign( expected_nodiag.aj.size(), 0 );
                APlusATSerial<int32_t, int32_t, false>( size, ai_cur.data(), aj_cur.data(),
                                                        ai_serial.data(), aj_serial.data() );
                compareCsrStructures( size, expected_nodiag.ai.data(),
                                      expected_nodiag.aj.data(), ai_serial.data(),
                                      aj_serial.data(), name + "_Serial_NoDiag" + suffix );
            }

            // Test with different thread counts
            for ( int nthreads : thread_counts )
            {
                std::cout << "\n  Testing with " << nthreads << " thread(s):" << suffix << std::endl;

                // Test with KEEPDIAG=true
                {
                    APlusATStruct<int32_t, int32_t, true> aplusatOp( nthreads );

                    // Allocate result with 2x original NNZ as upper bound
                    std::vector<int32_t> result_ai( size + 1 );
                    std::vector<int32_t> result_aj( 2 * ( ai_cur[size] - cur_base ) );

                    aplusatOp( size, ai_cur.data(), aj_cur.data(), result_ai.data(), result_aj.data() );

                    // Verify CSR structure is valid
                    verifyCsrStructure( size, result_ai.data(), result_aj.data(),
                                        cur_base );

                    // Verify against expected computation
                    compareCsrStructures( size, expected_keepdiag.ai.data(),
                                          expected_keepdiag.aj.data(), result_ai.data(),
                                          result_aj.data(),
                                          name + "_KeepDiag_" + std::to_string( nthreads ) + "threads" + suffix );

                    std::cout << "    KEEPDIAG=true: result nnz="
                              << ( result_ai[size] - result_ai[0] )
                              << ", symmetric=true, matches expected=true" << std::endl;
                }

                // Test with KEEPDIAG=false
                {
                    APlusATStruct<int32_t, int32_t, false> aplusatOp( nthreads );

                    // Allocate result with 2x original NNZ as upper bound
                    std::vector<int32_t> result_ai( size + 1 );
                    std::vector<int32_t> result_aj( 2 * ( ai_cur[size] - cur_base ) );

                    aplusatOp( size, ai_cur.data(), aj_cur.data(), result_ai.data(), result_aj.data() );

                    // Verify CSR structure is valid
                    verifyCsrStructure( size, result_ai.data(), result_aj.data(),
                                        cur_base );

                    // Verify against expected computation
                    compareCsrStructures( size, expected_nodiag.ai.data(),
                                          expected_nodiag.aj.data(), result_ai.data(),
                                          result_aj.data(),
                                          name + "_NoDiag_" + std::to_string( nthreads ) + "threads" + suffix );

                    std::cout << "    KEEPDIAG=false: result nnz="
                              << ( result_ai[size] - result_ai[0] )
                              << ", symmetric=true, matches expected=true" << std::endl;
                }
            }
        };

        // Run on original base
        run_all_checks( ai, aj, "_base" + std::to_string( base ) );

        // Shift base and rerun
        std::vector<int32_t> ai_shift = ai;
        std::vector<int32_t> aj_shift = aj;
        const int32_t new_base = ( base == 0 ) ? 1 : 0;
        matrix_utils::ShiftCSRBase<int32_t, int32_t>( size, new_base, ai_shift.data(), aj_shift.data() );
        run_all_checks( ai_shift, aj_shift, "_base" + std::to_string( new_base ) );
    }
}

// Partition tests
class PartitionTest : public testing::Test
{
};

TEST_F( PartitionTest, PartitionCSRMxN_Basic )
{
    // 4x4 matrix, base 0, CSR format
    // [ 1 2 0 0 ]
    // [ 3 4 5 0 ]
    // [ 0 6 7 8 ]
    // [ 0 0 9 10]
    std::vector<int> ai = {0, 2, 5, 8, 10};
    std::vector<int> aj = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> av = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const int rows = 4, cols = 4;
    const int base = ai[0];
    const int M = 2, N = 2;
    int row_splits[M + 1] = { base, 2 + base, 4 + base };
    int col_splits[N + 1] = { base, 2 + base, 4 + base };

    CSRMatrixVec<int, int, double> blocks[M * N];
    partitionCSRMxN<CSRMatrixVec<int, int, double>>(
        rows, cols, ai.data(), aj.data(), av.data(),
        M, row_splits, N, col_splits, blocks, /*nthreads=*/2 );

    auto& A11 = blocks[0];
    auto& A12 = blocks[1];
    auto& A21 = blocks[2];
    auto& A22 = blocks[3];

    EXPECT_EQ( A11.rows, 2 ); EXPECT_EQ( A11.cols, 2 );
    EXPECT_EQ( A12.rows, 2 ); EXPECT_EQ( A12.cols, 2 );
    EXPECT_EQ( A21.rows, 2 ); EXPECT_EQ( A21.cols, 2 );
    EXPECT_EQ( A22.rows, 2 ); EXPECT_EQ( A22.cols, 2 );

    EXPECT_EQ( A11.NNZ(), 4 );
    EXPECT_EQ( A12.NNZ(), 1 );
    EXPECT_EQ( A21.NNZ(), 1 );
    EXPECT_EQ( A22.NNZ(), 4 );
}

TEST_F( PartitionTest, PartitionCSRMxN_MixedSplits )
{
    // 5x5 matrix with non-uniform splits
    std::vector<int> ai = {0, 3, 5, 8, 10, 12};
    std::vector<int> aj = {0, 1, 2, 1, 3, 0, 2, 4, 1, 3, 2, 4};
    std::vector<double> av(12, 1.0);
    const int rows = 5, cols = 5;
    const int base = ai[0];
    const int M = 3, N = 2;
    int row_splits[M + 1] = { base, 2 + base, 3 + base, 5 + base };
    int col_splits[N + 1] = { base, 2 + base, 5 + base };

    CSRMatrixVec<int, int, double> blocks[M * N];
    partitionCSRMxN<CSRMatrixVec<int, int, double>>(
        rows, cols, ai.data(), aj.data(), av.data(),
        M, row_splits, N, col_splits, blocks, /*nthreads=*/2 );

    int total_nnz = 0;
    for ( int i = 0; i < M * N; ++i )
    {
        total_nnz += blocks[i].NNZ();
        ASSERT_EQ( blocks[i].rows, row_splits[i / N + 1] - row_splits[i / N] );
    }
    EXPECT_EQ( total_nnz, ai.back() - ai.front() );
}

TEST_F( PartitionTest, PartitionCSR2x2_Basic )
{
    // 4x4 matrix, base 0, CSR format
    // [ 1 2 0 0 ]
    // [ 3 4 5 0 ]
    // [ 0 6 7 8 ]
    // [ 0 0 9 10]
    std::vector<int> ai = {0, 2, 5, 8, 10};
    std::vector<int> aj = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> av = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int rows = 4, cols = 4;
    int row_split = 2, col_split = 2;
    int nthreads = 2;

    CSRMatrixVec<int, int, double> A11, A12, A21, A22;
    partitionCSR2x2<CSRMatrixVec<int, int, double>>(
        rows, cols, ai.data(), aj.data(), av.data(),
        row_split, col_split,
        A11, A12, A21, A22, nthreads );

    // Check dimensions
    EXPECT_EQ( A11.rows, 2 );
    EXPECT_EQ( A11.cols, 2 );
    EXPECT_EQ( A12.rows, 2 );
    EXPECT_EQ( A12.cols, 2 );
    EXPECT_EQ( A21.rows, 2 );
    EXPECT_EQ( A21.cols, 2 );
    EXPECT_EQ( A22.rows, 2 );
    EXPECT_EQ( A22.cols, 2 );

    // Check NNZ counts
    EXPECT_EQ( A11.NNZ() + A12.NNZ() + A21.NNZ() + A22.NNZ(), 10 );

    // Verify A11: top-left 2x2 block
    // [ 1 2 ]
    // [ 3 4 ]
    EXPECT_EQ( A11.NNZ(), 4 );
    const int* a11_ai = A11.AI();
    const int* a11_aj = A11.AJ();
    EXPECT_EQ( a11_ai[0], 0 );
    EXPECT_EQ( a11_ai[1], 2 );
    EXPECT_EQ( a11_ai[2], 4 );
    EXPECT_EQ( a11_aj[0], 0 );
    EXPECT_EQ( a11_aj[1], 1 );
    EXPECT_EQ( a11_aj[2], 0 );
    EXPECT_EQ( a11_aj[3], 1 );

    // Verify A12: top-right 2x2 block
    // [ 0 0 ]
    // [ 5 0 ]
    EXPECT_EQ( A12.NNZ(), 1 );
    const int* a12_ai = A12.AI();
    const int* a12_aj = A12.AJ();
    EXPECT_EQ( a12_ai[0], 0 );
    EXPECT_EQ( a12_ai[1], 0 );
    EXPECT_EQ( a12_ai[2], 1 );
    EXPECT_EQ( a12_aj[0], 0 ); // column 2 becomes column 0 in A12

    // Verify A21: bottom-left 2x2 block
    // [ 0 6 ]
    // [ 0 0 ]
    EXPECT_EQ( A21.NNZ(), 1 );
    const int* a21_ai = A21.AI();
    const int* a21_aj = A21.AJ();
    EXPECT_EQ( a21_ai[0], 0 );
    EXPECT_EQ( a21_ai[1], 1 );
    EXPECT_EQ( a21_ai[2], 1 );
    EXPECT_EQ( a21_aj[0], 1 );

    // Verify A22: bottom-right 2x2 block
    // [ 7 8 ]
    // [ 9 10]
    EXPECT_EQ( A22.NNZ(), 4 );
    const int* a22_ai = A22.AI();
    const int* a22_aj = A22.AJ();
    EXPECT_EQ( a22_ai[0], 0 );
    EXPECT_EQ( a22_ai[1], 2 );
    EXPECT_EQ( a22_ai[2], 4 );
    EXPECT_EQ( a22_aj[0], 0 ); // column 2 becomes column 0 in A22
    EXPECT_EQ( a22_aj[1], 1 ); // column 3 becomes column 1 in A22
    EXPECT_EQ( a22_aj[2], 0 );
    EXPECT_EQ( a22_aj[3], 1 );
}

TEST_F( PartitionTest, PartitionCSR2x2_DifferentSplits )
{
    // 5x5 matrix with non-uniform split
    std::vector<int> ai = {0, 3, 5, 8, 10, 12};
    std::vector<int> aj = {0, 1, 2, 1, 3, 0, 2, 4, 1, 3, 2, 4};
    std::vector<double> av(12, 1.0);
    int rows = 5, cols = 5;
    int row_split = 3, col_split = 2;
    int nthreads = 2;

    CSRMatrixVec<int, int, double> A11, A12, A21, A22;
    partitionCSR2x2<CSRMatrixVec<int, int, double>>(
        rows, cols, ai.data(), aj.data(), av.data(),
        row_split, col_split,
        A11, A12, A21, A22, nthreads );

    // Check dimensions
    EXPECT_EQ( A11.rows, 3 );
    EXPECT_EQ( A11.cols, 2 );
    EXPECT_EQ( A12.rows, 3 );
    EXPECT_EQ( A12.cols, 3 );
    EXPECT_EQ( A21.rows, 2 );
    EXPECT_EQ( A21.cols, 2 );
    EXPECT_EQ( A22.rows, 2 );
    EXPECT_EQ( A22.cols, 3 );

    // Total NNZ should be preserved
    EXPECT_EQ( A11.NNZ() + A12.NNZ() + A21.NNZ() + A22.NNZ(), 12 );
}

TEST_F( PartitionTest, PartitionCSR2x2_EmptyBlocks )
{
    // 4x4 matrix with some empty blocks
    // [ 1 2 0 0 ]
    // [ 3 4 0 0 ]
    // [ 0 0 0 0 ]
    // [ 0 0 0 0 ]
    std::vector<int> ai = {0, 2, 4, 4, 4};
    std::vector<int> aj = {0, 1, 0, 1};
    std::vector<double> av = {1, 2, 3, 4};
    int rows = 4, cols = 4;
    int row_split = 2, col_split = 2;
    int nthreads = 1;

    CSRMatrixVec<int, int, double> A11, A12, A21, A22;
    partitionCSR2x2<CSRMatrixVec<int, int, double>>(
        rows, cols, ai.data(), aj.data(), av.data(),
        row_split, col_split,
        A11, A12, A21, A22, nthreads );

    // Check dimensions
    EXPECT_EQ( A11.rows, 2 );
    EXPECT_EQ( A11.cols, 2 );
    EXPECT_EQ( A12.rows, 2 );
    EXPECT_EQ( A12.cols, 2 );
    EXPECT_EQ( A21.rows, 2 );
    EXPECT_EQ( A21.cols, 2 );
    EXPECT_EQ( A22.rows, 2 );
    EXPECT_EQ( A22.cols, 2 );

    // A11 should have all entries
    EXPECT_EQ( A11.NNZ(), 4 );
    // A12, A21, A22 should be empty
    EXPECT_EQ( A12.NNZ(), 0 );
    EXPECT_EQ( A21.NNZ(), 0 );
    EXPECT_EQ( A22.NNZ(), 0 );
}

TEST_F( PartitionTest, PartitionCSRMxN_EmptyBlocks )
{
    // 4x4 matrix with some empty blocks
    // [ 1 2 0 0 ]
    // [ 3 4 0 0 ]
    // [ 0 0 0 0 ]
    // [ 0 0 0 0 ]
    std::vector<int> ai = {0, 2, 4, 4, 4};
    std::vector<int> aj = {0, 1, 0, 1};
    std::vector<double> av = {1, 2, 3, 4};
    const int rows = 4, cols = 4;
    const int base = ai[0];
    const int M = 2, N = 2;
    int row_splits[M + 1] = { base, 2 + base, 4 + base };
    int col_splits[N + 1] = { base, 2 + base, 4 + base };
    CSRMatrixVec<int, int, double> blocks[M * N];

    partitionCSRMxN<CSRMatrixVec<int, int, double>>(
        rows, cols, ai.data(), aj.data(), av.data(),
        M, row_splits, N, col_splits, blocks, /*nthreads=*/1 );

    EXPECT_EQ( blocks[0].NNZ(), 4 );
    EXPECT_EQ( blocks[1].NNZ(), 0 );
    EXPECT_EQ( blocks[2].NNZ(), 0 );
    EXPECT_EQ( blocks[3].NNZ(), 0 );
}

TEST_F( PartitionTest, PartitionCSR2x2_RealMatrices )
{
    // Test with real matrices from tests/data
    std::vector<std::string> test_matrices = {
        "data/ex5.mtx",
        "data/bcsstk17.mtx",
        "data/s3rmt3m3.mtx"
    };

    for ( const auto& matrix_file : test_matrices )
    {
        std::cout << "Testing partition with matrix: " << matrix_file << std::endl;

        std::ifstream f( matrix_file );
        if ( !f.good() )
        {
            std::cout << "  Skipping (file not found): " << matrix_file << std::endl;
            continue;
        }

        std::vector<int> ai, aj;
        std::vector<double> av;
        utils::read_matrix_market_csr( f, ai, aj, av );
        f.close();

        if ( ai.empty() || ai.size() == 1 )
        {
            std::cout << "  Skipping (empty matrix): " << matrix_file << std::endl;
            continue;
        }

        const int size = ai.size() - 1;
        const int base = ai[0];
        const int nnz = ai[size] - base;

        // Test different split positions and thread counts
        std::vector<std::pair<int, int>> splits = {
            { size / 2, size / 2 },     // Mid-mid
            { size / 3, size / 3 },     // One-third
            { 2 * size / 3, 2 * size / 3 }, // Two-thirds
            { size / 4, 3 * size / 4 }  // Asymmetric
        };
        std::vector<int> thread_counts;
        for ( int t = 1; t <= 8; ++t ) thread_counts.push_back( t );

        for ( const auto& [row_split, col_split] : splits )
        {
            if ( row_split <= 0 || row_split >= size || col_split <= 0 || col_split >= size )
                continue;

            for ( int nthreads : thread_counts )
            {
                std::cout << "  Matrix size: " << size << "x" << size
                          << ", NNZ: " << nnz
                          << ", split: (" << row_split << ", " << col_split << ")"
                          << ", threads: " << nthreads << std::endl;

                CSRMatrixVec<int, int, double> A11, A12, A21, A22;
                
                // Partition the matrix
                partitionCSR2x2<CSRMatrixVec<int, int, double>>(
                    size, size, ai.data(), aj.data(), av.data(),
                    row_split, col_split,
                    A11, A12, A21, A22, nthreads );

                // Verify dimensions
                EXPECT_EQ( A11.rows, row_split );
                EXPECT_EQ( A11.cols, col_split );
                EXPECT_EQ( A12.rows, row_split );
                EXPECT_EQ( A12.cols, size - col_split );
                EXPECT_EQ( A21.rows, size - row_split );
                EXPECT_EQ( A21.cols, col_split );
                EXPECT_EQ( A22.rows, size - row_split );
                EXPECT_EQ( A22.cols, size - col_split );

                // Verify NNZ preservation
                const int total_nnz = A11.NNZ() + A12.NNZ() + A21.NNZ() + A22.NNZ();
                EXPECT_EQ( total_nnz, nnz ) << "Total NNZ should be preserved";

                std::cout << "    A11: " << A11.rows << "x" << A11.cols << ", NNZ=" << A11.NNZ() << std::endl;
                std::cout << "    A12: " << A12.rows << "x" << A12.cols << ", NNZ=" << A12.NNZ() << std::endl;
                std::cout << "    A21: " << A21.rows << "x" << A21.cols << ", NNZ=" << A21.NNZ() << std::endl;
                std::cout << "    A22: " << A22.rows << "x" << A22.cols << ", NNZ=" << A22.NNZ() << std::endl;

                // Get pointers to all blocks
                const int* a11_ai = A11.AI();
                const int* a11_aj = A11.AJ();
                const double* a11_av = A11.AV();
                
                const int* a12_ai = A12.AI();
                const int* a12_aj = A12.AJ();
                const double* a12_av = A12.AV();
                
                const int* a21_ai = A21.AI();
                const int* a21_aj = A21.AJ();
                const double* a21_av = A21.AV();
                
                const int* a22_ai = A22.AI();
                const int* a22_aj = A22.AJ();
                const double* a22_av = A22.AV();

                // Traverse every element in the original matrix and verify it appears in the correct sub-matrix
                for ( int i = 0; i < size; i++ )
                {
                    for ( int j_idx = ai[i] - base; j_idx < ai[i + 1] - base; j_idx++ )
                    {
                        const int col = aj[j_idx];
                        const double val = av[j_idx];
                        
                        // Determine which block this element belongs to
                        const bool in_top_rows = ( i < row_split );
                        const bool in_left_cols = ( col < col_split + base );
                        
                        if ( in_top_rows && in_left_cols )
                        {
                            // Should be in A11
                            const int local_row = i;
                            const int expected_col = col;
                            
                            // Find this entry in A11
                            bool found = false;
                            for ( int k = a11_ai[local_row] - base; k < a11_ai[local_row + 1] - base; k++ )
                            {
                                if ( a11_aj[k] == expected_col )
                                {
                                    EXPECT_DOUBLE_EQ( a11_av[k], val ) 
                                        << "Value mismatch in A11 at row " << i << ", col " << col;
                                    found = true;
                                    break;
                                }
                            }
                            EXPECT_TRUE( found ) << "Element (" << i << ", " << col << ") not found in A11";
                        }
                        else if ( in_top_rows && !in_left_cols )
                        {
                            // Should be in A12
                            const int local_row = i;
                            const int expected_col = col - col_split; // Column index is shifted
                            
                            // Find this entry in A12
                            bool found = false;
                            for ( int k = a12_ai[local_row] - base; k < a12_ai[local_row + 1] - base; k++ )
                            {
                                if ( a12_aj[k] == expected_col )
                                {
                                    EXPECT_DOUBLE_EQ( a12_av[k], val ) 
                                        << "Value mismatch in A12 at row " << i << ", col " << col;
                                    found = true;
                                    break;
                                }
                            }
                            EXPECT_TRUE( found ) << "Element (" << i << ", " << col << ") not found in A12";
                        }
                        else if ( !in_top_rows && in_left_cols )
                        {
                            // Should be in A21
                            const int local_row = i - row_split; // Row index is shifted
                            const int expected_col = col;
                            
                            // Find this entry in A21
                            bool found = false;
                            for ( int k = a21_ai[local_row] - base; k < a21_ai[local_row + 1] - base; k++ )
                            {
                                if ( a21_aj[k] == expected_col )
                                {
                                    EXPECT_DOUBLE_EQ( a21_av[k], val ) 
                                        << "Value mismatch in A21 at row " << i << ", col " << col;
                                    found = true;
                                    break;
                                }
                            }
                            EXPECT_TRUE( found ) << "Element (" << i << ", " << col << ") not found in A21";
                        }
                        else
                        {
                            // Should be in A22
                            const int local_row = i - row_split; // Row index is shifted
                            const int expected_col = col - col_split; // Column index is shifted
                            
                            // Find this entry in A22
                            bool found = false;
                            for ( int k = a22_ai[local_row] - base; k < a22_ai[local_row + 1] - base; k++ )
                            {
                                if ( a22_aj[k] == expected_col )
                                {
                                    EXPECT_DOUBLE_EQ( a22_av[k], val ) 
                                        << "Value mismatch in A22 at row " << i << ", col " << col;
                                    found = true;
                                    break;
                                }
                            }
                            EXPECT_TRUE( found ) << "Element (" << i << ", " << col << ") not found in A22";
                        }
                    }
                }

                std::cout << "    ✓ All elements verified in correct sub-matrices with correct values" << std::endl;
            }
        }

        std::cout << "  ✓ All checks passed for " << matrix_file << std::endl;
    }
}

TEST_F( PartitionTest, PartitionCSRMxN_RealMatrices )
{
    std::vector<std::string> test_matrices = {
        "data/ex5.mtx",
        "data/bcsstk17.mtx",
        "data/s3rmt3m3.mtx"
    };
    constexpr int max_grids = 5;

    for ( const auto& matrix_file : test_matrices )
    {
        std::cout << "Testing CSRMxN partition with matrix: " << matrix_file << std::endl;

        std::ifstream f( matrix_file );
        if ( !f.good() )
        {
            std::cout << "  Skipping (file not found): " << matrix_file << std::endl;
            continue;
        }

        std::vector<int> ai, aj;
        std::vector<double> av;
        utils::read_matrix_market_csr( f, ai, aj, av );
        f.close();

        if ( ai.empty() || ai.size() == 1 )
        {
            std::cout << "  Skipping (empty matrix): " << matrix_file << std::endl;
            continue;
        }

        const int size = static_cast<int>( ai.size() - 1 );
        const int base = ai[0];
        const int nnz = ai[size] - base;

        std::vector<std::pair<int, int>> grid_shapes;
        for ( int m = 1; m <= max_grids; ++m )
        {
            for ( int n = 1; n <= max_grids; ++n )
            {
                grid_shapes.emplace_back( m, n );
            }
        }
        std::vector<int> thread_counts;
        for ( int t = 1; t <= 4; ++t ) thread_counts.push_back( t );

        for ( const auto& [M, N] : grid_shapes )
        {
            if ( M <= 0 || N <= 0 ) continue;

            std::vector<int> row_splits( static_cast<size_t>( M + 1 ) );
            std::vector<int> col_splits( static_cast<size_t>( N + 1 ) );

            for ( int i = 0; i <= M; ++i )
            {
                row_splits[i] = base + static_cast<int>( ( static_cast<long long>( size ) * i ) / M );
            }
            row_splits.back() = size + base;

            for ( int j = 0; j <= N; ++j )
            {
                col_splits[j] = base + static_cast<int>( ( static_cast<long long>( size ) * j ) / N );
            }
            col_splits.back() = size + base;

            for ( int nthreads : thread_counts )
            {

                std::vector<CSRMatrixVec<int, int, double>> blocks( static_cast<size_t>( M * N ) );

                partitionCSRMxN<CSRMatrixVec<int, int, double>>(
                    size, size, ai.data(), aj.data(), av.data(),
                    M, row_splits.data(), N, col_splits.data(), blocks.data(), nthreads );

                int total_nnz = 0;
                for ( int idx = 0; idx < M * N; ++idx )
                {
                    total_nnz += blocks[idx].NNZ();
                }
                EXPECT_EQ( total_nnz, nnz );

                std::vector<int> combined_ai( size + 1, base );
                std::vector<int> combined_aj;
                std::vector<double> combined_av;

                for ( int rb = 0; rb < M; ++rb )
                {
                    const int global_row_start = row_splits[rb] - base;
                    const int block_rows = row_splits[rb + 1] - row_splits[rb];
                    for ( int local_r = 0; local_r < block_rows; ++local_r )
                    {
                        const int global_r = global_row_start + local_r;
                        for ( int cb = 0; cb < N; ++cb )
                        {
                            const int col_shift = col_splits[cb] - base;
                            auto& blk = blocks[rb * N + cb];
                            auto* ai_b = blk.AI();
                            auto* aj_b = blk.AJ();
                            auto* av_b = blk.AV();

                            const int start = ai_b[local_r] - base;
                            const int end   = ai_b[local_r + 1] - base;
                            for ( int k = start; k < end; ++k )
                            {
                                combined_aj.push_back( aj_b[k] + col_shift );
                                combined_av.push_back( av_b[k] );
                            }
                        }
                        combined_ai[global_r + 1] = static_cast<int>( combined_aj.size() ) + base;
                    }
                }

                ASSERT_EQ( combined_ai, ai );
                ASSERT_EQ( combined_aj, aj );
                ASSERT_EQ( combined_av, av );

                // Verify every element is placed into the correct block with correct value
                for ( int i = 0; i < size; ++i )
                {
                    for ( int j_idx = ai[i] - base; j_idx < ai[i + 1] - base; ++j_idx )
                    {
                        const int col = aj[j_idx];
                        const double val = av[j_idx];

                        const int rb = static_cast<int>(
                            std::upper_bound( row_splits.begin(), row_splits.end(), i + base )
                            - row_splits.begin() - 1 );
                        const int cb = static_cast<int>(
                            std::upper_bound( col_splits.begin(), col_splits.end(), col )
                            - col_splits.begin() - 1 );

                        auto& blk = blocks[rb * N + cb];
                        auto* ai_b = blk.AI();
                        auto* aj_b = blk.AJ();
                        auto* av_b = blk.AV();

                        const int local_row = i - ( row_splits[rb] - base );
                        const int expected_col = col - ( col_splits[cb] - base );

                        bool found = false;
                        for ( int k = ai_b[local_row] - base; k < ai_b[local_row + 1] - base; ++k )
                        {
                            if ( aj_b[k] == expected_col )
                            {
                                EXPECT_DOUBLE_EQ( av_b[k], val );
                                found = true;
                                break;
                            }
                        }
                        EXPECT_TRUE( found ) << "Element (" << i << "," << col
                                             << ") not found in block (" << rb << "," << cb
                                             << ") for grid " << M << "x" << N;
                    }
                }
            }
        }
    }
}

TEST(Block, Submatrix)
{
    for (int nthreads = 1; nthreads <= 4; nthreads++)
    {
        omp_set_num_threads(nthreads);
        constexpr int rows = 1000;
        constexpr int nnz_per_row = 15;

        matrix_utils::CSRMatrix<int, int, double> mat;
        mat.rows = rows;
        mat.cols = rows;
        mat.ResizeAI(rows + 1);
        mat.ResizeAJ(static_cast<std::size_t>(rows) * nnz_per_row);
        mat.ResizeAV(static_cast<std::size_t>(rows) * nnz_per_row);

        for (int iter = 0; iter < 4; iter++)
        {
            const int base = (iter % 2 == 0) ? 1 : 0;
            auto* ai = mat.AI();
            ai[0] = base;
            for (int r = 0; r < rows; ++r)
            {
                ai[r + 1] = ai[r] + nnz_per_row;
            }
            matrix_utils::RandomCSR(rows, rows, mat.AI(), mat.AJ(), mat.AV());

            const int start_row = 20;
            const int start_col = 32;
            const int p  = 123;
            const int q = 234;
            matrix_utils::CSRMatrix<int, int, double> block;
            matrix_utils::Block(mat.rows, mat.Base(), mat.AI(), mat.AJ(), mat.AV(), start_row,
                                start_col, p, q, block);

            auto* aj = mat.AJ();
            auto* av = mat.AV();
            EXPECT_EQ(block.rows, p);
            EXPECT_EQ(block.cols, q);
            EXPECT_EQ(block.Base(), base);

            for (int i = 0; i < block.rows; i++)
            {
                if (block.ai[i] != block.ai[i + 1])
                {
                    EXPECT_LT(block.aj[block.ai[i + 1] - 1 - block.Base()] - block.Base(), q);
                }
                auto* it = std::lower_bound(aj + ai[start_row + i] - base,
                                            aj + ai[start_row + i + 1] - base, start_col + base);
                for (int j = block.ai[i] - base; j < block.ai[i + 1] - base; j++)
                {
                    EXPECT_EQ(block.aj[j], *it - start_col);
                    EXPECT_EQ(block.av[j], av[it - aj]);
                    it++;
                }
            }
        }
    }
}

// SpADD Tests
class SpADDTest : public testing::Test
{
protected:
    const double _tol = 1e-10;

    // Helper function to verify CSR matrix validity
    template <typename CSRMatrixType>
    void verifyCsrMatrix( const CSRMatrixType& mat )
    {
        using ROWTYPE = typename CSRMatrixType::ROWTYPE;
        using COLTYPE = typename CSRMatrixType::COLTYPE;
        
        const ROWTYPE* ai = mat.AI();
        const COLTYPE* aj = mat.AJ();
        const ROWTYPE base = ai[0];

        ASSERT_EQ( ai[0], base ) << "Row pointer should start with base";

        for ( COLTYPE i = 0; i < mat.rows; i++ )
        {
            ASSERT_LE( ai[i], ai[i + 1] ) << "Row pointers must be non-decreasing";

            // Check column indices are sorted
            for ( ROWTYPE j = ai[i] - base; j < ai[i + 1] - base - 1; j++ )
            {
                ASSERT_LT( aj[j], aj[j + 1] ) << "Column indices must be sorted and unique";
            }
        }
    }

    // Helper function to compute C = alpha*A + beta*B naively
    template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
    CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE> computeSpAddNaive(
        const COLTYPE A_rows, const COLTYPE A_cols,
        const ROWTYPE* A_ai, const COLTYPE* A_aj, const VALTYPE* A_av, const VALTYPE alpha,
        const COLTYPE B_rows, const COLTYPE B_cols,
        const ROWTYPE* B_ai, const COLTYPE* B_aj, const VALTYPE* B_av, const VALTYPE beta )
    {
        EXPECT_EQ( A_rows, B_rows );
        EXPECT_EQ( A_cols, B_cols );

        const ROWTYPE A_base = A_ai[0];
        const ROWTYPE B_base = B_ai[0];

        CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE> result;
        result.rows = A_rows;
        result.cols = A_cols;
        result.ai.resize( A_rows + 1 );
        result.ai[0] = A_base;

        // Use map for each row
        std::map<COLTYPE, VALTYPE> row_map;

        for ( COLTYPE i = 0; i < A_rows; i++ )
        {
            row_map.clear();

            // Add entries from A
            for ( ROWTYPE ja = A_ai[i] - A_base; ja < A_ai[i + 1] - A_base; ja++ )
            {
                const COLTYPE col = A_aj[ja];
                row_map[col] += alpha * A_av[ja];
            }

            // Add entries from B
            for ( ROWTYPE jb = B_ai[i] - B_base; jb < B_ai[i + 1] - B_base; jb++ )
            {
                const COLTYPE col = B_aj[jb];
                row_map[col] += beta * B_av[jb];
            }

            // Write to result
            for ( const auto& [col, val] : row_map )
            {
                // Don't skip near-zero entries to match SpADD behavior
                result.aj.push_back( col );
                result.av.push_back( val );
            }

            result.ai[i + 1] = result.ai[0] + static_cast<ROWTYPE>( result.aj.size() );
        }

        return result;
    }

    // Compare two CSR matrices
    template <typename CSRMatrixType>
    void compareCsrMatrices( const CSRMatrixType& expected, const CSRMatrixType& actual, const std::string& test_name )
    {
        using ROWTYPE = typename CSRMatrixType::ROWTYPE;
        using COLTYPE = typename CSRMatrixType::COLTYPE;
        using VALTYPE = typename CSRMatrixType::VALTYPE;

        ASSERT_EQ( expected.rows, actual.rows ) << test_name << ": Row count mismatch";
        ASSERT_EQ( expected.cols, actual.cols ) << test_name << ": Column count mismatch";

        const ROWTYPE* exp_ai = expected.AI();
        const COLTYPE* exp_aj = expected.AJ();
        const VALTYPE* exp_av = expected.AV();

        const ROWTYPE* act_ai = actual.AI();
        const COLTYPE* act_aj = actual.AJ();
        const VALTYPE* act_av = actual.AV();

        const ROWTYPE base = exp_ai[0];
        ASSERT_EQ( base, act_ai[0] ) << test_name << ": Base mismatch";

        for ( COLTYPE i = 0; i <= expected.rows; i++ )
        {
            ASSERT_EQ( exp_ai[i], act_ai[i] ) << test_name << ": Row pointer mismatch at row " << i;
        }

        const ROWTYPE nnz = exp_ai[expected.rows] - base;
        for ( ROWTYPE j = 0; j < nnz; j++ )
        {
            ASSERT_EQ( exp_aj[j], act_aj[j] ) << test_name << ": Column index mismatch at position " << j;
            EXPECT_NEAR( exp_av[j], act_av[j], _tol ) << test_name << ": Value mismatch at position " << j;
        }
    }
};

// Test basic addition with small matrices
TEST_F( SpADDTest, SmallMatrix_ZeroBased )
{
    // 3x3 matrices:
    // A = [1 2 0]    B = [0 1 0]
    //     [0 3 4]        [2 0 0]
    //     [0 0 5]        [0 3 1]
    const int32_t rows = 3, cols = 3;
    const int32_t base = 0;

    std::vector<int32_t> A_ai = { 0, 2, 4, 5 };
    std::vector<int32_t> A_aj = { 0, 1, 1, 2, 2 };
    std::vector<double> A_av = { 1, 2, 3, 4, 5 };

    std::vector<int32_t> B_ai = { 0, 1, 2, 4 };
    std::vector<int32_t> B_aj = { 1, 0, 1, 2 };
    std::vector<double> B_av = { 1, 2, 3, 1 };

    const double alpha = 1.0, beta = 1.0;

    // Compute expected result
    auto expected = computeSpAddNaive( rows, cols,
                                       A_ai.data(), A_aj.data(), A_av.data(), alpha,
                                       rows, cols,
                                       B_ai.data(), B_aj.data(), B_av.data(), beta );

    // Test with SpADD
    SpADD<CSRMatrixVec<int32_t, int32_t, double>> spadd( 1 );
    CSRMatrixVec<int32_t, int32_t, double> C;

    // Analysis phase
    spadd.analysis( rows, cols, A_ai.data(), A_aj.data(),
                    rows, cols, B_ai.data(), B_aj.data(), C );

    verifyCsrMatrix( C );

    // Numerical phase
    spadd( rows, cols, A_ai.data(), A_aj.data(), A_av.data(), alpha,
           rows, cols, B_ai.data(), B_aj.data(), B_av.data(), beta, C );

    verifyCsrMatrix( C );
    compareCsrMatrices( expected, C, "SmallMatrix_ZeroBased" );
}

// Test with 1-based indexing
TEST_F( SpADDTest, SmallMatrix_OneBased )
{
    const int32_t rows = 3, cols = 3;
    const int32_t base = 1;

    std::vector<int32_t> A_ai = { 1, 3, 5, 6 };
    std::vector<int32_t> A_aj = { 1, 2, 2, 3, 3 };
    std::vector<double> A_av = { 1, 2, 3, 4, 5 };

    std::vector<int32_t> B_ai = { 1, 2, 3, 5 };
    std::vector<int32_t> B_aj = { 2, 1, 2, 3 };
    std::vector<double> B_av = { 1, 2, 3, 1 };

    const double alpha = 1.0, beta = 1.0;

    auto expected = computeSpAddNaive( rows, cols,
                                       A_ai.data(), A_aj.data(), A_av.data(), alpha,
                                       rows, cols,
                                       B_ai.data(), B_aj.data(), B_av.data(), beta );

    SpADD<CSRMatrixVec<int32_t, int32_t, double>> spadd( 1 );
    CSRMatrixVec<int32_t, int32_t, double> C;

    spadd.analysis( rows, cols, A_ai.data(), A_aj.data(),
                    rows, cols, B_ai.data(), B_aj.data(), C );
    spadd( rows, cols, A_ai.data(), A_aj.data(), A_av.data(), alpha,
           rows, cols, B_ai.data(), B_aj.data(), B_av.data(), beta, C );

    verifyCsrMatrix( C );
    compareCsrMatrices( expected, C, "SmallMatrix_OneBased" );
}

// Test with different alpha and beta scalars
TEST_F( SpADDTest, DifferentScalars )
{
    const int32_t rows = 2, cols = 2;
    std::vector<int32_t> A_ai = { 0, 2, 3 };
    std::vector<int32_t> A_aj = { 0, 1, 1 };
    std::vector<double> A_av = { 2, 3, 4 };

    std::vector<int32_t> B_ai = { 0, 1, 3 };
    std::vector<int32_t> B_aj = { 0, 0, 1 };
    std::vector<double> B_av = { 5, 6, 7 };

    const double alpha = 2.0, beta = -1.0;

    auto expected = computeSpAddNaive( rows, cols,
                                       A_ai.data(), A_aj.data(), A_av.data(), alpha,
                                       rows, cols,
                                       B_ai.data(), B_aj.data(), B_av.data(), beta );

    SpADD<CSRMatrixVec<int32_t, int32_t, double>> spadd( 1 );
    CSRMatrixVec<int32_t, int32_t, double> C;

    spadd.analysis( rows, cols, A_ai.data(), A_aj.data(),
                    rows, cols, B_ai.data(), B_aj.data(), C );
    spadd( rows, cols, A_ai.data(), A_aj.data(), A_av.data(), alpha,
           rows, cols, B_ai.data(), B_aj.data(), B_av.data(), beta, C );

    verifyCsrMatrix( C );
    compareCsrMatrices( expected, C, "DifferentScalars" );
}

// Test with disjoint sparsity patterns
TEST_F( SpADDTest, DisjointPatterns )
{
    const int32_t rows = 3, cols = 3;
    
    // A has entries in upper triangle
    std::vector<int32_t> A_ai = { 0, 2, 3, 3 };
    std::vector<int32_t> A_aj = { 0, 1, 2 };
    std::vector<double> A_av = { 1, 2, 3 };

    // B has entries in lower triangle
    std::vector<int32_t> B_ai = { 0, 0, 1, 2 };
    std::vector<int32_t> B_aj = { 0, 1 };
    std::vector<double> B_av = { 4, 5 };

    const double alpha = 1.0, beta = 1.0;

    auto expected = computeSpAddNaive( rows, cols,
                                       A_ai.data(), A_aj.data(), A_av.data(), alpha,
                                       rows, cols,
                                       B_ai.data(), B_aj.data(), B_av.data(), beta );

    SpADD<CSRMatrixVec<int32_t, int32_t, double>> spadd( 1 );
    CSRMatrixVec<int32_t, int32_t, double> C;

    spadd.analysis( rows, cols, A_ai.data(), A_aj.data(),
                    rows, cols, B_ai.data(), B_aj.data(), C );
    spadd( rows, cols, A_ai.data(), A_aj.data(), A_av.data(), alpha,
           rows, cols, B_ai.data(), B_aj.data(), B_av.data(), beta, C );

    verifyCsrMatrix( C );
    compareCsrMatrices( expected, C, "DisjointPatterns" );
}

// Test with identical sparsity patterns
TEST_F( SpADDTest, IdenticalPatterns )
{
    const int32_t rows = 3, cols = 3;
    
    std::vector<int32_t> A_ai = { 0, 2, 4, 5 };
    std::vector<int32_t> A_aj = { 0, 1, 1, 2, 2 };
    std::vector<double> A_av = { 1, 2, 3, 4, 5 };

    std::vector<int32_t> B_ai = { 0, 2, 4, 5 };
    std::vector<int32_t> B_aj = { 0, 1, 1, 2, 2 };
    std::vector<double> B_av = { 2, 3, 4, 5, 6 };

    const double alpha = 1.0, beta = 1.0;

    auto expected = computeSpAddNaive( rows, cols,
                                       A_ai.data(), A_aj.data(), A_av.data(), alpha,
                                       rows, cols,
                                       B_ai.data(), B_aj.data(), B_av.data(), beta );

    SpADD<CSRMatrixVec<int32_t, int32_t, double>> spadd( 1 );
    CSRMatrixVec<int32_t, int32_t, double> C;

    spadd.analysis( rows, cols, A_ai.data(), A_aj.data(),
                    rows, cols, B_ai.data(), B_aj.data(), C );
    spadd( rows, cols, A_ai.data(), A_aj.data(), A_av.data(), alpha,
           rows, cols, B_ai.data(), B_aj.data(), B_av.data(), beta, C );

    verifyCsrMatrix( C );
    compareCsrMatrices( expected, C, "IdenticalPatterns" );
}

// Test with empty matrices
TEST_F( SpADDTest, EmptyMatrices )
{
    const int32_t rows = 3, cols = 3;
    
    std::vector<int32_t> A_ai = { 0, 0, 0, 0 };
    std::vector<int32_t> A_aj = {};
    std::vector<double> A_av = {};

    std::vector<int32_t> B_ai = { 0, 0, 0, 0 };
    std::vector<int32_t> B_aj = {};
    std::vector<double> B_av = {};

    const double alpha = 1.0, beta = 1.0;

    SpADD<CSRMatrixVec<int32_t, int32_t, double>> spadd( 1 );
    CSRMatrixVec<int32_t, int32_t, double> C;

    spadd.analysis( rows, cols, A_ai.data(), A_aj.data(),
                    rows, cols, B_ai.data(), B_aj.data(), C );
    spadd( rows, cols, A_ai.data(), A_aj.data(), A_av.data(), alpha,
           rows, cols, B_ai.data(), B_aj.data(), B_av.data(), beta, C );

    verifyCsrMatrix( C );
    EXPECT_EQ( C.NNZ(), 0 );
}

// Test with one empty matrix
TEST_F( SpADDTest, OneEmptyMatrix )
{
    const int32_t rows = 2, cols = 2;
    
    std::vector<int32_t> A_ai = { 0, 0, 0 };
    std::vector<int32_t> A_aj = {};
    std::vector<double> A_av = {};

    std::vector<int32_t> B_ai = { 0, 1, 3 };
    std::vector<int32_t> B_aj = { 0, 0, 1 };
    std::vector<double> B_av = { 1, 2, 3 };

    const double alpha = 1.0, beta = 2.0;

    auto expected = computeSpAddNaive( rows, cols,
                                       A_ai.data(), A_aj.data(), A_av.data(), alpha,
                                       rows, cols,
                                       B_ai.data(), B_aj.data(), B_av.data(), beta );

    SpADD<CSRMatrixVec<int32_t, int32_t, double>> spadd( 1 );
    CSRMatrixVec<int32_t, int32_t, double> C;

    spadd.analysis( rows, cols, A_ai.data(), A_aj.data(),
                    rows, cols, B_ai.data(), B_aj.data(), C );
    spadd( rows, cols, A_ai.data(), A_aj.data(), A_av.data(), alpha,
           rows, cols, B_ai.data(), B_aj.data(), B_av.data(), beta, C );

    verifyCsrMatrix( C );
    compareCsrMatrices( expected, C, "OneEmptyMatrix" );
}

// Test with different thread counts
TEST_F( SpADDTest, MultipleThreads )
{
    const int32_t rows = 4, cols = 4;
    
    std::vector<int32_t> A_ai = { 0, 2, 4, 6, 7 };
    std::vector<int32_t> A_aj = { 0, 1, 1, 2, 2, 3, 3 };
    std::vector<double> A_av = { 1, 2, 3, 4, 5, 6, 7 };

    std::vector<int32_t> B_ai = { 0, 1, 2, 4, 6 };
    std::vector<int32_t> B_aj = { 0, 1, 0, 2, 1, 3 };
    std::vector<double> B_av = { 8, 9, 10, 11, 12, 13 };

    const double alpha = 1.5, beta = -0.5;

    auto expected = computeSpAddNaive( rows, cols,
                                       A_ai.data(), A_aj.data(), A_av.data(), alpha,
                                       rows, cols,
                                       B_ai.data(), B_aj.data(), B_av.data(), beta );

    for ( int nthreads : { 1, 2, 4, 8 } )
    {
        SpADD<CSRMatrixVec<int32_t, int32_t, double>> spadd( nthreads );
        CSRMatrixVec<int32_t, int32_t, double> C;

        spadd.analysis( rows, cols, A_ai.data(), A_aj.data(),
                        rows, cols, B_ai.data(), B_aj.data(), C );
        spadd( rows, cols, A_ai.data(), A_aj.data(), A_av.data(), alpha,
               rows, cols, B_ai.data(), B_aj.data(), B_av.data(), beta, C );

        verifyCsrMatrix( C );
        compareCsrMatrices( expected, C, "MultipleThreads_" + std::to_string( nthreads ) );
    }
}

// Test setNumThreads
TEST_F( SpADDTest, SetNumThreads )
{
    const int32_t rows = 3, cols = 3;
    
    std::vector<int32_t> A_ai = { 0, 2, 4, 5 };
    std::vector<int32_t> A_aj = { 0, 1, 1, 2, 2 };
    std::vector<double> A_av = { 1, 2, 3, 4, 5 };

    std::vector<int32_t> B_ai = { 0, 1, 2, 4 };
    std::vector<int32_t> B_aj = { 1, 0, 1, 2 };
    std::vector<double> B_av = { 1, 2, 3, 1 };

    const double alpha = 1.0, beta = 1.0;

    SpADD<CSRMatrixVec<int32_t, int32_t, double>> spadd( 2 );
    CSRMatrixVec<int32_t, int32_t, double> C1, C2;

    spadd.analysis( rows, cols, A_ai.data(), A_aj.data(),
                    rows, cols, B_ai.data(), B_aj.data(), C1 );
    spadd( rows, cols, A_ai.data(), A_aj.data(), A_av.data(), alpha,
           rows, cols, B_ai.data(), B_aj.data(), B_av.data(), beta, C1 );

    // Change thread count
    spadd.setNumThreads( 4 );

    spadd.analysis( rows, cols, A_ai.data(), A_aj.data(),
                    rows, cols, B_ai.data(), B_aj.data(), C2 );
    spadd( rows, cols, A_ai.data(), A_aj.data(), A_av.data(), alpha,
           rows, cols, B_ai.data(), B_aj.data(), B_av.data(), beta, C2 );

    verifyCsrMatrix( C1 );
    verifyCsrMatrix( C2 );
    compareCsrMatrices( C1, C2, "SetNumThreads" );
}

// Test with int64_t types
TEST_F( SpADDTest, Int64Types )
{
    const int64_t rows = 3, cols = 3;
    
    std::vector<int64_t> A_ai = { 0, 2, 4, 5 };
    std::vector<int64_t> A_aj = { 0, 1, 1, 2, 2 };
    std::vector<double> A_av = { 1, 2, 3, 4, 5 };

    std::vector<int64_t> B_ai = { 0, 1, 2, 4 };
    std::vector<int64_t> B_aj = { 1, 0, 1, 2 };
    std::vector<double> B_av = { 1, 2, 3, 1 };

    const double alpha = 1.0, beta = 1.0;

    auto expected = computeSpAddNaive( rows, cols,
                                       A_ai.data(), A_aj.data(), A_av.data(), alpha,
                                       rows, cols,
                                       B_ai.data(), B_aj.data(), B_av.data(), beta );

    SpADD<CSRMatrixVec<int64_t, int64_t, double>> spadd( 1 );
    CSRMatrixVec<int64_t, int64_t, double> C;

    spadd.analysis( rows, cols, A_ai.data(), A_aj.data(),
                    rows, cols, B_ai.data(), B_aj.data(), C );
    spadd( rows, cols, A_ai.data(), A_aj.data(), A_av.data(), alpha,
           rows, cols, B_ai.data(), B_aj.data(), B_av.data(), beta, C );

    verifyCsrMatrix( C );
    compareCsrMatrices( expected, C, "Int64Types" );
}

// Test with float types
TEST_F( SpADDTest, FloatTypes )
{
    const int32_t rows = 2, cols = 2;
    
    std::vector<int32_t> A_ai = { 0, 2, 3 };
    std::vector<int32_t> A_aj = { 0, 1, 1 };
    std::vector<float> A_av = { 1.5f, 2.5f, 3.5f };

    std::vector<int32_t> B_ai = { 0, 1, 3 };
    std::vector<int32_t> B_aj = { 0, 0, 1 };
    std::vector<float> B_av = { 4.5f, 5.5f, 6.5f };

    const float alpha = 2.0f, beta = -1.0f;

    auto expected = computeSpAddNaive( rows, cols,
                                       A_ai.data(), A_aj.data(), A_av.data(), alpha,
                                       rows, cols,
                                       B_ai.data(), B_aj.data(), B_av.data(), beta );

    SpADD<CSRMatrixVec<int32_t, int32_t, float>> spadd( 1 );
    CSRMatrixVec<int32_t, int32_t, float> C;

    spadd.analysis( rows, cols, A_ai.data(), A_aj.data(),
                    rows, cols, B_ai.data(), B_aj.data(), C );
    spadd( rows, cols, A_ai.data(), A_aj.data(), A_av.data(), alpha,
           rows, cols, B_ai.data(), B_aj.data(), B_av.data(), beta, C );

    verifyCsrMatrix( C );
    compareCsrMatrices( expected, C, "FloatTypes" );
}

// Test with larger matrices from file
TEST_F( SpADDTest, LargerMatrices )
{
    std::vector<std::string> matrix_names = { "ex5", "bcsstk17", "s3rmt3m3" };

    for ( const auto& name : matrix_names )
    {
        std::string filepath = "data/" + name + ".mtx";
        std::ifstream f( filepath );
        if ( !f.good() )
        {
            std::cout << "Skipping matrix " << name << " (file not found)" << std::endl;
            continue;
        }

        std::vector<int32_t> A_ai, A_aj;
        std::vector<double> A_av;
        utils::read_matrix_market_csr( f, A_ai, A_aj, A_av );
        f.close();

        if ( A_ai.size() == 0 )
        {
            std::cout << "Skipping matrix " << name << " (could not read)" << std::endl;
            continue;
        }

        const int32_t rows = A_ai.size() - 1;
        const int32_t cols = rows; // Assume square

        // Use same matrix for A and B with different scalars
        const double alpha = 1.5, beta = -0.5;

        std::cout << "\nTesting SpADD with matrix: " << name
                  << " (size=" << rows << ", nnz=" << A_ai[rows] - A_ai[0] << ")" << std::endl;

        // Compute expected result
        auto expected = computeSpAddNaive( rows, cols,
                                           A_ai.data(), A_aj.data(), A_av.data(), alpha,
                                           rows, cols,
                                           A_ai.data(), A_aj.data(), A_av.data(), beta );

        // Test with different thread counts
        for ( int nthreads : { 1, 2, 4, 8 } )
        {
            std::cout << "  Testing with " << nthreads << " thread(s)..." << std::endl;

            SpADD<CSRMatrixVec<int32_t, int32_t, double>> spadd( nthreads );
            CSRMatrixVec<int32_t, int32_t, double> C;

            spadd.analysis( rows, cols, A_ai.data(), A_aj.data(),
                            rows, cols, A_ai.data(), A_aj.data(), C );
            spadd( rows, cols, A_ai.data(), A_aj.data(), A_av.data(), alpha,
                   rows, cols, A_ai.data(), A_aj.data(), A_av.data(), beta, C );

            verifyCsrMatrix( C );
            compareCsrMatrices( expected, C, name + "_" + std::to_string( nthreads ) + "threads" );

            std::cout << "    Result nnz=" << C.NNZ() << ", passed" << std::endl;
        }
    }
}

// Test correctness with various patterns
TEST_F( SpADDTest, CorrectnessCheck_Various )
{
    struct TestCase
    {
        std::string name;
        std::vector<int32_t> A_ai, A_aj, B_ai, B_aj;
        std::vector<double> A_av, B_av;
        int32_t rows, cols;
        double alpha, beta;
    };

    std::vector<TestCase> cases = {
        {
            "Dense_2x2",
            { 0, 2, 4 }, { 0, 1, 0, 1 }, { 0, 2, 4 }, { 0, 1, 0, 1 },
            { 1, 2, 3, 4 }, { 5, 6, 7, 8 },
            2, 2, 1.0, 1.0
        },
        {
            "Diagonal",
            { 0, 1, 2, 3 }, { 0, 1, 2 }, { 0, 1, 2, 3 }, { 0, 1, 2 },
            { 1, 2, 3 }, { 4, 5, 6 },
            3, 3, 2.0, -1.0
        },
        {
            "UpperLower",
            { 0, 2, 3, 3 }, { 0, 1, 2 }, { 0, 0, 1, 2 }, { 0, 1 },
            { 1, 2, 3 }, { 4, 5 },
            3, 3, 1.0, 1.0
        },
        {
            "Cancellation",
            { 0, 1, 2 }, { 0, 1 }, { 0, 1, 2 }, { 0, 1 },
            { 5, 10 }, { 5, 10 },
            2, 2, 1.0, -1.0
        }
    };

    for ( const auto& tc : cases )
    {
        auto expected = computeSpAddNaive( tc.rows, tc.cols,
                                           tc.A_ai.data(), tc.A_aj.data(), tc.A_av.data(), tc.alpha,
                                           tc.rows, tc.cols,
                                           tc.B_ai.data(), tc.B_aj.data(), tc.B_av.data(), tc.beta );

        SpADD<CSRMatrixVec<int32_t, int32_t, double>> spadd( 1 );
        CSRMatrixVec<int32_t, int32_t, double> C;

        spadd.analysis( tc.rows, tc.cols, tc.A_ai.data(), tc.A_aj.data(),
                        tc.rows, tc.cols, tc.B_ai.data(), tc.B_aj.data(), C );
        spadd( tc.rows, tc.cols, tc.A_ai.data(), tc.A_aj.data(), tc.A_av.data(), tc.alpha,
               tc.rows, tc.cols, tc.B_ai.data(), tc.B_aj.data(), tc.B_av.data(), tc.beta, C );

        verifyCsrMatrix( C );
        compareCsrMatrices( expected, C, tc.name );
    }
}

// Jaccard Similarity Tests
class JaccardSimilarityTest : public testing::Test
{
protected:
    const double _tol = 1e-10;
};

// Test identical matrices (should give 1.0)
TEST_F( JaccardSimilarityTest, IdenticalMatrices )
{
    const int32_t rows = 3, cols = 3;
    
    std::vector<int32_t> ai = { 0, 2, 4, 5 };
    std::vector<int32_t> aj = { 0, 1, 1, 2, 2 };

    double similarity = jaccardSimilarity( rows, cols, ai.data(), aj.data(),
                                           rows, cols, ai.data(), aj.data(), 1 );

    EXPECT_NEAR( similarity, 1.0, _tol ) << "Identical matrices should have Jaccard similarity of 1.0";
}

// Test completely disjoint matrices (should give 0.0)
TEST_F( JaccardSimilarityTest, DisjointMatrices )
{
    const int32_t rows = 3, cols = 3;
    
    // A has entries in upper triangle
    std::vector<int32_t> A_ai = { 0, 2, 3, 3 };
    std::vector<int32_t> A_aj = { 0, 1, 2 };

    // B has entries in lower triangle
    std::vector<int32_t> B_ai = { 0, 0, 1, 2 };
    std::vector<int32_t> B_aj = { 0, 1 };

    double similarity = jaccardSimilarity( rows, cols, A_ai.data(), A_aj.data(),
                                           rows, cols, B_ai.data(), B_aj.data(), 1 );

    EXPECT_NEAR( similarity, 0.0, _tol ) << "Disjoint matrices should have Jaccard similarity of 0.0";
}

// Test partial overlap
TEST_F( JaccardSimilarityTest, PartialOverlap )
{
    const int32_t rows = 3, cols = 3;
    
    // A = [1 1 0]
    //     [0 1 1]
    //     [0 0 1]
    std::vector<int32_t> A_ai = { 0, 2, 4, 5 };
    std::vector<int32_t> A_aj = { 0, 1, 1, 2, 2 };

    // B = [0 1 0]
    //     [1 0 0]
    //     [0 1 1]
    std::vector<int32_t> B_ai = { 0, 1, 2, 4 };
    std::vector<int32_t> B_aj = { 1, 0, 1, 2 };

    // Intersection: {(0,1), (2,2)} = 2 entries
    // Union: {(0,0), (0,1), (1,0), (1,1), (1,2), (2,1), (2,2)} = 7 entries
    // Jaccard = 2/7
    double expected = 2.0 / 7.0;

    double similarity = jaccardSimilarity( rows, cols, A_ai.data(), A_aj.data(),
                                           rows, cols, B_ai.data(), B_aj.data(), 1 );

    EXPECT_NEAR( similarity, expected, _tol ) << "Partial overlap should give correct Jaccard similarity";
}

// Test with one empty matrix
TEST_F( JaccardSimilarityTest, OneEmptyMatrix )
{
    const int32_t rows = 2, cols = 2;
    
    std::vector<int32_t> A_ai = { 0, 0, 0 };
    std::vector<int32_t> A_aj = {};

    std::vector<int32_t> B_ai = { 0, 1, 3 };
    std::vector<int32_t> B_aj = { 0, 0, 1 };

    double similarity = jaccardSimilarity( rows, cols, A_ai.data(), A_aj.data(),
                                           rows, cols, B_ai.data(), B_aj.data(), 1 );

    EXPECT_NEAR( similarity, 0.0, _tol ) << "One empty matrix should give Jaccard similarity of 0.0";
}

// Test with both empty matrices
TEST_F( JaccardSimilarityTest, BothEmptyMatrices )
{
    const int32_t rows = 2, cols = 2;
    
    std::vector<int32_t> A_ai = { 0, 0, 0 };
    std::vector<int32_t> A_aj = {};

    std::vector<int32_t> B_ai = { 0, 0, 0 };
    std::vector<int32_t> B_aj = {};

    double similarity = jaccardSimilarity( rows, cols, A_ai.data(), A_aj.data(),
                                           rows, cols, B_ai.data(), B_aj.data(), 1 );

    EXPECT_NEAR( similarity, 1.0, _tol ) << "Both empty matrices should give Jaccard similarity of 1.0";
}

// Test with 1-based indexing
TEST_F( JaccardSimilarityTest, OneBasedIndexing )
{
    const int32_t rows = 3, cols = 3;
    
    std::vector<int32_t> A_ai = { 1, 3, 5, 6 };
    std::vector<int32_t> A_aj = { 1, 2, 2, 3, 3 };

    std::vector<int32_t> B_ai = { 1, 2, 3, 5 };
    std::vector<int32_t> B_aj = { 2, 1, 2, 3 };

    // Intersection: {(0,1), (2,2)} = 2 entries
    // Union: {(0,0), (0,1), (1,0), (1,1), (1,2), (2,1), (2,2)} = 7 entries
    // Jaccard = 2/7
    double expected = 2.0 / 7.0;

    double similarity = jaccardSimilarity( rows, cols, A_ai.data(), A_aj.data(),
                                           rows, cols, B_ai.data(), B_aj.data(), 1 );

    EXPECT_NEAR( similarity, expected, _tol ) << "One-based indexing should work correctly";
}

// Test with int64_t types
TEST_F( JaccardSimilarityTest, Int64Types )
{
    const int64_t rows = 2, cols = 2;
    
    std::vector<int64_t> A_ai = { 0, 2, 3 };
    std::vector<int64_t> A_aj = { 0, 1, 1 };

    std::vector<int64_t> B_ai = { 0, 1, 3 };
    std::vector<int64_t> B_aj = { 0, 0, 1 };

    // Intersection: {(0,0), (1,1)} = 2 entries
    // Union: {(0,0), (0,1), (1,0), (1,1)} = 4 entries
    // Jaccard = 2/4 = 0.5
    double expected = 0.5;

    double similarity = jaccardSimilarity( rows, cols, A_ai.data(), A_aj.data(),
                                           rows, cols, B_ai.data(), B_aj.data(), 1 );

    EXPECT_NEAR( similarity, expected, _tol ) << "int64_t types should work correctly";
}

// Test with multiple threads
TEST_F( JaccardSimilarityTest, MultipleThreads )
{
    const int32_t rows = 4, cols = 4;
    
    std::vector<int32_t> A_ai = { 0, 2, 4, 6, 7 };
    std::vector<int32_t> A_aj = { 0, 1, 1, 2, 2, 3, 3 };

    std::vector<int32_t> B_ai = { 0, 1, 2, 4, 6 };
    std::vector<int32_t> B_aj = { 0, 1, 0, 2, 1, 3 };

    // Compute with 1 thread as reference
    double ref_similarity = jaccardSimilarity( rows, cols, A_ai.data(), A_aj.data(),
                                               rows, cols, B_ai.data(), B_aj.data(), 1 );

    // Test with different thread counts
    for ( int nthreads : { 2, 4, 8 } )
    {
        double similarity = jaccardSimilarity( rows, cols, A_ai.data(), A_aj.data(),
                                               rows, cols, B_ai.data(), B_aj.data(), nthreads );

        EXPECT_NEAR( similarity, ref_similarity, _tol ) 
            << "Results should be consistent with " << nthreads << " threads";
    }
}

// Test with subset relationship
TEST_F( JaccardSimilarityTest, SubsetRelationship )
{
    const int32_t rows = 3, cols = 3;
    
    // A is subset of B
    std::vector<int32_t> A_ai = { 0, 1, 2, 3 };
    std::vector<int32_t> A_aj = { 0, 1, 2 };

    std::vector<int32_t> B_ai = { 0, 2, 4, 6 };
    std::vector<int32_t> B_aj = { 0, 1, 0, 1, 1, 2 };

    // Intersection: all 3 entries of A
    // Union: all 6 entries of B
    // Jaccard = 3/6 = 0.5
    double expected = 0.5;

    double similarity = jaccardSimilarity( rows, cols, A_ai.data(), A_aj.data(),
                                           rows, cols, B_ai.data(), B_aj.data(), 1 );

    EXPECT_NEAR( similarity, expected, _tol ) << "Subset relationship should give correct Jaccard similarity";
}

int main( int argc, char** argv )
{
    testing::InitGoogleTest( &argc, argv );
    return RUN_ALL_TESTS();
}
