#include "matrix_utils.hpp"
#include "sp_ops.hpp"
#include "utils.h"
#include <algorithm>
#include <fstream>
#include <gtest/gtest.h>
#include <set>

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

        // Compute expected result once (using naive method)
        auto expected_keepdiag = computeAPlusATNaive<int32_t, int32_t, true>(
            size, ai.data(), aj.data() );
        auto expected_nodiag = computeAPlusATNaive<int32_t, int32_t, false>(
            size, ai.data(), aj.data() );

        // Test with different thread counts
        for ( int nthreads : thread_counts )
        {
            std::cout << "\n  Testing with " << nthreads << " thread(s):" << std::endl;

            // Test with KEEPDIAG=true
            {
                APlusATStruct<int32_t, int32_t, true> aplusatOp( nthreads );

                // Allocate result with 2x original NNZ as upper bound
                std::vector<int32_t> result_ai( size + 1 );
                std::vector<int32_t> result_aj( 2 * nnz );

                aplusatOp( size, ai.data(), aj.data(), result_ai.data(), result_aj.data() );

                // Verify CSR structure is valid
                verifyCsrStructure( size, result_ai.data(), result_aj.data() );

                // Verify against expected computation
                compareCsrStructures( size, expected_keepdiag.ai.data(),
                                      expected_keepdiag.aj.data(), result_ai.data(),
                                      result_aj.data(),
                                      name + "_KeepDiag_" + std::to_string( nthreads ) + "threads" );

                std::cout << "    KEEPDIAG=true: result nnz="
                          << ( result_ai[size] - result_ai[0] )
                          << ", symmetric=true, matches expected=true" << std::endl;
            }

            // Test with KEEPDIAG=false
            {
                APlusATStruct<int32_t, int32_t, false> aplusatOp( nthreads );

                // Allocate result with 2x original NNZ as upper bound
                std::vector<int32_t> result_ai( size + 1 );
                std::vector<int32_t> result_aj( 2 * nnz );

                aplusatOp( size, ai.data(), aj.data(), result_ai.data(), result_aj.data() );

                // Verify CSR structure is valid
                verifyCsrStructure( size, result_ai.data(), result_aj.data() );

                // Verify against expected computation
                compareCsrStructures( size, expected_nodiag.ai.data(),
                                      expected_nodiag.aj.data(), result_ai.data(),
                                      result_aj.data(),
                                      name + "_NoDiag_" + std::to_string( nthreads ) + "threads" );

                std::cout << "    KEEPDIAG=false: result nnz="
                          << ( result_ai[size] - result_ai[0] )
                          << ", symmetric=true, matches expected=true" << std::endl;
            }
        }
    }
}

// Partition tests
class PartitionTest : public testing::Test
{
protected:
    const double _tol = 1e-10;
};

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
        std::vector<int> thread_counts = { 1, 2, 4, 8 };

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

int main( int argc, char** argv )
{
    testing::InitGoogleTest( &argc, argv );
    return RUN_ALL_TESTS();
}
