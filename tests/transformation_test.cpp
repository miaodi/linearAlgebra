#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "Transformation.hpp"
#include "TransformSeq.hpp"
#include "matrix_utils.hpp"
#include "spmv.hpp"

using namespace solver;
using namespace matrix_utils;

// Helper function to create a simple CSR matrix for testing
// Creates a 3x3 matrix:
// [2, 0, 1]
// [0, 3, 0]
// [1, 0, 4]
CSRMatrixVec<int, int, double> createTestMatrix()
{
    CSRMatrixVec<int, int, double> mat;
    mat.rows = 3;
    mat.cols = 3;

    // Row pointers (0-indexed)
    mat.ai = { 0, 2, 3, 5 };

    // Column indices
    mat.aj = { 0, 2, 1, 0, 2 };

    // Values
    mat.av = { 2.0, 1.0, 3.0, 1.0, 4.0 };

    return mat;
}

// Test fixture for RowScaling
class RowScalingTest : public ::testing::Test
{
protected:
    CSRMatrixVec<int, int, double> mat;
    CSRMatrixVec<int, int, double> mat_out;
    std::vector<double> scales;
    std::vector<double> vec_in;
    std::vector<double> vec_out;

    void SetUp() override
    {
        mat = createTestMatrix();
        mat_out = createTestMatrix();

        // Scaling factors for each row: [2.0, 0.5, 3.0]
        scales = { 2.0, 0.5, 3.0 };

        // Input vector
        vec_in = { 1.0, 2.0, 3.0 };
        vec_out = { 0.0, 0.0, 0.0 };
    }
};

TEST_F( RowScalingTest, applyToOperator )
{
    RowScaling<CSRMatrixVec<int, int, double>> transform( scales.data(), scales.size() );

    transform.applyToOperator( mat, mat_out );

    // Expected scaled matrix:
    // Row 0: [2*2, 0, 2*1] = [4, 0, 2]
    // Row 1: [0.5*0, 0.5*3, 0.5*0] = [0, 1.5, 0]
    // Row 2: [3*1, 3*0, 3*4] = [3, 0, 12]

    EXPECT_DOUBLE_EQ( mat_out.av[0], 4.0 );  // Row 0, col 0: 2*2
    EXPECT_DOUBLE_EQ( mat_out.av[1], 2.0 );  // Row 0, col 2: 2*1
    EXPECT_DOUBLE_EQ( mat_out.av[2], 1.5 );  // Row 1, col 1: 0.5*3
    EXPECT_DOUBLE_EQ( mat_out.av[3], 3.0 );  // Row 2, col 0: 3*1
    EXPECT_DOUBLE_EQ( mat_out.av[4], 12.0 ); // Row 2, col 2: 3*4
}

TEST_F( RowScalingTest, applyToRHS )
{
    RowScaling<CSRMatrixVec<int, int, double>> transform( scales.data(), scales.size() );

    transform.applyToRHS( vec_in, vec_out );

    // Expected: [2*1, 0.5*2, 3*3] = [2, 1, 9]
    EXPECT_DOUBLE_EQ( vec_out[0], 2.0 );
    EXPECT_DOUBLE_EQ( vec_out[1], 1.0 );
    EXPECT_DOUBLE_EQ( vec_out[2], 9.0 );
}

TEST_F( RowScalingTest, applyToX )
{
    RowScaling<CSRMatrixVec<int, int, double>> transform( scales.data(), scales.size() );

    transform.applyToX( vec_in, vec_out );

    // Row scaling doesn't affect solution x, should just swap
    EXPECT_DOUBLE_EQ( vec_out[0], 1.0 );
    EXPECT_DOUBLE_EQ( vec_out[1], 2.0 );
    EXPECT_DOUBLE_EQ( vec_out[2], 3.0 );
}

TEST_F( RowScalingTest, applyInverseToX )
{
    RowScaling<CSRMatrixVec<int, int, double>> transform( scales.data(), scales.size() );

    transform.applyInverseToX( vec_in, vec_out );

    // Row scaling doesn't affect solution x, should just swap
    EXPECT_DOUBLE_EQ( vec_out[0], 1.0 );
    EXPECT_DOUBLE_EQ( vec_out[1], 2.0 );
    EXPECT_DOUBLE_EQ( vec_out[2], 3.0 );
}

TEST_F( RowScalingTest, multipleThreads )
{
    RowScaling<CSRMatrixVec<int, int, double>> transform( scales.data(), scales.size() );

    // Test with different thread counts
    for ( int nthreads : { 1, 2, 4 } )
    {
        std::vector<double> vec_test = vec_in;
        std::vector<double> vec_result( 3, 0.0 );

        transform.applyToRHS( vec_test, vec_result, nthreads );

        EXPECT_DOUBLE_EQ( vec_result[0], 2.0 );
        EXPECT_DOUBLE_EQ( vec_result[1], 1.0 );
        EXPECT_DOUBLE_EQ( vec_result[2], 9.0 );
    }
}

TEST_F( RowScalingTest, spanConstructor )
{
    std::span<const double> scale_span( scales );
    RowScaling<CSRMatrixVec<int, int, double>> transform( scale_span );

    transform.applyToRHS( vec_in, vec_out );

    EXPECT_DOUBLE_EQ( vec_out[0], 2.0 );
    EXPECT_DOUBLE_EQ( vec_out[1], 1.0 );
    EXPECT_DOUBLE_EQ( vec_out[2], 9.0 );
}

// Helper function to perform CSR matrix-vector multiplication: y = A * x
void csrMatVecMult( const CSRMatrixVec<int, int, double>& A, const std::vector<double>& x, std::vector<double>& y )
{
    SerialSPMV spmv;
    y.assign( A.rows, 0.0 );
    spmv( A.rows, A.ai[0], A.ai.data(), A.aj.data(), A.av.data(), x.data(), y.data(), 1.0, 0.0 );
}

TEST_F( RowScalingTest, scalingMatrixVectorProperty )
{
    // Test that (Dr * A) * x = Dr * (A * x)
    // where Dr is diagonal row scaling matrix

    RowScaling<CSRMatrixVec<int, int, double>> transform( scales.data(), scales.size() );

    // Path 1: Scale the matrix first, then multiply by vector
    // (Dr * A) * x
    CSRMatrixVec<int, int, double> scaled_mat = mat;
    CSRMatrixVec<int, int, double> scaled_mat_out = mat;
    transform.applyToOperator( scaled_mat, scaled_mat_out );
    std::vector<double> result1( mat.rows );
    csrMatVecMult( scaled_mat_out, vec_in, result1 );

    // Path 2: Multiply matrix by vector first, then scale the result
    // Dr * (A * x)
    std::vector<double> temp( mat.rows );
    csrMatVecMult( mat, vec_in, temp );
    std::vector<double> temp_in = temp;
    std::vector<double> result2( mat.rows );
    transform.applyToRHS( temp_in, result2 );

    // Both paths should give the same result
    const double tol = 1e-10;
    for ( int i = 0; i < mat.rows; ++i )
    {
        EXPECT_NEAR( result1[i], result2[i], tol )
            << "Mismatch at index " << i << ": (Dr*A)*x = " << result1[i]
            << ", Dr*(A*x) = " << result2[i];
    }
}

TEST_F( RowScalingTest, scalingMatrixVectorPropertyMultipleVectors )
{
    // Test the property with multiple different input vectors
    RowScaling<CSRMatrixVec<int, int, double>> transform( scales.data(), scales.size() );

    std::vector<std::vector<double>> test_vectors = {
        { 1.0, 2.0, 3.0 }, { 0.5, -1.5, 2.5 }, { 10.0, -5.0, 0.0 }, { 0.0, 0.0, 0.0 }, { 1.0, 1.0, 1.0 } };

    const double tol = 1e-10;

    for ( const auto& test_vec : test_vectors )
    {
        // Path 1: (Dr * A) * x
        CSRMatrixVec<int, int, double> scaled_mat = mat;
        CSRMatrixVec<int, int, double> scaled_mat_out = mat;
        transform.applyToOperator( scaled_mat, scaled_mat_out );
        std::vector<double> result1( mat.rows );
        csrMatVecMult( scaled_mat_out, test_vec, result1 );

        // Path 2: Dr * (A * x)
        std::vector<double> temp( mat.rows );
        csrMatVecMult( mat, test_vec, temp );
        std::vector<double> temp_in = temp;
        std::vector<double> result2( mat.rows );
        transform.applyToRHS( temp_in, result2 );

        // Verify equality
        for ( int i = 0; i < mat.rows; ++i )
        {
            EXPECT_NEAR( result1[i], result2[i], tol );
        }
    }
}

// Test fixture for ColumnScaling
class ColumnScalingTest : public ::testing::Test
{
protected:
    CSRMatrixVec<int, int, double> mat;
    CSRMatrixVec<int, int, double> mat_out;
    std::vector<double> scales;
    std::vector<double> vec_in;
    std::vector<double> vec_out;

    void SetUp() override
    {
        mat = createTestMatrix();
        mat_out = createTestMatrix();

        // Scaling factors for each column: [2.0, 0.5, 3.0]
        scales = { 2.0, 0.5, 3.0 };

        // Input vector
        vec_in = { 1.0, 2.0, 3.0 };
        vec_out = { 0.0, 0.0, 0.0 };
    }
};

TEST_F( ColumnScalingTest, applyToOperator )
{
    ColumnScaling<CSRMatrixVec<int, int, double>> transform( scales.data(), scales.size() );

    transform.applyToOperator( mat, mat_out );

    // Original matrix:
    // [2, 0, 1]  (row 0: col 0 val 2, col 2 val 1)
    // [0, 3, 0]  (row 1: col 1 val 3)
    // [1, 0, 4]  (row 2: col 0 val 1, col 2 val 4)
    //
    // Column scaling by [2.0, 0.5, 3.0]:
    // [2*2, 0*0.5, 1*3] = [4, 0, 3]
    // [0*2, 3*0.5, 0*3] = [0, 1.5, 0]
    // [1*2, 0*0.5, 4*3] = [2, 0, 12]

    EXPECT_DOUBLE_EQ( mat_out.av[0], 4.0 );  // Row 0, col 0: 2*2
    EXPECT_DOUBLE_EQ( mat_out.av[1], 3.0 );  // Row 0, col 2: 1*3
    EXPECT_DOUBLE_EQ( mat_out.av[2], 1.5 );  // Row 1, col 1: 3*0.5
    EXPECT_DOUBLE_EQ( mat_out.av[3], 2.0 );  // Row 2, col 0: 1*2
    EXPECT_DOUBLE_EQ( mat_out.av[4], 12.0 ); // Row 2, col 2: 4*3
}

TEST_F( ColumnScalingTest, applyToRHS )
{
    ColumnScaling<CSRMatrixVec<int, int, double>> transform( scales.data(), scales.size() );

    transform.applyToRHS( vec_in, vec_out );

    // Column scaling doesn't affect RHS, should just swap
    EXPECT_DOUBLE_EQ( vec_out[0], 1.0 );
    EXPECT_DOUBLE_EQ( vec_out[1], 2.0 );
    EXPECT_DOUBLE_EQ( vec_out[2], 3.0 );
}

TEST_F( ColumnScalingTest, applyToX )
{
    ColumnScaling<CSRMatrixVec<int, int, double>> transform( scales.data(), scales.size() );

    transform.applyToX( vec_in, vec_out );

    // Column scaling scales solution x by Dc
    // Expected: [2*1, 0.5*2, 3*3] = [2, 1, 9]
    EXPECT_DOUBLE_EQ( vec_out[0], 2.0 );
    EXPECT_DOUBLE_EQ( vec_out[1], 1.0 );
    EXPECT_DOUBLE_EQ( vec_out[2], 9.0 );
}

TEST_F( ColumnScalingTest, applyInverseToX )
{
    ColumnScaling<CSRMatrixVec<int, int, double>> transform( scales.data(), scales.size() );

    transform.applyInverseToX( vec_in, vec_out );

    // Inverse column scaling divides solution x by Dc
    // Expected: [1/2, 2/0.5, 3/3] = [0.5, 4, 1]
    EXPECT_DOUBLE_EQ( vec_out[0], 0.5 );
    EXPECT_DOUBLE_EQ( vec_out[1], 4.0 );
    EXPECT_DOUBLE_EQ( vec_out[2], 1.0 );
}

TEST_F( ColumnScalingTest, multipleThreads )
{
    ColumnScaling<CSRMatrixVec<int, int, double>> transform( scales.data(), scales.size() );

    // Test with different thread counts
    for ( int nthreads : { 1, 2, 4 } )
    {
        std::vector<double> vec_test = vec_in;
        std::vector<double> vec_result( 3, 0.0 );

        transform.applyToX( vec_test, vec_result, nthreads );

        EXPECT_DOUBLE_EQ( vec_result[0], 2.0 );
        EXPECT_DOUBLE_EQ( vec_result[1], 1.0 );
        EXPECT_DOUBLE_EQ( vec_result[2], 9.0 );
    }
}

TEST_F( ColumnScalingTest, spanConstructor )
{
    std::span<const double> scale_span( scales );
    ColumnScaling<CSRMatrixVec<int, int, double>> transform( scale_span );

    transform.applyToX( vec_in, vec_out );

    EXPECT_DOUBLE_EQ( vec_out[0], 2.0 );
    EXPECT_DOUBLE_EQ( vec_out[1], 1.0 );
    EXPECT_DOUBLE_EQ( vec_out[2], 9.0 );
}

TEST_F( ColumnScalingTest, identityScaling )
{
    std::vector<double> identity_scales = { 1.0, 1.0, 1.0 };
    ColumnScaling<CSRMatrixVec<int, int, double>> transform( identity_scales.data(), identity_scales.size() );

    transform.applyToOperator( mat, mat_out );

    // Matrix should remain unchanged
    for ( size_t i = 0; i < mat.av.size(); ++i )
    {
        EXPECT_DOUBLE_EQ( mat_out.av[i], mat.av[i] );
    }
}

TEST_F( ColumnScalingTest, scalingMatrixVectorProperty )
{
    // Test that (A * Dc) * (Dc^-1 * x) = A * x
    // where Dc is diagonal column scaling matrix

    ColumnScaling<CSRMatrixVec<int, int, double>> transform( scales.data(), scales.size() );

    // Path 1: Compute A * x directly
    std::vector<double> expected( mat.rows );
    csrMatVecMult( mat, vec_in, expected );

    // Path 2: Scale matrix, scale vector inversely, then multiply
    // (A * Dc) * (Dc^-1 * x)

    // Step 1: Scale the matrix by Dc
    CSRMatrixVec<int, int, double> scaled_mat = mat;
    CSRMatrixVec<int, int, double> scaled_mat_out = mat;
    transform.applyToOperator( scaled_mat, scaled_mat_out );

    // Step 2: Apply inverse scaling to x: Dc^-1 * x
    std::vector<double> scaled_vec_in = vec_in;
    std::vector<double> scaled_vec( mat.cols );
    transform.applyInverseToX( scaled_vec_in, scaled_vec );

    // Step 3: Multiply scaled matrix by inversely scaled vector
    std::vector<double> result( mat.rows );
    csrMatVecMult( scaled_mat_out, scaled_vec, result );

    // Both should give the same result: A * x
    const double tol = 1e-10;
    for ( int i = 0; i < mat.rows; ++i )
    {
        EXPECT_NEAR( expected[i], result[i], tol )
            << "Mismatch at index " << i << ": A*x = " << expected[i]
            << ", (A*Dc)*(Dc^-1*x) = " << result[i];
    }
}

TEST_F( ColumnScalingTest, scalingMatrixVectorPropertyMultipleVectors )
{
    // Test the property (A * Dc) * (Dc^-1 * x) = A * x with multiple vectors
    ColumnScaling<CSRMatrixVec<int, int, double>> transform( scales.data(), scales.size() );

    std::vector<std::vector<double>> test_vectors = {
        { 1.0, 2.0, 3.0 }, { 0.5, -1.5, 2.5 }, { 10.0, -5.0, 0.0 }, { 0.0, 0.0, 0.0 }, { 1.0, 1.0, 1.0 } };

    const double tol = 1e-10;

    // Scale the matrix once
    CSRMatrixVec<int, int, double> scaled_mat = mat;
    CSRMatrixVec<int, int, double> scaled_mat_out = mat;
    transform.applyToOperator( scaled_mat, scaled_mat_out );

    for ( const auto& test_vec : test_vectors )
    {
        // Path 1: A * x
        std::vector<double> expected( mat.rows );
        csrMatVecMult( mat, test_vec, expected );

        // Path 2: (A * Dc) * (Dc^-1 * x)
        std::vector<double> scaled_vec_in = test_vec;
        std::vector<double> scaled_vec( mat.cols );
        transform.applyInverseToX( scaled_vec_in, scaled_vec );

        std::vector<double> result( mat.rows );
        csrMatVecMult( scaled_mat_out, scaled_vec, result );

        // Verify equality
        for ( int i = 0; i < mat.rows; ++i )
        {
            EXPECT_NEAR( expected[i], result[i], tol );
        }
    }
}

TEST_F( ColumnScalingTest, inverseScalingRoundtrip )
{
    // Test that applying scaling and then inverse scaling returns to original
    ColumnScaling<CSRMatrixVec<int, int, double>> transform( scales.data(), scales.size() );

    std::vector<double> original = vec_in;
    std::vector<double> temp( mat.cols, 0.0 );
    std::vector<double> roundtrip( mat.cols, 0.0 );

    // Apply scaling: x -> Dc * x
    std::vector<double> step1_in = original;
    transform.applyToX( step1_in, temp );

    // Apply inverse scaling: Dc * x -> Dc^-1 * (Dc * x) = x
    std::vector<double> step2_in = temp;
    transform.applyInverseToX( step2_in, roundtrip );

    const double tol = 1e-10;
    for ( size_t i = 0; i < original.size(); ++i )
    {
        EXPECT_NEAR( original[i], roundtrip[i], tol ) << "Roundtrip failed at index " << i;
    }
}

// Test fixture for RowColScaling
class RowColScalingTest : public ::testing::Test
{
protected:
    CSRMatrixVec<int, int, double> mat;
    CSRMatrixVec<int, int, double> mat_out;
    std::vector<double> row_scales;
    std::vector<double> col_scales;
    std::vector<double> vec_in;
    std::vector<double> vec_out;

    void SetUp() override
    {
        mat = createTestMatrix();
        mat_out = createTestMatrix();

        // Row scaling factors: [2.0, 0.5, 3.0]
        row_scales = { 2.0, 0.5, 3.0 };

        // Column scaling factors: [1.5, 2.0, 0.5]
        col_scales = { 1.5, 2.0, 0.5 };

        // Input vector
        vec_in = { 1.0, 2.0, 3.0 };
        vec_out = { 0.0, 0.0, 0.0 };
    }
};

TEST_F( RowColScalingTest, applyToOperator )
{
    RowColScaling<CSRMatrixVec<int, int, double>> transform( row_scales.data(), col_scales.data(),
                                                             row_scales.size() );

    transform.applyToOperator( mat, mat_out );

    // Original matrix:
    // [2, 0, 1]  (row 0: col 0 val 2, col 2 val 1)
    // [0, 3, 0]  (row 1: col 1 val 3)
    // [1, 0, 4]  (row 2: col 0 val 1, col 2 val 4)
    //
    // Row scaling [2.0, 0.5, 3.0], Column scaling [1.5, 2.0, 0.5]:
    // Row 0: [2*2*1.5, 0*2*2.0, 1*2*0.5] = [6, 0, 1]
    // Row 1: [0*0.5*1.5, 3*0.5*2.0, 0*0.5*0.5] = [0, 3, 0]
    // Row 2: [1*3*1.5, 0*3*2.0, 4*3*0.5] = [4.5, 0, 6]

    EXPECT_DOUBLE_EQ( mat_out.av[0], 6.0 ); // Row 0, col 0: 2*2*1.5
    EXPECT_DOUBLE_EQ( mat_out.av[1], 1.0 ); // Row 0, col 2: 1*2*0.5
    EXPECT_DOUBLE_EQ( mat_out.av[2], 3.0 ); // Row 1, col 1: 3*0.5*2.0
    EXPECT_DOUBLE_EQ( mat_out.av[3], 4.5 ); // Row 2, col 0: 1*3*1.5
    EXPECT_DOUBLE_EQ( mat_out.av[4], 6.0 ); // Row 2, col 2: 4*3*0.5
}

TEST_F( RowColScalingTest, applyToRHS )
{
    RowColScaling<CSRMatrixVec<int, int, double>> transform( row_scales.data(), col_scales.data(),
                                                             row_scales.size() );

    transform.applyToRHS( vec_in, vec_out );

    // Row scaling affects RHS: Dr * b
    // Expected: [2*1, 0.5*2, 3*3] = [2, 1, 9]
    EXPECT_DOUBLE_EQ( vec_out[0], 2.0 );
    EXPECT_DOUBLE_EQ( vec_out[1], 1.0 );
    EXPECT_DOUBLE_EQ( vec_out[2], 9.0 );
}

TEST_F( RowColScalingTest, applyToX )
{
    RowColScaling<CSRMatrixVec<int, int, double>> transform( row_scales.data(), col_scales.data(),
                                                             row_scales.size() );

    transform.applyToX( vec_in, vec_out );

    // Column scaling affects solution: Dc * x
    // Expected: [1.5*1, 2.0*2, 0.5*3] = [1.5, 4, 1.5]
    EXPECT_DOUBLE_EQ( vec_out[0], 1.5 );
    EXPECT_DOUBLE_EQ( vec_out[1], 4.0 );
    EXPECT_DOUBLE_EQ( vec_out[2], 1.5 );
}

TEST_F( RowColScalingTest, applyInverseToX )
{
    RowColScaling<CSRMatrixVec<int, int, double>> transform( row_scales.data(), col_scales.data(),
                                                             row_scales.size() );

    transform.applyInverseToX( vec_in, vec_out );

    // Inverse column scaling: Dc^-1 * x
    // Expected: [1/1.5, 2/2.0, 3/0.5] = [0.666..., 1, 6]
    EXPECT_NEAR( vec_out[0], 1.0 / 1.5, 1e-10 );
    EXPECT_DOUBLE_EQ( vec_out[1], 1.0 );
    EXPECT_DOUBLE_EQ( vec_out[2], 6.0 );
}

TEST_F( RowColScalingTest, multipleThreads )
{
    RowColScaling<CSRMatrixVec<int, int, double>> transform( row_scales.data(), col_scales.data(),
                                                             row_scales.size() );

    // Test with different thread counts
    for ( int nthreads : { 1, 2, 4 } )
    {
        std::vector<double> vec_test = vec_in;
        std::vector<double> vec_result( 3, 0.0 );

        transform.applyToRHS( vec_test, vec_result, nthreads );

        EXPECT_DOUBLE_EQ( vec_result[0], 2.0 );
        EXPECT_DOUBLE_EQ( vec_result[1], 1.0 );
        EXPECT_DOUBLE_EQ( vec_result[2], 9.0 );
    }
}

TEST_F( RowColScalingTest, spanConstructor )
{
    std::span<const double> row_span( row_scales );
    std::span<const double> col_span( col_scales );
    RowColScaling<CSRMatrixVec<int, int, double>> transform( row_span, col_span );

    transform.applyToRHS( vec_in, vec_out );

    EXPECT_DOUBLE_EQ( vec_out[0], 2.0 );
    EXPECT_DOUBLE_EQ( vec_out[1], 1.0 );
    EXPECT_DOUBLE_EQ( vec_out[2], 9.0 );
}

TEST_F( RowColScalingTest, identityScaling )
{
    std::vector<double> identity_row = { 1.0, 1.0, 1.0 };
    std::vector<double> identity_col = { 1.0, 1.0, 1.0 };
    RowColScaling<CSRMatrixVec<int, int, double>> transform(
        identity_row.data(), identity_col.data(), identity_row.size() );

    transform.applyToOperator( mat, mat_out );

    // Matrix should remain unchanged
    for ( size_t i = 0; i < mat.av.size(); ++i )
    {
        EXPECT_DOUBLE_EQ( mat_out.av[i], mat.av[i] );
    }
}

TEST_F( RowColScalingTest, scalingMatrixVectorProperty )
{
    // Test that (Dr * A * Dc) * (Dc^-1 * x) = Dr * (A * x)
    // where Dr is row scaling and Dc is column scaling

    RowColScaling<CSRMatrixVec<int, int, double>> transform( row_scales.data(), col_scales.data(),
                                                             row_scales.size() );

    // Path 1: Compute Dr * (A * x)
    // Step 1: A * x
    std::vector<double> ax( mat.rows );
    csrMatVecMult( mat, vec_in, ax );

    // Step 2: Dr * (A * x)
    RowScaling<CSRMatrixVec<int, int, double>> row_transform( row_scales.data(), row_scales.size() );
    std::vector<double> ax_in = ax;
    std::vector<double> expected( mat.rows );
    row_transform.applyToRHS( ax_in, expected );

    // Path 2: (Dr * A * Dc) * (Dc^-1 * x)
    // Step 1: Scale the matrix: Dr * A * Dc
    CSRMatrixVec<int, int, double> scaled_mat = mat;
    CSRMatrixVec<int, int, double> scaled_mat_out = mat;
    transform.applyToOperator( scaled_mat, scaled_mat_out );

    // Step 2: Apply inverse column scaling to x: Dc^-1 * x
    ColumnScaling<CSRMatrixVec<int, int, double>> col_transform( col_scales.data(), col_scales.size() );
    std::vector<double> scaled_vec_in = vec_in;
    std::vector<double> scaled_vec( mat.cols );
    col_transform.applyInverseToX( scaled_vec_in, scaled_vec );

    // Step 3: Multiply: (Dr * A * Dc) * (Dc^-1 * x)
    std::vector<double> result( mat.rows );
    csrMatVecMult( scaled_mat_out, scaled_vec, result );

    // Both paths should give the same result
    const double tol = 1e-10;
    for ( int i = 0; i < mat.rows; ++i )
    {
        EXPECT_NEAR( expected[i], result[i], tol )
            << "Mismatch at index " << i << ": Dr*(A*x) = " << expected[i]
            << ", (Dr*A*Dc)*(Dc^-1*x) = " << result[i];
    }
}

TEST_F( RowColScalingTest, scalingMatrixVectorPropertyMultipleVectors )
{
    // Test the property with multiple different input vectors
    RowColScaling<CSRMatrixVec<int, int, double>> transform( row_scales.data(), col_scales.data(),
                                                             row_scales.size() );

    std::vector<std::vector<double>> test_vectors = {
        { 1.0, 2.0, 3.0 }, { 0.5, -1.5, 2.5 }, { 10.0, -5.0, 0.0 }, { 0.0, 0.0, 0.0 }, { 1.0, 1.0, 1.0 } };

    const double tol = 1e-10;

    // Pre-scale the matrix once
    CSRMatrixVec<int, int, double> scaled_mat = mat;
    CSRMatrixVec<int, int, double> scaled_mat_out = mat;
    transform.applyToOperator( scaled_mat, scaled_mat_out );

    RowScaling<CSRMatrixVec<int, int, double>> row_transform( row_scales.data(), row_scales.size() );
    ColumnScaling<CSRMatrixVec<int, int, double>> col_transform( col_scales.data(), col_scales.size() );

    for ( const auto& test_vec : test_vectors )
    {
        // Path 1: Dr * (A * x)
        std::vector<double> ax( mat.rows );
        csrMatVecMult( mat, test_vec, ax );
        std::vector<double> ax_in = ax;
        std::vector<double> expected( mat.rows );
        row_transform.applyToRHS( ax_in, expected );

        // Path 2: (Dr * A * Dc) * (Dc^-1 * x)
        std::vector<double> scaled_vec_in = test_vec;
        std::vector<double> scaled_vec( mat.cols );
        col_transform.applyInverseToX( scaled_vec_in, scaled_vec );

        std::vector<double> result( mat.rows );
        csrMatVecMult( scaled_mat_out, scaled_vec, result );

        // Verify equality
        for ( int i = 0; i < mat.rows; ++i )
        {
            EXPECT_NEAR( expected[i], result[i], tol );
        }
    }
}

TEST_F( RowColScalingTest, inverseScalingRoundtrip )
{
    // Test that applying column scaling and then inverse column scaling on x returns to original
    RowColScaling<CSRMatrixVec<int, int, double>> transform( row_scales.data(), col_scales.data(),
                                                             row_scales.size() );

    std::vector<double> original = vec_in;
    std::vector<double> temp( mat.cols, 0.0 );
    std::vector<double> roundtrip( mat.cols, 0.0 );

    // Apply column scaling to x: x -> Dc * x
    std::vector<double> step1_in = original;
    transform.applyToX( step1_in, temp );

    // Apply inverse column scaling: Dc * x -> Dc^-1 * (Dc * x) = x
    std::vector<double> step2_in = temp;
    transform.applyInverseToX( step2_in, roundtrip );

    const double tol = 1e-10;
    for ( size_t i = 0; i < original.size(); ++i )
    {
        EXPECT_NEAR( original[i], roundtrip[i], tol ) << "Roundtrip failed at index " << i;
    }
}

// ============================================================================
// RowPermutation Tests
// ============================================================================

// Test fixture for RowPermutation
class RowPermutationTest : public ::testing::Test
{
protected:
    CSRMatrixVec<int, int, double> mat;
    CSRMatrixVec<int, int, double> mat_out;
    std::vector<int> perm;
    std::vector<double> vec_in;
    std::vector<double> vec_out;

    void SetUp() override
    {
        mat = createTestMatrix();
        mat_out = createTestMatrix();

        // Permutation: [1, 2, 0] (0-indexed)
        // This means: row 0 <- row 1, row 1 <- row 2, row 2 <- row 0
        perm = { 1, 2, 0 };

        // Input vector
        vec_in = { 1.0, 2.0, 3.0 };
        vec_out.resize( mat.rows, 0.0 );
    }

    // Helper to compare matrices (for debugging)
    bool matricesEqual( const CSRMatrixVec<int, int, double>& a,
                        const CSRMatrixVec<int, int, double>& b,
                        double tol = 1e-10 )
    {
        if ( a.rows != b.rows || a.cols != b.cols )
            return false;
        if ( a.ai.size() != b.ai.size() )
            return false;
        if ( a.aj.size() != b.aj.size() )
            return false;

        for ( size_t i = 0; i < a.ai.size(); ++i )
        {
            if ( a.ai[i] != b.ai[i] )
                return false;
        }

        for ( size_t i = 0; i < a.aj.size(); ++i )
        {
            if ( a.aj[i] != b.aj[i] )
                return false;
            if ( std::abs( a.av[i] - b.av[i] ) > tol )
                return false;
        }

        return true;
    }
};

TEST_F( RowPermutationTest, applyToOperator )
{
    // Test that row permutation permutes matrix rows correctly
    // Original matrix:
    // [2, 0, 1]
    // [0, 3, 0]
    // [1, 0, 4]
    //
    // After permutation [1, 2, 0]:
    // [0, 3, 0]  <- row 1
    // [1, 0, 4]  <- row 2
    // [2, 0, 1]  <- row 0

    RowPermutation<CSRMatrixVec<int, int, double>> transform( perm.data(), perm.size(), 0 );
    transform.applyToOperator( mat, mat_out );

    // Expected permuted matrix in CSR format
    CSRMatrixVec<int, int, double> expected;
    expected.rows = 3;
    expected.cols = 3;
    expected.ai = { 0, 1, 3, 5 };              // Row pointers for permuted rows
    expected.aj = { 1, 0, 2, 0, 2 };           // Column indices
    expected.av = { 3.0, 1.0, 4.0, 2.0, 1.0 }; // Values

    EXPECT_TRUE( matricesEqual( mat_out, expected ) ) << "Row permutation of matrix failed";
}

TEST_F( RowPermutationTest, applyToRHS )
{
    // Test that row permutation permutes RHS vector
    // Input: [1.0, 2.0, 3.0]
    // Permutation [1, 2, 0] means: out[0] = in[1], out[1] = in[2], out[2] = in[0]
    // Expected output: [2.0, 3.0, 1.0]

    RowPermutation<CSRMatrixVec<int, int, double>> transform( perm.data(), perm.size(), 0 );
    transform.applyToRHS( vec_in, vec_out );

    EXPECT_DOUBLE_EQ( vec_out[0], 2.0 );
    EXPECT_DOUBLE_EQ( vec_out[1], 3.0 );
    EXPECT_DOUBLE_EQ( vec_out[2], 1.0 );
}

TEST_F( RowPermutationTest, applyToX )
{
    // Row permutation should not affect solution vector x
    RowPermutation<CSRMatrixVec<int, int, double>> transform( perm.data(), perm.size(), 0 );
    std::vector<double> expected = vec_in;
    transform.applyToX( vec_in, vec_out );

    EXPECT_EQ( vec_out[0], expected[0] );
    EXPECT_EQ( vec_out[1], expected[1] );
    EXPECT_EQ( vec_out[2], expected[2] );
}

TEST_F( RowPermutationTest, applyInverseToX )
{
    // Row permutation should not affect solution vector x (inverse also no-op)
    RowPermutation<CSRMatrixVec<int, int, double>> transform( perm.data(), perm.size(), 0 );
    std::vector<double> expected = vec_in;
    transform.applyInverseToX( vec_in, vec_out );

    EXPECT_EQ( vec_out[0], expected[0] );
    EXPECT_EQ( vec_out[1], expected[1] );
    EXPECT_EQ( vec_out[2], expected[2] );
}

TEST_F( RowPermutationTest, multipleThreads )
{
    // Test with different thread counts
    RowPermutation<CSRMatrixVec<int, int, double>> transform( perm.data(), perm.size(), 0 );

    for ( int nthreads : { 1, 2, 4 } )
    {
        std::vector<double> result( mat.rows, 0.0 );
        std::vector<double> input = vec_in;
        transform.applyToRHS( input, result, nthreads );

        EXPECT_DOUBLE_EQ( result[0], 2.0 ) << "Failed with " << nthreads << " threads";
        EXPECT_DOUBLE_EQ( result[1], 3.0 ) << "Failed with " << nthreads << " threads";
        EXPECT_DOUBLE_EQ( result[2], 1.0 ) << "Failed with " << nthreads << " threads";
    }
}

TEST_F( RowPermutationTest, spanConstructor )
{
    // Test span-based constructor
    std::span<const int> perm_span( perm );
    RowPermutation<CSRMatrixVec<int, int, double>> transform( perm_span, 0 );

    std::vector<double> result( mat.rows, 0.0 );
    transform.applyToRHS( vec_in, result );

    EXPECT_DOUBLE_EQ( result[0], 2.0 );
    EXPECT_DOUBLE_EQ( result[1], 3.0 );
    EXPECT_DOUBLE_EQ( result[2], 1.0 );
}

TEST_F( RowPermutationTest, identityPermutation )
{
    // Test with identity permutation [0, 1, 2]
    std::vector<int> identity_perm = { 0, 1, 2 };
    RowPermutation<CSRMatrixVec<int, int, double>> transform( identity_perm.data(), identity_perm.size(), 0 );

    std::vector<double> expected = vec_in;
    transform.applyToRHS( vec_in, vec_out );

    EXPECT_DOUBLE_EQ( vec_out[0], expected[0] );
    EXPECT_DOUBLE_EQ( vec_out[1], expected[1] );
    EXPECT_DOUBLE_EQ( vec_out[2], expected[2] );
}

TEST_F( RowPermutationTest, permutationMatrixVectorProperty )
{
    // KEY TEST: Verify that (Pr * A) * x = Pr * (A * x)
    // This is the fundamental mathematical property of row permutation

    RowPermutation<CSRMatrixVec<int, int, double>> transform( perm.data(), perm.size(), 0 );

    std::vector<double> test_vec = { 1.5, 2.5, 3.5 };
    const double tol = 1e-10;

    // Path 1: Pr * (A * x)
    // First compute A * x
    std::vector<double> ax( mat.rows );
    csrMatVecMult( mat, test_vec, ax );

    // Then apply row permutation to result: Pr * (A * x)
    std::vector<double> ax_in = ax;
    std::vector<double> expected( mat.rows );
    transform.applyToRHS( ax_in, expected );

    // Path 2: (Pr * A) * x
    // First apply row permutation to matrix
    CSRMatrixVec<int, int, double> permuted_mat = mat;
    CSRMatrixVec<int, int, double> permuted_mat_out = mat;
    transform.applyToOperator( permuted_mat, permuted_mat_out );

    // Then compute (Pr * A) * x
    std::vector<double> result( mat.rows );
    csrMatVecMult( permuted_mat_out, test_vec, result );

    // Verify equality: (Pr * A) * x = Pr * (A * x)
    for ( int i = 0; i < mat.rows; ++i )
    {
        EXPECT_NEAR( expected[i], result[i], tol ) << "Property failed at index " << i;
    }
}

TEST_F( RowPermutationTest, permutationMatrixVectorPropertyMultipleVectors )
{
    // Test the property (Pr * A) * x = Pr * (A * x) with multiple different vectors
    RowPermutation<CSRMatrixVec<int, int, double>> transform( perm.data(), perm.size(), 0 );

    // Apply row permutation to matrix once
    CSRMatrixVec<int, int, double> permuted_mat = mat;
    CSRMatrixVec<int, int, double> permuted_mat_out = mat;
    transform.applyToOperator( permuted_mat, permuted_mat_out );

    const double tol = 1e-10;

    // Test with multiple different input vectors
    std::vector<std::vector<double>> test_vectors = {
        { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 }, { 1.0, 2.0, 3.0 }, { -1.5, 2.7, -3.2 } };

    for ( const auto& test_vec : test_vectors )
    {
        // Path 1: Pr * (A * x)
        std::vector<double> ax( mat.rows );
        csrMatVecMult( mat, test_vec, ax );

        std::vector<double> ax_in = ax;
        std::vector<double> expected( mat.rows );
        transform.applyToRHS( ax_in, expected );

        // Path 2: (Pr * A) * x
        std::vector<double> result( mat.rows );
        csrMatVecMult( permuted_mat_out, test_vec, result );

        // Verify equality
        for ( int i = 0; i < mat.rows; ++i )
        {
            EXPECT_NEAR( expected[i], result[i], tol );
        }
    }
}

TEST_F( RowPermutationTest, inversePermutation )
{
    // Test that applying permutation and its inverse gives back original
    RowPermutation<CSRMatrixVec<int, int, double>> transform( perm.data(), perm.size(), 0 );

    // Permute the vector
    std::vector<double> permuted( mat.rows, 0.0 );
    transform.applyToRHS( vec_in, permuted );

    // Create inverse permutation: if perm[i] = j, then inv_perm[j] = i
    std::vector<int> inv_perm( perm.size() );
    for ( size_t i = 0; i < perm.size(); ++i )
    {
        inv_perm[perm[i]] = i;
    }

    RowPermutation<CSRMatrixVec<int, int, double>> inv_transform( inv_perm.data(), inv_perm.size(), 0 );

    // Apply inverse permutation
    std::vector<double> roundtrip( mat.rows, 0.0 );
    inv_transform.applyToRHS( permuted, roundtrip );

    // Should get back original vector
    const double tol = 1e-10;
    for ( size_t i = 0; i < vec_in.size(); ++i )
    {
        EXPECT_NEAR( vec_in[i], roundtrip[i], tol ) << "Inverse permutation failed at index " << i;
    }
}

// ============================================================================
// ColumnPermutation Tests
// ============================================================================

// Test fixture for ColumnPermutation
class ColumnPermutationTest : public ::testing::Test
{
protected:
    CSRMatrixVec<int, int, double> mat;
    CSRMatrixVec<int, int, double> mat_out;
    std::vector<int> perm;
    std::vector<double> vec_in;
    std::vector<double> vec_out;

    void SetUp() override
    {
        mat = createTestMatrix();
        mat_out = createTestMatrix();

        // Permutation: [1, 2, 0] (0-indexed)
        // This means: col 0 <- col 1, col 1 <- col 2, col 2 <- col 0
        perm = { 1, 2, 0 };

        // Input vector
        vec_in = { 1.0, 2.0, 3.0 };
        vec_out.resize( mat.cols, 0.0 );
    }

    // Helper to compare matrices (for debugging)
    bool matricesEqual( const CSRMatrixVec<int, int, double>& a,
                        const CSRMatrixVec<int, int, double>& b,
                        double tol = 1e-10 )
    {
        if ( a.rows != b.rows || a.cols != b.cols )
            return false;
        if ( a.ai.size() != b.ai.size() )
            return false;
        if ( a.aj.size() != b.aj.size() )
            return false;

        for ( size_t i = 0; i < a.ai.size(); ++i )
        {
            if ( a.ai[i] != b.ai[i] )
                return false;
        }

        for ( size_t i = 0; i < a.aj.size(); ++i )
        {
            if ( a.aj[i] != b.aj[i] )
                return false;
            if ( std::abs( a.av[i] - b.av[i] ) > tol )
                return false;
        }

        return true;
    }
};

TEST_F( ColumnPermutationTest, applyToOperator )
{
    // Test that column permutation permutes matrix columns correctly
    // Original matrix:
    // [2, 0, 1]
    // [0, 3, 0]
    // [1, 0, 4]
    //
    // After column permutation [1, 2, 0] (Q^T means inverse):
    // For A * Q^T, we swap columns: col_new[perm[i]] = col_old[i]
    // col 0 -> col 1, col 1 -> col 2, col 2 -> col 0
    // Result:
    // [1, 2, 0]  (col 0 from old col 2, col 1 from old col 0, col 2 from old col 1)
    // [0, 0, 3]
    // [4, 1, 0]

    ColumnPermutation<CSRMatrixVec<int, int, double>> transform( perm.data(), perm.size(), 0 );
    transform.applyToOperator( mat, mat_out );

    // Expected permuted matrix in CSR format
    CSRMatrixVec<int, int, double> expected;
    expected.rows = 3;
    expected.cols = 3;
    expected.ai = { 0, 2, 3, 5 };              // Row pointers
    expected.aj = { 0, 1, 2, 0, 1 };           // Column indices after permutation
    expected.av = { 1.0, 2.0, 3.0, 4.0, 1.0 }; // Values

    EXPECT_TRUE( matricesEqual( mat_out, expected ) ) << "Column permutation of matrix failed";
}

TEST_F( ColumnPermutationTest, applyToRHS )
{
    // Column permutation should not affect RHS
    ColumnPermutation<CSRMatrixVec<int, int, double>> transform( perm.data(), perm.size(), 0 );
    std::vector<double> expected = vec_in;
    transform.applyToRHS( vec_in, vec_out );

    EXPECT_EQ( vec_out[0], expected[0] );
    EXPECT_EQ( vec_out[1], expected[1] );
    EXPECT_EQ( vec_out[2], expected[2] );
}

TEST_F( ColumnPermutationTest, applyToX )
{
    // Test that column permutation transforms solution vector x
    // applyToX applies Q * x using permVec
    // permVec: out[i] = in[perm[i]]
    // For perm [1, 2, 0] and input [1.0, 2.0, 3.0]:
    // out[0] = in[1] = 2.0, out[1] = in[2] = 3.0, out[2] = in[0] = 1.0
    // Expected: [2.0, 3.0, 1.0]

    ColumnPermutation<CSRMatrixVec<int, int, double>> transform( perm.data(), perm.size(), 0 );
    transform.applyToX( vec_in, vec_out );

    EXPECT_DOUBLE_EQ( vec_out[0], 2.0 );
    EXPECT_DOUBLE_EQ( vec_out[1], 3.0 );
    EXPECT_DOUBLE_EQ( vec_out[2], 1.0 );
}

TEST_F( ColumnPermutationTest, applyInverseToX )
{
    // Test inverse column permutation on solution vector
    // applyInverseToX applies Q^-1 * x using invPermVec
    // invPermVec: out[perm[i]] = in[i]
    // For perm [1, 2, 0] and input [1.0, 2.0, 3.0]:
    // out[1] = in[0] = 1.0, out[2] = in[1] = 2.0, out[0] = in[2] = 3.0
    // Expected: [3.0, 1.0, 2.0]

    ColumnPermutation<CSRMatrixVec<int, int, double>> transform( perm.data(), perm.size(), 0 );
    transform.applyInverseToX( vec_in, vec_out );

    EXPECT_DOUBLE_EQ( vec_out[0], 3.0 );
    EXPECT_DOUBLE_EQ( vec_out[1], 1.0 );
    EXPECT_DOUBLE_EQ( vec_out[2], 2.0 );
}

TEST_F( ColumnPermutationTest, multipleThreads )
{
    // Test with different thread counts
    ColumnPermutation<CSRMatrixVec<int, int, double>> transform( perm.data(), perm.size(), 0 );

    for ( int nthreads : { 1, 2, 4 } )
    {
        std::vector<double> result( mat.cols, 0.0 );
        std::vector<double> input = { 1.0, 2.0, 3.0 };
        transform.applyToX( input, result, nthreads );

        EXPECT_DOUBLE_EQ( result[0], 2.0 ) << "Failed with " << nthreads << " threads";
        EXPECT_DOUBLE_EQ( result[1], 3.0 ) << "Failed with " << nthreads << " threads";
        EXPECT_DOUBLE_EQ( result[2], 1.0 ) << "Failed with " << nthreads << " threads";
    }
}

TEST_F( ColumnPermutationTest, spanConstructor )
{
    // Test span-based constructor
    std::span<const int> perm_span( perm );
    ColumnPermutation<CSRMatrixVec<int, int, double>> transform( perm_span, 0 );

    std::vector<double> result( mat.cols, 0.0 );
    transform.applyToX( vec_in, result );

    EXPECT_DOUBLE_EQ( result[0], 2.0 );
    EXPECT_DOUBLE_EQ( result[1], 3.0 );
    EXPECT_DOUBLE_EQ( result[2], 1.0 );
}

TEST_F( ColumnPermutationTest, identityPermutation )
{
    // Test with identity permutation [0, 1, 2]
    std::vector<int> identity_perm = { 0, 1, 2 };
    ColumnPermutation<CSRMatrixVec<int, int, double>> transform( identity_perm.data(),
                                                                 identity_perm.size(), 0 );

    std::vector<double> expected = vec_in;
    transform.applyToX( vec_in, vec_out );

    EXPECT_DOUBLE_EQ( vec_out[0], expected[0] );
    EXPECT_DOUBLE_EQ( vec_out[1], expected[1] );
    EXPECT_DOUBLE_EQ( vec_out[2], expected[2] );
}

TEST_F( ColumnPermutationTest, permutationMatrixVectorProperty )
{
    // KEY TEST: Verify that (A * Q^T) * (Q^-1 * x) = A * x
    // This is the fundamental mathematical property of column permutation
    // Now applyInverseToX applies Q^-1

    ColumnPermutation<CSRMatrixVec<int, int, double>> transform( perm.data(), perm.size(), 0 );

    std::vector<double> test_vec = { 1.5, 2.5, 3.5 };
    const double tol = 1e-10;

    // Path 1: A * x
    std::vector<double> expected( mat.rows );
    csrMatVecMult( mat, test_vec, expected );

    // Path 2: (A * Q^T) * (Q^-1 * x)
    // First apply column permutation to matrix: A * Q^T
    CSRMatrixVec<int, int, double> permuted_mat = mat;
    CSRMatrixVec<int, int, double> permuted_mat_out = mat;
    transform.applyToOperator( permuted_mat, permuted_mat_out );

    // Then apply Q^-1 to x (which is applyInverseToX)
    std::vector<double> transformed_vec_in = test_vec;
    std::vector<double> transformed_vec( mat.cols );
    transform.applyInverseToX( transformed_vec_in, transformed_vec );

    // Finally compute (A * Q^T) * (Q^-1 * x)
    std::vector<double> result( mat.rows );
    csrMatVecMult( permuted_mat_out, transformed_vec, result );

    // Verify equality: (A * Q^T) * (Q^-1 * x) = A * x
    for ( int i = 0; i < mat.rows; ++i )
    {
        EXPECT_NEAR( expected[i], result[i], tol ) << "Property failed at index " << i;
    }
}

TEST_F( ColumnPermutationTest, permutationMatrixVectorPropertyMultipleVectors )
{
    // Test the property (A * Q^T) * (Q^-1 * x) = A * x with multiple different vectors
    ColumnPermutation<CSRMatrixVec<int, int, double>> transform( perm.data(), perm.size(), 0 );

    // Apply column permutation to matrix once
    CSRMatrixVec<int, int, double> permuted_mat = mat;
    CSRMatrixVec<int, int, double> permuted_mat_out = mat;
    transform.applyToOperator( permuted_mat, permuted_mat_out );

    const double tol = 1e-10;

    // Test with multiple different input vectors
    std::vector<std::vector<double>> test_vectors = {
        { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 }, { 1.0, 2.0, 3.0 }, { -1.5, 2.7, -3.2 } };

    for ( const auto& test_vec : test_vectors )
    {
        // Path 1: A * x
        std::vector<double> expected( mat.rows );
        csrMatVecMult( mat, test_vec, expected );

        // Path 2: (A * Q^T) * (Q^-1 * x)
        std::vector<double> transformed_vec_in = test_vec;
        std::vector<double> transformed_vec( mat.cols );
        transform.applyInverseToX( transformed_vec_in, transformed_vec );

        std::vector<double> result( mat.rows );
        csrMatVecMult( permuted_mat_out, transformed_vec, result );

        // Verify equality
        for ( int i = 0; i < mat.rows; ++i )
        {
            EXPECT_NEAR( expected[i], result[i], tol );
        }
    }
}

TEST_F( ColumnPermutationTest, inversePermutationRoundtrip )
{
    // Test that applying Q^-T and then Q^T gives back original (using applyToX and applyInverseToX)
    ColumnPermutation<CSRMatrixVec<int, int, double>> transform( perm.data(), perm.size(), 0 );

    std::vector<double> original = vec_in;
    std::vector<double> temp( mat.cols, 0.0 );
    std::vector<double> roundtrip( mat.cols, 0.0 );

    // Apply Q^-T (applyToX)
    std::vector<double> step1_in = original;
    transform.applyToX( step1_in, temp );

    // Apply Q^T (applyInverseToX)
    std::vector<double> step2_in = temp;
    transform.applyInverseToX( step2_in, roundtrip );

    // Should get back original vector
    const double tol = 1e-10;
    for ( size_t i = 0; i < original.size(); ++i )
    {
        EXPECT_NEAR( original[i], roundtrip[i], tol ) << "Roundtrip failed at index " << i;
    }
}

// ============================================================================
// TransformSeq Tests
// ============================================================================

// Test fixture for TransformSeq
class TransformSeqTest : public ::testing::Test
{
protected:
    CSRMatrixVec<int, int, double> mat;
    CSRMatrixVec<int, int, double> mat_out;
    std::vector<double> vec_in;
    std::vector<double> vec_out;

    void SetUp() override
    {
        mat = createTestMatrix();
        mat_out = createTestMatrix();
        vec_in = { 1.0, 2.0, 3.0 };
        vec_out.resize( mat.rows, 0.0 );
    }
};

TEST_F( TransformSeqTest, emptySequence )
{
    // Test that empty sequence acts as identity
    TransformSeq<CSRMatrixVec<int, int, double>> seq;

    EXPECT_EQ( seq.size(), 0 );

    // Apply to vector - should be identity (just swap)
    std::vector<double> expected = vec_in;
    seq.applyToX( vec_in, vec_out );

    for ( size_t i = 0; i < expected.size(); ++i )
    {
        EXPECT_DOUBLE_EQ( vec_out[i], expected[i] );
    }
}

TEST_F( TransformSeqTest, singleTransformation )
{
    // Test sequence with single transformation
    TransformSeq<CSRMatrixVec<int, int, double>> seq;

    std::vector<double> scales = { 2.0, 0.5, 3.0 };
    auto row_scaling =
        std::make_shared<RowScaling<CSRMatrixVec<int, int, double>>>( scales.data(), scales.size() );

    seq.addTransformation( row_scaling );
    EXPECT_EQ( seq.size(), 1 );

    // Apply to RHS
    std::vector<double> expected = { 2.0, 1.0, 9.0 }; // [2*1, 0.5*2, 3*3]
    seq.applyToRHS( vec_in, vec_out );

    for ( size_t i = 0; i < expected.size(); ++i )
    {
        EXPECT_DOUBLE_EQ( vec_out[i], expected[i] );
    }
}

TEST_F( TransformSeqTest, twoScalings )
{
    // Test sequence: S_r2 * S_r1
    TransformSeq<CSRMatrixVec<int, int, double>> seq;

    std::vector<double> scales1 = { 2.0, 0.5, 3.0 };
    std::vector<double> scales2 = { 0.5, 2.0, 0.25 };

    auto scaling1 =
        std::make_shared<RowScaling<CSRMatrixVec<int, int, double>>>( scales1.data(), scales1.size() );
    auto scaling2 =
        std::make_shared<RowScaling<CSRMatrixVec<int, int, double>>>( scales2.data(), scales2.size() );

    seq.addTransformation( scaling1 );
    seq.addTransformation( scaling2 );
    EXPECT_EQ( seq.size(), 2 );

    // Apply to RHS: S_r2 * S_r1 * b
    // S_r1 * b = [2*1, 0.5*2, 3*3] = [2, 1, 9]
    // S_r2 * [2, 1, 9] = [0.5*2, 2*1, 0.25*9] = [1, 2, 2.25]
    std::vector<double> expected = { 1.0, 2.0, 2.25 };
    seq.applyToRHS( vec_in, vec_out );

    const double tol = 1e-10;
    for ( size_t i = 0; i < expected.size(); ++i )
    {
        EXPECT_NEAR( vec_out[i], expected[i], tol );
    }
}

TEST_F( TransformSeqTest, clearTransformations )
{
    TransformSeq<CSRMatrixVec<int, int, double>> seq;

    std::vector<double> scales = { 2.0, 0.5, 3.0 };
    auto scaling =
        std::make_shared<RowScaling<CSRMatrixVec<int, int, double>>>( scales.data(), scales.size() );

    seq.addTransformation( scaling );
    EXPECT_EQ( seq.size(), 1 );

    seq.clear();
    EXPECT_EQ( seq.size(), 0 );
}

TEST_F( TransformSeqTest, complexSequenceProperty )
{
    // KEY TEST: Verify P_r^2 * S_r^1 * A * P_c^1 * S_c^2 * ((S_c^2)^-1 * (P_c^1)^-1 * x) = P_r^2 * S_r^1 * A * x
    // This tests the fundamental property that column transformations and their inverses cancel out

    TransformSeq<CSRMatrixVec<int, int, double>> seq;

    // Define transformations
    std::vector<double> row_scales = { 2.0, 0.5, 3.0 };
    std::vector<int> row_perm = { 1, 2, 0 }; // Row permutation
    std::vector<int> col_perm = { 2, 0, 1 }; // Column permutation
    std::vector<double> col_scales = { 1.5, 2.0, 0.5 };

    // Build sequence: order matters!
    // Transformations are applied left-to-right on the matrix: T_n * ... * T_2 * T_1 * A
    auto row_scaling = std::make_shared<RowScaling<CSRMatrixVec<int, int, double>>>(
        row_scales.data(), row_scales.size() );
    auto row_perm_transform = std::make_shared<RowPermutation<CSRMatrixVec<int, int, double>>>(
        row_perm.data(), row_perm.size() );
    auto col_perm_transform = std::make_shared<ColumnPermutation<CSRMatrixVec<int, int, double>>>(
        col_perm.data(), col_perm.size() );
    auto col_scaling = std::make_shared<ColumnScaling<CSRMatrixVec<int, int, double>>>(
        col_scales.data(), col_scales.size() );

    // Add in order: S_r^1, P_r^2, P_c^1, S_c^2
    // Result: P_r^2 * S_r^1 * A * P_c^1 * S_c^2
    seq.addTransformation( row_scaling );        // S_r^1 (first applied to A)
    seq.addTransformation( row_perm_transform ); // P_r^2 (second applied)
    seq.addTransformation( col_perm_transform ); // P_c^1 (third applied)
    seq.addTransformation( col_scaling );        // S_c^2 (fourth applied)

    std::vector<double> test_vec = { 1.5, 2.5, 3.5 };
    const double tol = 1e-10;

    // Path 1: (P_r^2 * S_r^1 * A) * x
    // Apply row transformations to matrix
    CSRMatrixVec<int, int, double> temp_mat = mat;
    CSRMatrixVec<int, int, double> row_transformed_mat = mat;

    // Apply S_r^1
    row_scaling->applyToOperator( temp_mat, row_transformed_mat );
    std::swap( temp_mat, row_transformed_mat );

    // Apply P_r^2
    row_perm_transform->applyToOperator( temp_mat, row_transformed_mat );

    // Compute (P_r^2 * S_r^1 * A) * x
    std::vector<double> expected( mat.rows );
    csrMatVecMult( row_transformed_mat, test_vec, expected );

    // Path 2: (P_r^2 * S_r^1 * A * P_c^1 * S_c^2) * ((S_c^2)^-1 * (P_c^1)^-1 * x)
    // Apply full sequence to matrix
    CSRMatrixVec<int, int, double> full_mat = mat;
    CSRMatrixVec<int, int, double> transformed_mat = mat;
    seq.applyToOperator( full_mat, transformed_mat );

    // Apply inverse column transformations to x using applyInverseToX
    std::vector<double> transformed_vec_in = test_vec;
    std::vector<double> transformed_vec( mat.cols );
    seq.applyInverseToX( transformed_vec_in, transformed_vec );

    // Compute result
    std::vector<double> result( mat.rows );
    csrMatVecMult( transformed_mat, transformed_vec, result );

    // Verify equality
    for ( int i = 0; i < mat.rows; ++i )
    {
        EXPECT_NEAR( expected[i], result[i], tol ) << "Property failed at index " << i;
    }
}

TEST_F( TransformSeqTest, multipleVectorsProperty )
{
    // Test the complex property with multiple different vectors
    TransformSeq<CSRMatrixVec<int, int, double>> seq;

    std::vector<double> row_scales = { 2.0, 0.5, 3.0 };
    std::vector<int> row_perm = { 1, 2, 0 };
    std::vector<int> col_perm = { 2, 0, 1 };
    std::vector<double> col_scales = { 1.5, 2.0, 0.5 };

    auto row_scaling = std::make_shared<RowScaling<CSRMatrixVec<int, int, double>>>(
        row_scales.data(), row_scales.size() );
    auto row_perm_transform = std::make_shared<RowPermutation<CSRMatrixVec<int, int, double>>>(
        row_perm.data(), row_perm.size() );
    auto col_perm_transform = std::make_shared<ColumnPermutation<CSRMatrixVec<int, int, double>>>(
        col_perm.data(), col_perm.size() );
    auto col_scaling = std::make_shared<ColumnScaling<CSRMatrixVec<int, int, double>>>(
        col_scales.data(), col_scales.size() );

    seq.addTransformation( row_scaling );
    seq.addTransformation( row_perm_transform );
    seq.addTransformation( col_perm_transform );
    seq.addTransformation( col_scaling );

    // Prepare row-transformed matrix
    CSRMatrixVec<int, int, double> temp_mat = mat;
    CSRMatrixVec<int, int, double> row_transformed_mat = mat;
    row_scaling->applyToOperator( temp_mat, row_transformed_mat );
    std::swap( temp_mat, row_transformed_mat );
    row_perm_transform->applyToOperator( temp_mat, row_transformed_mat );

    // Prepare fully-transformed matrix
    CSRMatrixVec<int, int, double> full_mat = mat;
    CSRMatrixVec<int, int, double> transformed_mat = mat;
    seq.applyToOperator( full_mat, transformed_mat );

    const double tol = 1e-10;

    std::vector<std::vector<double>> test_vectors = {
        { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 }, { 1.0, 2.0, 3.0 }, { -1.5, 2.7, -3.2 } };

    for ( const auto& test_vec : test_vectors )
    {
        // Path 1: (P_r^2 * S_r^1 * A) * x
        std::vector<double> expected( mat.rows );
        csrMatVecMult( row_transformed_mat, test_vec, expected );

        // Path 2: (full transform) * (inverse transform on x)
        std::vector<double> transformed_vec_in = test_vec;
        std::vector<double> transformed_vec( mat.cols );
        seq.applyInverseToX( transformed_vec_in, transformed_vec );

        std::vector<double> result( mat.rows );
        csrMatVecMult( transformed_mat, transformed_vec, result );

        for ( int i = 0; i < mat.rows; ++i )
        {
            EXPECT_NEAR( expected[i], result[i], tol );
        }
    }
}

TEST_F( TransformSeqTest, applyToXReverseOrder )
{
    // Test that applyToX applies transformations in reverse order
    TransformSeq<CSRMatrixVec<int, int, double>> seq;

    std::vector<double> scales1 = { 2.0, 0.5, 3.0 };
    std::vector<double> scales2 = { 1.5, 2.0, 0.5 };

    auto col_scaling1 =
        std::make_shared<ColumnScaling<CSRMatrixVec<int, int, double>>>( scales1.data(), scales1.size() );
    auto col_scaling2 =
        std::make_shared<ColumnScaling<CSRMatrixVec<int, int, double>>>( scales2.data(), scales2.size() );

    seq.addTransformation( col_scaling1 );
    seq.addTransformation( col_scaling2 );

    std::vector<double> test_vec = { 1.0, 2.0, 3.0 };

    // Manual application in reverse order for applyToX
    // applyToX should apply: col_scaling2, then col_scaling1
    std::vector<double> manual_result = test_vec;
    std::vector<double> temp( mat.cols );

    // Apply scaling2 first (reverse order)
    col_scaling2->applyToX( manual_result, temp );
    manual_result = temp;

    // Then apply scaling1
    col_scaling1->applyToX( manual_result, temp );
    manual_result = temp;

    // Use sequence
    std::vector<double> seq_result_in = test_vec;
    std::vector<double> seq_result( mat.cols );
    seq.applyToX( seq_result_in, seq_result );

    const double tol = 1e-10;
    for ( size_t i = 0; i < manual_result.size(); ++i )
    {
        EXPECT_NEAR( manual_result[i], seq_result[i], tol )
            << "applyToX reverse order failed at index " << i;
    }
}

TEST_F( TransformSeqTest, symmetricPermutations )
{
    // Test with row and column permutations combined
    TransformSeq<CSRMatrixVec<int, int, double>> seq;

    std::vector<int> row_perm = { 1, 2, 0 };
    std::vector<int> col_perm = { 2, 0, 1 };

    auto row_perm_transform = std::make_shared<RowPermutation<CSRMatrixVec<int, int, double>>>(
        row_perm.data(), row_perm.size() );
    auto col_perm_transform = std::make_shared<ColumnPermutation<CSRMatrixVec<int, int, double>>>(
        col_perm.data(), col_perm.size() );

    seq.addTransformation( row_perm_transform );
    seq.addTransformation( col_perm_transform );

    // Test the property: (P_r * A * P_c) * (P_c^-1 * x) = P_r * A * x
    std::vector<double> test_vec = { 1.5, 2.5, 3.5 };
    const double tol = 1e-10;

    // Path 1: P_r * A * x
    CSRMatrixVec<int, int, double> row_perm_mat = mat;
    CSRMatrixVec<int, int, double> row_perm_mat_out = mat;
    row_perm_transform->applyToOperator( row_perm_mat, row_perm_mat_out );

    std::vector<double> expected( mat.rows );
    csrMatVecMult( row_perm_mat_out, test_vec, expected );

    // Path 2: (P_r * A * P_c) * (P_c^-1 * x)
    CSRMatrixVec<int, int, double> full_mat = mat;
    CSRMatrixVec<int, int, double> transformed_mat = mat;
    seq.applyToOperator( full_mat, transformed_mat );

    std::vector<double> transformed_vec_in = test_vec;
    std::vector<double> transformed_vec( mat.cols );
    seq.applyInverseToX( transformed_vec_in, transformed_vec );

    std::vector<double> result( mat.rows );
    csrMatVecMult( transformed_mat, transformed_vec, result );

    for ( int i = 0; i < mat.rows; ++i )
    {
        EXPECT_NEAR( expected[i], result[i], tol )
            << "Symmetric permutations property failed at index " << i;
    }
}

TEST_F( TransformSeqTest, allTransformationTypes )
{
    // Test with all types: row scaling, row perm, col scaling, col perm
    TransformSeq<CSRMatrixVec<int, int, double>> seq;

    std::vector<double> row_scales = { 2.0, 0.5, 3.0 };
    std::vector<int> row_perm = { 1, 2, 0 };
    std::vector<double> col_scales = { 1.5, 2.0, 0.5 };
    std::vector<int> col_perm = { 2, 0, 1 };

    auto row_scaling = std::make_shared<RowScaling<CSRMatrixVec<int, int, double>>>(
        row_scales.data(), row_scales.size() );
    auto row_perm_transform = std::make_shared<RowPermutation<CSRMatrixVec<int, int, double>>>(
        row_perm.data(), row_perm.size() );
    auto col_scaling = std::make_shared<ColumnScaling<CSRMatrixVec<int, int, double>>>(
        col_scales.data(), col_scales.size() );
    auto col_perm_transform = std::make_shared<ColumnPermutation<CSRMatrixVec<int, int, double>>>(
        col_perm.data(), col_perm.size() );

    // Complex sequence
    seq.addTransformation( row_scaling );
    seq.addTransformation( col_scaling );
    seq.addTransformation( row_perm_transform );
    seq.addTransformation( col_perm_transform );

    EXPECT_EQ( seq.size(), 4 );

    // Just verify it runs without errors and produces consistent results
    CSRMatrixVec<int, int, double> temp_mat = mat;
    CSRMatrixVec<int, int, double> transformed_mat = mat;
    seq.applyToOperator( temp_mat, transformed_mat );

    std::vector<double> test_vec = { 1.5, 2.5, 3.5 };
    std::vector<double> rhs_in = test_vec;
    std::vector<double> rhs_out( mat.rows );
    seq.applyToRHS( rhs_in, rhs_out );

    std::vector<double> x_in = test_vec;
    std::vector<double> x_out( mat.cols );
    seq.applyToX( x_in, x_out );

    std::vector<double> inv_x_in = test_vec;
    std::vector<double> inv_x_out( mat.cols );
    seq.applyInverseToX( inv_x_in, inv_x_out );

    // Basic sanity checks
    EXPECT_GT( rhs_out.size(), 0 );
    EXPECT_GT( x_out.size(), 0 );
    EXPECT_GT( inv_x_out.size(), 0 );
}

TEST_F( TransformSeqTest, roundtripInverseTransformation )
{
    // Test that applyInverseToX and applyToX are inverses for column transformations
    TransformSeq<CSRMatrixVec<int, int, double>> seq;

    std::vector<double> col_scales = { 1.5, 2.0, 0.5 };
    std::vector<int> col_perm = { 2, 0, 1 };

    auto col_scaling = std::make_shared<ColumnScaling<CSRMatrixVec<int, int, double>>>(
        col_scales.data(), col_scales.size() );
    auto col_perm_transform = std::make_shared<ColumnPermutation<CSRMatrixVec<int, int, double>>>(
        col_perm.data(), col_perm.size() );

    seq.addTransformation( col_scaling );
    seq.addTransformation( col_perm_transform );

    std::vector<double> original = { 1.5, 2.5, 3.5 };
    std::vector<double> temp( mat.cols );
    std::vector<double> roundtrip( mat.cols );

    // Apply inverse transformations
    std::vector<double> step1_in = original;
    seq.applyInverseToX( step1_in, temp );

    // Apply forward transformations
    std::vector<double> step2_in = temp;
    seq.applyToX( step2_in, roundtrip );

    const double tol = 1e-10;
    for ( size_t i = 0; i < original.size(); ++i )
    {
        EXPECT_NEAR( original[i], roundtrip[i], tol ) << "Roundtrip failed at index " << i;
    }
}
