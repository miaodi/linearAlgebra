#include "LinearSolverSystem.hpp"
#include "Transformation.hpp"
#include "matrix_utils.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace solver;
using namespace matrix_utils;

class LinearSolverSystemTest : public ::testing::Test {
protected:
    using MatrixType = CSRMatrixVec<int, int, double>;
    
    // Create a simple 3x3 matrix
    // [2  -1   0]
    // [-1  2  -1]
    // [0  -1   2]
    MatrixType createTestMatrix() {
        MatrixType A;
        A.rows = 3;
        A.ai = {0, 2, 5, 7};
        A.aj = {0, 1, 0, 1, 2, 1, 2};
        A.av = {2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0};
        return A;
    }
};

TEST_F(LinearSolverSystemTest, BasicSetup) {
    auto A = createTestMatrix();
    LinearSolverSystem<MatrixType> system;
    
    system.setMatrix(A);
    EXPECT_EQ(system.numRowTransformations(), 0);
    EXPECT_EQ(system.numColumnTransformations(), 0);
    EXPECT_FALSE(system.isTransformed());
}

TEST_F(LinearSolverSystemTest, IdentityTransformation) {
    auto A = createTestMatrix();
    LinearSolverSystem<MatrixType> system;
    system.setMatrix(A);
    
    // Add identity transformations
    auto identity_row = std::make_shared<IdentityTransformation<double>>(A.rows);
    auto identity_col = std::make_shared<IdentityTransformation<double>>(A.rows);
    
    system.addRowTransformation(identity_row);
    system.addColumnTransformation(identity_col);
    
    // Transform RHS
    std::vector<double> b = {1.0, 2.0, 3.0};
    auto b_bar = system.transformRHS(b);
    
    // Should be unchanged
    ASSERT_EQ(b_bar.size(), 3);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(b_bar[i], b[i]);
    }
    
    // Transform solution
    std::vector<double> x = {0.1, 0.2, 0.3};
    auto x_bar = system.transformSolution(x);
    
    // Should be unchanged
    ASSERT_EQ(x_bar.size(), 3);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(x_bar[i], x[i]);
    }
    
    // Recover solution
    auto x_recovered = system.recoverSolution(x_bar);
    ASSERT_EQ(x_recovered.size(), 3);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(x_recovered[i], x[i]);
    }
}

TEST_F(LinearSolverSystemTest, DiagonalScaling) {
    auto A = createTestMatrix();
    LinearSolverSystem<MatrixType> system;
    system.setMatrix(A);
    
    // Scale rows by [2, 3, 4]
    std::vector<double> row_scales = {2.0, 3.0, 4.0};
    auto row_scaling = std::make_shared<DiagonalScaling<double>>(row_scales);
    system.addRowTransformation(row_scaling);
    
    // Scale columns by [0.5, 0.5, 0.5]
    std::vector<double> col_scales = {0.5, 0.5, 0.5};
    auto col_scaling = std::make_shared<DiagonalScaling<double>>(col_scales);
    system.addColumnTransformation(col_scaling);
    
    // Transform RHS: b_bar = S_r * b
    std::vector<double> b = {1.0, 1.0, 1.0};
    auto b_bar = system.transformRHS(b);
    
    ASSERT_EQ(b_bar.size(), 3);
    EXPECT_DOUBLE_EQ(b_bar[0], 2.0);  // 2 * 1
    EXPECT_DOUBLE_EQ(b_bar[1], 3.0);  // 3 * 1
    EXPECT_DOUBLE_EQ(b_bar[2], 4.0);  // 4 * 1
    
    // Transform solution: x_bar = S_c^{-1} * x
    std::vector<double> x = {1.0, 1.0, 1.0};
    auto x_bar = system.transformSolution(x);
    
    ASSERT_EQ(x_bar.size(), 3);
    EXPECT_DOUBLE_EQ(x_bar[0], 2.0);  // 1 / 0.5
    EXPECT_DOUBLE_EQ(x_bar[1], 2.0);  // 1 / 0.5
    EXPECT_DOUBLE_EQ(x_bar[2], 2.0);  // 1 / 0.5
    
    // Recover solution: x = S_c * x_bar
    auto x_recovered = system.recoverSolution(x_bar);
    
    ASSERT_EQ(x_recovered.size(), 3);
    EXPECT_DOUBLE_EQ(x_recovered[0], 1.0);  // 0.5 * 2
    EXPECT_DOUBLE_EQ(x_recovered[1], 1.0);  // 0.5 * 2
    EXPECT_DOUBLE_EQ(x_recovered[2], 1.0);  // 0.5 * 2
}

TEST_F(LinearSolverSystemTest, RowPermutation) {
    auto A = createTestMatrix();
    LinearSolverSystem<MatrixType> system;
    system.setMatrix(A);
    
    // Permutation: [2, 0, 1] (row 0 -> 2, row 1 -> 0, row 2 -> 1)
    std::vector<int> perm = {2, 0, 1};
    auto row_perm = std::make_shared<RowPermutation<double>>(perm);
    system.addRowTransformation(row_perm);
    
    // Transform RHS
    std::vector<double> b = {1.0, 2.0, 3.0};
    auto b_bar = system.transformRHS(b);
    
    ASSERT_EQ(b_bar.size(), 3);
    EXPECT_DOUBLE_EQ(b_bar[0], 3.0);  // b[perm[0]] = b[2]
    EXPECT_DOUBLE_EQ(b_bar[1], 1.0);  // b[perm[1]] = b[0]
    EXPECT_DOUBLE_EQ(b_bar[2], 2.0);  // b[perm[2]] = b[1]
}

TEST_F(LinearSolverSystemTest, ColumnPermutation) {
    auto A = createTestMatrix();
    LinearSolverSystem<MatrixType> system;
    system.setMatrix(A);
    
    // Permutation: [2, 0, 1]
    std::vector<int> perm = {2, 0, 1};
    auto col_perm = std::make_shared<ColumnPermutation<double>>(perm);
    system.addColumnTransformation(col_perm);
    
    // Transform solution (inverse of column permutation)
    std::vector<double> x = {1.0, 2.0, 3.0};
    auto x_bar = system.transformSolution(x);
    
    ASSERT_EQ(x_bar.size(), 3);
    // Q^{-1} is the inverse permutation: [1, 2, 0]
    EXPECT_DOUBLE_EQ(x_bar[0], 2.0);  // x[inv_perm[0]] = x[1]
    EXPECT_DOUBLE_EQ(x_bar[1], 3.0);  // x[inv_perm[1]] = x[2]
    EXPECT_DOUBLE_EQ(x_bar[2], 1.0);  // x[inv_perm[2]] = x[0]
    
    // Recover solution
    auto x_recovered = system.recoverSolution(x_bar);
    
    ASSERT_EQ(x_recovered.size(), 3);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(x_recovered[i], x[i]);
    }
}

TEST_F(LinearSolverSystemTest, MultipleTransformations) {
    auto A = createTestMatrix();
    LinearSolverSystem<MatrixType> system;
    system.setMatrix(A);
    
    // Add multiple transformations
    std::vector<double> row_scales = {2.0, 2.0, 2.0};
    auto row_scaling = std::make_shared<DiagonalScaling<double>>(row_scales);
    
    std::vector<int> row_perm = {1, 2, 0};
    auto row_permutation = std::make_shared<RowPermutation<double>>(row_perm);
    
    system.addRowTransformation(row_scaling);
    system.addRowTransformation(row_permutation);
    
    EXPECT_EQ(system.numRowTransformations(), 2);
    
    // Transform RHS: b_bar = P * S * b
    std::vector<double> b = {1.0, 2.0, 3.0};
    auto b_bar = system.transformRHS(b);
    
    // First S*b = [2, 4, 6], then P*[2,4,6] = [4, 6, 2]
    ASSERT_EQ(b_bar.size(), 3);
    EXPECT_DOUBLE_EQ(b_bar[0], 4.0);
    EXPECT_DOUBLE_EQ(b_bar[1], 6.0);
    EXPECT_DOUBLE_EQ(b_bar[2], 2.0);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
