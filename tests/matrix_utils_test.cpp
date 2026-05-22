#include "io.hpp"
#include "matrix_utils.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <filesystem>
#include <type_traits>

using namespace matrix_utils;

class MatrixUtilsTest : public testing::Test {
protected:
    const double tol = 1e-10;
};

// Test CSRMatrixRawPtr swap
TEST_F(MatrixUtilsTest, CSRMatrixRawPtrSwap) {
    // Create two CSRMatrixRawPtr instances
    CSRMatrixRawPtr<int, int, double> mat1;
    mat1.rows = 3;
    mat1.cols = 3;
    
    std::vector<int> ai1 = {0, 2, 4, 6};
    std::vector<int> aj1 = {0, 1, 1, 2, 0, 2};
    std::vector<double> av1 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    
    mat1.ai = ai1.data();
    mat1.aj = aj1.data();
    mat1.av = av1.data();
    
    CSRMatrixRawPtr<int, int, double> mat2;
    mat2.rows = 2;
    mat2.cols = 2;
    
    std::vector<int> ai2 = {0, 1, 2};
    std::vector<int> aj2 = {1, 0};
    std::vector<double> av2 = {7.0, 8.0};
    
    mat2.ai = ai2.data();
    mat2.aj = aj2.data();
    mat2.av = av2.data();
    
    // Swap using member function
    mat1.swap(mat2);
    
    // Verify mat1 now has mat2's original data
    EXPECT_EQ(mat1.rows, 2);
    EXPECT_EQ(mat1.cols, 2);
    EXPECT_EQ(mat1.ai, ai2.data());
    EXPECT_EQ(mat1.aj, aj2.data());
    EXPECT_EQ(mat1.av, av2.data());
    
    // Verify mat2 now has mat1's original data
    EXPECT_EQ(mat2.rows, 3);
    EXPECT_EQ(mat2.cols, 3);
    EXPECT_EQ(mat2.ai, ai1.data());
    EXPECT_EQ(mat2.aj, aj1.data());
    EXPECT_EQ(mat2.av, av1.data());
    
    // Swap back using free function
    swap(mat1, mat2);
    
    // Verify swap back worked
    EXPECT_EQ(mat1.rows, 3);
    EXPECT_EQ(mat1.cols, 3);
    EXPECT_EQ(mat2.rows, 2);
    EXPECT_EQ(mat2.cols, 2);
}

// Test CSRMatrix swap
TEST_F(MatrixUtilsTest, CSRMatrixSwap) {
    CSRMatrix<int, int, double> mat1;
    mat1.rows = 3;
    mat1.cols = 3;
    mat1.ai_size = 4;
    mat1.aj_size = 6;
    mat1.av_size = 6;
    mat1.diagonal_size = 3;
    
    mat1.ai.reset(new int[4]{0, 2, 4, 6});
    mat1.aj.reset(new int[6]{0, 1, 1, 2, 0, 2});
    mat1.av.reset(new double[6]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    mat1.diagonal.reset(new int[3]{0, 3, 5});
    
    auto ai1_ptr = mat1.ai.get();
    auto aj1_ptr = mat1.aj.get();
    auto av1_ptr = mat1.av.get();
    auto diag1_ptr = mat1.diagonal.get();
    
    CSRMatrix<int, int, double> mat2;
    mat2.rows = 2;
    mat2.cols = 2;
    mat2.ai_size = 3;
    mat2.aj_size = 2;
    mat2.av_size = 2;
    mat2.diagonal_size = 0;
    
    mat2.ai.reset(new int[3]{0, 1, 2});
    mat2.aj.reset(new int[2]{1, 0});
    mat2.av.reset(new double[2]{7.0, 8.0});
    
    auto ai2_ptr = mat2.ai.get();
    auto aj2_ptr = mat2.aj.get();
    auto av2_ptr = mat2.av.get();
    
    // Swap using member function
    mat1.swap(mat2);
    
    // Verify dimensions swapped
    EXPECT_EQ(mat1.rows, 2);
    EXPECT_EQ(mat1.cols, 2);
    EXPECT_EQ(mat1.ai_size, 3);
    EXPECT_EQ(mat1.aj_size, 2);
    EXPECT_EQ(mat1.av_size, 2);
    EXPECT_EQ(mat1.diagonal_size, 0);
    
    EXPECT_EQ(mat2.rows, 3);
    EXPECT_EQ(mat2.cols, 3);
    EXPECT_EQ(mat2.ai_size, 4);
    EXPECT_EQ(mat2.aj_size, 6);
    EXPECT_EQ(mat2.av_size, 6);
    EXPECT_EQ(mat2.diagonal_size, 3);
    
    // Verify shared_ptr ownership swapped (pointers themselves swap, not contents)
    EXPECT_EQ(mat1.ai.get(), ai2_ptr);
    EXPECT_EQ(mat1.aj.get(), aj2_ptr);
    EXPECT_EQ(mat1.av.get(), av2_ptr);
    EXPECT_EQ(mat1.diagonal.get(), nullptr);
    
    EXPECT_EQ(mat2.ai.get(), ai1_ptr);
    EXPECT_EQ(mat2.aj.get(), aj1_ptr);
    EXPECT_EQ(mat2.av.get(), av1_ptr);
    EXPECT_EQ(mat2.diagonal.get(), diag1_ptr);
    
    // Verify data values
    EXPECT_EQ(mat1.AI()[0], 0);
    EXPECT_EQ(mat1.AI()[1], 1);
    EXPECT_EQ(mat1.AI()[2], 2);
    EXPECT_EQ(mat1.AJ()[0], 1);
    EXPECT_EQ(mat1.AJ()[1], 0);
    EXPECT_NEAR(mat1.AV()[0], 7.0, tol);
    EXPECT_NEAR(mat1.AV()[1], 8.0, tol);
    
    // Swap back using free function
    swap(mat1, mat2);
    
    // Verify swap back
    EXPECT_EQ(mat1.rows, 3);
    EXPECT_EQ(mat2.rows, 2);
}

// Test CSRMatrixVec swap
TEST_F(MatrixUtilsTest, CSRMatrixVecSwap) {
    CSRMatrixVec<int, int, double> mat1;
    mat1.rows = 3;
    mat1.cols = 3;
    mat1.ai = {0, 2, 4, 6};
    mat1.aj = {0, 1, 1, 2, 0, 2};
    mat1.av = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    
    CSRMatrixVec<int, int, double> mat2;
    mat2.rows = 2;
    mat2.cols = 2;
    mat2.ai = {0, 1, 2};
    mat2.aj = {1, 0};
    mat2.av = {7.0, 8.0};
    
    // Swap using member function
    mat1.swap(mat2);
    
    // Verify dimensions swapped
    EXPECT_EQ(mat1.rows, 2);
    EXPECT_EQ(mat1.cols, 2);
    EXPECT_EQ(mat2.rows, 3);
    EXPECT_EQ(mat2.cols, 3);
    
    // Verify vector contents swapped
    ASSERT_EQ(mat1.ai.size(), 3);
    EXPECT_EQ(mat1.ai[0], 0);
    EXPECT_EQ(mat1.ai[1], 1);
    EXPECT_EQ(mat1.ai[2], 2);
    
    ASSERT_EQ(mat1.aj.size(), 2);
    EXPECT_EQ(mat1.aj[0], 1);
    EXPECT_EQ(mat1.aj[1], 0);
    
    ASSERT_EQ(mat1.av.size(), 2);
    EXPECT_NEAR(mat1.av[0], 7.0, tol);
    EXPECT_NEAR(mat1.av[1], 8.0, tol);
    
    ASSERT_EQ(mat2.ai.size(), 4);
    EXPECT_EQ(mat2.ai[0], 0);
    EXPECT_EQ(mat2.ai[1], 2);
    EXPECT_EQ(mat2.ai[2], 4);
    EXPECT_EQ(mat2.ai[3], 6);
    
    ASSERT_EQ(mat2.aj.size(), 6);
    EXPECT_EQ(mat2.aj[0], 0);
    EXPECT_EQ(mat2.aj[1], 1);
    EXPECT_EQ(mat2.aj[2], 1);
    EXPECT_EQ(mat2.aj[3], 2);
    
    ASSERT_EQ(mat2.av.size(), 6);
    EXPECT_NEAR(mat2.av[0], 1.0, tol);
    EXPECT_NEAR(mat2.av[1], 2.0, tol);
    EXPECT_NEAR(mat2.av[2], 3.0, tol);
    
    // Swap back using free function
    swap(mat1, mat2);
    
    // Verify swap back
    EXPECT_EQ(mat1.rows, 3);
    EXPECT_EQ(mat2.rows, 2);
    ASSERT_EQ(mat1.ai.size(), 4);
    ASSERT_EQ(mat2.ai.size(), 3);
}

// Test that swap is noexcept
TEST_F(MatrixUtilsTest, SwapIsNoexcept) {
    // CSRMatrixRawPtr
    EXPECT_TRUE(noexcept(std::declval<CSRMatrixRawPtr<int, int, double>&>().swap(
        std::declval<CSRMatrixRawPtr<int, int, double>&>())));
    
    // CSRMatrix
    EXPECT_TRUE(noexcept(std::declval<CSRMatrix<int, int, double>&>().swap(
        std::declval<CSRMatrix<int, int, double>&>())));
    
    // CSRMatrixVec
    EXPECT_TRUE(noexcept(std::declval<CSRMatrixVec<int, int, double>&>().swap(
        std::declval<CSRMatrixVec<int, int, double>&>())));
}

// Test that matrices are swappable (satisfies std::is_swappable_v)
TEST_F(MatrixUtilsTest, MatricesAreSwappable) {
    EXPECT_TRUE((std::is_swappable_v<CSRMatrixRawPtr<int, int, double>>));
    EXPECT_TRUE((std::is_swappable_v<CSRMatrix<int, int, double>>));
    EXPECT_TRUE((std::is_swappable_v<CSRMatrixVec<int, int, double>>));
}

TEST_F(MatrixUtilsTest, MatrixPathIORoundTrip) {
    CSRMatrixVec<int, int, double> mat;
    mat.rows = 2;
    mat.cols = 3;
    mat.ai = {0, 2, 3};
    mat.aj = {0, 2, 1};
    mat.av = {1.5, 2.5, 3.5};

    const auto tmp_dir = std::filesystem::temp_directory_path();
    const auto mtx_path = tmp_dir / "linear_algebra_matrix_io_roundtrip.mtx";
    const auto bin_path = tmp_dir / "linear_algebra_matrix_io_roundtrip.bin";

    writeMatrix(mat, mtx_path.string(), MatrixDataType::MatrixMarket);
    CSRMatrixVec<int, int, double> mtx_loaded;
    readMatrix(mtx_path.string(), mtx_loaded, MatrixDataType::MatrixMarket);
    EXPECT_EQ(mtx_loaded.rows, mat.rows);
    EXPECT_EQ(mtx_loaded.cols, mat.cols);
    EXPECT_EQ(mtx_loaded.ai, mat.ai);
    EXPECT_EQ(mtx_loaded.aj, mat.aj);
    EXPECT_EQ(mtx_loaded.av, mat.av);

    writeMatrix(mat, bin_path.string(), MatrixDataType::Binary);
    CSRMatrixVec<int, int, double> bin_loaded;
    readMatrix(bin_path.string(), bin_loaded, MatrixDataType::Binary);
    EXPECT_EQ(bin_loaded.rows, mat.rows);
    EXPECT_EQ(bin_loaded.cols, mat.cols);
    EXPECT_EQ(bin_loaded.ai, mat.ai);
    EXPECT_EQ(bin_loaded.aj, mat.aj);
    EXPECT_EQ(bin_loaded.av, mat.av);

    std::filesystem::remove(mtx_path);
    std::filesystem::remove(bin_path);
}
