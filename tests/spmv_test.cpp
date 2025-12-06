
#include "matrix_utils.hpp"
#include "spmv.hpp"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <numeric>

using namespace matrix_utils;

// The fixture for testing class Foo.
class spmv_Test : public testing::Test {
protected:
  std::vector<CSRMatrixVec<int, int, double>> _mats;

  const double _tol = 1e-8;

  spmv_Test() {

    std::vector<int> csr_rows;
    std::vector<int> csr_cols;
    std::vector<double> csr_vals;

    std::ifstream f;
    f.open("data/ex5.mtx"); // https://sparse.tamu.edu/FIDAP/ex5
    utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
    f.close();
    _mats.emplace_back();
    _mats.back().rows = csr_rows.size() - 1;
    _mats.back().ai = std::move(csr_rows);
    _mats.back().aj = std::move(csr_cols);
    _mats.back().av = std::move(csr_vals);

    f.open("data/nos5.mtx");
    utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
    f.close();
    _mats.emplace_back();
    _mats.back().rows = csr_rows.size() - 1;
    _mats.back().ai = std::move(csr_rows);
    _mats.back().aj = std::move(csr_cols);
    _mats.back().av = std::move(csr_vals);

    f.open("data/s3rmt3m3.mtx");
    utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
    f.close();
    _mats.emplace_back();
    _mats.back().rows = csr_rows.size() - 1;
    _mats.back().ai = std::move(csr_rows);
    _mats.back().aj = std::move(csr_cols);
    _mats.back().av = std::move(csr_vals);

    f.open("data/rdist1.mtx");
    utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
    f.close();
    _mats.emplace_back();
    _mats.back().rows = csr_rows.size() - 1;
    _mats.back().ai = std::move(csr_rows);
    _mats.back().aj = std::move(csr_cols);
    _mats.back().av = std::move(csr_vals);
  }

  ~spmv_Test() override {
    // You can do clean-up work that doesn't throw exceptions here.
  }

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  void SetUp() override {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  void TearDown() override {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }

  // Helper function to compute L2 norm
  double l2_norm(const std::vector<double> &vec) const {
    return std::sqrt(std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0));
  }

  // Class members declared here can be used by all tests in the test suite
  // for Foo.
};

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

// Test RowBalancedParallelSPMV with both Scalar and SIMD kernels, 1-8 threads
TEST_F(spmv_Test, row_balanced_parallel_spmv) {
  for (int nthreads = 1; nthreads <= 8; nthreads++) {
    for (auto &mat : _mats) {
      std::vector<double> b(mat.rows, 1.0);
      std::vector<double> x_scalar(mat.rows, 0.0);
      std::vector<double> x_simd(mat.rows, 0.0);
      std::vector<double> x_serial(mat.rows, 0.0);

      SerialSPMV<int, int, double> spmv;
      spmv.preprocess(mat.rows, mat.AI(), mat.AJ(), mat.AV());
      spmv(b.data(), x_serial.data(), 1., 0.);

      // Test Scalar kernel
      RowBalancedParallelSPMV<int, int, double, RowDotKernel::Scalar> p_spmv_scalar(nthreads);
      p_spmv_scalar.preprocess(mat.rows, mat.AI(), mat.AJ(), mat.AV());
      p_spmv_scalar(b.data(), x_scalar.data(), 1., 0.);
      
      // Test SIMD kernel
      RowBalancedParallelSPMV<int, int, double, RowDotKernel::Simd> p_spmv_simd(nthreads);
      p_spmv_simd.preprocess(mat.rows, mat.AI(), mat.AJ(), mat.AV());
      p_spmv_simd(b.data(), x_simd.data(), 1., 0.);

      // Both should match serial exactly (same summation order)
      for (int i = 0; i < mat.rows; i++) {
        EXPECT_NEAR(x_scalar[i], x_serial[i], _tol * std::abs(x_serial[i])) 
          << "Scalar failed at row " << i << " with " << nthreads << " threads";
        EXPECT_NEAR(x_simd[i], x_serial[i], _tol * std::abs(x_serial[i]))
          << "SIMD failed at row " << i << " with " << nthreads << " threads";
      }
    }
  }
}

// Test ALBUSSPMV with both Scalar and SIMD kernels, 1-8 threads
TEST_F(spmv_Test, ALBUS_spmv) {
  for (int nthreads = 1; nthreads <= 8; nthreads++) {
    for (auto &mat : _mats) {
      std::vector<double> b(mat.rows, 1.0);
      std::vector<double> x_scalar(mat.rows, 0.0);
      std::vector<double> x_simd(mat.rows, 0.0);
      std::vector<double> x_serial(mat.rows, 0.0);

      SerialSPMV<int, int, double> spmv;
      spmv.preprocess(mat.rows, mat.AI(), mat.AJ(), mat.AV());
      spmv(b.data(), x_serial.data(), 1., 0.);

      // Test Scalar kernel
      ALBUSSPMV<int, int, double, RowDotKernel::Scalar> albus_scalar(nthreads);
      albus_scalar.preprocess(mat.rows, mat.AI(), mat.AJ(), mat.AV());
      albus_scalar(b.data(), x_scalar.data(), 1., 0.);

      // Test SIMD kernel
      ALBUSSPMV<int, int, double, RowDotKernel::Simd> albus_simd(nthreads);
      albus_simd.preprocess(mat.rows, mat.AI(), mat.AJ(), mat.AV());
      albus_simd(b.data(), x_simd.data(), 1., 0.);

      // ALBUS uses different summation order (by nnz), allow small rounding differences
      const double serial_l2 = l2_norm(x_serial);
      for (int i = 0; i < mat.rows; i++) {
        EXPECT_NEAR(x_scalar[i], x_serial[i], _tol * serial_l2)
          << "Scalar failed at row " << i << " with " << nthreads << " threads";
        EXPECT_NEAR(x_simd[i], x_serial[i], _tol * serial_l2)
          << "SIMD failed at row " << i << " with " << nthreads << " threads";
      }

      // SIMD and Scalar should be very close to each other
      for (int i = 0; i < mat.rows; i++) {
        double abs_err = std::abs(x_simd[i] - x_scalar[i]);
        double rel_err = std::abs(x_scalar[i]) > 1e-15 ? abs_err / std::abs(x_scalar[i]) : abs_err;
        EXPECT_LT(rel_err, 1e-12) << "SIMD vs Scalar mismatch at row " << i 
                                   << " with " << nthreads << " threads";
      }
    }
  }
}

TEST_F(spmv_Test, CAMLB_spmv) {
  for (int nthreads = 1; nthreads <= 8; nthreads++) {
    for (auto &mat : _mats) {
      std::vector<double> b(mat.rows, 1.0);
      std::vector<double> x_scalar(mat.rows, 0.0);
      std::vector<double> x_simd(mat.rows, 0.0);
      std::vector<double> x_serial(mat.rows, 0.0);

      SerialSPMV<int, int, double> spmv;
      spmv.preprocess(mat.rows, mat.AI(), mat.AJ(), mat.AV());
      spmv(b.data(), x_serial.data(), 1., 0.);

      // Test Scalar kernel with CAMLB workload partitioning
      ALBUSSPMV<int, int, double, RowDotKernel::Scalar, WorkloadMode::CAMLB> camlb_scalar(nthreads);
      camlb_scalar.preprocess(mat.rows, mat.AI(), mat.AJ(), mat.AV());
      camlb_scalar(b.data(), x_scalar.data(), 1., 0.);

      // Test SIMD kernel with CAMLB workload partitioning
      ALBUSSPMV<int, int, double, RowDotKernel::Simd, WorkloadMode::CAMLB> camlb_simd(nthreads);
      camlb_simd.preprocess(mat.rows, mat.AI(), mat.AJ(), mat.AV());
      camlb_simd(b.data(), x_simd.data(), 1., 0.);

      // CAMLB uses different summation order (by cache-aware workload), allow small rounding differences
      const double serial_l2 = l2_norm(x_serial);
      for (int i = 0; i < mat.rows; i++) {
        EXPECT_NEAR(x_scalar[i], x_serial[i], _tol * serial_l2)
          << "CAMLB Scalar failed at row " << i << " with " << nthreads << " threads";
        EXPECT_NEAR(x_simd[i], x_serial[i], _tol * serial_l2)
          << "CAMLB SIMD failed at row " << i << " with " << nthreads << " threads";
      }

      // SIMD and Scalar should be very close to each other
      for (int i = 0; i < mat.rows; i++) {
        double abs_err = std::abs(x_simd[i] - x_scalar[i]);
        double rel_err = std::abs(x_scalar[i]) > 1e-15 ? abs_err / std::abs(x_scalar[i]) : abs_err;
        EXPECT_LT(rel_err, 1e-12) << "CAMLB SIMD vs Scalar mismatch at row " << i 
                                   << " with " << nthreads << " threads";
      }
    }
  }
}

// Small targeted test cases for ALBUS edge cases
TEST_F(spmv_Test, ALBUS_edge_cases) {
  struct TestCase {
    std::string name;
    std::vector<int> ai;
    std::vector<int> aj;
    std::vector<double> av;
    int nrows;
    int nvec;
    int nthreads;
  };

  std::vector<TestCase> test_cases = {
    {"diagonal", {0, 1, 2, 3, 4}, {0, 1, 2, 3}, {1.0, 2.0, 3.0, 4.0}, 4, 4, 2},
    {"tridiagonal", {0, 2, 5, 8, 11, 13}, {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4}, 
     {2, -1, -1, 2, -1, -1, 2, -1, -1, 2, -1, -1, 2}, 5, 5, 2},
    {"unbalanced_rows", {0, 1, 6, 7, 12}, {0, 0, 1, 2, 3, 4, 2, 0, 1, 2, 3, 4},
     {5.0, 1, 1, 1, 1, 1, 3.0, 1, 1, 1, 1, 1}, 4, 5, 2},
    {"partial_row_split", {0, 10, 11, 12}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3}, 3, 10, 3},
    {"single_row_many_threads", {0, 12}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
     {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 1, 12, 4}
  };

  for (const auto& tc : test_cases) {
    std::vector<double> b(tc.nvec, 1.0);
    std::vector<double> x(tc.nrows, 0.0);
    std::vector<double> x_serial(tc.nrows, 0.0);

    SerialSPMV<int, int, double> spmv;
    spmv.preprocess(tc.nrows, tc.ai.data(), tc.aj.data(), tc.av.data());
    spmv(b.data(), x_serial.data(), 1.0, 0.0);

    ALBUSSPMV<int, int, double> albus_spmv(tc.nthreads);
    albus_spmv.preprocess(tc.nrows, tc.ai.data(), tc.aj.data(), tc.av.data());
    albus_spmv(b.data(), x.data(), 1.0, 0.0);

    for (int i = 0; i < tc.nrows; i++) {
      EXPECT_DOUBLE_EQ(x[i], x_serial[i]) << "Test '" << tc.name << "' failed at row " << i;
    }
  }
}

// Test CAMLB edge cases
TEST_F(spmv_Test, CAMLB_edge_cases) {
  struct TestCase {
    std::string name;
    std::vector<int> ai;
    std::vector<int> aj;
    std::vector<double> av;
    int nrows;
    int nvec;
    int nthreads;
  };

  std::vector<TestCase> test_cases = {
    {"diagonal", {0, 1, 2, 3, 4}, {0, 1, 2, 3}, {1.0, 2.0, 3.0, 4.0}, 4, 4, 2},
    {"tridiagonal", {0, 2, 5, 8, 11, 13}, {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4}, 
     {2, -1, -1, 2, -1, -1, 2, -1, -1, 2, -1, -1, 2}, 5, 5, 2},
    {"unbalanced_rows", {0, 1, 6, 7, 12}, {0, 0, 1, 2, 3, 4, 2, 0, 1, 2, 3, 4},
     {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 4, 5, 3},
  };

  for (auto &tc : test_cases) {
    std::vector<double> b(tc.nvec, 1.0);
    std::vector<double> x(tc.nrows, 0.0);
    std::vector<double> x_serial(tc.nrows, 0.0);

    SerialSPMV<int, int, double> spmv;
    spmv.preprocess(tc.nrows, tc.ai.data(), tc.aj.data(), tc.av.data());
    spmv(b.data(), x_serial.data(), 1.0, 0.0);

    ALBUSSPMV<int, int, double, RowDotKernel::Scalar, WorkloadMode::CAMLB> camlb_spmv(tc.nthreads);
    camlb_spmv.preprocess(tc.nrows, tc.ai.data(), tc.aj.data(), tc.av.data());
    camlb_spmv(b.data(), x.data(), 1.0, 0.0);

    for (int i = 0; i < tc.nrows; i++) {
      EXPECT_DOUBLE_EQ(x[i], x_serial[i]) 
        << "CAMLB failed for case '" << tc.name << "' at row " << i;
    }
  }
}

// Test alpha/beta parameter handling
TEST_F(spmv_Test, ALBUS_alpha_beta) {
  std::vector<int> ai = {0, 10, 11, 12};
  std::vector<int> aj = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1};
  std::vector<double> av = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3};
  std::vector<double> b(10, 1.0);
  int n = 3;
  int base = 0;

  // Test cases: {alpha, beta, initial_x_value, description}
  std::vector<std::tuple<double, double, double, std::string>> test_cases = {
    {2.0, 0.0, 5.0, "alpha=2, beta=0"},
    {1.0, 1.0, 5.0, "alpha=1, beta=1"},
    {0.5, 2.0, 10.0, "fractional alpha/beta"},
    {3.0, -1.0, 8.0, "negative beta"},
    {-1.0, 1.0, 4.0, "negative alpha"},
    {0.0, 1.0, 7.0, "alpha=0"},
    {1.5, 0.5, 12.0, "both fractional"}
  };

  for (const auto& [alpha, beta, init_val, desc] : test_cases) {
    std::vector<double> x(n, init_val);
    std::vector<double> x_serial(n, init_val);

    SerialSPMV<int, int, double> spmv;
    spmv.preprocess(n, ai.data(), aj.data(), av.data());
    spmv(b.data(), x_serial.data(), alpha, beta);

    ALBUSSPMV<int, int, double> albus_spmv(3);
    albus_spmv.preprocess(n, ai.data(), aj.data(), av.data());
    albus_spmv(b.data(), x.data(), alpha, beta);

    for (int i = 0; i < n; i++) {
      EXPECT_DOUBLE_EQ(x[i], x_serial[i]) << "Failed for " << desc << " at row " << i;
    }
  }

  // Test row splitting with non-trivial beta
  std::vector<int> ai2 = {0, 8, 16};
  std::vector<int> aj2 = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<double> av2 = {1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> b2(8, 1.0);
  std::vector<double> x2(2, 100.0);
  std::vector<double> x2_serial(2, 100.0);

  SerialSPMV<int, int, double> spmv2;
  spmv2.preprocess(2, ai2.data(), aj2.data(), av2.data());
  spmv2(b2.data(), x2_serial.data(), 0.5, 1.5);

  ALBUSSPMV<int, int, double> albus_spmv2(4);
  albus_spmv2.preprocess(2, ai2.data(), aj2.data(), av2.data());
  albus_spmv2(b2.data(), x2.data(), 0.5, 1.5);

  for (int i = 0; i < 2; i++) {
    EXPECT_DOUBLE_EQ(x2[i], x2_serial[i]) << "Row split with beta failed at row " << i;
  }
}

// Test corner cases: empty matrix, null nnz rows, 1-indexing, extreme thread counts
TEST_F(spmv_Test, ALBUS_corner_cases) {
  // Case 1: Empty matrix (size=0)
  {
    std::vector<int> ai = {0};
    std::vector<int> aj;
    std::vector<double> av;
    std::vector<double> b;
    std::vector<double> x;
    
    ALBUSSPMV<int, int, double> albus(2);
    albus.preprocess(0, ai.data(), aj.data(), av.data());
    // Should not crash
    albus(b.data(), x.data(), 1.0, 0.0);
  }

  // Case 2: Matrix with nnz=0 but size>0
  {
    std::vector<int> ai = {0, 0, 0, 0, 0};  // 4 rows, all empty
    std::vector<int> aj;
    std::vector<double> av;
    std::vector<double> b(4, 1.0);
    std::vector<double> x(4, 5.0);
    std::vector<double> x_serial(4, 5.0);
    
    SerialSPMV<int, int, double> spmv;
    spmv.preprocess(4, ai.data(), aj.data(), av.data());
    spmv(b.data(), x_serial.data(), 1.0, 0.0);
    
    ALBUSSPMV<int, int, double> albus(2);
    albus.preprocess(4, ai.data(), aj.data(), av.data());
    albus(b.data(), x.data(), 1.0, 0.0);
    
    for (int i = 0; i < 4; i++) {
      EXPECT_DOUBLE_EQ(x[i], x_serial[i]) << "Empty matrix failed at row " << i;
    }
  }

  // Case 3: Matrix with empty rows in middle
  {
    std::vector<int> ai = {0, 2, 2, 2, 5};  // rows 1,2 are empty
    std::vector<int> aj = {0, 1, 2, 3, 4};
    std::vector<double> av = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> b(5, 1.0);
    std::vector<double> x(4, 0.0);
    std::vector<double> x_serial(4, 0.0);
    
    SerialSPMV<int, int, double> spmv;
    spmv.preprocess(4, ai.data(), aj.data(), av.data());
    spmv(b.data(), x_serial.data(), 1.0, 0.0);
    
    ALBUSSPMV<int, int, double> albus(3);
    albus.preprocess(4, ai.data(), aj.data(), av.data());
    albus(b.data(), x.data(), 1.0, 0.0);
    
    for (int i = 0; i < 4; i++) {
      EXPECT_DOUBLE_EQ(x[i], x_serial[i]) << "Empty rows in middle failed at row " << i;
    }
  }

  // Case 4: 1-indexed matrix (base=1)
  {
    std::vector<int> ai = {1, 3, 5, 7};  // 1-indexed
    std::vector<int> aj = {1, 2, 1, 3, 2, 3};  // 1-indexed columns
    std::vector<double> av = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<double> b = {0.0, 10.0, 20.0, 30.0};  // 0-indexed still for vector
    std::vector<double> x(3, 0.0);
    std::vector<double> x_serial(3, 0.0);
    
    SerialSPMV<int, int, double> spmv;
    spmv.preprocess(3, ai.data(), aj.data(), av.data());
    spmv(b.data(), x_serial.data(), 1.0, 0.0);
    
    ALBUSSPMV<int, int, double> albus(2);
    albus.preprocess(3, ai.data(), aj.data(), av.data());
    albus(b.data(), x.data(), 1.0, 0.0);
    
    for (int i = 0; i < 3; i++) {
      EXPECT_DOUBLE_EQ(x[i], x_serial[i]) << "1-indexed matrix failed at row " << i;
    }
  }

  // Case 5: More threads than nnz
  {
    std::vector<int> ai = {0, 1, 2, 3};  // 3 rows, 3 nnz
    std::vector<int> aj = {0, 1, 2};
    std::vector<double> av = {1.0, 2.0, 3.0};
    std::vector<double> b(3, 1.0);
    std::vector<double> x(3, 0.0);
    std::vector<double> x_serial(3, 0.0);
    
    SerialSPMV<int, int, double> spmv;
    spmv.preprocess(3, ai.data(), aj.data(), av.data());
    spmv(b.data(), x_serial.data(), 1.0, 0.0);
    
    ALBUSSPMV<int, int, double> albus(10);  // 10 threads for 3 nnz
    albus.preprocess(3, ai.data(), aj.data(), av.data());
    albus(b.data(), x.data(), 1.0, 0.0);
    
    for (int i = 0; i < 3; i++) {
      EXPECT_DOUBLE_EQ(x[i], x_serial[i]) << "More threads than nnz failed at row " << i;
    }
  }

  // Case 6: Single nnz in entire matrix
  {
    std::vector<int> ai = {0, 0, 1, 1, 1};  // only row 1 has one nnz
    std::vector<int> aj = {2};
    std::vector<double> av = {42.0};
    std::vector<double> b(3, 1.0);
    std::vector<double> x(4, 0.0);
    std::vector<double> x_serial(4, 0.0);
    
    SerialSPMV<int, int, double> spmv;
    spmv.preprocess(4, ai.data(), aj.data(), av.data());
    spmv(b.data(), x_serial.data(), 1.0, 0.0);
    
    ALBUSSPMV<int, int, double> albus(4);
    albus.preprocess(4, ai.data(), aj.data(), av.data());
    albus(b.data(), x.data(), 1.0, 0.0);
    
    for (int i = 0; i < 4; i++) {
      EXPECT_DOUBLE_EQ(x[i], x_serial[i]) << "Single nnz failed at row " << i;
    }
  }

  // Case 7: All nnz in first row (last row is endRow boundary)
  {
    std::vector<int> ai = {0, 10, 10, 10};  // all nnz in first row
    std::vector<int> aj = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<double> av(10, 1.0);
    std::vector<double> b(10, 2.0);
    std::vector<double> x(3, 0.0);
    std::vector<double> x_serial(3, 0.0);
    
    SerialSPMV<int, int, double> spmv;
    spmv.preprocess(3, ai.data(), aj.data(), av.data());
    spmv(b.data(), x_serial.data(), 1.0, 0.0);
    
    ALBUSSPMV<int, int, double> albus(4);
    albus.preprocess(3, ai.data(), aj.data(), av.data());
    albus(b.data(), x.data(), 1.0, 0.0);
    
    for (int i = 0; i < 3; i++) {
      EXPECT_DOUBLE_EQ(x[i], x_serial[i]) << "All nnz in first row failed at row " << i;
    }
  }

  // Case 8: Test with beta modes on empty matrix
  {
    std::vector<int> ai = {0, 0, 0};
    std::vector<int> aj;
    std::vector<double> av;
    std::vector<double> b(1, 1.0);
    
    // beta=0
    std::vector<double> x1(2, 5.0);
    ALBUSSPMV<int, int, double> albus1(2);
    albus1.preprocess(2, ai.data(), aj.data(), av.data());
    albus1(b.data(), x1.data(), 1.0, 0.0);
    EXPECT_DOUBLE_EQ(x1[0], 0.0);
    EXPECT_DOUBLE_EQ(x1[1], 0.0);
    
    // beta=1
    std::vector<double> x2(2, 5.0);
    ALBUSSPMV<int, int, double> albus2(2);
    albus2.preprocess(2, ai.data(), aj.data(), av.data());
    albus2(b.data(), x2.data(), 1.0, 1.0);
    EXPECT_DOUBLE_EQ(x2[0], 5.0);
    EXPECT_DOUBLE_EQ(x2[1], 5.0);
    
    // beta=2
    std::vector<double> x3(2, 5.0);
    ALBUSSPMV<int, int, double> albus3(2);
    albus3.preprocess(2, ai.data(), aj.data(), av.data());
    albus3(b.data(), x3.data(), 1.0, 2.0);
    EXPECT_DOUBLE_EQ(x3[0], 10.0);
    EXPECT_DOUBLE_EQ(x3[1], 10.0);
  }
}

// Test CAMLB corner cases
TEST_F(spmv_Test, CAMLB_corner_cases) {
  // Case 1: Empty matrix (size=0)
  {
    std::vector<int> ai = {0};
    std::vector<int> aj;
    std::vector<double> av;
    std::vector<double> b;
    std::vector<double> x;
    
    ALBUSSPMV<int, int, double, RowDotKernel::Scalar, WorkloadMode::CAMLB> camlb(2);
    camlb.preprocess(0, ai.data(), aj.data(), av.data());
    // Should not crash
    camlb(b.data(), x.data(), 1.0, 0.0);
  }

  // Case 2: Matrix with nnz=0 but size>0
  {
    std::vector<int> ai = {0, 0, 0, 0, 0};  // 4 rows, all empty
    std::vector<int> aj;
    std::vector<double> av;
    std::vector<double> b(4, 1.0);
    std::vector<double> x(4, 0.0);
    std::vector<double> x_serial(4, 0.0);
    
    SerialSPMV<int, int, double> spmv;
    spmv.preprocess(4, ai.data(), aj.data(), av.data());
    spmv(b.data(), x_serial.data(), 1.0, 0.0);
    
    ALBUSSPMV<int, int, double, RowDotKernel::Scalar, WorkloadMode::CAMLB> camlb(2);
    camlb.preprocess(4, ai.data(), aj.data(), av.data());
    camlb(b.data(), x.data(), 1.0, 0.0);
    
    for (int i = 0; i < 4; i++) {
      EXPECT_DOUBLE_EQ(x[i], x_serial[i]) << "CAMLB empty matrix failed at row " << i;
    }
  }

  // Case 3: More threads than nnz (workload-based partitioning should handle this)
  {
    std::vector<int> ai = {0, 1, 2, 3};  // 3 rows, 3 nnz
    std::vector<int> aj = {0, 1, 2};
    std::vector<double> av = {1.0, 2.0, 3.0};
    std::vector<double> b(3, 1.0);
    std::vector<double> x(3, 0.0);
    std::vector<double> x_serial(3, 0.0);
    
    SerialSPMV<int, int, double> spmv;
    spmv.preprocess(3, ai.data(), aj.data(), av.data());
    spmv(b.data(), x_serial.data(), 1.0, 0.0);
    
    ALBUSSPMV<int, int, double, RowDotKernel::Scalar, WorkloadMode::CAMLB> camlb(10);  // 10 threads for 3 nnz
    camlb.preprocess(3, ai.data(), aj.data(), av.data());
    camlb(b.data(), x.data(), 1.0, 0.0);
    
    for (int i = 0; i < 3; i++) {
      EXPECT_DOUBLE_EQ(x[i], x_serial[i]) << "CAMLB more threads than nnz failed at row " << i;
    }
  }

  // Case 4: Single element per row
  {
    std::vector<int> ai = {0, 1, 2, 3, 4};
    std::vector<int> aj = {0, 1, 2, 3};
    std::vector<double> av(4, 1.0);
    std::vector<double> b(4, 2.0);
    std::vector<double> x(4, 0.0);
    std::vector<double> x_serial(4, 0.0);
    
    SerialSPMV<int, int, double> spmv;
    spmv.preprocess(4, ai.data(), aj.data(), av.data());
    spmv(b.data(), x_serial.data(), 1.0, 0.0);
    
    ALBUSSPMV<int, int, double, RowDotKernel::Scalar, WorkloadMode::CAMLB> camlb(4);
    camlb.preprocess(4, ai.data(), aj.data(), av.data());
    camlb(b.data(), x.data(), 1.0, 0.0);
    
    for (int i = 0; i < 4; i++) {
      EXPECT_DOUBLE_EQ(x[i], x_serial[i]) << "CAMLB single nnz failed at row " << i;
    }
  }

  // Case 5: All nnz in first row
  {
    std::vector<int> ai = {0, 10, 10, 10};  // all nnz in first row
    std::vector<int> aj = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<double> av(10, 1.0);
    std::vector<double> b(10, 2.0);
    std::vector<double> x(3, 0.0);
    std::vector<double> x_serial(3, 0.0);
    
    SerialSPMV<int, int, double> spmv;
    spmv.preprocess(3, ai.data(), aj.data(), av.data());
    spmv(b.data(), x_serial.data(), 1.0, 0.0);
    
    ALBUSSPMV<int, int, double, RowDotKernel::Scalar, WorkloadMode::CAMLB> camlb(4);
    camlb.preprocess(3, ai.data(), aj.data(), av.data());
    camlb(b.data(), x.data(), 1.0, 0.0);
    
    for (int i = 0; i < 3; i++) {
      EXPECT_DOUBLE_EQ(x[i], x_serial[i]) << "CAMLB all nnz in first row failed at row " << i;
    }
  }

  // Case 6: Test with beta modes
  {
    std::vector<int> ai = {0, 0, 0};  // 2 empty rows
    std::vector<int> aj;
    std::vector<double> av;
    std::vector<double> b(1, 1.0);
    
    // beta=0
    std::vector<double> x1(2, 5.0);
    ALBUSSPMV<int, int, double, RowDotKernel::Scalar, WorkloadMode::CAMLB> camlb1(2);
    camlb1.preprocess(2, ai.data(), aj.data(), av.data());
    camlb1(b.data(), x1.data(), 1.0, 0.0);
    EXPECT_DOUBLE_EQ(x1[0], 0.0);
    EXPECT_DOUBLE_EQ(x1[1], 0.0);
    
    // beta=1
    std::vector<double> x2(2, 5.0);
    ALBUSSPMV<int, int, double, RowDotKernel::Scalar, WorkloadMode::CAMLB> camlb2(2);
    camlb2.preprocess(2, ai.data(), aj.data(), av.data());
    camlb2(b.data(), x2.data(), 1.0, 1.0);
    EXPECT_DOUBLE_EQ(x2[0], 5.0);
    EXPECT_DOUBLE_EQ(x2[1], 5.0);
    
    // beta=2
    std::vector<double> x3(2, 5.0);
    ALBUSSPMV<int, int, double, RowDotKernel::Scalar, WorkloadMode::CAMLB> camlb3(2);
    camlb3.preprocess(2, ai.data(), aj.data(), av.data());
    camlb3(b.data(), x3.data(), 1.0, 2.0);
    EXPECT_DOUBLE_EQ(x3[0], 10.0);
    EXPECT_DOUBLE_EQ(x3[1], 10.0);
  }
}
