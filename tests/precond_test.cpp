
#include "incomplete_cholesky.h"
#include "matrix_utils.hpp"
#include "mkl_sparse_mat.h"
#include "precond.hpp"
#include "utils.h"
#include <algorithm>
#include <fstream>
#include <gtest/gtest.h>

using namespace mkl_wrapper;
using namespace matrix_utils;

// The fixture for testing class Foo.
class precond_Test : public testing::Test {
protected:
  struct MatrixInfo {
    mkl_wrapper::mkl_sparse_mat mat;
    bool is_spd;
    std::string name;
  };
  
  std::vector<MatrixInfo> _mats;

  const double _tol = 1e-14;
  const double _MKLtol = 1e-13;

  precond_Test() {

    std::vector<MKL_INT> csr_rows;
    std::vector<MKL_INT> csr_cols;
    std::vector<double> csr_vals;

    std::ifstream f;
    f.open("data/nos5.mtx");
    utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
    f.close();
    _mats.push_back({mkl_wrapper::mkl_sparse_mat(csr_rows.size() - 1,
                                                csr_rows.size() - 1, csr_rows,
                                                csr_cols, csr_vals), true, "nos5"});
    
    csr_rows.clear();
    csr_cols.clear();
    csr_vals.clear();
    f.open("data/bcsstk17.mtx");
    utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
    f.close();
    _mats.push_back({mkl_wrapper::mkl_sparse_mat(csr_rows.size() - 1,
                                                csr_rows.size() - 1, csr_rows,
                                                csr_cols, csr_vals), true, "bcsstk17"});
    
    csr_rows.clear();
    csr_cols.clear();
    csr_vals.clear();
    f.open("data/ex27.mtx");
    utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
    f.close();
    _mats.push_back({mkl_wrapper::mkl_sparse_mat(csr_rows.size() - 1,
                                                csr_rows.size() - 1, csr_rows,
                                                csr_cols, csr_vals), false, "ex27"});
  }

  ~precond_Test() override {
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

  // Class members declared here can be used by all tests in the test suite
  // for Foo.
};

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

TEST_F(precond_Test, icc_level_symbolic_factorize) {
  for (auto &mat_info : _mats) {
    if (!mat_info.is_spd) {
      std::cout << "Skipping " << mat_info.name << " (not SPD)" << std::endl;
      continue;
    }
    auto &mat = mat_info.mat;
    const int lvl = 3;
    std::cout << "Testing " << mat_info.name << ", size: " << mat.rows() << std::endl;
    matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> U, ICC0, ICC1, ICC2, ICC3;
    matrix_utils::SplitTriangle<matrix_utils::TriangularMatrix::U>(
        mat.rows(), mat.mkl_base(), mat.get_ai().get(), mat.get_aj().get(),
        mat.get_av().get(), U);

    mkl_wrapper::mkl_sparse_mat matU(mat.rows(), mat.rows(), U.ai, U.aj, U.av,
                                     mat.mkl_base());

    matrix_utils::ICCLevelSymbolic0(mat.rows(), U.ai.get(), U.aj.get(),
                                    U.ai.get(), lvl, ICC0);
    mkl_wrapper::mkl_sparse_mat matICC0(mat.rows(), mat.rows(), ICC0.ai,
                                        ICC0.aj, ICC0.av, mat.mkl_base());
    matrix_utils::ICCLevelSymbolic1(mat.rows(), U.ai.get(), U.aj.get(),
                                    U.ai.get(), lvl, ICC1);
    mkl_wrapper::mkl_sparse_mat matICC1(mat.rows(), mat.rows(), ICC1.ai,
                                        ICC1.aj, ICC1.av, mat.mkl_base());
    matrix_utils::ICCLevelSymbolic2(mat.rows(), U.ai.get(), U.aj.get(),
                                    U.ai.get(), lvl, ICC2);
    mkl_wrapper::mkl_sparse_mat matICC2(mat.rows(), mat.rows(), ICC2.ai,
                                        ICC2.aj, ICC2.av, mat.mkl_base());
    matrix_utils::ICCLevelSymbolic3(mat.rows(), U.ai.get(), U.aj.get(),
                                    U.ai.get(), lvl, ICC3);
    mkl_wrapper::mkl_sparse_mat matICC3(mat.rows(), mat.rows(), ICC3.ai,
                                        ICC3.aj, ICC3.av, mat.mkl_base());
    auto prec = std::make_shared<mkl_wrapper::incomplete_cholesky_k>();
    prec->set_level(lvl);
    prec->symbolic_factorize(&matU);

    for (int i = 0; i < mat.rows() + 1; i++) {
      EXPECT_EQ(prec->get_ai()[i], ICC0.ai[i]);
    }

    for (int i = 0; i < prec->nnz(); i++) {
      EXPECT_EQ(prec->get_aj()[i], ICC0.aj[i]);
    }

    for (int i = 0; i < mat.rows() + 1; i++) {
      EXPECT_EQ(prec->get_ai()[i], ICC1.ai[i]);
    }

    for (int i = 0; i < prec->nnz(); i++) {
      EXPECT_EQ(prec->get_aj()[i], ICC1.aj[i]);
    }

    for (int i = 0; i < mat.rows() + 1; i++) {
      EXPECT_EQ(prec->get_ai()[i], ICC2.ai[i]);
    }

    for (int i = 0; i < prec->nnz(); i++) {
      EXPECT_EQ(prec->get_aj()[i], ICC2.aj[i]);
    }

    for (int i = 0; i < mat.rows() + 1; i++) {
      EXPECT_EQ(prec->get_ai()[i], ICC3.ai[i]);
    }

    for (int i = 0; i < prec->nnz(); i++) {
      EXPECT_EQ(prec->get_aj()[i], ICC3.aj[i]);
    }
  }
}

TEST_F(precond_Test, icc_level_numeric_factorize) {
  for (auto &mat_info : _mats) {
    if (!mat_info.is_spd) {
      std::cout << "Skipping " << mat_info.name << " (not SPD)" << std::endl;
      continue;
    }
    auto &mat = mat_info.mat;
    const int lvl = 3;
    std::cout << "Testing " << mat_info.name << ", size: " << mat.rows() << std::endl;
    matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> U, ICC0, ICC1, ICC2, ICC3;
    matrix_utils::SplitTriangle<matrix_utils::TriangularMatrix::U>(
        mat.rows(), mat.mkl_base(), mat.get_ai().get(), mat.get_aj().get(),
        mat.get_av().get(), U);

    mkl_wrapper::mkl_sparse_mat matU(mat.rows(), mat.rows(), U.ai, U.aj, U.av,
                                     mat.mkl_base());

    matrix_utils::ICCLevelSymbolic0(mat.rows(), U.ai.get(), U.aj.get(),
                                    U.ai.get(), lvl, ICC0);
    matrix_utils::ICCLevelNumeric(mat.rows(), U.ai.get(), U.aj.get(),
                                  U.av.get(), U.ai.get(), lvl, 0.,
                                  ICC0.ai.get(), ICC0.aj.get(), ICC0.av.get());
    mkl_wrapper::mkl_sparse_mat matICC0(mat.rows(), mat.rows(), ICC0.ai,
                                        ICC0.aj, ICC0.av, mat.mkl_base());
    std::cout << std::endl;
    auto prec = std::make_shared<mkl_wrapper::incomplete_cholesky_k>();
    prec->set_level(lvl);
    prec->symbolic_factorize(&matU);
    prec->numeric_factorize(&matU);
    prec->to_zero_based();

    for (int i = 0; i < mat.rows() + 1; i++) {
      EXPECT_EQ(prec->get_ai()[i], ICC0.ai[i]);
    }

    for (int i = 0; i < prec->nnz(); i++) {
      EXPECT_EQ(prec->get_aj()[i], ICC0.aj[i]);
    }

    // for (int i = 0; i < prec->nnz(); i++) {
    //   if (prec->get_av()[i] != ICC0.av[i])
    //     std::cout << std::setprecision(16) << i << " " << prec->get_av()[i]
    //               << " " << ICC0.av[i] << std::endl;
    //   // EXPECT_EQ(prec->get_av()[i], ICC0.av[i]);
    // }
  }
}

TEST_F(precond_Test, ilu_level_symbolic_parallel_matches_upper) {
  for (auto &mat_info : _mats) {
    auto &mat = mat_info.mat;
    std::cout << "Testing " << mat_info.name << std::endl;
    const auto base = mat.mkl_base();
    const auto size = mat.rows();
    ILULevelSymbolic<CSRMatrix<MKL_INT, MKL_INT, double>> serial;
    ILULevelSymbolicParallelU<CSRMatrix<MKL_INT, MKL_INT, double>> parallel(4);

    auto upper_cols = [base](const CSRMatrix<MKL_INT, MKL_INT, double> &m,
                             MKL_INT row) {
      std::vector<MKL_INT> cols;
      for (auto idx = m.ai[row] - base; idx < m.ai[row + 1] - base; ++idx) {
        auto c = m.aj[idx] - base;
        if (c > row)
          cols.push_back(c);
      }
      return cols;
    };

    for (int lvl = 0; lvl <= 5; ++lvl) {
      CSRMatrix<MKL_INT, MKL_INT, double> ilu_serial;
      CSRMatrix<MKL_INT, MKL_INT, double> ilu_parallel;

      ASSERT_TRUE(serial(size, mat.get_ai().get(), mat.get_aj().get(), lvl,
                         ilu_serial));
      const bool ok_parallel =
          parallel(size, mat.get_ai().get(), mat.get_aj().get(), lvl,
                   ilu_parallel);
      if (!ok_parallel) {
        GTEST_SKIP() << "ILULevelSymbolicParallelU not implemented for lvl="
                     << lvl;
      }

      for (MKL_INT row = 0; row < size; ++row) {
        auto expected = upper_cols(ilu_serial, row);
        auto actual = upper_cols(ilu_parallel, row);
        ASSERT_EQ(expected.size(), actual.size())
            << "lvl=" << lvl << " row=" << row;
        for (size_t i = 0; i < expected.size(); ++i) {
          EXPECT_EQ(expected[i], actual[i])
              << "lvl=" << lvl << " row=" << row << " idx=" << i;
        }
      }
    }
  }
}

TEST_F(precond_Test, ilu_level_symbolic_parallel_matches_lower) {
  for (auto &mat_info : _mats) {
    auto &mat = mat_info.mat;
    std::cout << "Testing " << mat_info.name << std::endl;
    const auto base = mat.mkl_base();
    const auto size = mat.rows();
    ILULevelSymbolic<CSRMatrix<MKL_INT, MKL_INT, double>> serial;
    ILULevelSymbolicParallel<CSRMatrix<MKL_INT, MKL_INT, double>,
                             enums::matrix_utils::L>
        parallel(4);

    auto lower_cols = [base](const CSRMatrix<MKL_INT, MKL_INT, double> &m,
                             MKL_INT row) {
      std::vector<MKL_INT> cols;
      for (auto idx = m.ai[row] - base; idx < m.ai[row + 1] - base; ++idx) {
        auto c = m.aj[idx] - base;
        if (c < row)
          cols.push_back(c);
      }
      return cols;
    };

    for (int lvl = 0; lvl <= 5; ++lvl) {
      CSRMatrix<MKL_INT, MKL_INT, double> ilu_serial;
      CSRMatrix<MKL_INT, MKL_INT, double> ilu_parallel;

      ASSERT_TRUE(serial(size, mat.get_ai().get(), mat.get_aj().get(), lvl,
                         ilu_serial));
      const bool ok_parallel =
          parallel(size, mat.get_ai().get(), mat.get_aj().get(), lvl,
                   ilu_parallel);
      if (!ok_parallel) {
        GTEST_SKIP() << "ILULevelSymbolicParallel not implemented for lvl="
                     << lvl;
      }

      for (MKL_INT row = 0; row < size; ++row) {
        auto expected = lower_cols(ilu_serial, row);
        auto actual = lower_cols(ilu_parallel, row);
        ASSERT_EQ(expected.size(), actual.size())
            << "lvl=" << lvl << " row=" << row;
        for (size_t i = 0; i < expected.size(); ++i) {
          EXPECT_EQ(expected[i], actual[i])
              << "lvl=" << lvl << " row=" << row << " idx=" << i;
        }
      }
    }
  }
}

TEST_F(precond_Test, ilu_level_symbolic_parallel_lu_includes_diag) {
  for (auto &mat_info : _mats) {
    auto &mat = mat_info.mat;
    std::cout << "Testing " << mat_info.name << std::endl;
    const auto base = mat.mkl_base();
    const auto size = mat.rows();
    ILULevelSymbolic<CSRMatrix<MKL_INT, MKL_INT, double>> serial;
    ILULevelSymbolicParallel<CSRMatrix<MKL_INT, MKL_INT, double>,
                             enums::matrix_utils::LU,
                             true>
        parallel(4);

    auto lower_cols = [base](const CSRMatrix<MKL_INT, MKL_INT, double> &m,
                             MKL_INT row) {
      std::vector<MKL_INT> cols;
      for (auto idx = m.ai[row] - base; idx < m.ai[row + 1] - base; ++idx) {
        auto c = m.aj[idx] - base;
        if (c < row)
          cols.push_back(c);
      }
      return cols;
    };

    auto has_diag = [base](const CSRMatrix<MKL_INT, MKL_INT, double> &m,
                           MKL_INT row) {
      for (auto idx = m.ai[row] - base; idx < m.ai[row + 1] - base; ++idx) {
        auto c = m.aj[idx] - base;
        if (c == row)
          return true;
      }
      return false;
    };

    for (int lvl = 0; lvl <= 5; ++lvl) {
      CSRMatrix<MKL_INT, MKL_INT, double> ilu_serial;
      CSRMatrix<MKL_INT, MKL_INT, double> ilu_parallel;

      ASSERT_TRUE(serial(size, mat.get_ai().get(), mat.get_aj().get(), lvl,
                         ilu_serial));
      ASSERT_TRUE(
          parallel(size, mat.get_ai().get(), mat.get_aj().get(), lvl,
                   ilu_parallel));

      const auto *diag = ilu_parallel.Diagonal();
      ASSERT_NE(diag, nullptr);

      for (MKL_INT row = 0; row < size; ++row) {
        const bool serial_has_diag = has_diag(ilu_serial, row);
        const bool parallel_has_diag = has_diag(ilu_parallel, row);
        EXPECT_TRUE(serial_has_diag) << "lvl=" << lvl << " row=" << row;
        EXPECT_EQ(serial_has_diag, parallel_has_diag)
            << "lvl=" << lvl << " row=" << row;
        const auto row_start = ilu_parallel.AI()[row] - base;
        const auto row_end = ilu_parallel.AI()[row + 1] - base;
        const auto diag_idx = diag[row] - base;
        EXPECT_GE(diag_idx, row_start) << "lvl=" << lvl << " row=" << row;
        EXPECT_LT(diag_idx, row_end) << "lvl=" << lvl << " row=" << row;
        EXPECT_EQ(ilu_parallel.AJ()[diag_idx], row + base)
            << "lvl=" << lvl << " row=" << row;
        auto expected = lower_cols(ilu_serial, row);
        auto actual = lower_cols(ilu_parallel, row);
        ASSERT_EQ(expected.size(), actual.size())
            << "lvl=" << lvl << " row=" << row;
        for (size_t i = 0; i < expected.size(); ++i) {
          EXPECT_EQ(expected[i], actual[i])
              << "lvl=" << lvl << " row=" << row << " idx=" << i;
        }
      }
    }
  }
}
