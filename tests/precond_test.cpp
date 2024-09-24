
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
  std::vector<mkl_wrapper::mkl_sparse_mat> _mats;

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
    _mats.push_back(mkl_wrapper::mkl_sparse_mat(csr_rows.size() - 1,
                                                csr_rows.size() - 1, csr_rows,
                                                csr_cols, csr_vals));
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
  for (auto &mat : _mats) {
    const int lvl = 3;
    std::cout << "size: " << mat.rows() << std::endl;
    matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> U, ICC0, ICC1, ICC2, ICC3;
    matrix_utils::SplitTriangle<matrix_utils::TriangularMatrix::U>(
        mat.rows(), mat.mkl_base(), mat.get_ai().get(), mat.get_aj().get(),
        mat.get_av().get(), U);

    mkl_wrapper::mkl_sparse_mat matU(mat.rows(), mat.rows(), U.ai, U.aj, U.av,
                                     mat.mkl_base());

    matrix_utils::ICCLevelSymbolic0(mat.rows(), mat.mkl_base(), U.ai.get(),
                                    U.aj.get(), U.ai.get(), lvl, ICC0);
    mkl_wrapper::mkl_sparse_mat matICC0(mat.rows(), mat.rows(), ICC0.ai,
                                        ICC0.aj, ICC0.av, mat.mkl_base());
    matrix_utils::ICCLevelSymbolic1(mat.rows(), mat.mkl_base(), U.ai.get(),
                                    U.aj.get(), U.ai.get(), lvl, ICC1);
    mkl_wrapper::mkl_sparse_mat matICC1(mat.rows(), mat.rows(), ICC1.ai,
                                        ICC1.aj, ICC1.av, mat.mkl_base());
    matrix_utils::ICCLevelSymbolic2(mat.rows(), mat.mkl_base(), U.ai.get(),
                                    U.aj.get(), U.ai.get(), lvl, ICC2);
    mkl_wrapper::mkl_sparse_mat matICC2(mat.rows(), mat.rows(), ICC2.ai,
                                        ICC2.aj, ICC2.av, mat.mkl_base());
    matrix_utils::ICCLevelSymbolic3(mat.rows(), mat.mkl_base(), U.ai.get(),
                                    U.aj.get(), U.ai.get(), lvl, ICC3);
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
