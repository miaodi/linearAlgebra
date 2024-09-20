
#include "matrix_utils.hpp"
#include "mkl_sparse_mat.h"
#include "precond_symbolic.hpp"
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
    f.open("data/ex5.mtx"); // https://sparse.tamu.edu/FIDAP/ex5
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

    matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> U, ICC;
    matrix_utils::SplitTriangle<matrix_utils::TriangularMatrix::U>(
        mat.rows(), mat.mkl_base(), mat.get_ai().get(), mat.get_aj().get(),
        mat.get_av().get(), U);
    matrix_utils::ICCLevelSymbolic(mat.rows(), mat.mkl_base(), U.ai.get(),
                                   U.aj.get(), U.aj.get(), 1, ICC);
    for(int i = 0;i<ICC.ai[mat.rows()];i++){
      std::cout << ICC.aj[i] << " ";
    }
    // mkl_wrapper::mkl_sparse_mat matICC(mat.rows(), mat.rows(), ICC.ai, ICC.aj,
    //                                    ICC.av, mat.mkl_base());
    // matICC.print();
  }
}
