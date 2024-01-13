#include "../utils/utils.h"
#include "mkl_solver.h"
#include "mkl_sparse_mat.h"
#include <gtest/gtest.h>
#include <memory>
// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}
using namespace mkl_wrapper;

/*
A = 1 2 3      B = 1 0 0
    0 4 5          2 4 0
    0 0 6          3 5 6
*/
TEST(sparse_matrix, add) {
  std::shared_ptr<MKL_INT[]> aiA(new MKL_INT[4]{0, 3, 5, 6});
  std::shared_ptr<MKL_INT[]> ajA(new MKL_INT[6]{0, 1, 2, 1, 2, 2});
  std::shared_ptr<double[]> avA(new double[6]{1, 2, 3, 4, 5, 6});

  std::shared_ptr<MKL_INT[]> aiB(new MKL_INT[4]{0, 1, 3, 6});
  std::shared_ptr<MKL_INT[]> ajB(new MKL_INT[6]{0, 0, 1, 0, 1, 2});
  std::shared_ptr<double[]> avB(new double[6]{1, 2, 4, 3, 5, 6});

  std::shared_ptr<MKL_INT[]> aiC(new MKL_INT[4]{0, 3, 6, 9});
  std::shared_ptr<MKL_INT[]> ajC(new MKL_INT[9]{0, 1, 2, 0, 1, 2, 0, 1, 2});
  std::shared_ptr<double[]> avC(new double[9]{2, 2, 3, 2, 8, 5, 3, 5, 12});

  mkl_wrapper::mkl_sparse_mat A(3, 3, aiA, ajA, avA);
  mkl_wrapper::mkl_sparse_mat B(3, 3, aiB, ajB, avB);
  auto C = mkl_sparse_sum(A, B);

  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(aiC[i], C.get_ai()[i]);
  }

  for (int i = 0; i < 9; i++) {
    EXPECT_EQ(ajC[i], C.get_aj()[i]);
    EXPECT_EQ(avC[i], C.get_av()[i]);
  }
}

/*
A = 1 2 3      B = 1 0 0
    0 4 5          2 4 0
    0 0 6          3 5 6
*/
TEST(sparse_matrix, mult_mat) {
  std::shared_ptr<MKL_INT[]> aiA(new MKL_INT[4]{0, 3, 5, 6});
  std::shared_ptr<MKL_INT[]> ajA(new MKL_INT[6]{0, 1, 2, 1, 2, 2});
  std::shared_ptr<double[]> avA(new double[6]{1, 2, 3, 4, 5, 6});

  std::shared_ptr<MKL_INT[]> aiB(new MKL_INT[4]{0, 1, 3, 6});
  std::shared_ptr<MKL_INT[]> ajB(new MKL_INT[6]{0, 0, 1, 0, 1, 2});
  std::shared_ptr<double[]> avB(new double[6]{1, 2, 4, 3, 5, 6});

  std::shared_ptr<MKL_INT[]> aiC(new MKL_INT[4]{0, 3, 6, 9});
  std::shared_ptr<MKL_INT[]> ajC(new MKL_INT[9]{0, 1, 2, 0, 1, 2, 0, 1, 2});
  std::shared_ptr<double[]> avC(
      new double[9]{14, 23, 18, 23, 41, 30, 18, 30, 36});

  mkl_wrapper::mkl_sparse_mat A(3, 3, aiA, ajA, avA);
  mkl_wrapper::mkl_sparse_mat B(3, 3, aiB, ajB, avB);
  auto C = mkl_sparse_mult(A, B);

  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(aiC[i], C.get_ai()[i]);
  }

  for (int i = 0; i < 9; i++) {
    EXPECT_EQ(ajC[i], C.get_aj()[i]);
    EXPECT_EQ(avC[i], C.get_av()[i]);
  }

  std::shared_ptr<MKL_INT[]> aiCT(new MKL_INT[4]{0, 1, 3, 6});
  std::shared_ptr<MKL_INT[]> ajCT(new MKL_INT[6]{0, 0, 1, 0, 1, 2});
  std::shared_ptr<double[]> avCT(new double[6]{1, 10, 16, 31, 50, 36});

  auto CT = mkl_sparse_mult(A, B, SPARSE_OPERATION_TRANSPOSE);

  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(aiCT[i], CT.get_ai()[i]);
  }

  for (int i = 0; i < 6; i++) {
    EXPECT_EQ(ajCT[i], CT.get_aj()[i]);
    EXPECT_EQ(avCT[i], CT.get_av()[i]);
  }
}

/*
A = 1 2 3
    0 4 5
    0 0 6
*/
TEST(sparse_matrix, mult_vec) {
  std::shared_ptr<MKL_INT[]> aiA(new MKL_INT[4]{0, 3, 5, 6});
  std::shared_ptr<MKL_INT[]> ajA(new MKL_INT[6]{0, 1, 2, 1, 2, 2});
  std::shared_ptr<double[]> avA(new double[6]{1, 2, 3, 4, 5, 6});

  mkl_wrapper::mkl_sparse_mat A(3, 3, aiA, ajA, avA);

  std::vector<double> rhs{1, 2, 3};
  std::vector<double> x(3);
  A.mult_vec(rhs.data(), x.data());
  for (auto i : x) {
    std::cout << i << std::endl;
  }
  A.to_one_based();
  A.mult_vec(rhs.data(), x.data());
  for (auto i : x) {
    std::cout << i << std::endl;
  }
}

// TEST(sparse_matrix, check) {
//   std::shared_ptr<MKL_INT[]> aiA(
//       new MKL_INT[4]{0, 1, 2, 3});
//   std::shared_ptr<MKL_INT[]> ajA(new MKL_INT[3]{1, 2, 3});
//   std::shared_ptr<double[]> avA(new double[3]{1, 2, 3});

//   mkl_wrapper::mkl_sparse_mat A(3, 4, aiA, ajA, avA);
//   A.print();
//   A.check();
// }

TEST(sparse_matrix, sym) {

  std::ifstream f("data/ex5.mtx"); // https://sparse.tamu.edu/FIDAP/ex5
  std::vector<MKL_INT> csr_rows, csr_cols;
  std::vector<double> csr_vals;
  utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);

  mkl_wrapper::mkl_sparse_mat mat(csr_rows.size() - 1, csr_rows.size() - 1,
                                  csr_rows, csr_cols, csr_vals);

  mat.to_zero_based();
  mkl_wrapper::mkl_sparse_mat_sym sym_zero(mat);
  mat.to_one_based();
  mkl_wrapper::mkl_sparse_mat_sym sym_one(mat);
  sym_one.to_zero_based();

  EXPECT_EQ(sym_zero.nnz(), sym_one.nnz());
  EXPECT_EQ(sym_zero.rows(), sym_one.rows());
  EXPECT_EQ(sym_zero.cols(), sym_one.cols());

  for (MKL_INT i = 0; i <= sym_zero.rows(); i++) {
    EXPECT_EQ(sym_zero.get_ai()[i], sym_one.get_ai()[i]);
  }

  for (MKL_INT i = 0; i < sym_zero.nnz(); i++) {
    EXPECT_EQ(sym_zero.get_aj()[i], sym_one.get_aj()[i]);
    EXPECT_EQ(sym_zero.get_av()[i], sym_one.get_av()[i]);
  }

  /*
      mat = U^T + D + U
      sym_one = D + U
      diff = U^T
      diff2 = D
  */
  mat.to_zero_based();
  auto diff = mkl_sparse_sum(sym_one, mat, -1.);
  diff.prune();
  diff.transpose();
  auto diff2 = mkl_sparse_sum(diff, sym_one, -1.);
  diff2.prune();
  auto diag_diff2 = diff2.get_diag();
  auto diag_sym = mat.get_diag();

  for (size_t i = 0; i < diff2.rows(); i++) {
    EXPECT_EQ(diag_diff2[i], diag_sym[i]);
  }
}

TEST(pardiso, full_vs_sym) {

  std::ifstream f("data/ex5.mtx"); // https://sparse.tamu.edu/FIDAP/ex5
  std::vector<MKL_INT> csr_rows, csr_cols;
  std::vector<double> csr_vals;
  utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);

  mkl_wrapper::mkl_sparse_mat mat(csr_rows.size() - 1, csr_rows.size() - 1,
                                  csr_rows, csr_cols, csr_vals);
  mat.set_positive_definite(true);
  std::vector<double> rhs(mat.rows(), 1.), x0(mat.rows()), x1(mat.rows());
  // for (int i = 0; i < mat.rows(); i++) {
  //   rhs[i] = utils::random<double>(-1.0, 1.0);
  // }
  auto solver0 =
      utils::singleton<mkl_wrapper::solver_factory>::instance().create("direct",
                                                                       mat);
  solver0->solve(rhs.data(), x0.data());
  mkl_wrapper::mkl_sparse_mat_sym sym(mat);
  auto solver1 =
      utils::singleton<mkl_wrapper::solver_factory>::instance().create("direct",
                                                                       sym);
  solver1->solve(rhs.data(), x1.data());

  for (size_t i = 0; i < x0.size(); i++) {
    EXPECT_NEAR(x0[i], x1[i], 1e-7);
  }
}
