#include "../utils/utils.h"
#include "mkl_solver.h"
#include "mkl_sparse_mat.h"
#include <gtest/gtest.h>
#include <iomanip>
#include <memory>
#include <omp.h>

using namespace mkl_wrapper;

// The fixture for testing class Foo.
class sparse_matrix_Test : public testing::Test {
protected:
  std::shared_ptr<MKL_INT[]> aiA;
  std::shared_ptr<MKL_INT[]> ajA;
  std::shared_ptr<double[]> avA;

  std::shared_ptr<MKL_INT[]> aiB;
  std::shared_ptr<MKL_INT[]> ajB;
  std::shared_ptr<double[]> avB;

  std::vector<MKL_INT> csr_rows;
  std::vector<MKL_INT> csr_cols;
  std::vector<double> csr_vals;

  sparse_matrix_Test() {
    /*
    A = 1 2 3      B = 1 0 0
        0 4 5          2 4 0
        0 0 6          3 5 6
    */
    aiA.reset(new MKL_INT[4]{0, 3, 5, 6});
    ajA.reset(new MKL_INT[6]{0, 1, 2, 1, 2, 2});
    avA.reset(new double[6]{1, 2, 3, 4, 5, 6});

    aiB.reset(new MKL_INT[4]{0, 1, 3, 6});
    ajB.reset(new MKL_INT[6]{0, 0, 1, 0, 1, 2});
    avB.reset(new double[6]{1, 2, 4, 3, 5, 6});

    std::ifstream f("data/ex5.mtx"); // https://sparse.tamu.edu/FIDAP/ex5
    utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
  }

  ~sparse_matrix_Test() override {
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

/*
A = 1 2 3      B = 1 0 0
    0 4 5          2 4 0
    0 0 6          3 5 6
*/
TEST_F(sparse_matrix_Test, add) {
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
TEST_F(sparse_matrix_Test, mult_mat) {

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
TEST_F(sparse_matrix_Test, mult_vec) {
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

// TEST_F(sparse_matrix_Test, check) {
//   std::shared_ptr<MKL_INT[]> aiA(
//       new MKL_INT[4]{0, 1, 2, 3});
//   std::shared_ptr<MKL_INT[]> ajA(new MKL_INT[3]{1, 2, 3});
//   std::shared_ptr<double[]> avA(new double[3]{1, 2, 3});

//   mkl_wrapper::mkl_sparse_mat A(3, 4, aiA, ajA, avA);
//   A.print();
//   A.check();
// }

TEST_F(sparse_matrix_Test, sym) {

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

TEST_F(sparse_matrix_Test, pardiso_full_vs_sym) {
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

TEST_F(sparse_matrix_Test, mult_full_vs_sym) {

  std::shared_ptr<MKL_INT[]> ai(new MKL_INT[6]{0, 2, 5, 6, 9, 11});
  std::shared_ptr<MKL_INT[]> aj(
      new MKL_INT[11]{0, 1, 0, 1, 3, 2, 3, 4, 1, 3, 4});
  std::shared_ptr<double[]> av(new double[11]{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

  mkl_wrapper::mkl_sparse_mat mat(5, 5, ai, aj, av);
  mat.set_positive_definite(true);
  std::vector<double> rhs(mat.rows(), 1.), x0(mat.rows()), x1(mat.rows());
  // for (int i = 0; i < mat.rows(); i++) {
  //   rhs[i] = utils::random<double>(-1.0, 1.0);
  // }

  mat.mult_vec(rhs.data(), x0.data());
  mkl_wrapper::mkl_sparse_mat_sym sym(mat);
  sym.mult_vec(rhs.data(), x1.data());
  for (size_t i = 0; i < x0.size(); i++) {
    EXPECT_EQ(x0[i], x1[i]);
  }
}

TEST_F(sparse_matrix_Test, mult_full_vs_sym2) {
  mkl_wrapper::mkl_sparse_mat mat(csr_rows.size() - 1, csr_rows.size() - 1,
                                  csr_rows, csr_cols, csr_vals);
  mkl_wrapper::mkl_sparse_mat_sym sym(mat);

  std::vector<double> rhs(mat.rows(), 1.), x0(mat.rows()), x1(mat.rows());
  mat.mult_vec(rhs.data(), x0.data());
  sym.mult_vec(rhs.data(), x1.data());

  for (size_t i = 0; i < x0.size(); i++) {
    EXPECT_NEAR(x0[i], x1[i], 1e-9);
  }
}

TEST_F(sparse_matrix_Test, dense_matrix_orthogonalize) {
  mkl_wrapper::dense_mat mat(3, 3);
  auto av = mat.get_av();
  av[0] = 1;
  av[1] = 0;
  av[2] = 0;
  av[3] = 1;
  av[4] = 1;
  av[5] = 0;
  av[6] = 1;
  av[7] = 1;
  av[8] = 1;
  mat.orthogonalize();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (i == j)
        EXPECT_NEAR(av[i + 3 * j], 1, 1e-14);
      else {
        EXPECT_NEAR(av[i + 3 * j], 0, 1e-14);
      }
    }
  }
}

TEST_F(sparse_matrix_Test, sparsifier_random) {
  auto spm = mkl_wrapper::random_sparse(5, 5);
  auto ai = spm.get_ai();
  auto aj = spm.get_aj();
  for (int i = 0; i <= 5; i++) {
    EXPECT_EQ(ai[i], i * 5);
  }
  for (int i = 0; i < 25; i++) {
    EXPECT_EQ(aj[i], i % 5);
  }

  auto spm1 = mkl_wrapper::random_sparse(1000, 23);
  EXPECT_EQ(0, spm1.check());
}

TEST_F(sparse_matrix_Test, permute) {
  omp_set_num_threads(3);
  mkl_wrapper::mkl_sparse_mat A(3, 3, aiA, ajA, avA);
  mkl_wrapper::mkl_sparse_mat A1(A);
  A1.to_one_based();
  for (int it = 0; it < 100; it++) {
    // test zero based
    std::vector<MKL_INT> perm0 = utils::randomPermute(3, A.mkl_base());
    auto [aiB, ajB, avB] = mkl_wrapper::permute(A, perm0.data(), perm0.data());
    mkl_wrapper::mkl_sparse_mat B(3, 3, aiB, ajB, avB);

    auto inv_perm = utils::inversePermute(perm0, A.mkl_base());
    auto [aiC, ajC, avC] =
        mkl_wrapper::permute(B, inv_perm.data(), inv_perm.data());
    mkl_wrapper::mkl_sparse_mat C(3, 3, aiC, ajC, avC);

    for (size_t i = 0; i < 4; i++) {
      EXPECT_EQ(A.get_ai()[i], C.get_ai()[i]);
    }
    for (size_t i = 0; i < A.nnz(); i++) {
      EXPECT_EQ(A.get_aj()[i], C.get_aj()[i]);
      EXPECT_EQ(A.get_av()[i], C.get_av()[i]);
    }
    // test one based
    {
      std::vector<MKL_INT> perm1(perm0.size());
      std::transform(perm0.cbegin(), perm0.cend(), perm1.begin(),
                     [](MKL_INT ind) { return ind + 1; });
      auto [aiB, ajB, avB] =
          mkl_wrapper::permute(A1, perm1.data(), perm1.data());
      mkl_wrapper::mkl_sparse_mat B(3, 3, aiB, ajB, avB, SPARSE_INDEX_BASE_ONE);

      auto inv_perm = utils::inversePermute(perm1, A1.mkl_base());
      auto [aiC, ajC, avC] =
          mkl_wrapper::permute(B, inv_perm.data(), inv_perm.data());
      mkl_wrapper::mkl_sparse_mat C(3, 3, aiC, ajC, avC, SPARSE_INDEX_BASE_ONE);

      for (size_t i = 0; i < 4; i++) {
        EXPECT_EQ(A1.get_ai()[i], C.get_ai()[i]);
      }
      for (size_t i = 0; i < A.nnz(); i++) {
        EXPECT_EQ(A1.get_aj()[i], C.get_aj()[i]);
        EXPECT_EQ(A1.get_av()[i], C.get_av()[i]);
      }
    }
  }
}