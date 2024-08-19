
#include "incomplete_lu.h"
#include "matrix_utils.hpp"
#include "mkl_sparse_mat.h"
#include "triangle_solve.hpp"
#include "utils.h"
#include <algorithm>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <omp.h>

TEST(transpose_and_partranspose, base0) {
  auto mat = mkl_wrapper::random_sparse(100, 16);
  mat.randomVals();
  auto t_csr = matrix_utils::AllocateCSRData(mat.cols(), mat.nnz());
  matrix_utils::SerialTranspose(
      mat.rows(), mat.cols(), (int)mat.mkl_base(), mat.get_ai().get(),
      mat.get_aj().get(), mat.get_av().get(), std::get<0>(t_csr).get(),
      std::get<1>(t_csr).get(), std::get<2>(t_csr).get());

  auto tt_csr = matrix_utils::AllocateCSRData(mat.cols(), mat.nnz());
  matrix_utils::SerialTranspose(
      mat.cols(), mat.rows(), (int)mat.mkl_base(), std::get<0>(t_csr).get(),
      std::get<1>(t_csr).get(), std::get<2>(t_csr).get(),
      std::get<0>(tt_csr).get(), std::get<1>(tt_csr).get(),
      std::get<2>(tt_csr).get());

  for (int i = 0; i <= mat.rows(); i++) {
    EXPECT_EQ(std::get<0>(tt_csr)[i], mat.get_ai()[i]);
  }
  for (size_t i = 0; i < mat.nnz(); i++) {
    EXPECT_EQ(std::get<1>(tt_csr)[i], mat.get_aj()[i]);
    EXPECT_EQ(std::get<2>(tt_csr)[i], mat.get_av()[i]);
  }

  auto ptt_csr = matrix_utils::AllocateCSRData(mat.cols(), mat.nnz());
  matrix_utils::ParallelTranspose(
      mat.cols(), mat.rows(), (int)mat.mkl_base(), std::get<0>(t_csr).get(),
      std::get<1>(t_csr).get(), std::get<2>(t_csr).get(),
      std::get<0>(ptt_csr).get(), std::get<1>(ptt_csr).get(),
      std::get<2>(ptt_csr).get());

  for (int i = 0; i <= mat.rows(); i++) {
    EXPECT_EQ(std::get<0>(tt_csr)[i], std::get<0>(ptt_csr)[i]);
  }
  for (size_t i = 0; i < mat.nnz(); i++) {
    EXPECT_EQ(std::get<1>(tt_csr)[i], std::get<1>(ptt_csr)[i]);
    EXPECT_EQ(std::get<2>(tt_csr)[i], std::get<2>(ptt_csr)[i]);
  }

  auto pt2_csr = matrix_utils::AllocateCSRData(mat.cols(), mat.nnz());
  matrix_utils::ParallelTranspose2(
      mat.rows(), mat.cols(), (int)mat.mkl_base(), mat.get_ai().get(),
      mat.get_aj().get(), mat.get_av().get(), std::get<0>(pt2_csr).get(),
      std::get<1>(pt2_csr).get(), std::get<2>(pt2_csr).get());

  for (int i = 0; i <= mat.rows(); i++) {
    EXPECT_EQ(std::get<0>(t_csr)[i], std::get<0>(pt2_csr)[i]);
  }
  for (size_t i = 0; i < mat.nnz(); i++) {
    EXPECT_EQ(std::get<1>(t_csr)[i], std::get<1>(pt2_csr)[i]);
    EXPECT_EQ(std::get<2>(t_csr)[i], std::get<2>(pt2_csr)[i]);
  }
}

TEST(transpose_and_partranspose, base1) {
  auto mat = mkl_wrapper::random_sparse(10, 3);
  mat.randomVals();
  mat.to_one_based();

  //   std::ofstream myfile;
  //   myfile.open("origin.svg");
  //   mat.print_svg(myfile);
  //   myfile.close();

  auto t_csr = matrix_utils::AllocateCSRData(mat.cols(), mat.nnz());
  matrix_utils::SerialTranspose(
      mat.rows(), mat.cols(), (int)mat.mkl_base(), mat.get_ai().get(),
      mat.get_aj().get(), mat.get_av().get(), std::get<0>(t_csr).get(),
      std::get<1>(t_csr).get(), std::get<2>(t_csr).get());

  auto tt_csr = matrix_utils::AllocateCSRData(mat.cols(), mat.nnz());
  matrix_utils::SerialTranspose(
      mat.cols(), mat.rows(), (int)mat.mkl_base(), std::get<0>(t_csr).get(),
      std::get<1>(t_csr).get(), std::get<2>(t_csr).get(),
      std::get<0>(tt_csr).get(), std::get<1>(tt_csr).get(),
      std::get<2>(tt_csr).get());

  for (int i = 0; i <= mat.rows(); i++) {
    EXPECT_EQ(std::get<0>(tt_csr)[i], mat.get_ai()[i]);
  }
  for (size_t i = 0; i < mat.nnz(); i++) {
    EXPECT_EQ(std::get<1>(tt_csr)[i], mat.get_aj()[i]);
    EXPECT_EQ(std::get<2>(tt_csr)[i], mat.get_av()[i]);
  }

  auto pt_csr = matrix_utils::AllocateCSRData(mat.cols(), mat.nnz());
  matrix_utils::ParallelTranspose(
      mat.rows(), mat.cols(), (int)mat.mkl_base(), mat.get_ai().get(),
      mat.get_aj().get(), mat.get_av().get(), std::get<0>(pt_csr).get(),
      std::get<1>(pt_csr).get(), std::get<2>(pt_csr).get());

  for (int i = 0; i <= mat.rows(); i++) {
    EXPECT_EQ(std::get<0>(t_csr)[i], std::get<0>(pt_csr)[i]);
  }
  for (size_t i = 0; i < mat.nnz(); i++) {
    EXPECT_EQ(std::get<1>(t_csr)[i], std::get<1>(pt_csr)[i]);
    EXPECT_EQ(std::get<2>(t_csr)[i], std::get<2>(pt_csr)[i]);
  }

  auto pt2_csr = matrix_utils::AllocateCSRData(mat.cols(), mat.nnz());
  matrix_utils::ParallelTranspose2(
      mat.rows(), mat.cols(), (int)mat.mkl_base(), mat.get_ai().get(),
      mat.get_aj().get(), mat.get_av().get(), std::get<0>(pt2_csr).get(),
      std::get<1>(pt2_csr).get(), std::get<2>(pt2_csr).get());

  // for (int i = 0; i <= mat.rows(); i++) {
  //   EXPECT_EQ(std::get<0>(t_csr)[i], std::get<0>(pt2_csr)[i]);
  // }
  // for (size_t i = 0; i < mat.nnz(); i++) {
  //   EXPECT_EQ(std::get<1>(t_csr)[i], std::get<1>(pt2_csr)[i]);
  //   EXPECT_EQ(std::get<2>(t_csr)[i], std::get<2>(pt2_csr)[i]);
  // }
  //   mkl_wrapper::mkl_sparse_mat t_mat(mat.cols(), mat.rows(),
  //   std::get<0>(pt_csr),
  //                                     std::get<1>(pt_csr),
  //                                     std::get<2>(pt_csr),
  //                                     SPARSE_INDEX_BASE_ONE);
  //   myfile.open("transpose.svg");
  //   t_mat.print_svg(myfile);
  //   myfile.close();
}

TEST(transpose_and_partranspose, no_av) {
  auto mat = mkl_wrapper::random_sparse(10, 3);
  mat.randomVals();
  mat.to_one_based();

  //   std::ofstream myfile;
  //   myfile.open("origin.svg");
  //   mat.print_svg(myfile);
  //   myfile.close();

  auto t_csr = matrix_utils::AllocateCSRData(mat.cols(), mat.nnz());
  matrix_utils::SerialTranspose(
      mat.rows(), mat.cols(), (int)mat.mkl_base(), mat.get_ai().get(),
      mat.get_aj().get(), (double *)nullptr, std::get<0>(t_csr).get(),
      std::get<1>(t_csr).get(), std::get<2>(t_csr).get());
}

TEST(SplitLDU, base0) {
  omp_set_num_threads(5);
  auto mat = mkl_wrapper::random_sparse(1000, 32);
  mat.randomVals();

  matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> L, U;
  std::vector<double> D;
  matrix_utils::SplitLDU(mat.rows(), (int)mat.mkl_base(), mat.get_ai().get(),
                         mat.get_aj().get(), mat.get_av().get(), L, D, U);
  mkl_wrapper::mkl_sparse_mat matL(mat.rows(), mat.rows(), L.ai, L.aj, L.av);
  mkl_wrapper::mkl_sparse_mat matU(mat.rows(), mat.rows(), U.ai, U.aj, U.av);
  auto tmp = mkl_wrapper::mkl_sparse_sum(matU, mat, -1.);
  auto matD = mkl_wrapper::mkl_sparse_sum(matL, tmp, -1.);
  matD.prune(1e-11);

  std::vector<double> ones(mat.rows(), 1);
  std::vector<double> diag(mat.rows());

  matD.mult_vec(ones.data(), diag.data());
  for (int i = 0; i < mat.rows(); i++) {
    EXPECT_NEAR(diag[i], D[i], 2e-11);
  }

  // std::ofstream myfile;
  // myfile.open("origin.svg");
  // mat.print_svg(myfile);
  // myfile.close();

  // myfile.open("L.svg");
  // matL.print_svg(myfile);
  // myfile.close();

  // myfile.open("U.svg");
  // matU.print_svg(myfile);
  // myfile.close();

  // myfile.open("D.svg");
  // matD.print_svg(myfile);
  // myfile.close();
}

TEST(SplitLDU, base1) {
  omp_set_num_threads(5);
  auto mat = mkl_wrapper::random_sparse(1000, 32);
  mat.randomVals();
  mat.to_one_based();

  matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> L, U;
  std::vector<double> D;
  matrix_utils::SplitLDU(mat.rows(), (int)mat.mkl_base(), mat.get_ai().get(),
                         mat.get_aj().get(), mat.get_av().get(), L, D, U);
  mkl_wrapper::mkl_sparse_mat matL(mat.rows(), mat.rows(), L.ai, L.aj, L.av,
                                   SPARSE_INDEX_BASE_ONE);
  mkl_wrapper::mkl_sparse_mat matU(mat.rows(), mat.rows(), U.ai, U.aj, U.av,
                                   SPARSE_INDEX_BASE_ONE);
  auto tmp = mkl_wrapper::mkl_sparse_sum(matU, mat, -1.);
  auto matD = mkl_wrapper::mkl_sparse_sum(matL, tmp, -1.);
  matD.prune(1e-11);

  std::vector<double> ones(mat.rows(), 1);
  std::vector<double> diag(mat.rows());

  matD.mult_vec(ones.data(), diag.data());
  for (int i = 0; i < mat.rows(); i++) {
    EXPECT_NEAR(diag[i], D[i], 2e-11);
  }

  // std::ofstream myfile;
  // myfile.open("origin.svg");
  // mat.print_svg(myfile);
  // myfile.close();

  // myfile.open("L.svg");
  // matL.print_svg(myfile);
  // myfile.close();

  // myfile.open("U.svg");
  // matU.print_svg(myfile);
  // myfile.close();

  // myfile.open("D.svg");
  // matD.print_svg(myfile);
  // myfile.close();
}

TEST(triangular_solve, forward_substitution) {
  omp_set_num_threads(4);

  std::ifstream f("data/ex5.mtx");
  f.clear();
  f.seekg(0, std::ios::beg);
  std::vector<MKL_INT> csr_rows, csr_cols;
  std::vector<double> csr_vals;
  utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
  mkl_wrapper::mkl_sparse_mat mat(csr_rows.size() - 1, csr_rows.size() - 1,
                                  csr_rows, csr_cols, csr_vals);
  const MKL_INT size = csr_rows.size() - 1;
  mat.to_zero_based();
  mkl_wrapper::incomplete_lu_k prec;
  prec.set_level(5);
  prec.symbolic_factorize(&mat);
  prec.numeric_factorize(&mat);

  // std::ofstream myfile;
  // myfile.open("prec.svg");
  // prec.print_svg(myfile);
  // myfile.close();

  std::vector<double> b(mat.rows());
  std::iota(std::begin(b), std::end(b), 0);
  std::vector<double> x_serial(mat.rows(), 0.0);
  std::vector<double> x_par(mat.rows(), 0.0);
  std::vector<double> x_mkl(mat.rows(), 0.0);

  matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  descr.mode = SPARSE_FILL_MODE_LOWER;
  descr.diag = SPARSE_DIAG_UNIT;
  mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, prec.mkl_handler(),
                    descr, b.data(), x_mkl.data());

  matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> L, U;
  std::vector<double> D;

  matrix_utils::SplitLDU(prec.rows(), (int)prec.mkl_base(), prec.get_ai().get(),
                         prec.get_aj().get(), prec.get_av().get(), L, D, U);

  matrix_utils::ForwardSubstitution(L.rows, L.base, L.ai.get(), L.aj.get(),
                                    L.av.get(), b.data(), x_serial.data());
  for (int i = 0; i < mat.rows(); i++) {
    EXPECT_NEAR(x_serial[i], x_mkl[i], 1e-13);
  }

  std::vector<int> iperm(L.rows);
  std::vector<int> prefix;
  matrix_utils::TopologicalSort2<matrix_utils::TriangularSolve::L>(
      L.rows, L.base, L.ai.get(), L.aj.get(), iperm, prefix);
  matrix_utils::LevelScheduleForwardSubstitution(
      iperm, prefix, L.rows, L.base, L.ai.get(), L.aj.get(), L.av.get(),
      b.data(), x_par.data());
  for (int i = 0; i < mat.rows(); i++) {
    EXPECT_NEAR(x_par[i], x_mkl[i], 1e-13);
  }

  auto Lt_data = matrix_utils::AllocateCSRData(L.cols, L.nnz);
  matrix_utils::ParallelTranspose(
      L.rows, L.cols, L.base, L.ai.get(), L.aj.get(), L.av.get(),
      std::get<0>(Lt_data).get(), std::get<1>(Lt_data).get(),
      std::get<2>(Lt_data).get());

  std::vector<double> x_serial_t(mat.rows(), 0.0);

  matrix_utils::ForwardSubstitutionT(
      L.rows, L.base, std::get<0>(Lt_data).get(), std::get<1>(Lt_data).get(),
      std::get<2>(Lt_data).get(), b.data(), x_serial_t.data());

  for (int i = 0; i < mat.rows(); i++) {
    EXPECT_NEAR(x_serial_t[i], x_mkl[i], 1e-13);
  }
}

TEST(triangular_solve, forward_substitution1) {
  omp_set_num_threads(2);

  std::ifstream f("data/nos5.mtx");

  // std::ifstream f("/home/dimiao/matrix_lib/thermal2.mtx");

  f.clear();
  f.seekg(0, std::ios::beg);
  std::vector<MKL_INT> csr_rows, csr_cols;
  std::vector<double> csr_vals;
  utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
  mkl_wrapper::mkl_sparse_mat mat(csr_rows.size() - 1, csr_rows.size() - 1,
                                  csr_rows, csr_cols, csr_vals);
  const MKL_INT size = csr_rows.size() - 1;
  mat.to_zero_based();
  mkl_wrapper::incomplete_lu_k prec;
  prec.set_level(0);
  prec.symbolic_factorize(&mat);
  prec.numeric_factorize(&mat);

  // std::ofstream myfile;
  // myfile.open("prec.svg");
  // prec.print_svg(myfile);
  // myfile.close();

  std::vector<double> b(mat.rows());
  std::iota(std::begin(b), std::end(b), 0);
  std::vector<double> x(mat.rows(), 0.0);
  std::vector<double> x_mkl(mat.rows(), 0.0);
  std::vector<double> x_serial(mat.rows(), 0.0);

  matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  descr.mode = SPARSE_FILL_MODE_LOWER;
  descr.diag = SPARSE_DIAG_UNIT;
  mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, prec.mkl_handler(),
                    descr, b.data(), x_mkl.data());

  matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> L, U;
  std::vector<double> D;

  matrix_utils::SplitLDU(prec.rows(), (int)prec.mkl_base(), prec.get_ai().get(),
                         prec.get_aj().get(), prec.get_av().get(), L, D, U);

  matrix_utils::ForwardSubstitution(L.rows, L.base, L.ai.get(), L.aj.get(),
                                    L.av.get(), b.data(), x_serial.data());

  matrix_utils::OptimizedForwardSubstitution<true, int, int, int, double>
      forwardsweep;
  forwardsweep.analysis(L.rows, L.base, L.ai.get(), L.aj.get(), L.av.get());
  forwardsweep.build_task_graph();
  for (int i = 0; i < 1000; i++) {
    forwardsweep(b.data(), x.data());
    for (int i = 0; i < x.size(); i++) {
      EXPECT_EQ(x[i], x_serial[i]);
    }
  }
}