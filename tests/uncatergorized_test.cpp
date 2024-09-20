
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

TEST(UpperTrigToFull, small) {
  omp_set_num_threads(5);
  for (int i = 0; i < 20; i++) {
    auto mat = mkl_wrapper::random_sparse(50, 13);
    mat.randomVals();
    if (i % 2 == 0)
      mat.to_one_based();
    else
      mat.to_zero_based();
    matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> U, F;
    matrix_utils::SplitTriangle<matrix_utils::TriangularMatrix::U>(
        mat.rows(), mat.mkl_base(), mat.get_ai().get(), mat.get_aj().get(),
        mat.get_av().get(), U);
    matrix_utils::TriangularToFull<matrix_utils::TriangularMatrix::U>(
        U.rows, U.Base(), U.AI(), U.AJ(), U.AV(), F);

    mkl_wrapper::mkl_sparse_mat full(mat.rows(), mat.rows(), F.ai, F.aj, F.av,
                                     mat.mkl_base());
    mkl_wrapper::mkl_sparse_mat transpose_full = full;
    transpose_full.transpose();
    for (int i = 0; i <= full.rows(); i++) {
      EXPECT_EQ(full.get_ai()[i], transpose_full.get_ai()[i]);
    }
  }
}

TEST(UpperTrigToFull, medium) {
  omp_set_num_threads(10);
  for (int i = 0; i < 10; i++) {
    auto mat = mkl_wrapper::random_sparse(10000, 30);
    mat.randomVals();
    if (i % 2 == 0)
      mat.to_one_based();
    else
      mat.to_zero_based();
    matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> U, F;
    matrix_utils::SplitTriangle<matrix_utils::TriangularMatrix::U>(
        mat.rows(), mat.mkl_base(), mat.get_ai().get(), mat.get_aj().get(),
        mat.get_av().get(), U);
    matrix_utils::TriangularToFull<matrix_utils::TriangularMatrix::U>(
        U.rows, U.Base(), U.AI(), U.AJ(), U.AV(), F);

    mkl_wrapper::mkl_sparse_mat full(mat.rows(), mat.rows(), F.ai, F.aj, F.av,
                                     mat.mkl_base());
    mkl_wrapper::mkl_sparse_mat transpose_full = full;
    transpose_full.transpose();
    for (int i = 0; i <= full.rows(); i++) {
      EXPECT_EQ(full.get_ai()[i], transpose_full.get_ai()[i]);
    }
  }
}