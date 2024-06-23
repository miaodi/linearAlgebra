
#include "matrix_utils.hpp"
#include "mkl_sparse_mat.h"
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
  auto t_csr = matrix_utils::SerialTranspose(mat.rows(), mat.cols(), mat.nnz(),
                                             (int)mat.mkl_base(), mat.get_ai(),
                                             mat.get_aj(), mat.get_av());

  auto tt_csr = matrix_utils::SerialTranspose(
      mat.cols(), mat.rows(), mat.nnz(), (int)mat.mkl_base(),
      std::get<0>(t_csr), std::get<1>(t_csr), std::get<2>(t_csr));

  for (int i = 0; i <= mat.rows(); i++) {
    EXPECT_EQ(std::get<0>(tt_csr)[i], mat.get_ai()[i]);
  }
  for (size_t i = 0; i < mat.nnz(); i++) {
    EXPECT_EQ(std::get<1>(tt_csr)[i], mat.get_aj()[i]);
    EXPECT_EQ(std::get<2>(tt_csr)[i], mat.get_av()[i]);
  }

  auto ptt_csr = matrix_utils::ParallelTranspose(
      mat.cols(), mat.rows(), mat.nnz(), (int)mat.mkl_base(),
      std::get<0>(t_csr), std::get<1>(t_csr), std::get<2>(t_csr));

  for (int i = 0; i <= mat.rows(); i++) {
    EXPECT_EQ(std::get<0>(tt_csr)[i], std::get<0>(ptt_csr)[i]);
  }
  for (size_t i = 0; i < mat.nnz(); i++) {
    EXPECT_EQ(std::get<1>(tt_csr)[i], std::get<1>(ptt_csr)[i]);
    EXPECT_EQ(std::get<2>(tt_csr)[i], std::get<2>(ptt_csr)[i]);
  }

  auto pt2_csr = matrix_utils::ParallelTranspose2(
      mat.rows(), mat.cols(), mat.nnz(), (int)mat.mkl_base(), mat.get_ai(),
      mat.get_aj(), mat.get_av());

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

  auto t_csr = matrix_utils::SerialTranspose(mat.rows(), mat.cols(), mat.nnz(),
                                             (int)mat.mkl_base(), mat.get_ai(),
                                             mat.get_aj(), mat.get_av());

  auto tt_csr = matrix_utils::SerialTranspose(
      mat.cols(), mat.rows(), mat.nnz(), (int)mat.mkl_base(),
      std::get<0>(t_csr), std::get<1>(t_csr), std::get<2>(t_csr));

  for (int i = 0; i <= mat.rows(); i++) {
    EXPECT_EQ(std::get<0>(tt_csr)[i], mat.get_ai()[i]);
  }
  for (size_t i = 0; i < mat.nnz(); i++) {
    EXPECT_EQ(std::get<1>(tt_csr)[i], mat.get_aj()[i]);
    EXPECT_EQ(std::get<2>(tt_csr)[i], mat.get_av()[i]);
  }

  auto pt_csr = matrix_utils::ParallelTranspose(
      mat.rows(), mat.cols(), mat.nnz(), (int)mat.mkl_base(), mat.get_ai(),
      mat.get_aj(), mat.get_av());

  for (int i = 0; i <= mat.rows(); i++) {
    EXPECT_EQ(std::get<0>(t_csr)[i], std::get<0>(pt_csr)[i]);
  }
  for (size_t i = 0; i < mat.nnz(); i++) {
    EXPECT_EQ(std::get<1>(t_csr)[i], std::get<1>(pt_csr)[i]);
    EXPECT_EQ(std::get<2>(t_csr)[i], std::get<2>(pt_csr)[i]);
  }

  auto pt2_csr = matrix_utils::ParallelTranspose2(
      mat.rows(), mat.cols(), mat.nnz(), (int)mat.mkl_base(), mat.get_ai(),
      mat.get_aj(), mat.get_av());

  for (int i = 0; i <= mat.rows(); i++) {
    EXPECT_EQ(std::get<0>(t_csr)[i], std::get<0>(pt2_csr)[i]);
  }
  for (size_t i = 0; i < mat.nnz(); i++) {
    EXPECT_EQ(std::get<1>(t_csr)[i], std::get<1>(pt2_csr)[i]);
    EXPECT_EQ(std::get<2>(t_csr)[i], std::get<2>(pt2_csr)[i]);
  }
  //   mkl_wrapper::mkl_sparse_mat t_mat(mat.cols(), mat.rows(),
  //   std::get<0>(pt_csr),
  //                                     std::get<1>(pt_csr),
  //                                     std::get<2>(pt_csr),
  //                                     SPARSE_INDEX_BASE_ONE);
  //   myfile.open("transpose.svg");
  //   t_mat.print_svg(myfile);
  //   myfile.close();
}
