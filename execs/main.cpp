
#include "../mkl_wrapper/mkl_sparse_mat.h"
#include "../utils/utils.h"
#include <Eigen/Sparse>
#include <algorithm>
#include <fast_matrix_market/app/Eigen.hpp>
#include <fstream>
#include <ios>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

using SpMat = typename Eigen::SparseMatrix<double, Eigen::RowMajor, MKL_INT>;
using SpMatMap = typename Eigen::Map<const SpMat>;
int main() {

  std::ifstream f("../../data/cage15.mtx");

  SpMat mat;
  fast_matrix_market::read_matrix_market_eigen(f, mat);
  // std::cout << mat.rows() << " , " << mat.cols() << std::endl;
  // std::cout << Eigen::MatrixXd(mat) << std::endl << std::endl;
  f.clear();
  f.seekg(0, std::ios::beg);
  std::vector<MKL_INT> csr_rows, csr_cols;
  std::vector<double> csr_vals;
  utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
  SpMatMap mat_csr(csr_rows.size() - 1, csr_rows.size() - 1, csr_cols.size(),
                   csr_rows.data(), csr_cols.data(), csr_vals.data());
  // std::cout << mat_csr.rows() << " , " << mat_csr.cols() << std::endl;
  // std::cout << mat_csr << std::endl << std::endl;
  SpMat res = mat_csr - mat;
  // Eigen::MatrixXd dense_csr(mat_csr);
  // Eigen::MatrixXd res = dense_csr - mat;
  std::cout << res.norm() << std::endl;
  mkl_wrapper::mkl_sparse_mat mkl_mat(csr_rows.size() - 1, csr_rows.size() - 1,
                                      csr_rows.data(), csr_cols.data(),
                                      csr_vals.data());

  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_int_distribution<int> dist{0, 10000};

  auto gen = [&dist, &mersenne_engine]() {
    return dist(mersenne_engine) * 1. / 10000;
  };

  std::vector<double> rhs(mkl_mat.rows());
  std::generate(std::begin(rhs), std::end(rhs), gen);
  std::vector<double> x(mkl_mat.rows());
  std::cout << "m: " << mkl_mat.rows() << " , n: " << mkl_mat.cols()
            << std::endl;
  for (int i = 0; i < 100; i++) {
    std::cout << i << std::endl;
    mkl_mat.mult_vec(rhs.data(), x.data());
  }
  return 0;
}