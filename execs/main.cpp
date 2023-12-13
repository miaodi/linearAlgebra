
#include "../utils/utils.h"
#include <Eigen/Sparse>
#include <fast_matrix_market/app/Eigen.hpp>
#include <fstream>
#include <ios>
#include <iostream>
#include <vector>

using SpMat = typename Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
using SpMatMap = typename Eigen::Map<const SpMat>;
int main() {

  std::ifstream f("../../data/bcsstk01.mtx");

  SpMat mat;
  fast_matrix_market::read_matrix_market_eigen(f, mat);
  std::cout << mat.rows() << " , " << mat.cols() << std::endl;
  std::cout << Eigen::MatrixXd(mat) << std::endl << std::endl;
  f.clear();
  f.seekg(0, std::ios::beg);
  std::vector<int> csr_rows, csr_cols;
  std::vector<double> csr_vals;
  utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
  SpMatMap mat_csr(csr_rows.size() - 1, csr_rows.size() - 1, csr_cols.size(),
                   csr_rows.data(), csr_cols.data(), csr_vals.data());
  std::cout << mat_csr.rows() << " , " << mat_csr.cols() << std::endl;
  std::cout << Eigen::MatrixXd(mat_csr) << std::endl << std::endl;
  // SpMat res = mat_csr - mat;
  // Eigen::MatrixXd dense_csr(mat_csr);
  // Eigen::MatrixXd res = dense_csr - mat;
  // std::cout<<res.norm()<<std::endl;
  return 0;
}