
#include <fast_matrix_market/app/Eigen.hpp>
#include <fstream>
#include <iostream>
#include <Eigen/Sparse>

int main() {

  std::ifstream f("/home/dimiao/repo/linearAgebra/data/bcsstk28.mtx");

  Eigen::SparseMatrix<double> mat;

  fast_matrix_market::read_matrix_market_eigen(f, mat);

  std::cout<<mat<<std::endl;
  return 0;
}