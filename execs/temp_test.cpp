
#include "../sparse_mat_op/matrix_utils.hpp"
#include "MinimumDegree.hpp"
#include "ObjectPool.hpp"
#include "aat.hpp"
#include "circularbuffer.hpp"
#include "mkl_sparse_mat.h"
#include <vector>

int main() {

  // std::string filename = "../tests/data/rdist1.mtx";
  std::string filename = "../data/GD01_c.mtx";
  // std::string filename = "/home/dimiao/matrix_lib/thermal1.mtx";

  std::ifstream f(filename);
  f.clear();
  f.seekg(0, std::ios::beg);
  std::vector<int> csr_rows, csr_cols;
  std::vector<double> csr_vals;
  utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);

  mkl_wrapper::mkl_sparse_mat mat(csr_rows.size() - 1, csr_rows.size() - 1,
                                  csr_rows, csr_cols, csr_vals);
  std::ofstream out("mat.svg");
  mat.print_svg(out);
  out.close();
  int size = csr_rows.size() - 1;
  std::vector<int> ai_AAT(csr_rows.size());
  std::cout << csr_rows.size() << " " << csr_cols.size() << " "
            << csr_vals.size() << std::endl;
  matrix_utils::AATSymbolic(size, csr_rows.data(), csr_cols.data(),
                            ai_AAT.data());

  std::vector<int> aj_AAT(ai_AAT.back());
  std::vector<double> av_AAT(ai_AAT.back());
  matrix_utils::AATNumeric(size, csr_rows.data(), csr_cols.data(),
                           ai_AAT.data(), aj_AAT.data());
  mkl_wrapper::mkl_sparse_mat mat1(size, size, ai_AAT, aj_AAT, av_AAT);

  std::ofstream out1("mat_AAT.svg");
  mat1.print_svg(out1);
  out1.close();
  std::cout << "start\n";
  reordering::QuotientGraph<int> qg;
  std::vector<int> iperm(size);
  std::vector<int> perm(size);
  qg(size, ai_AAT.data(), aj_AAT.data(), perm.data(), iperm.data());

  for(auto& it: qg._degree_to_principle){
    std::cout << it.first << " " << it.second->size() << std::endl;
  }
  std::cout << "base: " << csr_rows[0] << std::endl;
  std::cout << "is permutation: " << utils::isPermutation(perm) << "\n";
  for (auto i : perm)
    std::cout << i << " ";
  std::cout << std::endl;
  return 0;
}