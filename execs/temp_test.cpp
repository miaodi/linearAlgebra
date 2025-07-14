#include "../sparse_mat_op/io.hpp"
#include "../sparse_mat_op/matrix_utils.hpp"
#include "../sparse_mat_op/permutation.hpp"
#include "MinimumDegree.hpp"
#include "ObjectPool.hpp"
#include "aat.hpp"
#include "circularbuffer.hpp"
#include "mkl_sparse_mat.h"
#include <vector>

int main() {

  std::string filename = "../tests/data/rdist1.mtx";
  // std::string filename = "../data/GD01_c.mtx";
  // std::string filename = "/home/dimiao/matrix_lib/thermal1.mtx";

  std::ifstream f(filename);
  f.clear();
  f.seekg(0, std::ios::beg);
  matrix_utils::CSRMatrix<int, int, double> csr_matrix;
  matrix_utils::readMatrixMarket(f, csr_matrix);

  std::ofstream out0("mat_csr.mtx");
  matrix_utils::writeMatrixMarket(csr_matrix.rows, csr_matrix.cols,
                                  csr_matrix.AI(), csr_matrix.AJ(),
                                  csr_matrix.AV(), out0);
  out0.close();
  int size = csr_matrix.rows;
  std::vector<int> ai_AAT(size + 1);
  matrix_utils::AATSymbolic(size, csr_matrix.AI(), csr_matrix.AJ(),
                            ai_AAT.data());

  std::vector<int> aj_AAT(ai_AAT.back());
  std::vector<double> av_AAT(ai_AAT.back(), 1.);
  matrix_utils::AATNumeric(size, csr_matrix.AI(), csr_matrix.AJ(),
                           ai_AAT.data(), aj_AAT.data());
  std::ofstream out1("mat_AAT.mtx");
  matrix_utils::writeMatrixMarket(size, size, ai_AAT.data(), aj_AAT.data(),
                                  av_AAT.data(), out1);
  out1.close();

  std::cout << "start\n";
  reordering::QuotientGraph<int> qg;
  std::vector<int> iperm(size);
  std::vector<int> perm(size);
  qg(size, ai_AAT.data(), aj_AAT.data(), perm.data(), iperm.data());

  for (auto &it : qg._degree_to_principle) {
    std::cout << it.first << " " << it.second->size() << std::endl;
  }
  std::cout << "base: " << ai_AAT[0] << std::endl;
  std::cout << "is permutation: "
            << matrix_utils::isPermutation(size, ai_AAT[0], perm.data())
            << "\n";
  std::cout << "first row size: " << ai_AAT[1] - ai_AAT[0] << "\n";
  std::cout << "second row size: " << ai_AAT[2] - ai_AAT[1] << "\n";
  std::cout << "third row size: " << ai_AAT[3] - ai_AAT[2] << "\n";
  std::cout << "fourth row size: " << ai_AAT[4] - ai_AAT[3] << "\n";
  std::cout << "fifth row size: " << ai_AAT[5] - ai_AAT[4] << "\n";
  std::cout << "sixth row size: " << ai_AAT[6] - ai_AAT[5] << "\n";
  std::cout << "seventh row size: " << ai_AAT[7] - ai_AAT[6] << "\n";
  std::cout << "eighth row size: " << ai_AAT[8] - ai_AAT[7] << "\n";
  std::cout << "ninth row size: " << ai_AAT[9] - ai_AAT[8] << "\n";
  std::cout << "tenth row size: " << ai_AAT[10] - ai_AAT[9] << "\n";

  std::vector<int> perm_ai(size + 1);
  std::vector<int> perm_aj(ai_AAT.back());
  std::vector<double> perm_av(ai_AAT.back(), 1.);

  matrix_utils::permuteMat(size, size, perm.data(), iperm.data(), ai_AAT.data(),
                           aj_AAT.data(), perm_ai.data(), perm_aj.data(),
                           av_AAT.data(), perm_av.data());

  std::ofstream out2("mat_AAT_perm.mtx");
  matrix_utils::writeMatrixMarket(size, size, perm_ai.data(), perm_aj.data(),
                                  perm_av.data(), out2);
  out2.close();

  std::cout << "first row size: " << perm_ai[1] - perm_ai[0] << "\n";
  std::cout << "second row size: " << perm_ai[2] - perm_ai[1] << "\n";
  std::cout << "third row size: " << perm_ai[3] - perm_ai[2] << "\n";
  std::cout << "fourth row size: " << perm_ai[4] - perm_ai[3] << "\n";
  std::cout << "fifth row size: " << perm_ai[5] - perm_ai[4] << "\n";
  std::cout << "sixth row size: " << perm_ai[6] - perm_ai[5] << "\n";
  std::cout << "seventh row size: " << perm_ai[7] - perm_ai[6] << "\n";
  std::cout << "eighth row size: " << perm_ai[8] - perm_ai[7] << "\n";
  std::cout << "ninth row size: " << perm_ai[9] - perm_ai[8] << "\n";
  std::cout << "tenth row size: " << perm_ai[10] - perm_ai[9] << "\n";

  std::ofstream out3("mat_AAT_iperm.mtx");
  matrix_utils::permuteMat(size, size, iperm.data(), perm.data(),
                           perm_ai.data(), perm_aj.data(), ai_AAT.data(),
                           aj_AAT.data(), perm_av.data(), av_AAT.data());
  matrix_utils::writeMatrixMarket(size, size, ai_AAT.data(), aj_AAT.data(),
                                  av_AAT.data(), out3);
  out3.close();
}