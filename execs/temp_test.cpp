#include "../reordering/quotient_graph.h"
#include "../sparse_mat_op/io.hpp"
#include "../sparse_mat_op/matrix_utils.hpp"
#include "../sparse_mat_op/permutation.hpp"
#include "MinimumDegree.hpp"
#include "sp_ops.hpp"
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
  std::vector<int> perm_ai(size + 1);
  std::vector<int> perm_aj(ai_AAT.back());
  std::vector<double> perm_av(ai_AAT.back(), 1.);

  matrix_utils::permuteMat<int, int, double>(size, size, perm.data(), iperm.data(), ai_AAT.data(),
                           aj_AAT.data(), av_AAT.data(), perm_ai.data(), perm_aj.data(),
                           perm_av.data());

  std::ofstream out2("mat_AAT_perm.mtx");
  matrix_utils::writeMatrixMarket(size, size, perm_ai.data(), perm_aj.data(),
                                  perm_av.data(), out2);
  out2.close();

  std::ofstream out3("mat_AAT_iperm.mtx");
  matrix_utils::permuteMat<int, int, double>(size, size, iperm.data(), perm.data(),
                           perm_ai.data(), perm_aj.data(), perm_av.data(), ai_AAT.data(),
                           aj_AAT.data(), av_AAT.data());
  matrix_utils::writeMatrixMarket(size, size, ai_AAT.data(), aj_AAT.data(),
                                  av_AAT.data(), out3);
  out3.close();

  {
    std::vector<std::pair<int, int>> pairs;
    for (int i = 0; i < size; i++) {
      for (int j = ai_AAT[i]; j < ai_AAT[i + 1]; j++) {
        pairs.emplace_back(i, aj_AAT[j]);
      }
    }
    srrg2_solver::QuotientGraph qg2(pairs, size);
    std::vector<int> perm2(size);
    std::vector<int> iperm2(size);
    qg2.setPolicy(srrg2_solver::QuotientGraph::External);
    qg2.mdo(perm2);
    for (int i = 0; i < size; i++) {
      iperm2[perm2[i]] = i;
    }

    std::cout << "is permutation: "
              << matrix_utils::isPermutation(size, ai_AAT[0], perm2.data())
              << "\n";
    std::ofstream out4("mat_AAT_qg.mtx");
    matrix_utils::permuteMat<int, int, double>(size, size, perm2.data(), iperm2.data(),
                             ai_AAT.data(), aj_AAT.data(), perm_ai.data(),
                             perm_aj.data(), av_AAT.data(), perm_av.data());
    matrix_utils::writeMatrixMarket(size, size, perm_ai.data(), perm_aj.data(),
                                    perm_av.data(), out4);
    out4.close();
  }
}
