
#include "Cholesky.hpp"
#include "aat.hpp"
#include "io.hpp"
#include "matrix_utils.hpp"
#include <fstream>
#include <string>
#include <vector>

int main() {

  std::string filename = "../data/matrix_fig_4_2.mtx";

  std::ifstream f(filename);
  f.clear();
  f.seekg(0, std::ios::beg);
  matrix_utils::CSRMatrix<int, int, double> csr_matrix;
  matrix_utils::readMatrixMarket(f, csr_matrix);

  std::ofstream out0("mat_csr.svg");
  matrix_utils::writeSVG(csr_matrix.rows, csr_matrix.cols, csr_matrix.AI(),
                         csr_matrix.AJ(), out0);
  out0.close();
  std::vector<int> parent(csr_matrix.rows);
  std::vector<int> ancestor(csr_matrix.rows);

  factorization::EliminationTree(csr_matrix.rows, csr_matrix.AI(),
                                 csr_matrix.AJ(), parent.data(),
                                 ancestor.data());
  for (size_t i = 0; i < csr_matrix.rows; i++) {
    std::cout << i + 1 << "->" << parent[i] + 1 << std::endl;
  }

  std::vector<int> row_size(csr_matrix.rows);
  factorization::RowSubtreeSize(csr_matrix.rows, csr_matrix.AI()[0], parent.data(), row_size.data());
  for (size_t i = 0; i < csr_matrix.rows; i++) {
    std::cout << i + 1 << "->" << row_size[i] << std::endl;
  }

  return 0;
}