
#include "Cholesky.hpp"
#include "aat.hpp"
#include "config.h"
#include "io.hpp"
#include "matrix_utils.hpp"
#include "precond.hpp"
#include "utils.h"
#include <cxxopts.hpp>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  cxxopts::Options options("Cholesky Example", "Example of using Cholesky");
  options.add_options()("f,filename", "Matrix Market file to read",
                        cxxopts::value<std::string>()->default_value(
                            "../data/matrix_fig_4_2.mtx"));
  auto result = options.parse(argc, argv);
  std::string filename = result["filename"].as<std::string>();

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
  // for (size_t i = 0; i < csr_matrix.rows; i++) {
  //   std::cout << i + 1 << "->" << parent[i] + 1 << std::endl;
  // }

  // std::vector<int> row_size(csr_matrix.rows);
  // factorization::RowCount(csr_matrix.rows, csr_matrix.AI(), csr_matrix.AJ(),
  //                         parent.data(), row_size.data(), ancestor.data());
  // for (size_t i = 0; i < csr_matrix.rows; i++) {
  //   std::cout << i + 1 << "->" << row_size[i] << std::endl;
  // }
#ifdef USE_BOOST_LIB
  utils::printEliminationTree(csr_matrix.rows, csr_matrix.AI()[0],
                              parent.data(), "tree.dot");
#endif

  std::vector<int> perm(csr_matrix.rows);
  std::vector<int> iperm(csr_matrix.rows);
  std::vector<int> permed_parent(csr_matrix.rows);
  factorization::PostOrder<int> postorder;
  postorder(csr_matrix.rows, 0, parent.data(), permed_parent.data(),
            perm.data(), iperm.data());
  for (size_t i = 0; i < csr_matrix.rows; i++) {
    std::cout << perm[i] << " ";
  }

#ifdef USE_BOOST_LIB
  utils::printEliminationTree(csr_matrix.rows, csr_matrix.AI()[0],
                              permed_parent.data(), "post_order_tree.dot");
#endif
  std::cout << std::endl;
  std::cout << "postorder: \n";
  for (size_t i = 0; i < csr_matrix.rows; i++) {
    if (i != perm[i] - csr_matrix.Base()) {
      std::cout << i << " " << perm[i] - csr_matrix.Base() << std::endl;
    }
  }
  {
    std::cout << "check postordering" << std::endl;
    std::vector<int> perm2(csr_matrix.rows);
    std::vector<int> iperm2(csr_matrix.rows);
    std::vector<int> permed_parent2(csr_matrix.rows);
    postorder(csr_matrix.rows, 0, permed_parent.data(), permed_parent2.data(),
              perm2.data(), iperm2.data());
    for (size_t i = 0; i < csr_matrix.rows; i++) {
      if (i != perm2[i] - csr_matrix.Base()) {
        std::cout << i << " " << perm2[i] - csr_matrix.Base() << std::endl;
      }
    }
  }

  std::vector<int> row_count(csr_matrix.rows + 1);
  std::vector<int> col_count(csr_matrix.rows + 1);
  row_count[0] = col_count[0] = csr_matrix.Base();
  std::vector<int> mark(csr_matrix.rows);
  factorization::NNZCount(csr_matrix.rows, csr_matrix.AI(), csr_matrix.AJ(),
                          parent.data(), row_count.data() + 1,
                          col_count.data() + 1, mark.data());
  std::inclusive_scan(col_count.begin(), col_count.end(), col_count.begin());

  {
    csr_matrix.ResizeDiagonal(csr_matrix.rows);
    std::cout << matrix_utils::Diagonal(csr_matrix.rows, csr_matrix.AI(),
                                        csr_matrix.AJ(), csr_matrix.AV(),
                                        csr_matrix.Diagonal(),
                                        static_cast<double *>(nullptr))
              << std::endl;

    matrix_utils::CSRMatrix<int, int, double> icc;
    matrix_utils::ICCLevelSymbolic2(csr_matrix.rows, csr_matrix.AI(),
                                    csr_matrix.AJ(), csr_matrix.Diagonal(),
                                    10000, icc);
    std::cout << std::noboolalpha;
    for (size_t i = 0; i < icc.rows + 1; i++) {
      if (icc.AI()[i] != col_count[i]) {
        std::cout << i << " " << icc.AI()[i] << " " << col_count[i]
                  << std::endl;
      }
    }
  }
  return 0;
}