#include "UnsymmReordering.hpp"
#include "io.hpp"
#include "matrix_utils.hpp"
#include "permutation.hpp"
#include <cxxopts.hpp>
#include <fstream>
#include <mkl.h>
#include <string>
#include <vector>

int main(int argc, char **argv) {

  cxxopts::Options options("GMRES Example",
                           "Example of using GMRES with a CSR matrix");
  options.add_options()(
      "f,filename", "Matrix Market file to read",
      cxxopts::value<std::string>()->default_value("../tests/data/ex5.mtx"))(
      "l,level", "ILU level",
      cxxopts::value<int>()->default_value("0"))("h,help", "Print usage");
  auto result = options.parse(argc, argv);
  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }
  std::string filename = result["filename"].as<std::string>();
  int level = result["level"].as<int>();
  std::ifstream f(filename);
  f.clear();
  f.seekg(0, std::ios::beg);
  matrix_utils::CSRMatrix<int, int, double> csr_matrix, ilu_matrix;
  matrix_utils::readMatrixMarket(f, csr_matrix);
  std::cout << "size: " << csr_matrix.rows << std::endl;
  std::ofstream out0("mat_csr.svg");
  matrix_utils::writeSVG(csr_matrix.rows, csr_matrix.cols, csr_matrix.AI(),
                         csr_matrix.AJ(), out0);
  out0.close();

  std::vector<int> matching_row(csr_matrix.rows);
  std::vector<int> matching_col(csr_matrix.rows);
  reordering::MaximumMatching(csr_matrix.rows, csr_matrix.AI(), csr_matrix.AJ(),
                              matching_row.data(), matching_col.data());
  // for (int i = 0; i < csr_matrix.rows; i++) {
  //   std::cout << matching_row[i] << " ";
  // }
  // std::cout << std::endl;
  // for (int i = 0; i < csr_matrix.rows; i++) {
  //   std::cout << matching_col[i] << " ";
  // }
  // std::cout << std::endl;

  // // Row permute the matrix according to matching_row
  // matrix_utils::CSRMatrix<int, int, double> permuted_matrix;

  // permuted_matrix.rows = csr_matrix.rows;
  // permuted_matrix.cols = csr_matrix.cols;
  // permuted_matrix.ResizeAI(csr_matrix.rows + 1);
  // permuted_matrix.ResizeAJ(csr_matrix.NNZ());
  // permuted_matrix.ResizeAV(csr_matrix.NNZ());
  // matrix_utils::permuteMat(csr_matrix.rows, csr_matrix.cols, matching_col.data(),
  //                          (int*)nullptr, csr_matrix.AI(),
  //                          csr_matrix.AJ(), permuted_matrix.AI(),
  //                          permuted_matrix.AJ());

  // std::ofstream out1("mat_csr_rowperm.svg");
  // matrix_utils::writeSVG(permuted_matrix.rows, permuted_matrix.cols,
  //                        permuted_matrix.AI(), permuted_matrix.AJ(), out1);
  // out1.close();

  {
    reordering::HungarianAlgorithm<int, int, double> hungarian;
    std::vector<int> matching_row(csr_matrix.rows);
    std::vector<int> matching_col(csr_matrix.rows);
    std::vector<double> potential_row(csr_matrix.rows);
    std::vector<double> potential_col(csr_matrix.rows);
    hungarian(csr_matrix.rows, csr_matrix.AI(), csr_matrix.AJ(),
              csr_matrix.AV(), matching_row.data(), matching_col.data(),
              potential_row.data(), potential_col.data());

    for (int i = 0; i < csr_matrix.rows; i++) {
      std::cout << matching_row[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < csr_matrix.rows; i++) {
      std::cout << potential_row[i] << " ";
    }
    std::cout << std::endl;
  }
  return 0;
}