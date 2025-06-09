
#include "../sparse_mat_op/aat.hpp"
#include "../sparse_mat_op/matrix_utils.hpp"
#include "mkl_sparse_mat.h"

#include <vector>

int main() {

  std::string filename = "../data/GD01_c.mtx";

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
  return 0;
}