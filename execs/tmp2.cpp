
#include "../mkl_wrapper/incomplete_lu.h"
#include "../mkl_wrapper/mkl_eigen.h"
#include "../mkl_wrapper/mkl_solver.h"
#include "../mkl_wrapper/mkl_sparse_mat.h"
#include "../utils/timer.h"
#include "../utils/utils.h"
#include "Reordering.h"
#include <execution>
#include <fstream>
#include <iostream>
#include <memory>
#include <omp.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

int main(int argc, char **argv) {

  std::string k_mat("../../data/shared/K2.bin");
  std::vector<MKL_INT> k_csr_rows, k_csr_cols;
  std::vector<double> k_csr_vals;
  std::cout << "read K\n";
  utils::ReadFromBinaryCSR(k_mat, k_csr_rows, k_csr_cols, k_csr_vals,
                           SPARSE_INDEX_BASE_ONE);
  std::shared_ptr<MKL_INT[]> k_csr_rows_ptr(k_csr_rows.data(),
                                            [](MKL_INT[]) {});
  std::shared_ptr<MKL_INT[]> k_csr_cols_ptr(k_csr_cols.data(),
                                            [](MKL_INT[]) {});
  std::shared_ptr<double[]> k_csr_vals_ptr(k_csr_vals.data(), [](double[]) {});

  const MKL_INT size = k_csr_rows.size() - 1;
  mkl_wrapper::mkl_sparse_mat mat(size, size, k_csr_rows_ptr, k_csr_cols_ptr,
                                  k_csr_vals_ptr, SPARSE_INDEX_BASE_ONE);
  mat.to_zero_based();

  // std::ifstream f("../tests/data/ex5.mtx");
  // f.clear();
  // f.seekg(0, std::ios::beg);
  // std::vector<MKL_INT> csr_rows, csr_cols;
  // std::vector<double> csr_vals;
  // utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
  // mkl_wrapper::mkl_sparse_mat mat(csr_rows.size() - 1, csr_rows.size() - 1,
  //                                 csr_rows, csr_cols, csr_vals);

  std::cout << "bandwidth before reordering: " << mat.bandwidth() << std::endl;
  std::vector<double> rhs(size, 1.);
  std::vector<double> res(size, 0);
  std::ofstream myfile;
  // myfile.open("mat.svg");
  // mat.print_svg(myfile);
  // myfile.close();
  mkl_wrapper::incomplete_lu_k iluk;
  iluk.set_level(0);
  iluk.symbolic_factorize(&mat);
  iluk.numeric_factorize(&mat);
  // myfile.open("mat_iluk.svg");
  // iluk.print_svg(myfile);
  // myfile.close();
  // iluk.solve(rhs.data(), res.data());
  std::cout<<iluk.nnz()<<std::endl;
  // for (auto i : res)
  //   std::cout << i << " ";
  // std::cout << std::endl;
  // iluk.print();
  // mkl_wrapper::mkl_ilu0 ilu0;
  // ilu0.symbolic_factorize(&mat);
  // ilu0.numeric_factorize(&mat);
  // myfile.open("mat_ilu0.svg");
  // ilu0.print_svg(myfile);
  // myfile.close();
  // ilu0.solve(rhs.data(), res.data());
  // for (auto i : res)
  //   std::cout << i << " ";
  // std::cout << std::endl;
  // ilu0.print();
  return 0;
}
