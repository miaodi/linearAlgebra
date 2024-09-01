#include "../config.h"

#ifdef USE_CUDA

#include "cudss_wrapper.h"
#include "matrix_utils.hpp"
#include "mkl_solver.h"
#include "mkl_sparse_mat.h"
#include "utils.h"
#include <algorithm>
#include <deque>
#include <gtest/gtest.h>
#include <omp.h>
#include <unordered_map>

TEST(Solve, pardiso_vs_cudss) {
  omp_set_num_threads(3);
  const double tol = 1e-10;
  std::vector<std::string> files{"data/ex5.mtx"};
  std::ofstream myfile;
  for (const auto &fn : files) {
    std::ifstream f(fn);
    f.clear();
    f.seekg(0, std::ios::beg);
    std::vector<MKL_INT> csr_rows, csr_cols;
    std::vector<double> csr_vals;
    utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);

    mkl_wrapper::mkl_sparse_mat mat(csr_rows.size() - 1, csr_rows.size() - 1,
                                    csr_rows, csr_cols, csr_vals);

    std::vector<double> x_pardiso(mat.rows(), 0.0);
    std::vector<double> b(mat.rows(), 1.0);
    std::vector<double> x_cudss(mat.rows(), 0.0);
    std::vector<double> x_cudss2(mat.rows(), 0.0);

    mkl_wrapper::mkl_direct_solver pardiso_solver(&mat);
    pardiso_solver.factorize();
    pardiso_solver.solve(b.data(), x_pardiso.data());

    mkl_wrapper::cudss_solver cudss_solver(&mat);
    cudss_solver.factorize();
    cudss_solver.solve(b.data(), x_cudss.data());
    mkl_wrapper::cudss_solver cudss_solver2(&mat);
    cudss_solver2.factorize();
    cudss_solver2.solve(b.data(), x_cudss2.data());

    for (int i = 0; i < mat.rows(); i++) {
      EXPECT_NEAR(x_cudss2[i], x_cudss[i], tol * std::abs(x_cudss[i]));
    }
  }
}
#endif