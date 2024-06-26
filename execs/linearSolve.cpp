
#include "../config.h"
#include "../mkl_wrapper/mkl_solver.h"
#include "../mkl_wrapper/mkl_sparse_mat.h"
#ifdef USE_MUMPS_LIB
#include "../mkl_wrapper/mumps_solver.h"
#endif
#include "../utils/timer.h"
#include "../utils/utils.h"
#include <Eigen/Sparse>
#include <algorithm>
#include <execution>
#include <fast_matrix_market/app/Eigen.hpp>
#include <fstream>
#include <ios>
#include <iostream>
#include <iterator>
#include <memory>
#include <mkl.h>
#include <omp.h>
#include <random>
#include <vector>

using SpMat = typename Eigen::SparseMatrix<double, Eigen::RowMajor, MKL_INT>;
using SpMatMap = typename Eigen::Map<const SpMat>;
int main() {
  // {

  //   std::string m_mat("../../data/eigenvalue/M_sparse.bin");

  //   std::vector<MKL_INT> csr_rows, csr_cols;
  //   std::vector<double> csr_vals;
  //   utils::ReadFromBinaryCSR(m_mat, csr_rows, csr_cols, csr_vals,
  //                            SPARSE_INDEX_BASE_ONE);

  //   std::shared_ptr<double[]> csr_vals_ptr(new double[csr_vals.size()]);

  //   utils::Elapse<>::execute("par_unseq: ", [&csr_vals_ptr, &csr_vals]() {
  //     for (int i = 0; i < 1000; i++)
  //       std::copy(std::execution::par_unseq, csr_vals.begin(),
  //       csr_vals.end(),
  //                 csr_vals_ptr.get());
  //   });

  //   utils::Elapse<>::execute("seq: ", [&csr_vals_ptr, &csr_vals]() {
  //     for (int i = 0; i < 1000; i++)
  //       std::copy(std::execution::seq, csr_vals.begin(), csr_vals.end(),
  //                 csr_vals_ptr.get());
  //   });

  //   utils::Elapse<>::execute("par: ", [&csr_vals_ptr, &csr_vals]() {
  //     for (int i = 0; i < 1000; i++)
  //       std::copy(std::execution::par, csr_vals.begin(), csr_vals.end(),
  //                 csr_vals_ptr.get());
  //   });
  // }

  std::ifstream f("../../data/linear_system/inline_1.mtx");

  // SpMat mat;
  // fast_matrix_market::read_matrix_market_eigen(f, mat);
  // std::cout << mat.rows() << " , " << mat.cols() << std::endl;
  // std::cout << Eigen::MatrixXd(mat) << std::endl << std::endl;
  f.clear();
  f.seekg(0, std::ios::beg);
  std::vector<MKL_INT> csr_rows, csr_cols;
  std::vector<double> csr_vals;
  utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
  SpMatMap mat_csr(csr_rows.size() - 1, csr_rows.size() - 1, csr_cols.size(),
                   csr_rows.data(), csr_cols.data(), csr_vals.data());

  // std::cout << mat_csr.rows() << " , " << mat_csr.cols() << std::endl;
  // std::cout << mat_csr << std::endl << std::endl;
  // SpMat res = mat_csr - mat;
  // Eigen::MatrixXd dense_csr(mat_csr);
  // Eigen::MatrixXd res = dense_csr - mat;
  // std::cout << res.norm() << std::endl;

  std::shared_ptr<MKL_INT[]> csr_rows_ptr(csr_rows.data(), [](MKL_INT[]) {});
  std::shared_ptr<MKL_INT[]> csr_cols_ptr(csr_cols.data(), [](MKL_INT[]) {});
  std::shared_ptr<double[]> csr_vals_ptr(csr_vals.data(), [](double[]) {});
  mkl_wrapper::mkl_sparse_mat mkl_mat(csr_rows.size() - 1, csr_rows.size() - 1,
                                      csr_rows_ptr, csr_cols_ptr, csr_vals_ptr);

  // mkl_wrapper::mkl_sparse_mat_sym mkl_mat_sym(&mkl_mat);
  // std::cout << mkl_mat.nnz() << " " << mkl_mat_sym.nnz() << std::endl;
  // SpMatMap mat_sym_csr(mkl_mat_sym.rows(), mkl_mat_sym.cols(),
  //                      mkl_mat_sym.nnz(), mkl_mat_sym.get_ai().get(),
  //                      mkl_mat_sym.get_aj().get(),
  //                      mkl_mat_sym.get_av().get());
  // std::cout << mat_sym_csr << std::endl << std::endl;

  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_int_distribution<int> dist{0, 10000};

  auto gen = [&dist, &mersenne_engine]() {
    return dist(mersenne_engine) * 1. / 10000;
  };

  std::vector<double> rhs(mkl_mat.rows());
  std::generate(std::begin(rhs), std::end(rhs), gen);
  std::vector<double> x_iter(mkl_mat.rows());
  std::vector<double> x_direct(mkl_mat.rows());
  std::vector<double> x_direct2(mkl_mat.rows());
  std::cout << "m: " << mkl_mat.rows() << " , n: " << mkl_mat.cols()
            << std::endl;

  // {

  //   mkl_set_num_threads_local(10);
  //   auto prec = std::make_shared<mkl_wrapper::mkl_ilut>(&mkl_mat);
  //   prec->set_tau(1e-6);
  //   prec->set_max_fill(200);
  //   utils::Elapse<>::execute("ilut factorize: ",
  //                            [&prec]() { prec->factorize(); });
  //   mkl_wrapper::mkl_fgmres_solver pcg(&mkl_mat, prec);
  //   pcg.set_max_iters(1e5);
  //   pcg.set_rel_tol(1e-10);
  //   pcg.set_restart_steps(20);
  //   utils::Elapse<>::execute("fgmres solve: ", [&pcg, &rhs, &x_iter]() {
  //     pcg.solve(rhs.data(), x_iter.data());
  //   });
  // }

  mkl_wrapper::mkl_sparse_mat_sym mkl_mat_sym(mkl_mat);
  mkl_mat_sym.set_positive_definite(true);

#ifdef USE_MUMPS_LIB
  {
    omp_set_num_threads(1);
    mkl_wrapper::mumps_solver mumps(&mkl_mat_sym);
    utils::Elapse<>::execute("mumps factorize: ",
                             [&mumps]() { mumps.factorize(); });
    utils::Elapse<>::execute("mumps solve: ", [&mumps, &rhs, &x_direct2]() {
      mumps.solve(rhs.data(), x_direct2.data());
    });
  }
#endif
  {
    mkl_set_num_threads_local(1);
    mkl_wrapper::mkl_direct_solver pardiso(&mkl_mat);
    utils::Elapse<>::execute("pardiso factorize: ",
                             [&pardiso]() { pardiso.factorize(); });
    utils::Elapse<>::execute("pardiso solve: ", [&pardiso, &rhs, &x_direct]() {
      pardiso.solve(rhs.data(), x_direct.data());
    });
  }
  // cblas_daxpy(mkl_mat.rows(), -1., x_direct.data(), 1, x_iter.data(), 1);
  // std::cout << "norm: "
  //           << cblas_dnrm2(mkl_mat.rows(), x_iter.data(), 1) /
  //                  cblas_dnrm2(mkl_mat.rows(), x_direct.data(), 1)
  //           << std::endl;

  cblas_daxpy(mkl_mat.rows(), -1., x_direct2.data(), 1, x_direct.data(), 1);
  std::cout << "norm: "
            << cblas_dnrm2(mkl_mat.rows(), x_direct.data(), 1) /
                   cblas_dnrm2(mkl_mat.rows(), x_direct2.data(), 1)
            << std::endl;

  return 0;
}