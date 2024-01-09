
#include "../mkl_wrapper/mkl_eigen.h"
#include "../mkl_wrapper/mkl_solver.h"
#include "../mkl_wrapper/mkl_sparse_mat.h"
#include "../utils/timer.h"
#include "../utils/utils.h"
#include "arpack.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

void register_solvers() {
  using solver_ptr = typename mkl_wrapper::solver_factory::solver_ptr;
  using create_method = typename mkl_wrapper::solver_factory::create_method;
  create_method func = [](mkl_wrapper::mkl_sparse_mat &A) {
    auto prec = std::make_shared<mkl_wrapper::mkl_ilut>(&A);
    prec->set_tau(1e-8);
    prec->set_max_fill(std::min(A.max_nz() * 3, A.cols()));
    auto solver = std::make_unique<mkl_wrapper::mkl_fgmres_solver>(&A, prec);

    solver->set_max_iters(1e5);
    solver->set_rel_tol(1e-10);
    solver->set_restart_steps(20);
    return std::move(solver);
  };
  utils::singleton<mkl_wrapper::solver_factory>::instance().reg("gmres+ilut",
                                                                func);
}

int main(int argc, char **argv) {
  std::string m_mat("../../data/shared/M.bin");
  std::string k_mat("../../data/shared/K.bin");
  std::string g_mat("../../data/shared/G.bin");

  std::vector<MKL_INT> k_csr_rows, k_csr_cols;
  std::vector<double> k_csr_vals;
  utils::ReadFromBinaryCSR(k_mat, k_csr_rows, k_csr_cols, k_csr_vals,
                           SPARSE_INDEX_BASE_ONE);
  std::shared_ptr<MKL_INT[]> k_csr_rows_ptr(k_csr_rows.data(),
                                            [](MKL_INT[]) {});
  std::shared_ptr<MKL_INT[]> k_csr_cols_ptr(k_csr_cols.data(),
                                            [](MKL_INT[]) {});
  std::shared_ptr<double[]> k_csr_vals_ptr(k_csr_vals.data(), [](double[]) {});

  std::vector<MKL_INT> m_csr_rows, m_csr_cols;
  std::vector<double> m_csr_vals;
  utils::ReadFromBinaryCSR(m_mat, m_csr_rows, m_csr_cols, m_csr_vals,
                           SPARSE_INDEX_BASE_ONE);
  std::shared_ptr<MKL_INT[]> m_csr_rows_ptr(m_csr_rows.data(),
                                            [](MKL_INT[]) {});
  std::shared_ptr<MKL_INT[]> m_csr_cols_ptr(m_csr_cols.data(),
                                            [](MKL_INT[]) {});
  std::shared_ptr<double[]> m_csr_vals_ptr(m_csr_vals.data(), [](double[]) {});

  std::vector<MKL_INT> g_csr_rows, g_csr_cols;
  std::vector<double> g_csr_vals;
  utils::ReadFromBinaryCSR(g_mat, g_csr_rows, g_csr_cols, g_csr_vals,
                           SPARSE_INDEX_BASE_ONE);

  for (int i = g_csr_rows.size(); i < m_csr_rows.size(); i++) {
    g_csr_rows.push_back(g_csr_rows.back());
  }
  // g_csr_rows.back() += 1;
  // g_csr_cols.push_back(2);
  // g_csr_vals.push_back(0);

  std::shared_ptr<MKL_INT[]> g_csr_rows_ptr(g_csr_rows.data(),
                                            [](MKL_INT[]) {});
  std::shared_ptr<MKL_INT[]> g_csr_cols_ptr(g_csr_cols.data(),
                                            [](MKL_INT[]) {});
  std::shared_ptr<double[]> g_csr_vals_ptr(g_csr_vals.data(), [](double[]) {});

  const MKL_INT size = m_csr_rows.size() - 1;

  mkl_wrapper::mkl_sparse_mat k(size, size, k_csr_rows_ptr, k_csr_cols_ptr,
                                k_csr_vals_ptr, SPARSE_INDEX_BASE_ONE);
  mkl_wrapper::mkl_sparse_mat m(size, size, m_csr_rows_ptr, m_csr_cols_ptr,
                                m_csr_vals_ptr, SPARSE_INDEX_BASE_ONE);

  mkl_wrapper::mkl_sparse_mat g(size, 2, g_csr_rows_ptr, g_csr_cols_ptr,
                                g_csr_vals_ptr, SPARSE_INDEX_BASE_ONE);
  std::vector<double> rhs(size);
  std::vector<double> select(2, 0);
  std::vector<double> res(size);
  select[1] = 1;
  g.mult_vec(select.data(), rhs.data());

  double max, min;
  {

    mkl_wrapper::arpack_gv ar_gv(&k, &m);

    std::vector<double> eigenvalues(1, 0);
    std::vector<double> eigenvectors(1 * size, 0);
    ar_gv.which("LM");
    utils::Elapse<>::execute(
        "arpack max eigen: ", [&ar_gv, &eigenvalues, &eigenvectors]() {
          ar_gv.eigen_solve(eigenvalues.data(), eigenvectors.data());
        });
    for (auto i : eigenvalues) {
      std::cout << i << std::endl;
    }
    max = eigenvalues[0];
  }
  {

    mkl_wrapper::arpack_gv ar_gv(&m, &k);

    std::vector<double> eigenvalues(1, 0);
    std::vector<double> eigenvectors(1 * size, 0);
    ar_gv.which("LM");
    utils::Elapse<>::execute(
        "arpack min eigen: ", [&ar_gv, &eigenvalues, &eigenvectors]() {
          ar_gv.eigen_solve(eigenvalues.data(), eigenvectors.data());
        });
    for (auto i : eigenvalues) {
      std::cout << 1. / i << std::endl;
    }
    min = 1. / eigenvalues[0];
  }
  register_solvers();

  int freq_size = 2;
  std::vector<double> frequencies(freq_size);
  for (int i = 0; i < freq_size; i++) {
    frequencies[i] = min + i * (max - min) / (freq_size - 1);
  }
  for (int i = 0; i < select.size(); i++) {
    if (i - 1 >= 0)
      select[i - 1] = 0;
    select[i] = 1;
    g.mult_vec(select.data(), rhs.data());
    for (auto freq : frequencies) {
      auto mat = mkl_wrapper::mkl_sparse_sum(k, m, freq);
      auto solver =
          utils::singleton<mkl_wrapper::solver_factory>::instance().create(
              "gmres+ilut", mat);
      solver->solve(rhs.data(), res.data());
      for(auto i:res){
        std::cout<<i<<" ";
      }
      std::cout<<std::endl;
      std::cout<<std::endl;
    }
  }
  return 0;
}