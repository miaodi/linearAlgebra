
#include "../mkl_wrapper/mkl_eigen.h"
#include "../mkl_wrapper/mkl_solver.h"
#include "../mkl_wrapper/mkl_sparse_mat.h"
#include "../utils/timer.h"
#include "../utils/utils.h"
#include <execution>
#include <fstream>
#include <iostream>
#include <memory>
#include <omp.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

void register_solvers() {
  using solver_ptr = typename mkl_wrapper::solver_factory::solver_ptr;
  using create_method = typename mkl_wrapper::solver_factory::create_method;

  create_method gmres_ilut = [](mkl_wrapper::mkl_sparse_mat &A) {
    auto prec = std::make_shared<mkl_wrapper::mkl_ilut>(&A);
    prec->set_tau(1e-3);
    prec->set_max_fill(std::min((MKL_INT)(A.avg_nz() * 2), A.cols()));
    prec->factorize();
    auto solver = std::make_unique<mkl_wrapper::mkl_fgmres_solver>(&A, prec);
    // prec->print();
    solver->set_max_iters(1e5);
    solver->set_rel_tol(1e-8);
    solver->set_restart_steps(50);
    return std::move(solver);
  };
  utils::singleton<mkl_wrapper::solver_factory>::instance().reg("gmres+ilut",
                                                                gmres_ilut);

  create_method cg_ic0 = [](mkl_wrapper::mkl_sparse_mat &A) {
    auto prec = std::make_shared<mkl_wrapper::mkl_ic0>(A);
    prec->factorize();
    auto solver = std::make_unique<mkl_wrapper::mkl_pcg_solver>(&A, prec);
    // prec->print();
    solver->set_max_iters(1e5);
    solver->set_rel_tol(1e-8);
    return std::move(solver);
  };
  utils::singleton<mkl_wrapper::solver_factory>::instance().reg("cg+ic0",
                                                                cg_ic0);

  create_method cg = [](mkl_wrapper::mkl_sparse_mat &A) {
    auto solver = std::make_unique<mkl_wrapper::mkl_pcg_solver>(&A);
    // prec->print();
    solver->set_max_iters(1e5);
    solver->set_rel_tol(1e-8);
    return std::move(solver);
  };
  utils::singleton<mkl_wrapper::solver_factory>::instance().reg("cg", cg);
}

int main(int argc, char **argv) {
  // mkl_cbwr_set(MKL_CBWR_COMPATIBLE);
  std::cout << sizeof(MKL_INT) << std::endl;

  std::string m_mat("../../data/shared/M_sparse.bin");
  std::string k_mat("../../data/shared/K_sparse.bin");
  std::string g_mat("../../data/shared/G_sparse.bin");
  // std::string v_mat("../../data/shared/V_sparse.bin");

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

  std::cout << "read M\n";
  std::vector<MKL_INT> m_csr_rows, m_csr_cols;
  std::vector<double> m_csr_vals;
  utils::ReadFromBinaryCSR(m_mat, m_csr_rows, m_csr_cols, m_csr_vals,
                           SPARSE_INDEX_BASE_ONE);
  std::shared_ptr<MKL_INT[]> m_csr_rows_ptr(m_csr_rows.data(),
                                            [](MKL_INT[]) {});
  std::shared_ptr<MKL_INT[]> m_csr_cols_ptr(m_csr_cols.data(),
                                            [](MKL_INT[]) {});
  std::shared_ptr<double[]> m_csr_vals_ptr(m_csr_vals.data(), [](double[]) {});

  std::cout << "read G\n";
  std::vector<MKL_INT> g_csr_rows, g_csr_cols;
  std::vector<double> g_csr_vals;
  utils::ReadFromBinaryCSR(g_mat, g_csr_rows, g_csr_cols, g_csr_vals,
                           SPARSE_INDEX_BASE_ONE);
  for (int i = g_csr_rows.size(); i < m_csr_rows.size(); i++) {
    g_csr_rows.push_back(g_csr_rows.back());
  }
  std::shared_ptr<MKL_INT[]> g_csr_rows_ptr(g_csr_rows.data(),
                                            [](MKL_INT[]) {});
  std::shared_ptr<MKL_INT[]> g_csr_cols_ptr(g_csr_cols.data(),
                                            [](MKL_INT[]) {});
  std::shared_ptr<double[]> g_csr_vals_ptr(g_csr_vals.data(), [](double[]) {});

  // std::cout << "read V\n";
  // std::vector<MKL_INT> v_csr_rows, v_csr_cols;
  // std::vector<double> v_csr_vals;
  // utils::ReadFromBinaryCSR(v_mat, v_csr_rows, v_csr_cols, v_csr_vals,
  //                          SPARSE_INDEX_BASE_ONE);
  // for (int i = v_csr_rows.size(); i < m_csr_rows.size(); i++) {
  //   v_csr_rows.push_back(v_csr_rows.back());
  // }
  // std::shared_ptr<MKL_INT[]> v_csr_rows_ptr(v_csr_rows.data(),
  //                                           [](MKL_INT[]) {});
  // std::shared_ptr<MKL_INT[]> v_csr_cols_ptr(v_csr_cols.data(),
  //                                           [](MKL_INT[]) {});
  // std::shared_ptr<double[]> v_csr_vals_ptr(v_csr_vals.data(), [](double[])
  // {});

  const MKL_INT size = m_csr_rows.size() - 1;

  mkl_wrapper::mkl_sparse_mat k(size, size, k_csr_rows_ptr, k_csr_cols_ptr,
                                k_csr_vals_ptr, SPARSE_INDEX_BASE_ONE);
  mkl_wrapper::mkl_sparse_mat m(size, size, m_csr_rows_ptr, m_csr_cols_ptr,
                                m_csr_vals_ptr, SPARSE_INDEX_BASE_ONE);
  // {
  //   mkl_wrapper::mkl_sparse_mat v(size, 4500, v_csr_rows_ptr, v_csr_cols_ptr,
  //                                 v_csr_vals_ptr, SPARSE_INDEX_BASE_ONE);

  //   v.check();
  //   std::cout<<v.nnz()<<std::endl;
  // }
  mkl_wrapper::mkl_sparse_mat_sym sym_k(k);
  mkl_wrapper::mkl_sparse_mat_sym sym_m(m);
  m.clear();
  k.clear();

  const int num_ports = 900;

  mkl_wrapper::mkl_sparse_mat g(size, num_ports, g_csr_rows_ptr, g_csr_cols_ptr,
                                g_csr_vals_ptr, SPARSE_INDEX_BASE_ONE);

  // std::cout << *std::max_element(g_csr_cols.data(),
  //                                g_csr_cols.data() + g_csr_rows[size])
  //           << std::endl;

  // sym_k.to_zero_based();
  // sym_m.to_zero_based();
  // g.to_zero_based();
  std::cout << "compute eigenvalues\n";
  double max{1.28061e+09}, min{26141.9};
  {
    mkl_set_num_threads(10);
    mkl_wrapper::mkl_eigen_sparse_d_gv ar_gv(&sym_k, &sym_m);

    ar_gv.set_tol(3);
    ar_gv.set_num_eigen(1);
    ar_gv.set_ncv(10);
    std::vector<double> eigenvalues(1, 0);
    std::vector<double> eigenvectors(1 * size, 0);
    ar_gv.which("L");
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
    mkl_set_num_threads(10);
    mkl_wrapper::mkl_eigen_sparse_d_gv ar_gv(&sym_m, &sym_k);

    ar_gv.set_tol(3);
    ar_gv.set_num_eigen(1);
    ar_gv.set_ncv(10);
    std::vector<double> eigenvalues(1, 0);
    std::vector<double> eigenvectors(1 * size, 0);
    ar_gv.which("L");
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

  int freq_size = 5;
  std::vector<double> frequencies(freq_size);
  for (int i = 0; i < freq_size; i++) {
    frequencies[i] = min + i * (max - min) / (freq_size - 1);
  }

  const int total_work = num_ports * freq_size;
  // csr data for the V^T  matrix
  auto vTAI = std::shared_ptr<MKL_INT[]>(new MKL_INT[total_work + 1]);
  vTAI[0] = 0;
  std::shared_ptr<MKL_INT[]> vTAJ;
  std::shared_ptr<double[]> vTAV;

  int count = 0;

  omp_set_max_active_levels(2);
#pragma omp parallel num_threads(1)
  {
    mkl_set_num_threads_local(10);
    mkl_set_dynamic(0);
    const int total_omp_threads = omp_get_num_threads();
    const int local_port_size = num_ports / total_omp_threads + 1;
    const int rank = omp_get_thread_num();
    std::vector<double> select(num_ports, 0);
    auto [start, end] = utils::LoadBalancedPartition(
        select.begin(), select.end(), rank, total_omp_threads);
    std::vector<MKL_INT> vTAI_local;
    vTAI_local.push_back(0);
    std::vector<MKL_INT> vTAJ_local;
    std::vector<double> vTAV_local;
    std::vector<MKL_INT> local_to_global;
#pragma omp critical
    {
      std::cout << "omp_get_thread_num: " << omp_get_thread_num()
                << " mkl_max: " << mkl_get_max_threads()
                << " omp_get_num_threads: " << omp_get_num_threads()
                << std::endl;
    }

    std::vector<double> rhs(size);
    std::vector<double> res(size);
    double norm = 0;
    for (size_t f = 0; f < frequencies.size(); f++) {
      auto mat = mkl_wrapper::mkl_sparse_sum(sym_m, sym_k, frequencies[f]);
      mat.set_positive_definite(true);
      auto solver =
          utils::singleton<mkl_wrapper::solver_factory>::instance().create(
              "cg+ic0", mat);
      for (auto it = start; it < end; it++) {
        local_to_global.push_back((it - select.begin()) * frequencies.size() +
                                  f);
        *it = 1;
        g.mult_vec(select.data(), rhs.data());
        solver->solve(rhs.data(), res.data());
        norm = 0.;
        // #pragma omp parallel for reduction(max : norm)
        for (MKL_INT col = 0; col < size; col++) {
          norm = std::abs(res[col]) > norm ? std::abs(res[col]) : norm;
        }
        for (MKL_INT col = 0; col < size; col++) {
          res[col] /= norm;
          if (std::abs(res[col]) > 1e-3) {
            vTAJ_local.push_back(col);
            vTAV_local.push_back(res[col]);
          }
        }
        vTAI_local.push_back(vTAV_local.size());

        vTAI[local_to_global.back() + 1] =
            *vTAI_local.crbegin() - *(vTAI_local.crbegin() + 1);
        *it = 0;
#pragma omp critical
        { count += 1; }
        // #pragma omp single
        { utils::printProgress(count * 1. / total_work); }
      }
    }
    // One thread indicates that the barrier is complete.
#pragma omp barrier
#pragma omp master
    {
      std::inclusive_scan(std::execution::seq, vTAI.get(),
                          vTAI.get() + total_work + 1, vTAI.get(),
                          std::plus<>());
      std::cout << "\nnnz: " << vTAI[total_work] << std::endl;
      vTAJ.reset(new MKL_INT[vTAI[total_work]]);
      vTAV.reset(new double[vTAI[total_work]]);
    }
#pragma omp barrier
    for (size_t i = 0; i < local_to_global.size(); i++) {
      std::copy(std::execution::seq, vTAJ_local.cbegin() + vTAI_local[i],
                vTAJ_local.cbegin() + vTAI_local[i + 1],
                vTAJ.get() + vTAI[local_to_global[i]]);
      std::copy(std::execution::seq, vTAV_local.cbegin() + vTAI_local[i],
                vTAV_local.cbegin() + vTAI_local[i + 1],
                vTAV.get() + vTAI[local_to_global[i]]);
    }
  }
  // mkl_set_num_threads_local(0); // make master thread use global setting
  mkl_wrapper::mkl_sparse_mat vt(freq_size * num_ports, size, vTAI, vTAJ, vTAV);
  vt.to_one_based();
  {
    mkl_set_num_threads(10);
    auto m_red = mkl_sparse_mult_papt(sym_m, vt);
    auto k_red = mkl_sparse_mult_papt(sym_k, vt);
    auto g_red = mkl_sparse_mult(vt, g);
  }
  return 0;
}
