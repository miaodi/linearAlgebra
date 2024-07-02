
#include "../mkl_wrapper/amgcl_precond.h"
#include "../mkl_wrapper/incomplete_cholesky.h"
#include "../mkl_wrapper/incomplete_lu.h"
#include "../mkl_wrapper/matrix_utils.hpp"
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
#include <mkl.h>
#include <omp.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

mkl_wrapper::mkl_sparse_mat identity(const MKL_INT size) {
  std::vector<MKL_INT> ai(size + 1);
  std::vector<MKL_INT> aj(size);
  std::vector<double> av(size);
  for (MKL_INT i = 0; i < size; i++) {
    ai[i] = i;
    aj[i] = i;
    av[i] = 1;
  }
  ai[size] = size;
  return mkl_wrapper::mkl_sparse_mat(size, size, ai, aj, av);
}

void register_solvers() {
  using solver_ptr = typename mkl_wrapper::solver_factory::solver_ptr;
  using create_method = typename mkl_wrapper::solver_factory::create_method;

  create_method gmres_ilut = [](mkl_wrapper::mkl_sparse_mat &A) {
    auto prec = std::make_shared<mkl_wrapper::mkl_ilut>();
    prec->set_tau(1e-2);
    prec->set_max_fill(std::min((MKL_INT)(A.avg_nz() * 4), A.cols()));

    utils::Elapse<>::execute("mkl_ilut symbolic factorization: ",
                             [&A, &prec]() { prec->symbolic_factorize(&A); });

    utils::Elapse<>::execute("mkl_ilut numeric factorization: ",
                             [&A, &prec]() { prec->numeric_factorize(&A); });
    auto solver = std::make_unique<mkl_wrapper::mkl_fgmres_solver>(&A, prec);
    // prec->print();
    solver->set_max_iters(1e5);
    solver->set_rel_tol(1e-8);
    solver->set_restart_steps(100);
    return std::move(solver);
  };
  utils::singleton<mkl_wrapper::solver_factory>::instance().reg("gmres+ilut",
                                                                gmres_ilut);

  create_method gmres_ilu0 = [](mkl_wrapper::mkl_sparse_mat &A) {
    auto prec = std::make_shared<mkl_wrapper::mkl_ilu0>();
    utils::Elapse<>::execute("mkl_ilu0 symbolic factorization: ",
                             [&A, &prec]() { prec->symbolic_factorize(&A); });

    utils::Elapse<>::execute("mkl_ilu0 numeric factorization: ",
                             [&A, &prec]() { prec->numeric_factorize(&A); });
    auto solver = std::make_unique<mkl_wrapper::mkl_fgmres_solver>(&A, prec);
    // prec->print();
    solver->set_max_iters(1e5);
    solver->set_rel_tol(1e-8);
    // solver->set_restart_steps(50);
    return std::move(solver);
  };
  utils::singleton<mkl_wrapper::solver_factory>::instance().reg("gmres+ilu0",
                                                                gmres_ilu0);

  create_method gmres_iluk = [](mkl_wrapper::mkl_sparse_mat &A) {
    auto prec = std::make_shared<mkl_wrapper::incomplete_lu_k>();
    prec->set_level(4);
    utils::Elapse<>::execute("incomplete_lu_k symbolic factorization: ",
                             [&A, &prec]() { prec->symbolic_factorize(&A); });

    utils::Elapse<>::execute("incomplete_lu_k numeric factorization: ",
                             [&A, &prec]() { prec->numeric_factorize(&A); });
    auto solver = std::make_unique<mkl_wrapper::mkl_fgmres_solver>(&A, prec);
    // prec->print();
    solver->set_max_iters(1e5);
    solver->set_rel_tol(1e-8);
    solver->set_restart_steps(150);
    return std::move(solver);
  };
  utils::singleton<mkl_wrapper::solver_factory>::instance().reg("gmres+iluk",
                                                                gmres_iluk);

  create_method cg_ic0 = [](mkl_wrapper::mkl_sparse_mat &A) {
    auto prec = std::make_shared<mkl_wrapper::incomplete_cholesky_k>();
    prec->set_level(2);
    prec->shift(true);
    utils::Elapse<>::execute("incomplete_cholesky_k symbolic factorization: ",
                             [&A, &prec]() { prec->symbolic_factorize(&A); });
    utils::Elapse<>::execute("incomplete_cholesky_k numeric factorization: ",
                             [&A, &prec]() {
                               if (!prec->numeric_factorize(&A))
                                 std::abort();
                             });
    auto solver = std::make_unique<mkl_wrapper::mkl_pcg_solver>(&A, prec);
    // prec->print();
    solver->set_max_iters(1e5);
    solver->set_rel_tol(1e-8);
    return std::move(solver);
  };
  utils::singleton<mkl_wrapper::solver_factory>::instance().reg(
      "cg+incomplete_cholesky_k", cg_ic0);

  create_method cg_incomplete_cholesky_fm = [](mkl_wrapper::mkl_sparse_mat &A) {
    auto prec = std::make_shared<mkl_wrapper::incomplete_cholesky_fm>();
    prec->set_lsize(40);
    prec->set_rsize(0);
    prec->shift(true);
    // std::abort();
    utils::Elapse<>::execute("incomplete_cholesky_fm symbolic factorization: ",
                             [&A, &prec]() { prec->symbolic_factorize(&A); });

    utils::Elapse<>::execute("incomplete_cholesky_fm numeric factorization: ",
                             [&A, &prec]() { prec->numeric_factorize(&A); });
    auto solver = std::make_unique<mkl_wrapper::mkl_pcg_solver>(&A, prec);
    // std::ofstream myfile;
    // myfile.open("icc.svg");
    // prec->print_svg(myfile);
    // myfile.close();
    // prec->print();
    solver->set_max_iters(1e5);
    solver->set_rel_tol(1e-8);
    return std::move(solver);
  };
  utils::singleton<mkl_wrapper::solver_factory>::instance().reg(
      "cg+incomplete_cholesky_fm", cg_incomplete_cholesky_fm);

  create_method cg = [](mkl_wrapper::mkl_sparse_mat &A) {
    auto solver = std::make_unique<mkl_wrapper::mkl_pcg_solver>(&A);
    // prec->print();
    solver->set_max_iters(1e5);
    solver->set_rel_tol(1e-8);
    return std::move(solver);
  };
  utils::singleton<mkl_wrapper::solver_factory>::instance().reg("cg", cg);
}

std::vector<double> permuteVec(const std::vector<MKL_INT> &iperm,
                               const std::vector<double> &v,
                               const MKL_INT base = 0) {
  std::vector<double> res(v.size());
#pragma omp parallel for
  for (size_t i = 0; i < v.size(); i++) {
    res[i] = v[iperm[i] - base];
  }
  return res;
}

int main(int argc, char **argv) {
  mkl_set_num_threads(16);
  std::string k_mat(
      "/remote/tcad25/dimiao/work/examples/xiaopeng_czm/failedMat.bin");
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

  // std::ifstream f(
  //     "/remote/tcad25/dimiao/work/examples/xiaopeng_czm/successMat.mtx");
  // f.clear();
  // f.seekg(0, std::ios::beg);
  // std::vector<MKL_INT> csr_rows, csr_cols;
  // std::vector<double> csr_vals;
  // utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
  // const MKL_INT size = csr_rows.size() - 1;
  // mkl_wrapper::mkl_sparse_mat mat(csr_rows.size() - 1, csr_rows.size() - 1,
  //                                 csr_rows, csr_cols, csr_vals);
  // mat.to_zero_based();
  std::cout << "size: " << size << std::endl;

  std::cout << "bandwidth before reordering: " << mat.bandwidth() << std::endl;
  std::vector<MKL_INT> iperm, perm;
  reordering::SerialCM(&mat, iperm, perm);
  std::cout << "is permutation: "
            << (utils::isPermutation(iperm, mat.mkl_base())) << std::endl;
  auto [ai, aj, av] = matrix_utils::AllocateCSRData(mat.rows(), mat.nnz());
  matrix_utils::permute(mat.rows(), (int)mat.mkl_base(), mat.get_ai().get(),
                        mat.get_aj().get(), mat.get_av().get(), iperm.data(),
                        perm.data(), ai.get(), aj.get(), av.get());
  mkl_wrapper::mkl_sparse_mat perm_mat(mat.rows(), mat.cols(), ai, aj, av,
                                       SPARSE_INDEX_BASE_ONE);
  std::cout << "bandwidth after rcm reordering: " << perm_mat.bandwidth()
            << std::endl;

  std::vector<double> rhs(size);
  std::vector<double> x(size);

  std::string rhs_vec(
      "/remote/tcad25/dimiao/work/examples/xiaopeng_czm/failedRHS.bin");
  utils::ReadFromBinaryVec(rhs_vec, rhs);
  auto perm_rhs = permuteVec(iperm, rhs);

  auto sqrt_D = perm_mat.rowwiseSqrtNorm();
  perm_mat.DtAD(sqrt_D);
  for (size_t i = 0; i < perm_rhs.size(); i++) {
    perm_rhs[i] *= sqrt_D[i];
  }

  auto identity_mat = identity(size);
  // {
  //   mkl_wrapper::mkl_eigen_sparse_d_gv gv(&perm_mat, &identity_mat);
  //   gv.set_tol(1);
  //   gv.set_num_eigen(1);
  //   gv.set_ncv(3);
  //   gv.which("L");
  //   std::vector<double> eigenvalues(1, 0);
  //   std::vector<double> eigenvectors(1 * size, 0);
  //   utils::Elapse<>::execute(
  //       "mkl max eigen: ", [&gv, &eigenvalues, &eigenvectors]() {
  //         gv.eigen_solve(eigenvalues.data(), eigenvectors.data());
  //       });
  //   for (auto i : eigenvalues) {
  //     std::cout << i << std::endl;
  //   }
  // }

  // {
  //   mkl_wrapper::mkl_eigen_sparse_d_gv gv(&identity_mat, &perm_mat);
  //   gv.set_tol(1);
  //   gv.set_num_eigen(1);
  //   gv.set_ncv(3);
  //   gv.which("L");
  //   std::vector<double> eigenvalues(1, 0);
  //   std::vector<double> eigenvectors(1 * size, 0);
  //   utils::Elapse<>::execute(
  //       "mkl min eigen: ", [&gv, &eigenvalues, &eigenvectors]() {
  //         gv.eigen_solve(eigenvalues.data(), eigenvectors.data());
  //       });
  //   for (auto i : eigenvalues) {
  //     std::cout << 1. / i << std::endl;
  //   }
  // }

  std::cout << "problem size: " << size << std::endl;
  register_solvers();
  int incx = 1;
  std::cout << "rhs norm: " << dnrm2(&size, rhs.data(), &incx) << std::endl;

  // auto solver =
  //     utils::singleton<mkl_wrapper::solver_factory>::instance().create(
  //         "cg+incomplete_cholesky_k", perm_mat);
  // std::cout << "cg+incomplete_cholesky_k: \n";
  // solver->set_print_level(1);
  // x = std::vector<double>(size, 0);

  // utils::Elapse<>::execute("cg solve: ",
  //                          [&]() { solver->solve(perm_rhs.data(), x.data());
  //                          });

  auto solver =
      utils::singleton<mkl_wrapper::solver_factory>::instance().create(
          "cg+incomplete_cholesky_fm", perm_mat);
  std::cout << "cg+incomplete_cholesky_fm: \n";
  solver->set_print_level(1);
  x = std::vector<double>(size, 0);

  utils::Elapse<>::execute("iterative solve: ",
                           [&]() { solver->solve(perm_rhs.data(), x.data()); });
  utils::Elapse<>::execute("pardiso solve: ", [&]() {
    auto solver = std::make_unique<mkl_wrapper::mkl_direct_solver>(&perm_mat);
    solver->factorize();
    x = std::vector<double>(size, 0);
    solver->solve(perm_rhs.data(), x.data());
  });

  return 0;
}
