#include "../mkl_wrapper/incomplete_cholesky.h"
#include "../mkl_wrapper/incomplete_lu.h"
#include "../mkl_wrapper/mkl_solver.h"
#include "../mkl_wrapper/mkl_sparse_mat.h"
#include "../reordering/Reordering.h"
#include "../utils/timer.h"
#include "../utils/utils.h"
#include <algorithm>
#include <execution>
#include <fstream>
#include <ios>
#include <iostream>
#include <iterator>
#include <memory>
#include <mkl.h>
#include <random>
#include <vector>

void register_solvers() {
  using solver_ptr = typename mkl_wrapper::solver_factory::solver_ptr;
  using create_method = typename mkl_wrapper::solver_factory::create_method;

  create_method gmres_ilut = [](mkl_wrapper::mkl_sparse_mat &A) {
    auto prec = std::make_shared<mkl_wrapper::mkl_ilut>();
    prec->set_tau(1e-3);
    prec->set_max_fill(std::min((MKL_INT)(A.avg_nz() * 2), A.cols()));

    utils::Elapse<>::execute("mkl_ilut symbolic factorization: ",
                             [&A, &prec]() { prec->symbolic_factorize(&A); });

    utils::Elapse<>::execute("mkl_ilut numeric factorization: ",
                             [&A, &prec]() { prec->numeric_factorize(&A); });
    auto solver = std::make_unique<mkl_wrapper::mkl_fgmres_solver>(&A, prec);
    // prec->print();
    solver->set_max_iters(1e5);
    solver->set_rel_tol(1e-8);
    solver->set_restart_steps(50);
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
    solver->set_restart_steps(50);
    return std::move(solver);
  };
  utils::singleton<mkl_wrapper::solver_factory>::instance().reg("gmres+ilu0",
                                                                gmres_ilu0);

  create_method gmres_iluk = [](mkl_wrapper::mkl_sparse_mat &A) {
    auto prec = std::make_shared<mkl_wrapper::incomplete_lu_k>();
    prec->set_level(0);

    utils::Elapse<>::execute("incomplete_lu_k symbolic factorization: ",
                             [&A, &prec]() { prec->symbolic_factorize(&A); });
    utils::Elapse<>::execute("incomplete_lu_k numeric factorization: ",
                             [&A, &prec]() { prec->numeric_factorize(&A); });
    auto solver = std::make_unique<mkl_wrapper::mkl_fgmres_solver>(&A, prec);
    // prec->print();
    solver->set_max_iters(1e5);
    solver->set_rel_tol(1e-8);
    solver->set_restart_steps(50);
    return std::move(solver);
  };
  utils::singleton<mkl_wrapper::solver_factory>::instance().reg("gmres+iluk",
                                                                gmres_iluk);

  create_method cg_incomplete_cholesky_k = [](mkl_wrapper::mkl_sparse_mat &A) {
    auto prec = std::make_shared<mkl_wrapper::incomplete_cholesky_k>();
    prec->set_level(3);
    prec->shift(true);
    // std::abort();
    utils::Elapse<>::execute("incomplete_cholesky_k symbolic factorization: ",
                             [&A, &prec]() { prec->symbolic_factorize(&A); });

    utils::Elapse<>::execute("incomplete_cholesky_k numeric factorization: ",
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
      "cg+incomplete_cholesky_k", cg_incomplete_cholesky_k);

  create_method cg = [](mkl_wrapper::mkl_sparse_mat &A) {
    auto solver = std::make_unique<mkl_wrapper::mkl_pcg_solver>(&A);
    // prec->print();
    solver->set_max_iters(1e5);
    solver->set_rel_tol(1e-8);
    return std::move(solver);
  };
  utils::singleton<mkl_wrapper::solver_factory>::instance().reg("cg", cg);
}

int main() {
  mkl_set_num_threads(8);
  std::ifstream f("/home/dimiao/matrix_lib/Fault_639.mtx");
  f.clear();
  f.seekg(0, std::ios::beg);
  std::vector<MKL_INT> csr_rows, csr_cols;
  std::vector<double> csr_vals;
  utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
  mkl_wrapper::mkl_sparse_mat mat(csr_rows.size() - 1, csr_rows.size() - 1,
                                  csr_rows, csr_cols, csr_vals);

  // std::string k_mat("../../data/shared/K.bin");
  // std::vector<MKL_INT> k_csr_rows, k_csr_cols;
  // std::vector<double> k_csr_vals;
  // std::cout << "read K\n";
  // utils::ReadFromBinaryCSR(k_mat, k_csr_rows, k_csr_cols, k_csr_vals,
  //                          SPARSE_INDEX_BASE_ONE);
  // std::shared_ptr<MKL_INT[]> k_csr_rows_ptr(k_csr_rows.data(),
  //                                           [](MKL_INT[]) {});
  // std::shared_ptr<MKL_INT[]> k_csr_cols_ptr(k_csr_cols.data(),
  //                                           [](MKL_INT[]) {});
  // std::shared_ptr<double[]> k_csr_vals_ptr(k_csr_vals.data(), [](double[]) {});
  // mkl_wrapper::mkl_sparse_mat mat(k_csr_rows.size() - 1, k_csr_rows.size() - 1,
  //                                 k_csr_rows_ptr, k_csr_cols_ptr,
  //                                 k_csr_vals_ptr, SPARSE_INDEX_BASE_ONE);
  mat.to_zero_based();
  const MKL_INT size = mat.rows();
  std::cout << "problem size: " << size << std::endl;

  auto inv_perm = reordering::SerialCM(&mat);
  std::cout << (utils::isPermutation(inv_perm, mat.mkl_base())) << std::endl;
  auto perm = utils::inversePermute(inv_perm, mat.mkl_base());
  std::cout<<"permute matrix\n";
  auto [ai, aj, av] = mkl_wrapper::permute(mat, inv_perm.data(), perm.data());
  mkl_wrapper::mkl_sparse_mat perm_mat(mat.rows(), mat.cols(), ai, aj, av);
  std::cout << "before permute bandwidth: " << mat.bandwidth()
            << " , after permute bandwidth: " << perm_mat.bandwidth()
            << std::endl;
  //   mkl_wrapper::mkl_sparse_mat_sym sym_k(k);
  //   k.clear();
  register_solvers();
  std::vector<double> rhs(size, 1.);
  std::vector<double> res(size, 0);

  auto sqrt_D = perm_mat.rowwiseSqrtNorm();
  perm_mat.DtAD(sqrt_D);
  for (int i = 0; i < rhs.size(); i++) {
    rhs[i] *= sqrt_D[i];
  }

  std::cout<<"creating solver\n";
  auto solver =
      utils::singleton<mkl_wrapper::solver_factory>::instance().create(
          "cg+incomplete_cholesky_k", perm_mat);
  std::cout << "cg+incomplete_cholesky_k: \n";
  res = std::vector<double>(size, 0);
  solver->set_print_level(1);
  solver->solve(rhs.data(), res.data());

  // solver = utils::singleton<mkl_wrapper::solver_factory>::instance().create(
  //     "gmres+iluk", mat);
  // std::cout << "gmres+iluk: \n";
  // res = std::vector<double>(size, 0);
  // solver->set_print_level(1);
  // solver->solve(rhs.data(), res.data());

  return 0;
}