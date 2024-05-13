
#include "../mkl_wrapper/mkl_solver.h"
#include "../mkl_wrapper/mkl_sparse_mat.h"
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

  create_method gmres_ilut2 = [](mkl_wrapper::mkl_sparse_mat &A) {
    auto prec = std::make_shared<mkl_wrapper::mkl_ilut>(&A);
    prec->set_tau(1e-8);
    prec->set_max_fill(std::min((MKL_INT)(A.avg_nz() * 4), A.cols()));
    prec->factorize();
    auto solver = std::make_unique<mkl_wrapper::mkl_fgmres_solver>(&A, prec);
    // prec->print();
    solver->set_max_iters(1e5);
    solver->set_rel_tol(1e-8);
    solver->set_restart_steps(50);
    return std::move(solver);
  };
  utils::singleton<mkl_wrapper::solver_factory>::instance().reg("gmres+ilut2",
                                                                gmres_ilut2);

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

int main() {

  mkl_set_num_threads(10);
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
  mkl_wrapper::mkl_sparse_mat k(size, size, k_csr_rows_ptr, k_csr_cols_ptr,
                                k_csr_vals_ptr, SPARSE_INDEX_BASE_ONE);

  std::ofstream myfile;
  myfile.open("example.gnuplot");
  k.print_gnuplot(myfile);
  myfile.close();
  //   mkl_wrapper::mkl_sparse_mat_sym sym_k(k);
  //   k.clear();
  register_solvers();
  auto solver =
      utils::singleton<mkl_wrapper::solver_factory>::instance().create("cg", k);

  std::vector<double> rhs(size, 1.);
  std::vector<double> res(size);
  solver->solve(rhs.data(), res.data());
  solver = utils::singleton<mkl_wrapper::solver_factory>::instance().create(
      "cg+ic0", k);

  solver->solve(rhs.data(), res.data());
  solver = utils::singleton<mkl_wrapper::solver_factory>::instance().create(
      "gmres+ilut", k);

  solver->solve(rhs.data(), res.data());
  solver = utils::singleton<mkl_wrapper::solver_factory>::instance().create(
      "gmres", k);

  solver->solve(rhs.data(), res.data());
  return 0;
}