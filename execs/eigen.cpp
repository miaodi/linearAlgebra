
#include "../mkl_wrapper/mkl_eigen.h"
#include "../mkl_wrapper/mkl_solver.h"
#include "../mkl_wrapper/mkl_sparse_mat.h"
#include "../utils/timer.h"
#include "../utils/utils.h"
#include <Eigen/Sparse>
#include <algorithm>
#include <fast_matrix_market/app/Eigen.hpp>
#include <fstream>
#include <ios>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <vector>

#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/SymEigsSolver.h>

using SpMat = typename Eigen::SparseMatrix<double, Eigen::RowMajor, MKL_INT>;
using SpMatMap = typename Eigen::Map<const SpMat>;
int main() {

  std::ifstream fm("../../data/eigenvalue/bcsstk36.mtx");
  std::ifstream fk("../../data/eigenvalue/bcsstk36.mtx");

  std::vector<MKL_INT> k_csr_rows, k_csr_cols;
  std::vector<double> k_csr_vals;
  utils::read_matrix_market_csr(fk, k_csr_rows, k_csr_cols, k_csr_vals);
  std::shared_ptr<MKL_INT[]> k_csr_rows_ptr(k_csr_rows.data(),
                                            [](MKL_INT[]) {});
  std::shared_ptr<MKL_INT[]> k_csr_cols_ptr(k_csr_cols.data(),
                                            [](MKL_INT[]) {});
  std::shared_ptr<double[]> k_csr_vals_ptr(k_csr_vals.data(), [](double[]) {});

  std::vector<MKL_INT> m_csr_rows, m_csr_cols;
  std::vector<double> m_csr_vals;
  utils::read_matrix_market_csr(fm, m_csr_rows, m_csr_cols, m_csr_vals);
  std::shared_ptr<MKL_INT[]> m_csr_rows_ptr(m_csr_rows.data(),
                                            [](MKL_INT[]) {});
  std::shared_ptr<MKL_INT[]> m_csr_cols_ptr(m_csr_cols.data(),
                                            [](MKL_INT[]) {});
  std::shared_ptr<double[]> m_csr_vals_ptr(m_csr_vals.data(), [](double[]) {});

  const MKL_INT size = m_csr_rows.size() - 1;

  //   std::cout << mat_csr << std::endl;

  mkl_wrapper::mkl_sparse_mat k(size, size, k_csr_rows_ptr, k_csr_cols_ptr,
                                k_csr_vals_ptr);
  mkl_wrapper::mkl_sparse_mat m(size, size, m_csr_rows_ptr, m_csr_cols_ptr,
                                m_csr_vals_ptr);

  std::cout << "m: " << k.rows() << " , n: " << k.cols() << std::endl;
  // {
  //   mkl_wrapper::mkl_eigen_sparse_d_gv gv(&k);
  //   gv.set_tol(3);
  //   gv.set_num_eigen(5);
  //   gv.set_ncv(20);
  //   gv.which('L');
  //   std::vector<double> eigenvalues(5, 0);
  //   std::vector<double> eigenvectors(5 * size, 0);
  //   utils::Elapse<>::execute(
  //       "mkl max eigen: ", [&gv, &eigenvalues, &eigenvectors]() {
  //         gv.eigen_solve(eigenvalues.data(), eigenvectors.data());
  //       });
  //   for (auto i : eigenvalues) {
  //     std::cout << i << std::endl;
  //   }
  //   gv.which('S');
  //   utils::Elapse<>::execute(
  //       "mkl min eigen: ", [&gv, &eigenvalues, &eigenvectors]() {
  //         gv.eigen_solve(eigenvalues.data(), eigenvectors.data());
  //       });
  //   for (auto i : eigenvalues) {
  //     std::cout << i << std::endl;
  //   }
  // }
  {
    mkl_wrapper::power_sparse_gv gv(&k);
    gv.set_tol(1e-5);
    gv.which('L');
    std::vector<double> eigenvalues(1, 0);
    std::vector<double> eigenvectors(1 * size, 0);
    utils::Elapse<>::execute(
        "mkl max eigen: ", [&gv, &eigenvalues, &eigenvectors]() {
          gv.eigen_solve(eigenvalues.data(), eigenvectors.data());
        });
    for (auto i : eigenvalues) {
      std::cout << i << std::endl;
    }
    gv.which('S');
    utils::Elapse<>::execute(
        "mkl min eigen: ", [&gv, &eigenvalues, &eigenvectors]() {
          gv.eigen_solve(eigenvalues.data(), eigenvectors.data());
        });
    for (auto i : eigenvalues) {
      std::cout << i << std::endl;
    }
  }

  SpMatMap mat_csr(size, size, k_csr_cols.size(), k_csr_rows.data(),
                   k_csr_cols.data(), k_csr_vals.data());

  {

    Spectra::SparseSymMatProd<double, Eigen::Lower | Eigen::Upper,
                              Eigen::RowMajor, MKL_INT>
        op(mat_csr);

    Spectra::SymEigsSolver<Spectra::SparseSymMatProd<
        double, Eigen::Lower | Eigen::Upper, Eigen::RowMajor, MKL_INT>>
        eigs(op, 1, 2);

    eigs.init();
    int nconv;
    nconv = utils::Elapse<>::execute("eigen max eigen: ", [&eigs]() {
      return eigs.compute(Spectra::SortRule::LargestMagn);
    });
    if (eigs.info() == Spectra::CompInfo::Successful) {
      Eigen::VectorXd evalues = eigs.eigenvalues();
      // Will get (10, 9, 8)
      std::cout << "Eigenvalues found:\n" << evalues << std::endl;
    }

    nconv = utils::Elapse<>::execute("eigen min eigen: ", [&eigs]() {
      return eigs.compute(Spectra::SortRule::SmallestMagn);
    });
    if (eigs.info() == Spectra::CompInfo::Successful) {
      Eigen::VectorXd evalues = eigs.eigenvalues();
      // Will get (10, 9, 8)
      std::cout << "Eigenvalues found:\n" << evalues << std::endl;
    }
  }
  {
    Spectra::SparseSymShiftSolve<double, Eigen::Lower | Eigen::Upper,
                                 Eigen::RowMajor, MKL_INT>
        op(mat_csr);

    // Construct eigen solver object, requesting the largest three eigenvalues

    Spectra::SymEigsShiftSolver<Spectra::SparseSymShiftSolve<
        double, Eigen::Lower | Eigen::Upper, Eigen::RowMajor, MKL_INT>>
        eigs(op, 1, 2, 0);

    // Initialize and compute
    eigs.init();
    int nconv = utils::Elapse<>::execute("eigen+shift min eigen: ", [&eigs]() {
      return eigs.compute(Spectra::SortRule::LargestMagn);
    });

    // Retrieve results
    Eigen::VectorXd evalues;
    if (eigs.info() == Spectra::CompInfo::Successful)
      evalues = eigs.eigenvalues();

    std::cout << "Eigenvalues found:\n" << evalues << std::endl;
  }

  return 0;
}