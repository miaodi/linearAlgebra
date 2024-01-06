#include "mkl_eigen.h"
#include "../utils/utils.h"
#include "mkl_solver.h"
#include "mkl_sparse_mat.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <mkl.h>

namespace mkl_wrapper {
mkl_eigen_sparse_d_gv::mkl_eigen_sparse_d_gv(mkl_sparse_mat *A,
                                             mkl_sparse_mat *B)
    : mkl_eigen(A, B) {}
bool mkl_eigen_sparse_d_gv::eigen_solve(double *eigenValues,
                                        double *eigenVectors) {

  _pm[1] = _tol;
  _pm[2] = 0; // select Krylov-Schur method
  _pm[3] = _ncv;
  _pm[4] = _maxiter;
  _pm[6] = 0;

  MKL_INT info;
  std::vector<double> res(_num_req, 0);

  /* Step 1. Call mkl_sparse_ee_init to define default input values */
  mkl_sparse_ee_init(_pm);
  if (_B) {
    _A->to_one_based();
    _B->to_one_based();
    info = mkl_sparse_d_gv(&_which, _pm, _A->mkl_handler(), _A->mkl_descr(),
                           _B->mkl_handler(), _B->mkl_descr(), _num_req,
                           &_num_found, eigenValues, eigenVectors, res.data());
    _A->to_zero_based();
    _B->to_zero_based();
  } else {
    _A->to_one_based();
    info = mkl_sparse_d_ev(&_which, _pm, _A->mkl_handler(), _A->mkl_descr(),
                           _num_req, &_num_found, eigenValues, eigenVectors,
                           res.data());
    _A->to_zero_based();
  }
  if (info != 0) {
    std::cout << "mkl_sparse_d_gv output info: " << info << std::endl;
    return false;
  }
  return true;
}

power_sparse_gv::power_sparse_gv(mkl_sparse_mat *A, mkl_sparse_mat *B)
    : mkl_eigen(A, B) {}

void power_sparse_gv::set_num_eigen(const MKL_INT num) { _num_req = 1; }

bool power_sparse_gv::eigen_solve(double *eigenValues, double *eigenVectors) {
  MKL_INT size = _A->rows();
  mkl_sparse_mat *mat{nullptr};
  mkl_direct_solver *solver{nullptr};
  if (_which == 'L') {
    mat = _A;
    if (_B) {
      solver = new mkl_direct_solver(_B);
      solver->factorize();
    }
  } else {
    solver = new mkl_direct_solver(_A);
    solver->factorize();
    if (_B) {
      mat = _B;
    }
  }
  std::vector<double> cur(size), prev(size), tmp(size);
  double norm2;
  for (int i = 0; i < size; i++) {
    cur[i] = utils::random<double>(-1.0, 1.0);
  }
  norm2 = cblas_dnrm2(size, cur.data(), 1);
  cblas_dscal(size, 1. / norm2, cur.data(), 1);
  int iter;
  for (iter = 0; iter < _maxiter; iter++) {
    std::swap(cur, prev);
    if (mat) {
      mat->mult_vec(prev.data(), tmp.data());
    }
    if (solver) {
      if (mat)
        solver->solve(tmp.data(), cur.data());
      else
        solver->solve(prev.data(), cur.data());
    } else if (mat) {
      std::swap(tmp, cur);
    }

    norm2 = cblas_dnrm2(size, cur.data(), 1);
    cblas_dscal(size, 1. / norm2, cur.data(), 1);
    double angle = std::acos(cblas_ddot(size, cur.data(), 1, prev.data(), 1));
    if (angle < _tol)
      break;
  }
  std::cout << "niter: " << iter + 1 << std::endl;
  eigenValues[0] = _which == 'L' ? norm2 : 1. / norm2;
  if (solver)
    delete solver;
  return true;
}
} // namespace mkl_wrapper