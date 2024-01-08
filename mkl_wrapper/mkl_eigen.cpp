#include "mkl_eigen.h"
#include "../utils/utils.h"
#include "mkl_solver.h"
#include <arpack.h>
#include <cmath>
#include <iostream>
#include <mkl.h>
#include <vector>

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

arpack_gv::arpack_gv(mkl_sparse_mat *A, mkl_sparse_mat *B) : mkl_eigen(A, B) {}

int arpack_gv::mode() const {
  bool shift = _shift != 0.;
  if (_B == nullptr) {
    return 1;
  } else {
    if (shift) {
      return 3;
    }
    return 2;
  }
}

std::function<bool(double const *const b, double *const x)>
arpack_gv::op(const int mode) {
  switch (mode) {
  case 1: {
    return [this](double const *const b, double *const x) -> bool {
      return this->_A->mult_vec(b, x);
    };
  }
  case 2: {
    _solver =
        utils::singleton<solver_factory>::instance().create("direct", *_A);
    return [this](double const *const b, double *const x) -> bool {
      this->_B->mult_vec(b, tmp.data());
      return this->_solver->solve(tmp.data(), x);
    };
  }
  case 3: {
    _op = std::move(mkl_sparse_sum(*_A, *_B, -_shift));
    _solver =
        utils::singleton<solver_factory>::instance().create("direct", _op);
    tmp.resize(_A->rows());
    return [this](double const *const b, double *const x) -> bool {
      this->_B->mult_vec(b, tmp.data());
      return this->_solver->solve(tmp.data(), x);
    };
  }
  default:
    return [](double const *const b, double *const x) -> bool { return false; };
  }
}

bool arpack_gv::eigen_solve(double *eigenValues, double *eigenVectors) {

  const MKL_INT size = _A->rows();

  // arpack params
  a_int ido{0};
  a_int n = size;
  char which[2] = {'L', 'A'};
  char bMat = _B == nullptr ? 'I' : 'G';

  std::vector<double> resid(n);
  a_int ldv = n;
  std::vector<double> v(ldv * _ncv);
  std::vector<a_int> iparam(11, 0);
  iparam[0] =
      1; // exact shifts with respect to the reduced tridiagonal matrix T.

  /*iparam[1] No longer referenced.*/
  int max_it = _maxiter;
  int mode = this->mode();

  iparam[2] = max_it;
  iparam[3] = 1; // NB: blocksize to be used in the recurrence. The code
                 // currently works only for N = 1.
  iparam[4] = 0; // NCONV: number of "converged" Ritz values.
  /*iparam[5] IUPD No longer referenced. Implicit restarting is ALWAYS used. */
  iparam[6] = mode; // MODE On INPUT determines what type of eigenproblem is
                    // being solved. Must be 1,2,3,4,5;

  /*rest iparam are output*/

  std::vector<a_int> ipntr(11, 0);

  std::vector<double> workd(
      3 * n, 0.); // Distributed array to be used in the basic Arnoldi iteration
  // for reverse communication. The user should not use WORKD as
  // temporary workspace during the iteration.
  a_int lworkl = _ncv * _ncv + 8 * _ncv;
  std::vector<double> workl(lworkl); // Private (replicated) array on each PE
                                     // or array allocated on the front end.

  a_int info = 0;

  /* prepare op*/

  return true;
}
} // namespace mkl_wrapper