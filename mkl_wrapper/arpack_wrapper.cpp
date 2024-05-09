#include "arpack_wrapper.h"

#ifdef USE_ARPACK_LIB

#include "../utils/utils.h"
#include "mkl_solver.h"
#include <arpack.h>
#include <arpackdef.h>
#include <iostream>

namespace mkl_wrapper {
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

std::function<bool(double *const b, double *const x)>
arpack_gv::op(const int mode) {
  // https://help.scilab.org/docs/6.1.0/en_US/dnaupd.html
  switch (mode) {
  case 1: {
    return [this](double *const b, double *const x) -> bool {
      return this->_A->mult_vec(b, x);
    };
  }
  case 2: {
    _solver =
        utils::singleton<solver_factory>::instance().create("direct", *_B);
    tmp.resize(_A->rows());
    return [this](double *const b, double *const x) -> bool {
      this->_A->mult_vec(b, this->tmp.data());

      std::copy(tmp.begin(), tmp.end(), b);
      return this->_solver->solve(this->tmp.data(), x);
    };
  }
  case 3: {
    _op = std::move(mkl_sparse_sum(*_A, *_B, -_shift));
    _solver =
        utils::singleton<solver_factory>::instance().create("direct", _op);
    tmp.resize(_A->rows());
    return [this](double *const b, double *const x) -> bool {
      this->_B->mult_vec(b, this->tmp.data());
      return this->_solver->solve(this->tmp.data(), x);
    };
  }
  default:
    std::cerr << "incorrect mode type" << std::endl;
    return [](double *const b, double *const x) -> bool { return false; };
  }
}

bool arpack_gv::eigen_solve(double *eigenValues, double *eigenVectors) {

  const MKL_INT size = _A->rows();

  // arpack params
  a_int ido{0};
  a_int n = size;
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
  auto op_func = this->op(mode);
  do {
    dsaupd_c(&ido, &bMat, n, _which.c_str(), _nev, _tol, resid.data(), _ncv,
             v.data(), ldv, iparam.data(), ipntr.data(), workd.data(),
             workl.data(), lworkl, &info);

    if (info == 1)
      std::cerr << "Error: dsaupd: maximum number of iterations taken. "
                   "Increase --maxIt..."
                << std::endl;
    if (info == 3)
      std::cerr << "Error: dsaupd: no shifts could be applied. Increase "
                   "--nbCV..."
                << std::endl;
    if (info == -9)
      std::cerr << "Error: dsaupd: starting vector is zero. Retry: play "
                   "with shift..."
                << std::endl;
    if (info < 0) {
      std::cerr << "Error: dsaupd with info " << info << ", nbIt " << iparam[2]
                << std::endl;
      return false;
    }

    a_int x_idx = ipntr[0] - 1; // 0-based (Fortran is 1-based).
    a_int y_idx = ipntr[1] - 1; // 0-based (Fortran is 1-based).

    double *X = workd.data() + x_idx; // Arpack provides X.
    double *Y = workd.data() + y_idx; // Arpack provides Y.

    // std::cout << "ido: " << ido << std::endl;
    if (ido == -1) {
      op_func(X, Y);
    } else if (ido == 1) {
      op_func(X, Y);
    } else if (ido == 2) {
      if (iparam[6] == 1)
        std::copy(X, X + n, Y);
      else if (iparam[6] == 2)
        this->_B->mult_vec(X, Y); // Y = B * X.
      else if (iparam[6] == 3)
        this->_B->mult_vec(X, Y); // Y = B * X.
    } else if (ido != 99) {
      std::cerr << "Error: unexpected ido " << ido << std::endl;
      return false;
    }

  } while (ido != 99);

  // Get arpack results (computed eigen values and vectors).

  // Extract eigen pairs
  std::cout << "nconv: " << iparam[4] << std::endl;
  std::cout << "niter: " << iparam[2] << std::endl;

  a_int rvec = 0;
  char howmny = 'A';
  std::vector<a_int> select(_ncv, 1);

  dseupd_c(rvec, &howmny, select.data(), eigenValues, eigenVectors, n * _nev,
           _shift, &bMat, n, _which.c_str(), _nev, _tol, resid.data(), _ncv,
           v.data(), v.size(), iparam.data(), ipntr.data(), workd.data(),
           workl.data(), lworkl, &info);
  return true;
}
} // namespace mkl_wrapper

#endif
