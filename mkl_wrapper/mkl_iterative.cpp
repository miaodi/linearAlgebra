#include "mkl_iterative.h"
#include "mkl_sparse_mat.h"
#include <cassert>
#include <cstdio>
#include <iostream>
#include <vector>

namespace mkl_wrapper {

bool mkl_pcg_solver::solve(double *b, double *x) {
  // make sure we have a matrix
  if (_A == 0)
    return false;

  // get number of equations
  MKL_INT n = _A->rows();

  // zero solution vector
#pragma omp parallel for
  for (int i = 0; i < n; ++i)
    x[i] = 0.0;

  // output parameters
  MKL_INT rci_request;
  MKL_INT ipar[128] = {0};
  double dpar[128] = {0.};
  std::vector<double> tmp(n * 4);
  std::vector<double> temp(n);
  auto ptmp = tmp.data();

  // initialize parameters
  dcg_init(&n, x, b, &rci_request, ipar, dpar, ptmp);
  if (rci_request != 0)
    return false;
  double residual0 = cblas_dnrm2(n, b, 1);
  double residual = 0.;

  // set the desired parameters:
  ipar[4] = _maxiter; // max nr of iterations
  // ipar[8] = 1;         // do residual stopping test
  // ipar[9] = 0;             // do not request for the user defined stopping

  ipar[9] = 1;             // use user defined stopping test
  ipar[10] = (_P ? 1 : 0); // preconditioning
  // dpar[0] = m_tol;         // set the relative tolerance

  // check the consistency of the newly set parameters
  dcg_check(&n, x, b, &rci_request, ipar, dpar, ptmp);
  if (rci_request != 0)
    return false;

  // loop until converged
  bool bsuccess = false;
  bool bdone = false;
  do {
    // compute the solution by RCI
    dcg(&n, x, b, &rci_request, ipar, dpar, ptmp);

    switch (rci_request) {
    case 0: // solution converged!
    {
      bsuccess = true;
      bdone = true;
      break;
    }
    case 1: // compute vector A*tmp[0] and store in tmp[n]
    {
      _A->mult_vec(ptmp, ptmp + n);

      // if (_print_level == 1) {
      //   fprintf(stderr, "%3d = %lg (%lg), %lg (%lg)\n", ipar[3], dpar[4],
      //           dpar[3], dpar[6], dpar[7]);
      // }
      break;
    }
    case 2: // then do the user-defined stopping test
    {
      _A->mult_vec(x, temp.data());
      cblas_daxpy(n, -1., b, 1, temp.data(), 1);
      residual = cblas_dnrm2(n, temp.data(), 1);

      fprintf(stderr, "%3d = %lg/%lg=%lg, %lg (%lg)\n", ipar[3], residual,
              residual0, residual / residual0, dpar[3], dpar[6], dpar[7]);
      if (residual / residual0 < _rel_tol || residual < _abs_tol) {
        bsuccess = true;
        bdone = true;
      }
      break;
    }
    case 3: {
      assert(_P);
      // apply preconditioner
      _P->solve(ptmp + n * 2, ptmp + n * 3);
      break;
    }
    default: {
      bsuccess = false;
      bdone = true;
      break;
    }
    }
  } while (!bdone);

  // get convergence information
  MKL_INT niter;
  dcg_get(&n, x, b, &rci_request, ipar, dpar, ptmp, &niter);

  // if (_print_level > 0) {
  //   fprintf(stderr, "%3d = %lg (%lg), %lg (%lg)\n", ipar[3], dpar[4],
  //   dpar[3],
  //           dpar[6], dpar[7]);
  // }

  return (m_fail_max_iters ? bsuccess : true);
}

bool mkl_fgmres_solver::solve(double *b, double *x) {
  // make sure we have a matrix
  if (_A == 0)
    return false;

  // get number of equations
  MKL_INT n = _A->rows();

  // zero solution vector
#pragma omp parallel for
  for (int i = 0; i < n; ++i)
    x[i] = 0.0;

  MKL_INT niter;

  // output parameters
  MKL_INT rci_request;
  MKL_INT ipar[128] = {0};
  double dpar[128] = {0.};
  int m = _maxiter;
  if (_num_restart > 0)
    m = _num_restart;
  std::vector<double> tmp(n * (2 * m + 1) + (m * (m + 9)) / 2 + 1, 0.);
  auto ptmp = tmp.data();
  std::vector<double> residual_vec(n, 0.);
  std::vector<double> rhs(n);
  cblas_dcopy(n, b, 1, rhs.data(), 1);

  // set the desired parameters:
  ipar[4] = _maxiter; // max nr of iterations
  ipar[7] = 1;        // do the stopping test for maximal number of iterations
  ipar[8] = 1;         // do residual stopping test
  // ipar[9] = 0;             // do not request for the user defined stopping

  ipar[9] = 0;           // use user defined stopping test
  ipar[10] = _P ? 1 : 0; // preconditioning
  std::cout << "ipar[10]: " << ipar[10] << std::endl;
  ipar[14] = _num_restart != 0 ? _num_restart
                               : _maxiter; // number of non-restarted iterations
  dpar[0] = _rel_tol;         // set the relative tolerance
  dpar[1] = _abs_tol;         // set the absolute tolerance


  // initialize parameters
  dfgmres_init(&n, x, rhs.data(), &rci_request, ipar, dpar, ptmp);
  if (rci_request != 0)
    return false;
  double residual0 = cblas_dnrm2(n, b, 1);
  double residual = 0.;

  // check the consistency of the newly set parameters
  dfgmres_check(&n, x, rhs.data(), &rci_request, ipar, dpar, ptmp);
  if (rci_request != 0)
    return false;

  // loop until converged
  bool bsuccess = false;
  bool bdone = false;
  do {
    // compute the solution by RCI
    dfgmres(&n, x, rhs.data(), &rci_request, ipar, dpar, ptmp);
    std::cout << "rci_request: " << rci_request << std::endl;
    switch (rci_request) {
    case 0: // solution converged!
    {
      bsuccess = true;
      bdone = true;
      break;
    }
    case 1: // compute vector A*tmp[0] and store in tmp[n]
    {
      if (_R) {
        // first apply the right preconditioner
        _R->solve(&ptmp[ipar[21] - 1], &residual_vec[0]);
        // then multiply with matrix
        _A->mult_vec(&residual_vec[0], &ptmp[ipar[22] - 1]);
      } else {
        _A->mult_vec(&ptmp[ipar[21] - 1], &ptmp[ipar[22] - 1]);
      }

      // if (_print_level == 1) {
      fprintf(stderr, "%3d = %lg (%lg), %lg (%lg)\n", ipar[3], dpar[4], dpar[3],
              dpar[6], dpar[7]);
      // }
      break;
    }
    case 2: // then do the user-defined stopping test
    {
      /* Request to the dfgmres_get routine to put the solution into b[N] via
      ipar[12]
      --------------------------------------------------------------------------------
      WARNING: beware that the call to dfgmres_get routine with ipar[12]=0 at
      this stage may destroy the convergence of the FGMRES method, therefore,
      only advanced users should exploit this option with care */
      ipar[12] = 1;
      /* Get the current FGMRES solution in the vector b[N] */
      dfgmres_get(&n, x, b, &rci_request, ipar, dpar, ptmp, &niter);
      /* Compute the current true residual via  Intel oneMKL (Sparse) BLAS
       * routines */
      _A->mult_vec(b, residual_vec.data());
      cblas_daxpy(n, -1., rhs.data(), 1, residual_vec.data(), 1);
      residual = cblas_dnrm2(n, residual_vec.data(), 1);

      fprintf(stderr, "%3d = %lg/%lg=%lg, %lg (%lg)\n", ipar[3], residual,
              residual0, residual / residual0, dpar[3], dpar[6], dpar[7]);
      if (residual / residual0 < _rel_tol || residual < _abs_tol) {
        bsuccess = true;
        bdone = true;
      }
      break;
    }
    case 3: {
      assert(_P);
      // apply preconditioner
      _P->solve(&ptmp[ipar[21] - 1], &ptmp[ipar[22] - 1]);
      break;
    }
    case 4: {
      /*---------------------------------------------------------------------------
       * If RCI_request=4, then check if the norm of the next generated vector
       *is not zero up to rounding and computational errors. The norm is
       *contained in dpar[6] parameter
       *---------------------------------------------------------------------------*/
      // std::cout<<"dpar[6]: "<<dpar[6]<<std::endl;
      if (dpar[6] < 1.0E-13) {
        bsuccess = true;
        bdone = true;
      }
      break;
    }
    default: {
      bsuccess = false;
      bdone = true;
      break;
    }
    }
  } while (!bdone);

  // get convergence information
  ipar[12] = 0;
  dfgmres_get(&n, x, rhs.data(), &rci_request, ipar, dpar, ptmp, &niter);

  // if (_print_level > 0) {
    fprintf(stderr, "%3d = %lg (%lg), %lg (%lg)\n", ipar[3], dpar[4],
    dpar[3],
            dpar[6], dpar[7]);
  // }

  return (m_fail_max_iters ? bsuccess : true);
}
} // namespace mkl_wrapper