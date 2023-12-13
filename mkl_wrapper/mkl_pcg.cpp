#include "mkl_pcg.h"
#include "mkl_sparse_mat.h"
#include <vector>

namespace mkl_wrapper {

bool mkl_pcg_solver::solve(double const *const b, double *const x) {
  // make sure we have a matrix
  if (_A == 0)
    return false;

  // get number of equations
  MKL_INT n = _A->rows();

  // zero solution vector
  for (int i = 0; i < n; ++i)
    x[i] = 0.0;

  // output parameters
  MKL_INT rci_request;
  MKL_INT ipar[128];
  double dpar[128];
  std::vector<double> tmp(n * 4);
  double *ptmp = &tmp[0];

  // initialize parameters
  dcg_init(&n, x, b, &rci_request, ipar, dpar, ptmp);
  if (rci_request != 0)
    return false;

  // set the desired parameters:
  if (m_maxiter > 0)
    ipar[4] = m_maxiter;   // max nr of iterations
  ipar[8] = 1;             // do residual stopping test
  ipar[9] = 0;             // do not request for the user defined stopping test
  ipar[10] = (_P ? 1 : 0); // preconditioning
  dpar[0] = m_tol;         // set the relative tolerance

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
      bsuccess = true;
      bdone = true;
      break;
    case 1: // compute vector A*tmp[0] and store in tmp[n]
    {
      _A->mult_vec(ptmp, ptmp + n);

      // if (_Print_level == 1) {
      //   fprintf(stderr, "%3d = %lg (%lg), %lg (%lg)\n", ipar[3], dpar[4],
      //           dpar[3], dpar[6], dpar[7]);
      // }
    } break;
    case 3: {
      // assert(_P);
      _P->mult_vec(ptmp + n * 2, ptmp + n * 3);
    } break;
    default:
      bsuccess = false;
      bdone = true;
      break;
    }
  } while (!bdone);

  // get convergence information
  MKL_INT niter;
  dcg_get(&n, x, b, &rci_request, ipar, dpar, ptmp, &niter);

  // if (_Print_level > 0) {
  //   fprintf(stderr, "%3d = %lg (%lg), %lg (%lg)\n", ipar[3], dpar[4],
  //   dpar[3],
  //           dpar[6], dpar[7]);
  // }

  // release internal MKL buffers
  //	MKL_Free_Buffers();

  return (m_fail_max_iters ? bsuccess : true);
}
} // namespace mkl_wrapper