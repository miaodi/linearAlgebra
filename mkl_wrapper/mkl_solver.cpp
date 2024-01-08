#include "mkl_solver.h"
#include "mkl_sparse_mat.h"
#include <cassert>
#include <cstdio>
#include <iostream>
#include <mkl.h>
#include <vector>

#define IFORMAT "%lli"
namespace mkl_wrapper {

bool mkl_direct_solver::factorize() {
  // init
  if (_factorized)
    return true;

  /* -------------------------------------------------------------------- */
  /* .. Setup Pardiso control parameters. */
  /* -------------------------------------------------------------------- */

  _iparm[0] = 0; // iparm[1] - iparm[63] are filled with default values.

  _maxfct = 1; // Maximum number of factors with identical sparsity structure
               // that must be kept in memory at the same time. In most
               // applications this value is equal to 1

  _mnum = 1; // Indicates the actual matrix for the solution phase. With this
             // scalar you can define which matrix to factorize. The value must
             // be: 1 ≤mnum≤maxfct.

  _msglvl = 0; // the solver prints statistical information to the screen.
  if (_A->mkl_descr().type == SPARSE_MATRIX_TYPE_SYMMETRIC) {
    _mtype = _A->positive_definite() ? 2 : -2;
  } else {
    _mtype = 11;
  }
  pardisoinit(_pt, &_mtype, _iparm);

  /* -------------------------------------------------------------------- */
  /* .. Reordering and Symbolic Factorization. This step also allocates */
  /* all memory that is necessary for the factorization. */
  /* -------------------------------------------------------------------- */

  MKL_INT phase = 12; // Analysis, numerical factorization
  MKL_INT error = 0;
  MKL_INT nrhs = 1; /* Number of right hand sides. */
  MKL_INT n = _A->rows();

  _iparm[34] = _A->mkl_base() == SPARSE_INDEX_BASE_ZERO ? 1 : 0;
  pardiso(_pt, &_maxfct, &_mnum, &_mtype, &phase, &n, _A->get_av().get(),
          _A->get_ai().get(), _A->get_aj().get(), NULL, &nrhs, _iparm, &_msglvl,
          NULL, NULL, &error);

  if (error) {
    fprintf(stderr, "\nERROR during factorization: ");
    std::cout << "error: " << error << std::endl;
    return false;
  }
  _factorized = true;
  return true;
}

bool mkl_direct_solver::solve(double const *const b, double *const x) {

  MKL_INT phase = 33;
  MKL_INT error = 0;

  _iparm[7] = _max_iter_ref; /* Maximum number of iterative refinement steps */

  MKL_INT nrhs = 1; /* Number of right hand sides. */
  MKL_INT n = _A->rows();

  pardiso(_pt, &_maxfct, &_mnum, &_mtype, &phase, &n, _A->get_av().get(),
          _A->get_ai().get(), _A->get_aj().get(), NULL, &nrhs, _iparm, &_msglvl,
          const_cast<double *const>(b), x, &error);

  if (error) {
    fprintf(stderr, "\nERROR during solution: ");
    std::cout << "error: " << error << std::endl;
    return false;
  }
  return true;
}

bool mkl_direct_solver::forward_substitution(double const *const b,
                                             double *const x) {

  MKL_INT phase = 331;
  MKL_INT error = 0;

  _iparm[7] = _max_iter_ref; /* Maximum number of iterative refinement steps */

  MKL_INT nrhs = 1; /* Number of right hand sides. */
  MKL_INT n = _A->rows();

  pardiso(_pt, &_maxfct, &_mnum, &_mtype, &phase, &n, _A->get_av().get(),
          _A->get_ai().get(), _A->get_aj().get(), NULL, &nrhs, _iparm, &_msglvl,
          const_cast<double *const>(b), x, &error);

  if (error) {
    fprintf(stderr, "\nERROR during solution: ");
    std::cout << "error: " << error << std::endl;
    return false;
  }
  return true;
}

bool mkl_direct_solver::backward_substitution(double const *const b,
                                              double *const x) {

  MKL_INT phase = 333;
  MKL_INT error = 0;

  _iparm[7] = _max_iter_ref; /* Maximum number of iterative refinement steps */

  MKL_INT nrhs = 1; /* Number of right hand sides. */
  MKL_INT n = _A->rows();

  pardiso(_pt, &_maxfct, &_mnum, &_mtype, &phase, &n, _A->get_av().get(),
          _A->get_ai().get(), _A->get_aj().get(), NULL, &nrhs, _iparm, &_msglvl,
          const_cast<double *const>(b), x, &error);

  if (error) {
    fprintf(stderr, "\nERROR during solution: ");
    std::cout << "error: " << error << std::endl;
    return false;
  }
  return true;
}

mkl_direct_solver::~mkl_direct_solver() {

  if (_factorized) {

    MKL_INT phase = -1;
    MKL_INT error = 0;
    MKL_INT nrhs = 1; /* Number of right hand sides. */
    MKL_INT n = _A->rows();

    pardiso(_pt, &_maxfct, &_mnum, &_mtype, &phase, &n, _A->get_av().get(),
            _A->get_ai().get(), _A->get_aj().get(), NULL, &nrhs, _iparm,
            &_msglvl, NULL, NULL, &error);
  }
}

bool mkl_pcg_solver::solve(double const *const b, double *const x) {
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

bool mkl_fgmres_solver::solve(double const *const b, double *const x) {
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

  // initialize parameters
  dfgmres_init(&n, x, b, &rci_request, ipar, dpar, ptmp);
  if (rci_request != 0)
    return false;
  double residual0 = cblas_dnrm2(n, b, 1);
  double residual = 0.;

  // set the desired parameters:
  ipar[4] = _maxiter; // max nr of iterations
  ipar[7] = 1;        // do the stopping test for maximal number of iterations
  // ipar[8] = 1;        // do residual stopping test
  // ipar[9] = 0;             // do not request for the user defined stopping

  ipar[9] = 1;           // use user defined stopping test
  ipar[10] = _P ? 1 : 0; // preconditioning
  ipar[14] = _num_restart != 0 ? _num_restart
                               : _maxiter; // number of non-restarted iterations
  dpar[0] = _rel_tol;                      // set the relative tolerance
  dpar[1] = _abs_tol;                      // set the absolute tolerance

  // check the consistency of the newly set parameters
  dfgmres_check(&n, x, b, &rci_request, ipar, dpar, ptmp);
  if (rci_request != 0 && rci_request != -1001)
    return false;

  // loop until converged
  bool bsuccess = false;
  bool bdone = false;
  do {
    // compute the solution by RCI
    dfgmres(&n, x, const_cast<double *const>(b), &rci_request, ipar, dpar,
            ptmp);
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
      // fprintf(stderr, "%3d = %lg (%lg), %lg (%lg)\n", ipar[3], dpar[4],
      // dpar[3],
      //         dpar[6], dpar[7]);
      // }
      break;
    }
    case 2: // then do the user-defined stopping test
    {
      residual = dpar[4];
      fprintf(stderr, "%3d = %lg/%lg=%lg\n", ipar[3], residual, residual0,
              residual / residual0);
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
  dfgmres_get(&n, x, const_cast<double *const>(b), &rci_request, ipar, dpar,
              ptmp, &niter);

  // if (_print_level > 0) {
  // fprintf(stderr, "%3d = %lg (%lg), %lg (%lg)\n", ipar[3], dpar[4], dpar[3],
  //         dpar[6], dpar[7]);
  // }

  return (m_fail_max_iters ? bsuccess : true);
}

solver_factory::solver_factory() {
  _methods["direct"] = [](mkl_sparse_mat &m) {
    auto solver = std::make_unique<mkl_direct_solver>(&m);
    solver->factorize();
    return std::move(solver);
  };

  _methods["gmres"] = [](mkl_sparse_mat &m) {
    return std::make_unique<mkl_fgmres_solver>(&m);
  };
}

bool solver_factory::reg(const std::string &name, create_method func) {
  if (auto it = _methods.find(name); it == _methods.end()) {
    _methods[name] = func;
    return true;
  }
  return false;
}

solver_factory::solver_ptr solver_factory::create(const std::string &name,
                                                  mkl_sparse_mat &m) {
  if (auto it = _methods.find(name); it != _methods.end())
    return it->second(m);
  return nullptr;
}
} // namespace mkl_wrapper