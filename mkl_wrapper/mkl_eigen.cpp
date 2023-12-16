#include "mkl_eigen.h"
#include "mkl_sparse_mat.h"
#include <iostream>
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
  _pm[6] = eigenVectors == nullptr ? 0 : 1;

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
} // namespace mkl_wrapper