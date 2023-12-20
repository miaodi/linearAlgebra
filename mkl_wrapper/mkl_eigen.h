#pragma once

#include <memory>
#include <mkl.h>

namespace mkl_wrapper {
class mkl_sparse_mat;

class mkl_eigen {
public:
  mkl_eigen(mkl_sparse_mat *A, mkl_sparse_mat *B) : _A(A), _B(B) {}
  virtual bool eigen_solve(double *eigenValues, double *eigenVectors) = 0;
  void set_num_eigen(const MKL_INT num) { _num_req = num; }

protected:
  mkl_sparse_mat *_A;
  mkl_sparse_mat *_B;

  MKL_INT _num_req{1};
  MKL_INT _num_found{0};
};

/*
 eigenvalue solver from mkl
    standard, Ax = λx
        A complex Hermitian
        A real symmetric
    generalized, Ax = λBx
        A complex Hermitian, B Hermitian positive definite (hpd)
        A real symmetric and B real symmetric positive definite (spd)
*/
class mkl_eigen_sparse_d_gv : public mkl_eigen {
public:
  mkl_eigen_sparse_d_gv(mkl_sparse_mat *A, mkl_sparse_mat *B = nullptr);

  virtual bool eigen_solve(double *eigenValues, double *eigenVectors) override;
  void set_tol(const double tol) { _tol = tol; }
  void which(const char s) { _which = s; }
  void set_ncv(const MKL_INT ncv) { _ncv = ncv; }

protected:
  MKL_INT _pm[128] = {0};
  double _tol{6}; // 1e-{_tol}+1
  MKL_INT _ncv{10};
  MKL_INT _maxiter{10000}; // max num of iterations
  char _which{'S'};
};
} // namespace mkl_wrapper