#pragma once
#include "mkl_sparse_mat.h"
#include <arpackdef.h>
#include <functional>
#include <memory>
#include <mkl_types.h>
namespace mkl_wrapper {
class mkl_sparse_mat;
class mkl_solver;

class mkl_eigen {
public:
  mkl_eigen(mkl_sparse_mat *A, mkl_sparse_mat *B) : _A(A), _B(B) {}
  virtual bool eigen_solve(double *eigenValues, double *eigenVectors) = 0;
  virtual void set_num_eigen(const MKL_INT num) { _num_req = num; }

  void set_max_iter(const MKL_INT iters) { _maxiter = iters; }
  void set_ncv(const MKL_INT ncv) { _ncv = ncv; }

protected:
  mkl_sparse_mat *_A{nullptr};
  mkl_sparse_mat *_B{nullptr};

  MKL_INT _num_req{1};
  MKL_INT _num_found{0};
  MKL_INT _maxiter{10000}; // max num of iterations
  MKL_INT _ncv{10};
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

protected:
  MKL_INT _pm[128] = {0};
  double _tol{6}; // 1e-{_tol}+1
  char _which{'S'};
};

// power method
class power_sparse_gv : public mkl_eigen {
public:
  power_sparse_gv(mkl_sparse_mat *A, mkl_sparse_mat *B = nullptr);
  virtual void set_num_eigen(const MKL_INT num);

  void which(const char s) { _which = s; }
  virtual bool eigen_solve(double *eigenValues, double *eigenVectors) override;

  void set_tol(const double tol) { _tol = tol; }

protected:
  double _tol{1e-7};
  char _which{'S'};
  // mkl_solver *_solver;
};

// arpack wrapper
class arpack_gv : public mkl_eigen {
public:
  arpack_gv(mkl_sparse_mat *A, mkl_sparse_mat *B = nullptr);
  virtual bool eigen_solve(double *eigenValues, double *eigenVectors) override;

private:
  int mode() const;
  std::function<bool(double const *const b, double *const x)>
  op(const int mode);

private:
  double _shift{0.};

  mkl_sparse_mat _op;
  std::vector<double> tmp;
  std::unique_ptr<mkl_solver> _solver{nullptr};
};
} // namespace mkl_wrapper