#pragma once
#include <mkl.h>
namespace mkl_wrapper {
class mkl_sparse_mat;

class mkl_iterative_solver {
public:
  mkl_iterative_solver() {}

  void SetMaxIterations(int n) { _maxiter = n; }
  void SetRelTol(double tol) { _rel_tol = tol; }
  void SetAbsTol(double tol) { _abs_tol = tol; }

  virtual bool solve(double *b, double *x) = 0;

protected:
  int _maxiter{1000};    // max nr of iterations
  double _rel_tol{1e-8}; // residual relative tolerance
  double _abs_tol{1e-16};
  bool m_fail_max_iters{true};
  int _print_level{1}; // output level
};

class mkl_pcg_solver : public mkl_iterative_solver {
public:
  mkl_pcg_solver(mkl_sparse_mat *A, mkl_sparse_mat *P = nullptr)
      : mkl_iterative_solver(), _A(A), _P(P) {}

  virtual bool solve(double *b, double *x) override;

protected:
  mkl_sparse_mat *_A;
  mkl_sparse_mat *_P;
};

class mkl_fgmres_solver : public mkl_iterative_solver {
public:
  mkl_fgmres_solver(mkl_sparse_mat *A, mkl_sparse_mat *P = nullptr,
                    mkl_sparse_mat *R = nullptr)
      : mkl_iterative_solver(), _A(A), _P(P), _R(R) {}

  virtual bool solve(double *b, double *x) override;

protected:
  mkl_sparse_mat *_A;
  mkl_sparse_mat *_P; // left preconditioner
  mkl_sparse_mat *_R; // right preconditioner

  MKL_INT _num_restart{100};
};
} // namespace mkl_wrapper