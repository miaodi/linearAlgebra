#pragma once
#include <mkl.h>
namespace mkl_wrapper {
class mkl_sparse_mat;

class mkl_pcg_solver {
public:
  mkl_pcg_solver(mkl_sparse_mat *A, mkl_sparse_mat *P = nullptr)
      : _A(A), _P(P) {}

  void SetMaxIterations(int n) { m_maxiter = n; }
  void SetTolerance(double tol) { m_tol = tol; }
  bool solve(double const *const b, double *const x);

protected:
  mkl_sparse_mat *_A;
  mkl_sparse_mat *_P;

  int m_maxiter{1000}; // max nr of iterations
  double m_tol{1e-8};  // residual relative tolerance
  bool m_fail_max_iters{true};
  int _print_level{1}; // output level
};
} // namespace mkl_wrapper