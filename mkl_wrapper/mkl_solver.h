#pragma once
#include <functional>
#include <map>
#include <memory>
#include <mkl_types.h>
#include <string>
namespace mkl_wrapper {
class mkl_sparse_mat;

class mkl_solver {
public:
  mkl_solver() = default;
  virtual bool solve(double const *const b, double *const x) = 0;
  virtual ~mkl_solver() {}

protected:
  int _print_level{1}; // output level
};

class mkl_direct_solver : public mkl_solver {
public:
  virtual bool solve(double const *const b, double *const x) override;
  bool forward_substitution(double const *const b, double *const x);
  bool backward_substitution(double const *const b, double *const x);
  mkl_direct_solver(mkl_sparse_mat *A) : mkl_solver(), _A(A) {}
  bool factorize();
  void set_max_iter_ref(int n) { _max_iter_ref = n; }
  virtual ~mkl_direct_solver() override;

protected:
  bool _factorized{false};
  mkl_sparse_mat *_A;
  MKL_INT _iparm[64];
  MKL_INT _maxfct;
  MKL_INT _mnum;
  MKL_INT _mtype;
  MKL_INT _msglvl;
  void *_pt[64];
  MKL_INT _max_iter_ref{1};
};

class mkl_iterative_solver : public mkl_solver {
public:
  mkl_iterative_solver() : mkl_solver() {}

  void set_max_iters(int n) { _maxiter = n; }
  void set_rel_tol(double tol) { _rel_tol = tol; }
  void set_abs_tol(double tol) { _abs_tol = tol; }

protected:
  int _maxiter{1000};    // max nr of iterations
  double _rel_tol{1e-8}; // residual relative tolerance
  double _abs_tol{1e-16};
  bool m_fail_max_iters{true};
};

class mkl_pcg_solver : public mkl_iterative_solver {
public:
  mkl_pcg_solver(mkl_sparse_mat *A, mkl_sparse_mat *P = nullptr)
      : mkl_iterative_solver(), _A(A), _P(P) {}

  virtual bool solve(double const *const b, double *const x) override;

protected:
  mkl_sparse_mat *_A;
  mkl_sparse_mat *_P;
};

class mkl_fgmres_solver : public mkl_iterative_solver {
public:
  mkl_fgmres_solver(mkl_sparse_mat *A, mkl_sparse_mat *P = nullptr,
                    mkl_sparse_mat *R = nullptr)
      : mkl_iterative_solver(), _A(A), _P(P), _R(R) {}

  virtual bool solve(double const *const b, double *const x) override;
  void set_restart_steps(const int n) { _num_restart = n; }

protected:
  mkl_sparse_mat *_A;
  mkl_sparse_mat *_P{nullptr}; // left preconditioner
  mkl_sparse_mat *_R{nullptr}; // right preconditioner

  MKL_INT _num_restart{0};
};

class solver_factory {
public:
  using solver_ptr = std::unique_ptr<mkl_solver>;
  using create_method = std::function<solver_ptr(mkl_sparse_mat &)>;

public:
  solver_factory();
  bool reg(const std::string &name, create_method func);
  solver_ptr create(const std::string &name, mkl_sparse_mat &);

private:
  std::map<std::string, create_method> _methods;
};
} // namespace mkl_wrapper