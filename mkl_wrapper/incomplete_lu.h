#pragma once
#include "mkl_sparse_mat.h"

namespace mkl_wrapper {

class incomplete_lu_base : public incomplete_fact {
public:
  incomplete_lu_base() : incomplete_fact() {}

  virtual bool solve(double const *const b, double *const x) override;
};

// mkl ilu0
class mkl_ilu0 : public incomplete_lu_base {
public:
  mkl_ilu0();

  virtual bool symbolic_factorize(mkl_sparse_mat const *const A) override;

  virtual bool numeric_factorize(mkl_sparse_mat const *const A) override;

protected:
  bool _check_zero_diag{false}; // check for zero diagonals
  double _zero_tol{1e-16};      // threshold for zero diagonal check
  double _zero_rep{1e-10};      // replacement value for zero diagonal
};

// mkl ilut
class mkl_ilut : public incomplete_lu_base {
public:
  mkl_ilut();
  virtual bool symbolic_factorize(mkl_sparse_mat const *const A) override;
  virtual bool numeric_factorize(mkl_sparse_mat const *const A) override;
  void set_tau(const double tau) { _tau = tau; }
  void set_max_fill(const MKL_INT fill) { _max_fill = fill; }

protected:
  bool _check_zero_diag{false}; // check for zero diagonals
  double _zero_tol{1e-16};      // threshold for zero diagonal check
  // double _zero_rep{1e-10};      // replacement value for zero diagonal
  double _tau{1e-6};
  MKL_INT _max_fill{2};
};

// Incomplete LU k level
class incomplete_lu_k : public incomplete_lu_base {
public:
  incomplete_lu_k();

  virtual bool symbolic_factorize(mkl_sparse_mat const *const A) override;

  virtual bool numeric_factorize(mkl_sparse_mat const *const A) override;

  void set_level(const int level) { _level = level; }

protected:
  int _level;
};
} // namespace mkl_wrapper