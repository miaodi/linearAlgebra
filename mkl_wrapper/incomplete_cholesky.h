#pragma once
#include "mkl_sparse_mat.h"

namespace mkl_wrapper {

class incomplete_choleksy_base : public mkl_sparse_mat_sym {
public:
  incomplete_choleksy_base() : mkl_sparse_mat_sym() {}

  virtual bool numeric_factorize(mkl_sparse_mat const *const A) {
    return false;
  }

  virtual bool symbolic_factorize(mkl_sparse_mat const *const A) {
    return true;
  }

  virtual bool solve(double const *const b, double *const x) override;

  virtual void optimize() override;

  void shift(const bool shift) { _shift = shift; }

protected:
  std::vector<double> _interm_vec;
  double _initial_shift{1e-3};
  int _nrestart{20};
  bool _shift{false};
  std::vector<MKL_INT> _diagPos;
};

// Incomplete Cholesky k level
class incomplete_cholesky_k : public incomplete_choleksy_base {
public:
  incomplete_cholesky_k();

  virtual bool symbolic_factorize(mkl_sparse_mat const *const A) override;

  virtual bool numeric_factorize(mkl_sparse_mat const *const A) override;

  void set_level(const int level) { _level = level; }

protected:
  int _level;
};

class incomplete_cholesky_fm : public incomplete_choleksy_base {
public:
  incomplete_cholesky_fm();

  virtual bool symbolic_factorize(mkl_sparse_mat const *const A) override;

  virtual bool numeric_factorize(mkl_sparse_mat const *const A) override;

  void set_p(const int p) { _p = p; }

protected:
  int _p{0};
  int _capacity{0};
};
} // namespace mkl_wrapper