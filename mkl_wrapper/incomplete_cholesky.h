#pragma once
#include "mkl_sparse_mat.h"

namespace mkl_wrapper {

// Incomplete Cholesky k level
class incomplete_cholesky_k : public precond {
public:
  incomplete_cholesky_k();

  virtual bool symbolic_factorize(mkl_sparse_mat const *const A) override;

  virtual bool numeric_factorize(mkl_sparse_mat const *const A) override;

  virtual bool solve(double const *const b, double *const x) override;

  void set_level(const int level) { _level = level; }

  virtual void optimize() override;
protected:
  int _level;
  std::vector<MKL_INT> _diagPos;
};
} // namespace mkl_wrapper