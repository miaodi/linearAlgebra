#pragma once

#include "../config.h"

#ifdef USE_ARPACK_LIB

#include "mkl_eigen.h"

namespace mkl_wrapper {
// arpack wrapper
class arpack_gv : public mkl_eigen {
public:
  arpack_gv(mkl_sparse_mat *A, mkl_sparse_mat *B = nullptr);
  virtual bool eigen_solve(double *eigenValues, double *eigenVectors) override;

private:
  int mode() const;
  std::function<bool(double *const b, double *const x)> op(const int mode);

private:
  double _shift{0.};

  mkl_sparse_mat _op;
  std::vector<double> tmp;
  std::unique_ptr<mkl_solver> _solver{nullptr};
};
} // namespace mkl_wrapper

#endif