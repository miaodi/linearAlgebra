#pragma once
#include "mkl_solver.h"
#include <dmumps_c.h>
#include <memory>

namespace mkl_wrapper {

class mumps_solver : public mkl_solver {
public:
  mumps_solver(mkl_sparse_mat const *A);
  virtual bool solve(double const *const b, double *const x) override;
  virtual ~mumps_solver() override;
  bool factorize();

protected:
  void set_params();
  DMUMPS_STRUC_C id;
  std::unique_ptr<int[]> row_ptr{nullptr};
  std::unique_ptr<int[]> col_ptr{nullptr};
};
} // namespace mkl_wrapper