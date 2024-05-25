#pragma once
#include "../config.h"

#ifdef USE_AMGCL_LIB

#include "mkl_sparse_mat.h"
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/spai1.hpp>

namespace mkl_wrapper {

// Incomplete Cholesky k level
class amgcl_precond : public precond {
public:
  amgcl_precond();

  virtual bool symbolic_factorize(mkl_sparse_mat const *const A) override;

  virtual bool numeric_factorize(mkl_sparse_mat const *const A) override;

  virtual bool solve(double const *const b, double *const x) override;

  //   void set_level(const int level) { _level = level; }

protected:
  std::unique_ptr<amgcl::amg<amgcl::backend::builtin<double>,
                             amgcl::coarsening::smoothed_aggregation,
                             amgcl::relaxation::spai1>>
      _amgclPrecond{nullptr};
  //   std::unique_ptr<amgcl::relaxation::as_preconditioner<
  //       amgcl::backend::builtin<double>, amgcl::relaxation::ilu0>>
  //       _amgclPrecond{nullptr};
  std::vector<double> _b;
  std::vector<double> _x;
};
} // namespace mkl_wrapper

#endif