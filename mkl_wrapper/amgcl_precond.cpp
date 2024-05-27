#include "amgcl_precond.h"

#ifdef USE_AMGCL_LIB
#include <amgcl/adapter/crs_tuple.hpp>
#include <execution>
namespace mkl_wrapper {
amgcl_precond::amgcl_precond() : mkl_sparse_mat() {}

bool amgcl_precond::symbolic_factorize(mkl_sparse_mat const *const A) {
  _nrow = A->rows();
  _ncol = A->cols();
  _mkl_base = A->mkl_base();
  _nnz = A->nnz();
  const MKL_INT n = A->rows();
  _ai.reset(new MKL_INT[n + 1]);
  _aj.reset(new MKL_INT[_nnz]);
  _av.reset(new double[_nnz]);
  _b.resize(n);
  _x.resize(n);

  _amgclPrecond.reset(new amgcl::amg<amgcl::backend::builtin<double>,
                                     amgcl::coarsening::smoothed_aggregation,
                                     amgcl::relaxation::spai1>(std::make_tuple(
      n, amgcl::make_iterator_range(_ai.get(), _ai.get() + n + 1),
      amgcl::make_iterator_range(_aj.get(), _aj.get() + _nnz),
      amgcl::make_iterator_range(_av.get(), _av.get() + _nnz))));

  //   _amgclPrecond.reset(
  //       new
  //       amgcl::relaxation::as_preconditioner<amgcl::backend::builtin<double>,
  //                                                amgcl::relaxation::ilu0>(
  //           std::make_tuple(
  //               n, amgcl::make_iterator_range(_ai.get(), _ai.get() + n + 1),
  //               amgcl::make_iterator_range(_aj.get(), _aj.get() + _nnz),
  //               amgcl::make_iterator_range(_av.get(), _av.get() + _nnz))));
  return true;
}

bool amgcl_precond::numeric_factorize(mkl_sparse_mat const *const A) {

  auto ai = A->get_ai();
  auto aj = A->get_aj();
  auto av = A->get_av();
  const MKL_INT base = _mkl_base;
  const MKL_INT n = A->rows();

  std::transform(std::execution::par_unseq, ai.get(), ai.get() + _nrow + 1,
                 _ai.get(), [base](const MKL_INT n) { return n - base; });

  std::transform(std::execution::par_unseq, aj.get(), aj.get() + _nnz,
                 _aj.get(), [base](const MKL_INT n) { return n - base; });

  std::copy(std::execution::par_unseq, av.get(), av.get() + _nnz, _av.get());

  //   _amgclPrecond.reset(
  //       new
  //       amgcl::relaxation::as_preconditioner<amgcl::backend::builtin<double>,
  //                                                amgcl::relaxation::ilu0>(
  //           std::make_tuple(
  //               n, amgcl::make_iterator_range(_ai.get(), _ai.get() + n + 1),
  //               amgcl::make_iterator_range(_aj.get(), _aj.get() + _nnz),
  //               amgcl::make_iterator_range(_av.get(), _av.get() + _nnz))));
  _amgclPrecond->rebuild(std::make_tuple(
      _nrow, amgcl::make_iterator_range(_ai.get(), _ai.get() + _nrow + 1),
      amgcl::make_iterator_range(_aj.get(), _aj.get() + _nnz),
      amgcl::make_iterator_range(_av.get(), _av.get() + _nnz)));
  return true;
}

bool amgcl_precond::solve(double const *const b, double *const x) {

  std::copy(std::execution::par_unseq, b, b + _nrow, _b.data());
  _amgclPrecond->apply(_b, _x);
  std::copy(std::execution::par_unseq, _x.cbegin(), _x.cend(), x);
  return true;
}
} // namespace mkl_wrapper
#endif
