#include "incomplete_lu.h"
#include <cmath>
#include <cstdio>
#include <execution>
#include <fstream>
#include <iostream>
#include <limits>
#include <mkl_rci.h>
#include <mkl_sparse_handle.h>
#include <mkl_spblas.h>

namespace mkl_wrapper {

bool incomplete_lu_base::solve(double const *const b, double *const x) {
  sparse_operation_t transA = SPARSE_OPERATION_NON_TRANSPOSE;
  _mkl_descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  _mkl_descr.mode = SPARSE_FILL_MODE_LOWER;
  _mkl_descr.diag = SPARSE_DIAG_UNIT;
  _mkl_stat = mkl_sparse_d_trsv(transA, 1.0, _mkl_mat, _mkl_descr, b,
                                _interm_vec.data());

  _mkl_descr.mode = SPARSE_FILL_MODE_UPPER;
  _mkl_descr.diag = SPARSE_DIAG_NON_UNIT;
  _mkl_stat = mkl_sparse_d_trsv(transA, 1.0, _mkl_mat, _mkl_descr,
                                _interm_vec.data(), x);
  return true;
}

mkl_ilu0::mkl_ilu0() : incomplete_lu_base() {}

bool mkl_ilu0::symbolic_factorize(mkl_sparse_mat const *const A) {
  _nrow = A->rows();
  _ncol = A->cols();
  _nnz = A->nnz();
  _interm_vec.resize(_nrow);

  _ai.reset(new MKL_INT[_nrow + 1]);
  _aj.reset(new MKL_INT[_nnz]);
  _av.reset(new double[_nnz]);

  std::copy(std::execution::seq, A->get_ai().get(),
            A->get_ai().get() + _nrow + 1, _ai.get());

  std::copy(std::execution::seq, A->get_aj().get(), A->get_aj().get() + _nnz,
            _aj.get());
  _mkl_base = A->mkl_base();
  to_one_based();
  return true;
}

bool mkl_ilu0::numeric_factorize(mkl_sparse_mat const *const A) {
  if (A == nullptr || A->rows() != A->cols() || A->rows() != _nrow)
    return false;

  MKL_INT ipar[128] = {0};
  double dpar[128] = {0.0};

  // parameters affecting the pre-conditioner
  if (_check_zero_diag) {
    ipar[30] = 1;
    dpar[30] = _zero_tol;
    dpar[31] = _zero_rep;
  }

  MKL_INT ierr = 0;
  dcsrilu0(&_nrow, A->get_av().get(), _ai.get(), _aj.get(), _av.get(), ipar,
           dpar, &ierr);

  if (ierr != 0)
    return false;

  sp_fill();
  return true;
}

mkl_ilut::mkl_ilut() : incomplete_lu_base() {}

bool mkl_ilut::symbolic_factorize(mkl_sparse_mat const *const A) {
  _nrow = A->rows();
  _ncol = A->cols();
  _nnz = (2 * _max_fill + 1) * _nrow;
  _interm_vec.resize(_nrow);

  _ai.reset(new MKL_INT[_nrow + 1]);
  _aj.reset(new MKL_INT[_nnz]);
  _av.reset(new double[_nnz]);
  return true;
}

bool mkl_ilut::numeric_factorize(mkl_sparse_mat const *const A) {
  if (A == nullptr || A->rows() != A->cols() || A->rows() != _nrow)
    return false;

  MKL_INT ipar[128] = {0};
  double dpar[128] = {0.0};

  // parameters affecting the pre-conditioner
  if (_check_zero_diag) {
    ipar[30] = 1;
    dpar[30] = _zero_tol;
    // dpar[31] = _zero_rep;
  } else {
    // do this to avoid a warning from the preconditioner
    dpar[30] = _tau;
  }

  MKL_INT ierr = 0;
  std::shared_ptr<const MKL_INT[]> Aai = A->get_ai();
  std::shared_ptr<const MKL_INT[]> Aaj = A->get_aj();
  if (A->mkl_base() == 0) {
    std::shared_ptr<MKL_INT[]> tmp_ai(new MKL_INT[_nrow + 1]);
    std::transform(Aai.get(), Aai.get() + _nrow + 1, tmp_ai.get(),
                   [](const MKL_INT i) { return i + 1; });
    std::shared_ptr<MKL_INT[]> tmp_aj(new MKL_INT[A->nnz()]);
    std::transform(Aaj.get(), Aaj.get() + _nnz, tmp_aj.get(),
                   [](const MKL_INT i) { return i + 1; });
    Aai = tmp_ai;
    Aaj = tmp_aj;
  }
  dcsrilut(&_nrow, A->get_av().get(), Aai.get(), Aaj.get(), _av.get(),
           _ai.get(), _aj.get(), &_tau, &_max_fill, ipar, dpar, &ierr);
  if (ierr != 0)
    return false;

  _mkl_base = SPARSE_INDEX_BASE_ONE;
  sp_fill();
  return true;
}

incomplete_lu_k::incomplete_lu_k() : incomplete_lu_base() {}

bool incomplete_lu_k::symbolic_factorize(mkl_sparse_mat const *const A) {
  _nrow = A->rows();
  _ncol = A->cols();
  _mkl_base = A->mkl_base();
  _interm_vec.resize(_nrow);
  const MKL_INT n = A->rows();
  _ai.reset(new MKL_INT[n + 1]);
  if (_level == 0) {

    _nnz = A->nnz();
    _aj.reset(new MKL_INT[_nnz]);
    _av.reset(new double[_nnz]);
    std::copy(std::execution::par_unseq, A->get_ai().get(),
              A->get_ai().get() + n + 1, _ai.get());
    std::copy(std::execution::par_unseq, A->get_aj().get(),
              A->get_aj().get() + _nnz, _aj.get());
  } else {
    const MKL_INT base = A->mkl_base();
    std::vector<MKL_INT> rowLevels(n);
    auto ai = A->get_ai();
    auto aj = A->get_aj();
    _ai[0] = base;
    std::vector<MKL_INT> aj_vec;
    std::vector<MKL_INT> av_levels;
    aj_vec.reserve(A->nnz());
    av_levels.reserve(A->nnz());

    for (MKL_INT i = 0; i < n; i++) {
// initialize levels
#pragma omp parallel for
      for (MKL_INT k = 0; k != n; k++)
        rowLevels[k] = std::numeric_limits<MKL_INT>::max();
#pragma omp parallel for
      for (MKL_INT k = ai[i] - base; k != ai[i + 1] - base; k++)
        rowLevels[aj[k] - base] = 0;

      for (MKL_INT k = 0; k < i; k++) {
        auto cur_level = rowLevels[k];
        if (cur_level == std::numeric_limits<MKL_INT>::max())
          continue;
        // #pragma omp parallel for
        for (MKL_INT j = _ai[k] - base; j != _ai[k + 1] - base; j++) {
          if (aj_vec[j] <= k)
            continue;
          if (av_levels[j] != std::numeric_limits<MKL_INT>::max()) {
            rowLevels[aj_vec[j] - base] = std::min(
                rowLevels[aj_vec[j] - base], cur_level + av_levels[j] + 1);
          }
        }
      }

      // #pragma omp parallel for
      // push current row level back to aj av
      for (MKL_INT k = 0; k != n; k++)
        if (rowLevels[k] <= _level) {
          aj_vec.push_back(k + base);
          av_levels.push_back(rowLevels[k]);
        }
      _ai[i + 1] = aj_vec.size() + base;
    }
    _nnz = _ai[n] - base;
    _aj.reset(new MKL_INT[_nnz]);
    _av.reset(new double[_nnz]);
    std::copy(std::execution::par_unseq, aj_vec.begin(), aj_vec.end(),
              _aj.get());
  }
  return true;
}

bool incomplete_lu_k::numeric_factorize(mkl_sparse_mat const *const A) {
  auto ai = A->get_ai();
  auto aj = A->get_aj();
  auto av = A->get_av();
  const MKL_INT n = rows();
  const MKL_INT base = mkl_base();
  std::vector<double> rowVals(n);
  std::vector<double> diag(n);
  for (MKL_INT i = 0; i < n; i++) {
    // initialize levels
#pragma omp parallel for
    for (MKL_INT k = 0; k != n; k++)
      rowVals[k] = 0.;
#pragma omp parallel for
    for (MKL_INT k = ai[i] - base; k != ai[i + 1] - base; k++)
      rowVals[aj[k] - base] = av[k];

    MKL_INT k_idx;
    for (k_idx = _ai[i] - base; _aj[k_idx] - base < i; k_idx++) {
      MKL_INT k = _aj[k_idx] - base;
      rowVals[k] /= diag[k];
      const double aik = rowVals[k];
#pragma omp parallel for
      for (MKL_INT j = _ai[k] - base; j != _ai[k + 1] - base; j++) {
        if (_aj[j] - base <= k)
          continue;
        rowVals[_aj[j] - base] -= aik * _av[j];
      }
    }
    // copy diagonal aii
    if (_aj[k_idx] - base != i) {
      std::cerr << "no element on diagonal!\n";
    } else {
      diag[i] = rowVals[i];
    }

#pragma omp parallel for
    // push current row back to av
    for (k_idx = _ai[i] - base; k_idx != _ai[i + 1] - base; k_idx++) {
      _av[k_idx] = rowVals[_aj[k_idx] - base];
    }
  }
  if (_mkl_base == SPARSE_INDEX_BASE_ONE)
    sp_fill();
  else
    to_one_based();
  return true;
}
} // namespace mkl_wrapper