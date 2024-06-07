#include "incomplete_lu.h"
#include <cmath>
#include <cstdio>
#include <execution>
#include <forward_list>
#include <fstream>
#include <iostream>
#include <limits>
#include <mkl_rci.h>
#include <mkl_sparse_handle.h>
#include <mkl_spblas.h>
#include <unordered_map>

#include "../config.h"

#ifdef USE_BOOST_LIB
#include <boost/pool/pool_alloc.hpp>
#endif
namespace mkl_wrapper {

void incomplete_lu_base::optimize() {

  _mkl_descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  _mkl_descr.mode = SPARSE_FILL_MODE_LOWER;
  _mkl_descr.diag = SPARSE_DIAG_UNIT;
  mkl_sparse_set_sv_hint(_mkl_mat, SPARSE_OPERATION_NON_TRANSPOSE, _mkl_descr,
                         1000);
  _mkl_descr.mode = SPARSE_FILL_MODE_UPPER;
  _mkl_descr.diag = SPARSE_DIAG_NON_UNIT;
  mkl_sparse_set_sv_hint(_mkl_mat, SPARSE_OPERATION_NON_TRANSPOSE, _mkl_descr,
                         1000);
  mkl_sparse_set_memory_hint(_mkl_mat, SPARSE_MEMORY_AGGRESSIVE);
  mkl_sparse_optimize(_mkl_mat);
}

bool incomplete_lu_base::solve(double const *const b, double *const x) {
  _mkl_descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  _mkl_descr.mode = SPARSE_FILL_MODE_LOWER;
  _mkl_descr.diag = SPARSE_DIAG_UNIT;
  _mkl_stat = mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, _mkl_mat,
                                _mkl_descr, b, _interm_vec.data());

  _mkl_descr.mode = SPARSE_FILL_MODE_UPPER;
  _mkl_descr.diag = SPARSE_DIAG_NON_UNIT;
  _mkl_stat = mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, _mkl_mat,
                                _mkl_descr, _interm_vec.data(), x);
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
    std::transform(Aaj.get(), Aaj.get() + A->nnz(), tmp_aj.get(),
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

  const MKL_INT base = A->mkl_base();
  auto ai = A->get_ai();
  auto aj = A->get_aj();
  _firstUpperPos.resize(_nrow);

  if (_level == 0) {
    _nnz = A->nnz();
    _aj.reset(new MKL_INT[_nnz]);
    _av.reset(new double[_nnz]);
    std::copy(std::execution::par_unseq, A->get_ai().get(),
              A->get_ai().get() + n + 1, _ai.get());
    std::copy(std::execution::par_unseq, A->get_aj().get(),
              A->get_aj().get() + _nnz, _aj.get());
  } else {
    _ai[0] = base;
    MKL_INT aj_size = A->nnz();
    _aj.reset(new MKL_INT[aj_size]);
    auto av_levels = std::make_unique<MKL_INT[]>(aj_size);

#ifdef USE_BOOST_LIB
    std::forward_list<std::pair<MKL_INT, MKL_INT>,
                      boost::fast_pool_allocator<std::pair<MKL_INT, MKL_INT>>>
        _rowLevels;
#else
    std::forward_list<std::pair<MKL_INT, MKL_INT>> _rowLevels;
#endif

    MKL_INT list_size = 0;
    MKL_INT j, k;

    for (MKL_INT i = 0; i < n; i++) {
      // initialize levels
      auto rowIt = _rowLevels.before_begin();
      // std::cout << "origin mat: ";
      for (MKL_INT k = ai[i] - base; k != ai[i + 1] - base; k++) {
        rowIt = _rowLevels.insert_after(rowIt, std::make_pair(aj[k] - base, 0));
        // std::cout << aj[k] << " ";
      }
      list_size = ai[i + 1] - ai[i];
      // use n as the list end to prevent from branch prediction
      _rowLevels.insert_after(rowIt, std::make_pair(n, 0));

      rowIt = _rowLevels.begin();
      for (MKL_INT kk = 0; kk < list_size; kk++) {
        k = rowIt->first;
        if (k >= i)
          break;
        auto lik = rowIt->second;
        auto eij = rowIt;
        MKL_INT nextIdx = std::next(eij)->first;
        j = _firstUpperPos[k];

        for (; j < _ai[k + 1] - base; j++) {
          while (nextIdx <= _aj[j] - base) {
            eij = std::next(eij);
            nextIdx = std::next(eij)->first;
          }
          if (lik + av_levels[j] + 1 <= _level) {
            if (eij->first == _aj[j] - base) {
              if (eij->second > lik + av_levels[j] + 1) {
                eij->second = lik + av_levels[j] + 1;
              }
            } else {
              eij = _rowLevels.insert_after(
                  eij, std::make_pair(_aj[j] - base, lik + av_levels[j] + 1));
              nextIdx = std::next(eij)->first;
              list_size++;
            }
          }
        }
        rowIt++;
      }
      // #pragma omp parallel for
      // push current row level back to aj av

      rowIt = _rowLevels.begin();
      MKL_INT pos = _ai[i] - base;
      if (_ai[i] + list_size - base > aj_size) {
        MKL_INT new_aj_size;
        if (2 * i >= n)
          new_aj_size = 2 * aj_size;
        else
          new_aj_size = aj_size * std::ceil(n * 1. / i);
        std::shared_ptr<MKL_INT[]> new_aj(new MKL_INT[new_aj_size]);
        auto new_levels = std::make_unique<MKL_INT[]>(new_aj_size);

        std::copy(std::execution::seq, _aj.get(), _aj.get() + aj_size,
                  new_aj.get());
        std::copy(std::execution::seq, av_levels.get(),
                  av_levels.get() + aj_size, new_levels.get());
        std::swap(new_aj, _aj);
        std::swap(new_levels, av_levels);
        std::swap(new_aj_size, aj_size);
        // std::cout<<"copy\n";
      }
      for (MKL_INT kk = 0; kk < list_size; kk++) {
        _aj[pos] = rowIt->first + base;
        av_levels[pos++] = rowIt->second;
        rowIt++;
      }
      _ai[i + 1] = _ai[i] + list_size;
      _rowLevels.clear();

      auto mid = std::upper_bound(_aj.get() + _ai[i] - base,
                                  _aj.get() + _ai[i + 1] - base, i + base);
      if (mid == _aj.get() + _ai[i + 1] - base)
        _firstUpperPos[i] = _ai[i + 1] - base;
      else
        _firstUpperPos[i] = mid - _aj.get();
    }
    _nnz = _ai[n] - base;
    _av.reset(new double[_nnz]);
  }
  return true;
}

bool incomplete_lu_k::numeric_factorize(mkl_sparse_mat const *const A) {
  auto ai = A->get_ai();
  auto aj = A->get_aj();
  auto av = A->get_av();
  const MKL_INT n = rows();
  const MKL_INT base = mkl_base();
  std::vector<double> diag(n);
  MKL_INT k_idx, A_k_idx, k, _j_idx, j_idx;

  for (MKL_INT i = 0; i < n; i++) {

    for (k_idx = _ai[i] - base, A_k_idx = ai[i] - base;
         k_idx != _ai[i + 1] - base; k_idx++) {
      if (A_k_idx == ai[i + 1] - base || _aj[k_idx] != aj[A_k_idx]) {
        _av[k_idx] = 0;
      } else {
        _av[k_idx] = av[A_k_idx++];
      }
    }

    for (k_idx = _ai[i] - base;
         _aj[k_idx] - base != i && k_idx != _ai[i + 1] - base; k_idx++) {
      k = _aj[k_idx] - base;
      _av[k_idx] /= diag[k];
      const double aik = _av[k_idx];
      _j_idx = k_idx;
      j_idx = _firstUpperPos[k];
      for (; j_idx != _ai[k + 1] - base && _j_idx != _ai[i + 1] - base;) {
        if (_aj[_j_idx] == _aj[j_idx]) {
          _av[_j_idx++] -= aik * _av[j_idx++];
        } else if (_aj[_j_idx] < _aj[j_idx])
          _j_idx++;
        else
          j_idx++;
      }
    }
    // copy diagonal aii
    if (_aj[k_idx] - base != i) {
      std::cerr << "no element on diagonal!\n";
    } else {
      diag[i] = _av[k_idx];
    }
  }
  if (_mkl_base == SPARSE_INDEX_BASE_ONE)
    sp_fill();
  else
    to_one_based();
  return true;
}

// bool incomplete_lu_k::solve(double const *const b, double *const x) {
//   const MKL_INT base = mkl_base();
//   // std::fill(_interm_vec.begin(), _interm_vec.end(), 0.0);
//   for (MKL_INT i = 0; i < rows(); i++) {
//     _interm_vec[i] = b[i];
//     for (MKL_INT j = _ai[i] - base; j < _firstUpperPos[i]-1; j++) {
//       _interm_vec[i] -= _av[j] * _interm_vec[_aj[j] - base];
//     }
//   }
//   // for (auto i : _interm_vec)
//   //   std::cout << i << std::endl;
//   // std::abort();
//   for (MKL_INT i = rows() - 1; i >= 0; i--) {
//     x[i] = _interm_vec[i];
//     for (MKL_INT j = _firstUpperPos[i]; j < _ai[i + 1] - base; j++) {
//       x[i] -= _av[j] * x[_aj[j] - base];
//     }
//     x[i] /= _av[_firstUpperPos[i] - 1];
//   }

//   return true;
// }
} // namespace mkl_wrapper