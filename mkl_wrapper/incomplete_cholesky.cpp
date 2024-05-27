#include "incomplete_cholesky.h"
#include <cmath>
#include <cstdio>
#include <execution>
#include <forward_list>
#include <fstream>
#include <iostream>
#include <mkl_sparse_handle.h>
#include <mkl_spblas.h>

#include "../config.h"

#ifdef USE_BOOST_LIB
#include <boost/pool/pool_alloc.hpp>
#endif

namespace mkl_wrapper {
bool incomplete_choleksy_base::solve(double const *const b, double *const x) {
  mkl_sparse_d_trsv(SPARSE_OPERATION_TRANSPOSE, 1.0, _mkl_mat, _mkl_descr, b,
                    _interm_vec.data());

  mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, _mkl_mat, _mkl_descr,
                    _interm_vec.data(), x);
  return true;
}

void incomplete_choleksy_base::optimize() {

  _mkl_descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  _mkl_descr.mode = SPARSE_FILL_MODE_UPPER;
  _mkl_descr.diag = SPARSE_DIAG_NON_UNIT;

  // mkl_sparse_set_mv_hint(_mkl_mat, SPARSE_OPERATION_NON_TRANSPOSE,
  // _mkl_descr,
  //                        1000);
  mkl_sparse_set_sv_hint(_mkl_mat, SPARSE_OPERATION_NON_TRANSPOSE, _mkl_descr,
                         1000);
  mkl_sparse_set_sv_hint(_mkl_mat, SPARSE_OPERATION_TRANSPOSE, _mkl_descr,
                         1000);
  mkl_sparse_set_memory_hint(_mkl_mat, SPARSE_MEMORY_AGGRESSIVE);
  mkl_sparse_optimize(_mkl_mat);
}

incomplete_cholesky_k::incomplete_cholesky_k() : incomplete_choleksy_base() {}

bool incomplete_cholesky_k::symbolic_factorize(mkl_sparse_mat const *const A) {
  _nrow = A->rows();
  _ncol = A->cols();
  _mkl_base = A->mkl_base();
  _interm_vec.resize(_nrow);
  const MKL_INT n = A->rows();
  const MKL_INT base = A->mkl_base();
  auto ai = A->get_ai();
  auto aj = A->get_aj();
  _ai.reset(new MKL_INT[n + 1]);
  const bool sym = A->mkl_descr().type == SPARSE_MATRIX_TYPE_SYMMETRIC;
  _diagPos.resize(_nrow);
  if (sym) {
#pragma omp parallel for
    for (MKL_INT i = 0; i < _nrow; i++) {
      _diagPos[i] = ai[i] - base;
    }
  } else {
    volatile bool missing_diag = false;
#pragma omp parallel for shared(missing_diag)
    for (MKL_INT i = 0; i < _nrow; i++) {
      if (missing_diag)
        continue;
      auto mid = std::find(aj.get() + ai[i] - base, aj.get() + ai[i + 1] - base,
                           i + base);
      if (mid == aj.get() + ai[i + 1] - base) {
        std::cerr << "Could not find diagonal!" << std::endl;
        missing_diag = true;
      }
      _diagPos[i] = mid - aj.get();
    }
    if (missing_diag)
      return false;
  }
  if (_level == 0 && sym) {
    _nnz = A->nnz();
    _aj.reset(new MKL_INT[_nnz]);
    _av.reset(new double[_nnz]);
    std::copy(std::execution::seq, A->get_ai().get(), A->get_ai().get() + n + 1,
              _ai.get());
    std::copy(std::execution::seq, A->get_aj().get(), A->get_aj().get() + _nnz,
              _aj.get());
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
    MKL_INT j_idx;
    MKL_INT k;
    std::vector<std::vector<std::pair<MKL_INT, MKL_INT>>> jKRow(n);
    MKL_INT availableJKRow = 0;

    for (MKL_INT i = 0; i < n; i++) {
      auto rowIt = _rowLevels.before_begin();
      k = _diagPos[i];
      list_size = ai[i + 1] - base - k;
      for (; k != ai[i + 1] - base; k++) {
        rowIt = _rowLevels.insert_after(rowIt, std::make_pair(aj[k] - base, 0));
      }

      // use n as the list end to prevent from branch prediction
      rowIt = _rowLevels.insert_after(rowIt, std::make_pair(n, 0));

      for (auto &k_pair : jKRow[i]) {
        k = k_pair.first;
        j_idx = k_pair.second;

        if (j_idx + 1 != _ai[k + 1] - base) {
          if (jKRow[_aj[j_idx + 1] - base].empty() && availableJKRow < i) {
            std::swap(jKRow[_aj[j_idx + 1] - base], jKRow[availableJKRow++]);
          }
          jKRow[_aj[j_idx + 1] - base].push_back({k, j_idx + 1});
        }
        auto lik = av_levels[j_idx];
        auto eij = _rowLevels.begin();
        MKL_INT nextIdx = std::next(eij)->first;
        for (; j_idx != _ai[k + 1] - base; j_idx++) {
          MKL_INT jk_idx = _aj[j_idx] - base;
          while (nextIdx <= jk_idx) {
            eij = std::next(eij);
            nextIdx = std::next(eij)->first;
          }
          MKL_INT level = lik + av_levels[j_idx] + 1;
          if (level <= _level) {
            if (eij->first == jk_idx) {
              if (eij->second > level) {
                eij->second = level;
              }
            } else {
              eij = _rowLevels.insert_after(eij, std::make_pair(jk_idx, level));
              nextIdx = std::next(eij)->first;
              list_size++;
            }
          }
        }
      }
      jKRow[i].clear();

      _ai[i + 1] = _ai[i] + list_size;
      if (_ai[i + 1] - base > aj_size) {
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
      }
      rowIt = _rowLevels.begin();
      MKL_INT pos = _ai[i] - base;
      for (MKL_INT ii = 0; ii < list_size; ii++) {
        _aj[pos] = rowIt->first + base;
        if (pos == _ai[i] - base + 1) {
          if (jKRow[_aj[pos] - base].empty()) {
            std::swap(jKRow[availableJKRow++], jKRow[_aj[pos] - base]);
          }
          jKRow[_aj[pos] - base].push_back({i, pos});
        }
        av_levels[pos++] = rowIt->second;
        rowIt++;
      }
      _rowLevels.clear();
    }
    _nnz = _ai[n] - base;
    _av.reset(new double[_nnz]);
  }
  return true;
}

bool incomplete_cholesky_k::numeric_factorize(mkl_sparse_mat const *const A) {
  auto ai = A->get_ai();
  auto aj = A->get_aj();
  auto av = A->get_av();
  const MKL_INT n = rows();
  const MKL_INT base = mkl_base();
  MKL_INT k_idx, A_k_idx, k, _j_idx, j_idx;
  std::vector<std::vector<std::pair<MKL_INT, MKL_INT>>> jKRow(n);
  MKL_INT availableJKRow = 0;
  for (MKL_INT i = 0; i < n; i++) {

    k_idx = _ai[i] - base;
    A_k_idx = _diagPos[i];

    for (; k_idx != _ai[i + 1] - base; k_idx++) {
      if (A_k_idx == ai[i + 1] - base || _aj[k_idx] != aj[A_k_idx]) {
        _av[k_idx] = 0;
      } else {
        _av[k_idx] = av[A_k_idx++];
      }
    }
    for (auto &k_pair : jKRow[i]) {
      k = k_pair.first;
      j_idx = k_pair.second;

      if (j_idx + 1 != _ai[k + 1] - base) {
        if (jKRow[_aj[j_idx + 1] - base].empty() && availableJKRow < i)
          std::swap(jKRow[_aj[j_idx + 1] - base], jKRow[availableJKRow++]);
        jKRow[_aj[j_idx + 1] - base].push_back({k, j_idx + 1});
      }

      const double aki = _av[j_idx];
      _j_idx = _ai[i] - base;
      for (; j_idx != _ai[k + 1] - base && _j_idx != _ai[i + 1] - base;) {
        if (_aj[_j_idx] == _aj[j_idx]) {
          _av[_j_idx++] -= aki * _av[j_idx++];
        } else if (_aj[_j_idx] < _aj[j_idx])
          _j_idx++;
        else
          j_idx++;
      }
    }
    jKRow[i].clear();
    k_idx = _ai[i] - base;
    if (_ai[i + 1] - _ai[i] != 1) {
      if (jKRow[_aj[k_idx + 1] - base].empty())
        std::swap(jKRow[availableJKRow++], jKRow[_aj[k_idx + 1] - base]);
      jKRow[_aj[k_idx + 1] - base].push_back({i, k_idx + 1});
    }

    if (_av[k_idx] <= 0) {
      std::cerr << "non-positive diagonal!\n";
      return false;
    }
    const double aii = std::sqrt(_av[k_idx]);
    _av[k_idx++] = aii;
    for (; k_idx != _ai[i + 1] - base; k_idx++)
      _av[k_idx] /= aii;
  }
  if (_mkl_base == SPARSE_INDEX_BASE_ONE)
    sp_fill();
  else
    to_one_based();
  return true;
}
} // namespace mkl_wrapper