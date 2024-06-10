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
#include "utils.h"

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

template <typename LIST, typename IDX, typename VAL>
void incomplete_choleksy_base::aij_update(IDX _ai, IDX _aj, VAL _av,
                                          MKL_INT j_idx, MKL_INT k,
                                          MKL_INT base, const double aki,
                                          int &list_size, LIST &list) {
  auto eij = list.begin();
  MKL_INT nextIdx = std::next(eij)->first;
  for (; j_idx < _ai[k + 1] - base; j_idx++) {
    MKL_INT jk_idx = _aj[j_idx] - base;
    while (nextIdx <= jk_idx) {
      eij = std::next(eij);
      nextIdx = std::next(eij)->first;
    }
    const double val = aki * _av[j_idx];
    if (eij->first == jk_idx) {
      eij->second -= val;
    } else {
      eij = list.insert_after(eij, std::make_pair(jk_idx, -val));
      nextIdx = std::next(eij)->first;
      list_size++;
    }
  }
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
  A->diag_pos(_diagPos);

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
    utils::CacheFriendlyVectors<std::pair<MKL_INT, MKL_INT>> jKRow(n);
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
          jKRow.push_back(_aj[j_idx + 1] - base, {k, j_idx + 1});
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
      jKRow.to_next();

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
          jKRow.push_back(_aj[pos] - base, {i, pos});
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
  MKL_INT i, k_idx, A_k_idx, k, _j_idx, j_idx;
  utils::CacheFriendlyVectors<std::pair<MKL_INT, MKL_INT>> jKRow(n);
  bool success_flag = false;
  int iter = 0;

  double shift = 0.;
  if (_shift) {
    // initialize shift
    double minDiag = std::numeric_limits<double>::max();
#pragma omp parallel for reduction(min : minDiag)
    for (i = 0; i < n; i++) {
      minDiag = std::min(minDiag, av[_diagPos[i]]);
    }
    if (minDiag <= 0.)
      shift = _initial_shift - minDiag;
    std::cout << "init shift: " << shift << std::endl;
  }
  do {
    for (i = 0; i < n; i++) {

      k_idx = _ai[i] - base;
      A_k_idx = _diagPos[i];

      for (; k_idx != _ai[i + 1] - base; k_idx++) {
        if (A_k_idx == ai[i + 1] - base || _aj[k_idx] != aj[A_k_idx]) {
          _av[k_idx] = 0;
        } else {
          _av[k_idx] = av[A_k_idx++];
        }
      }
      k_idx = _ai[i] - base;
      _av[k_idx] += shift;
      for (auto &k_pair : jKRow[i]) {
        k = k_pair.first;
        j_idx = k_pair.second;

        if (j_idx + 1 != _ai[k + 1] - base) {
          jKRow.push_back(_aj[j_idx + 1] - base, {k, j_idx + 1});
        }

        const double aki = _av[j_idx];
        _j_idx = k_idx;
        for (; j_idx != _ai[k + 1] - base && _j_idx != _ai[i + 1] - base;) {
          if (_aj[_j_idx] == _aj[j_idx]) {
            _av[_j_idx++] -= aki * _av[j_idx++];
          } else if (_aj[_j_idx] < _aj[j_idx])
            _j_idx++;
          else
            j_idx++;
        }
      }
      jKRow.to_next();
      if (_ai[i + 1] - _ai[i] != 1) {
        jKRow.push_back(_aj[k_idx + 1] - base, {i, k_idx + 1});
      }

      if (_av[k_idx] <= 0) {
        if (!_shift || ++iter > _nrestart)
          return false;
        shift = std::max(_initial_shift, 2. * shift);
        std::cout << "shift: " << shift << std::endl;
        jKRow.clear();
        break;
      }
      const double aii = std::sqrt(_av[k_idx]);
      _av[k_idx++] = aii;
      for (; k_idx != _ai[i + 1] - base; k_idx++)
        _av[k_idx] /= aii;
    }
    if (i == n)
      success_flag = true;
  } while (!success_flag);

  if (_mkl_base == SPARSE_INDEX_BASE_ONE)
    sp_fill();
  else
    to_one_based();
  return true;
}

incomplete_cholesky_fm::incomplete_cholesky_fm() : incomplete_choleksy_base() {}

bool incomplete_cholesky_fm::symbolic_factorize(mkl_sparse_mat const *const A) {
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
  if (!A->diag_pos(_diagPos))
    return false;
  const MKL_INT l_tails =
      std::min(_lsize * (_lsize + 1) / 2, _nrow * (_nrow + 1) / 2);
  const MKL_INT l_capacity =
      sym ? (A->nnz() + _nrow * _lsize - l_tails)
          : ((A->nnz() + _nrow) / 2 + _nrow * _lsize - l_tails);
  _aj.reset(new MKL_INT[l_capacity]);
  _av.reset(new double[l_capacity]);
  if (_rsize) {
    _ai_r.reset(new MKL_INT[n + 1]);

    const MKL_INT r_tails =
        std::min(_rsize * (_rsize + 1) / 2, _nrow * (_nrow + 1) / 2);

    const MKL_INT r_capacity = _nrow * _rsize - r_tails;

    _aj_r.reset(new MKL_INT[r_capacity]);
    _av_r.reset(new double[r_capacity]);
  }

  return true;
}

bool incomplete_cholesky_fm::numeric_factorize(mkl_sparse_mat const *const A) {
  if (_rsize)
    return numeric_factorize<true>(A);
  else
    return numeric_factorize<false>(A);
}

template <bool buildR>
bool incomplete_cholesky_fm::numeric_factorize(mkl_sparse_mat const *const A) {
  auto ai = A->get_ai();
  auto aj = A->get_aj();
  auto av = A->get_av();
  const MKL_INT n = rows();
  const MKL_INT base = mkl_base();
  MKL_INT i, k_idx, A_k_idx, k, j_idx;

  auto compMax = [](const std::pair<MKL_INT, double> &v1,
                    const std::pair<MKL_INT, double> &v2) {
    return std::abs(v1.second) > std::abs(v2.second);
  };
  auto compMin = [](const std::pair<MKL_INT, double> &v1,
                    const std::pair<MKL_INT, double> &v2) {
    return std::abs(v1.second) < std::abs(v2.second);
  };

  utils::MaxHeap<std::pair<MKL_INT, double>, decltype(compMax)> max_heap_L(
      compMax);
  utils::MaxHeap<std::pair<MKL_INT, double>, decltype(compMax)> max_heap_R(
      compMax);
  utils::MaxHeap<std::pair<MKL_INT, double>, decltype(compMin)> min_heap(
      compMin);

  utils::CacheFriendlyVectors<std::pair<MKL_INT, MKL_INT>> jKRowU(n);
  utils::CacheFriendlyVectors<MKL_INT> jKRowR(n);

  std::vector<MKL_INT> curR(n, std::numeric_limits<MKL_INT>::max());
  std::vector<MKL_INT> curU(n, std::numeric_limits<MKL_INT>::max());

#ifdef USE_BOOST_LIB
  std::forward_list<std::pair<MKL_INT, double>,
                    boost::fast_pool_allocator<std::pair<MKL_INT, double>>>
      _rowVals;
#else
  std::forward_list<std::pair<MKL_INT, double>> _rowVals;
#endif
  MKL_INT list_size = 0;
  bool success_flag = false;
  int iter = 0;

  double shift = 0.;
  if (_shift) {
    // initialize shift
    double minDiag = std::numeric_limits<double>::max();
#pragma omp parallel for reduction(min : minDiag)
    for (i = 0; i < n; i++) {
      minDiag = std::min(minDiag, av[_diagPos[i]]);
    }
    if (minDiag <= 0.)
      shift = _initial_shift - minDiag;
    std::cout << "init shift: " << shift << std::endl;
  }

  do {
    _ai[0] = base;
    if constexpr (buildR) {
      _ai_r[0] = base;
    }
    for (i = 0; i < n; i++) {
      _rowVals.clear();
      auto rowIt = _rowVals.before_begin();
      k = _diagPos[i]; // get diagonal position of row i on A
      list_size = ai[i + 1] - base - k;
      // copy row i of A to _ai_i
      for (A_k_idx = _diagPos[i]; A_k_idx < ai[i + 1] - base; A_k_idx++) {
        rowIt = _rowVals.insert_after(
            rowIt, std::make_pair(aj[A_k_idx] - base, av[A_k_idx]));
      }

      // use n as the list end to prevent from branch prediction
      _rowVals.insert_after(rowIt, std::make_pair(n + base, 0));

      // compute _a_ij = _a_ij - sum_{j(i,n)}(_a_ki * _a_kj)
      for (auto &k_pair : jKRowU[i]) {
        k = k_pair.first;
        j_idx = k_pair.second;

        if (j_idx + 1 != _ai[k + 1] - base) {
          jKRowU.push_back(_aj[j_idx + 1] - base, {k, j_idx + 1});
        }
        if constexpr (buildR) {
          curU[k] = j_idx + 1;
        }

        const double aki = _av[j_idx];
        aij_update(_ai, _aj, _av, j_idx, k, base, aki, list_size, _rowVals);

        if constexpr (buildR) {
          aij_update(_ai_r, _aj_r, _av_r, curR[k], k, base, aki, list_size,
                     _rowVals);
        }
      }
      jKRowU.to_next();

      if constexpr (buildR) {
        for (auto k : jKRowR[i]) {
          j_idx = curR[k]++;
          const double aki = _av_r[j_idx];
          aij_update(_ai, _aj, _av, curU[k], k, base, aki, list_size, _rowVals);
          if (j_idx + 1 != _ai_r[k + 1] - base) {
            jKRowR.push_back(_aj_r[j_idx + 1] - base, k);
          }
        }
        jKRowR.to_next();
      }

      // treat a_ii
      rowIt = _rowVals.begin();
      k_idx = _ai[i] - base;
      _aj[k_idx] = rowIt->first + base;
      _av[k_idx] = (rowIt++)->second + shift;

      // restart with new shift if negative a_ii
      if (_av[k_idx] <= 0) {
        if (!_shift || ++iter > _nrestart)
          return false;
        shift = std::max(_initial_shift, 2. * shift);
        std::cout << "shift: " << shift << std::endl;
        jKRowU.clear();
        if constexpr (buildR)
          jKRowR.clear();
        break;
      }

      const double aii = std::sqrt(_av[k_idx]);
      _av[k_idx++] = aii;
      const MKL_INT row_l_size =
          ai[i + 1] - _diagPos[i] + std::min(_lsize, _nrow - i - 1);
      if (list_size < row_l_size) {
        for (int ii = 1; ii < list_size; ii++) {
          _aj[k_idx] = rowIt->first + base;
          _av[k_idx++] = rowIt->second / aii;
          rowIt++;
        }
        _ai[i + 1] = _ai[i] + list_size;
        if constexpr (buildR)
          _ai_r[i + 1] = _ai_r[i];
      } else {
        max_heap_L.clear();
        if constexpr (buildR)
          max_heap_R.clear();
          
        for (int ii = 1; ii < list_size; ii++) {
          rowIt->second /= aii;
          max_heap_L.push(*rowIt++);
          if (max_heap_L.size() > row_l_size - 1) {
            if constexpr (buildR) {
              max_heap_R.push(*max_heap_L.top());
              if (max_heap_R.size() > _rsize) {
                max_heap_R.pop();
              }
            }
            max_heap_L.pop();
          }
        }
        auto &heapL = max_heap_L.getHeap();
        std::sort(heapL.begin(), heapL.end(),
                  [](const std::pair<MKL_INT, double> &a,
                     const std::pair<MKL_INT, double> &b) {
                    return a.first < b.first;
                  });
        for (size_t ii = 0; ii < heapL.size(); ii++) {
          _aj[k_idx] = heapL[ii].first + base;
          _av[k_idx++] = heapL[ii].second;
        }
        _ai[i + 1] = _ai[i] + row_l_size;

        if constexpr (buildR) {
          k_idx = _ai_r[i] - base;
          auto &heapR = max_heap_R.getHeap();
          std::sort(heapR.begin(), heapR.end(),
                    [](const std::pair<MKL_INT, double> &a,
                       const std::pair<MKL_INT, double> &b) {
                      return a.first < b.first;
                    });
          for (size_t ii = 0; ii < heapR.size(); ii++) {
            _aj_r[k_idx] = heapR[ii].first + base;
            _av_r[k_idx++] = heapR[ii].second;
          }
          _ai_r[i + 1] = _ai_r[i] + max_heap_R.size();
          k_idx = _ai_r[i] - base;
          if (_ai_r[i + 1] - _ai_r[i] != 0) {
            jKRowR.push_back(_aj_r[k_idx] - base, i);
          }
          curR[i] = k_idx;
        }
      }

      // append row i to _aj[diagiI+1] row
      k_idx = _ai[i] - base;
      if (_ai[i + 1] - _ai[i] > 1) {
        jKRowU.push_back(_aj[k_idx + 1] - base, {i, k_idx + 1});
      }
      if constexpr (buildR) {
        curU[i] = k_idx + 1;
      }
    }
    if (i == n)
      success_flag = true;
  } while (!success_flag);
  _nnz = _ai[_nrow] - base;
  if (_mkl_base == SPARSE_INDEX_BASE_ONE)
    sp_fill();
  else
    to_one_based();
  return true;
}

} // namespace mkl_wrapper