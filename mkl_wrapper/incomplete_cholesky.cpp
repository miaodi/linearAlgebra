#include "incomplete_cholesky.h"
#include <cmath>
#include <cstdio>
#include <execution>
#include <forward_list>
#include <fstream>
#include <iostream>
#include <mkl_sparse_handle.h>
#include <mkl_spblas.h>

namespace mkl_wrapper {
incomplete_cholesky_k::incomplete_cholesky_k() : incomplete_fact() {}

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
    std::copy(std::execution::par_unseq, A->get_ai().get(),
              A->get_ai().get() + n + 1, _ai.get());
    std::copy(std::execution::par_unseq, A->get_aj().get(),
              A->get_aj().get() + _nnz, _aj.get());
  } else {
    _ai[0] = base;
    MKL_INT aj_size = A->nnz();
    _aj.reset(new MKL_INT[aj_size]);
    auto av_levels = std::make_unique<MKL_INT[]>(aj_size);
    std::forward_list<std::pair<MKL_INT, MKL_INT>> _rowLevels;
    MKL_INT list_size = 0;
    MKL_INT j;
    MKL_INT k;
    std::vector<MKL_INT> jStart(n);

    for (MKL_INT i = 0; i < n; i++) {
      jStart[i] = _ai[i] - base + 1;
      // initialize levels
      auto rowIt = _rowLevels.before_begin();
      k = _diagPos[i];
      list_size = k;
      for (; k != ai[i + 1] - base; k++) {
        rowIt = _rowLevels.insert_after(rowIt, std::make_pair(aj[k] - base, 0));
      }
      list_size = k - list_size;

      for (k = 0; k < i; k++) {
        if (_aj[jStart[k]] - base != i || jStart[k] == _ai[k + 1] - base)
          continue;
        j = jStart[k]++;
        // std::cout<<"hello\n";
        auto eij = _rowLevels.begin();
        auto lik = av_levels[j];
        MKL_INT nextIdx = std::next(eij) == _rowLevels.end()
                              ? std::numeric_limits<MKL_INT>::max()
                              : std::next(eij)->first;
        for (; j != _ai[k + 1] - base; j++) {
          // std::cout << i << ":" << j << " insert: ";
          while (nextIdx <= _aj[j] - base) {
            eij = std::next(eij);
            nextIdx = std::next(eij) == _rowLevels.end()
                          ? std::numeric_limits<MKL_INT>::max()
                          : std::next(eij)->first;
          }
          // std::cout << "inner j: " << j << std::endl;
          if (lik + av_levels[j] + 1 <= _level) {
            if (eij->first == _aj[j] - base) {
              if (eij->second > lik + av_levels[j] + 1) {
                eij->second = lik + av_levels[j] + 1;
              }
            } else {
              eij = _rowLevels.insert_after(
                  eij, std::make_pair(_aj[j] - base, lik + av_levels[j] + 1));
              nextIdx = std::next(eij) == _rowLevels.end()
                            ? std::numeric_limits<MKL_INT>::max()
                            : std::next(eij)->first;
              list_size++;
              // std::cout << _aj[j] - base << std::endl;
            }
            // std::cout << eij->first + base << " ";
          }
        }
      }
      // std::cout << "step 3: \n";
      rowIt = _rowLevels.begin();
      // std::cout << "i: " << i << " j: " << rowIt->first << std::endl;
      MKL_INT pos = _ai[i] - base;
      while (rowIt != _rowLevels.end()) {
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
        _aj[pos] = rowIt->first + base;
        av_levels[pos++] = rowIt->second;
        // std::cout << rowIt->first + base << " ";
        rowIt++;
      }
      // std::cout<<"heihei\n";
      // std::cout << std::endl;
      _ai[i + 1] = _ai[i] + list_size;
      _rowLevels.clear();
      // std::cout << _ai[i + 1] << std::endl;
    }
    // std::abort();
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
  std::vector<MKL_INT> jStart(n);
  std::transform(_ai.get(), _ai.get() + _nrow, jStart.begin(),
                 [base](const MKL_INT i) { return i - base + 1; });
  // std::cout << jStart.size() << std::endl;

  for (MKL_INT i = 0; i < n; i++) {

    k_idx = _ai[i] - base;
    A_k_idx = _diagPos[i];

    // std::cout << "hello\n";
    for (; k_idx != _ai[i + 1] - base; k_idx++) {
      if (A_k_idx == ai[i + 1] - base || _aj[k_idx] != aj[A_k_idx]) {
        _av[k_idx] = 0;
      } else {
        _av[k_idx] = av[A_k_idx++];
      }
    }

    for (k = 0; k < i; k++) {
      if (_aj[jStart[k]] - base != i || jStart[k] == _ai[k + 1] - base)
        continue;
      j_idx = jStart[k]++;

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

    k_idx = _ai[i] - base;
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

bool incomplete_cholesky_k::solve(double const *const b, double *const x) {
  sparse_operation_t transA = SPARSE_OPERATION_TRANSPOSE;
  _mkl_descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  _mkl_descr.mode = SPARSE_FILL_MODE_UPPER;
  _mkl_descr.diag = SPARSE_DIAG_NON_UNIT;
  mkl_sparse_d_trsv(transA, 1.0, _mkl_mat, _mkl_descr, b, _interm_vec.data());

  transA = SPARSE_OPERATION_NON_TRANSPOSE;
  mkl_sparse_d_trsv(transA, 1.0, _mkl_mat, _mkl_descr, _interm_vec.data(), x);
  return true;
}
} // namespace mkl_wrapper