#pragma once

#include "BitVector.hpp"
#include "utils.h"
#include <chrono>
#include <execution>
#include <memory>
#include <numeric>
#include <omp.h>
#include <span>
#include <thread>
#include <tuple>
#include <type_traits>

#include "matrix_utils.hpp"

namespace matrix_utils {

/// @brief Forward-substitution algorithm for low triangular csr matrix L. Note
/// that the diagonal term is assumed to be 1
/// @tparam SIZE
/// @tparam R
/// @tparam C
/// @tparam V
/// @tparam VALTYPE
/// @param size
/// @param base
/// @param ai
/// @param aj
/// @param av
/// @param rhs
/// @param x
template <typename SIZE, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void ForwardSubstitution(const SIZE size, const int base, ROWTYPE const *ai,
                         COLTYPE const *aj, VALTYPE const *av,
                         VALTYPE const *const b, VALTYPE *const x) {
  ROWTYPE j;
  for (SIZE i = 0; i < size; i++) {
    x[i] = b[i];
    for (j = ai[i] - base; j < ai[i + 1] - base; j++) {
      x[i] -= av[j] * x[aj[j] - base];
    }
  }
}

/// @brief Forward-substitution algorithm for low triangular csr matrix L
/// obtained by the transpose of a strict upper triangular csr matrix. Note that
/// the diagonal term is assumed to be 1
/// @tparam SIZE
/// @tparam R
/// @tparam C
/// @tparam V
/// @tparam VALTYPE
/// @param size
/// @param base
/// @param ai
/// @param aj
/// @param av
/// @param rhs
/// @param x
template <typename SIZE, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void ForwardSubstitutionT(const SIZE size, const int base, ROWTYPE const *ai,
                          COLTYPE const *aj, VALTYPE const *av,
                          VALTYPE const *const b, VALTYPE *const x) {
  ROWTYPE j;
  std::copy(b, b + size, x);
  for (SIZE i = 0; i < size; i++) {
    for (j = ai[i] - base; j < ai[i + 1] - base; j++) {
      x[aj[j] - base] -= av[j] * x[i];
    }
  }
}

template <typename SIZE, typename ROWTYPE, typename COLTYPE, typename VALTYPE,
          typename VEC>
void LevelScheduleForwardSubstitution(const VEC &iperm, const VEC &prefix,
                                      const SIZE rows, const int base,
                                      ROWTYPE const *ai, COLTYPE const *aj,
                                      VALTYPE const *av, VALTYPE const *const b,
                                      VALTYPE *const x) {
#pragma omp parallel
  {
    for (int l = 0; l < prefix.size() - 1; l++) {
#pragma omp for
      for (SIZE i = prefix[l]; i < prefix[l + 1]; i++) {
        const SIZE idx = iperm[i] - base;
        x[idx] = b[idx];
        for (auto j = ai[idx] - base; j < ai[idx + 1] - base; j++) {
          x[idx] -= av[j] * x[aj[j] - base];
        }
      }
#pragma omp barrier
    }
  }
}

template <bool WithBarrier = true, typename SIZE = int, typename ROWTYPE = int,
          typename COLTYPE = int, typename VALTYPE = double>
class OptimizedForwardSubstitution {
public:
  OptimizedForwardSubstitution() : _nthreads{omp_get_max_threads()} {}

  void analysis(const SIZE rows, const int base, ROWTYPE const *ai,
                COLTYPE const *aj, VALTYPE const *av) {
    if constexpr (!WithBarrier)
      _bv.resize(rows);
    const auto nnz = ai[rows] - base;
    _reorderedMat.ai.resize(rows + 1);
    _reorderedMat.aj.resize(nnz);
    _reorderedMat.av.resize(nnz);
    _reorderedMat.base = base;
    _reorderedMat.rows = rows;
    _nthreads = omp_get_max_threads();
    matrix_utils::TopologicalSort2<matrix_utils::TriangularSolve::L>(
        rows, base, ai, aj, _iperm, _levels);
    _numLevels = _levels.size() - 1;
    _threadlevels.resize(_nthreads);
    _threadiperm.resize(rows);

#pragma omp parallel num_threads(_nthreads)
    {
      const int tid = omp_get_thread_num();
      const int nthreads = omp_get_num_threads();
      // #pragma omp single
      //       std::cout << "nthreads: " << nthreads << std::endl;

      _threadlevels[tid].resize(_numLevels + 1);
      _threadlevels[tid][0] = 0;

      for (COLTYPE l = 0; l < _numLevels; l++) {
        auto [start, end] = utils::LoadBalancedPartition(
            _iperm.data() + _levels[l], _iperm.data() + _levels[l + 1], tid,
            nthreads);
        const COLTYPE size = std::distance(start, end);
        // #pragma omp critical
        //         std::cout << "tid: " << tid << " , size: " << size <<
        //         std::endl;
        _threadlevels[tid][l + 1] = _threadlevels[tid][l] + size;
      }

#pragma omp barrier
#pragma omp single
      {
        COLTYPE size = 0;
        for (int tid = 1; tid < nthreads; tid++) {
          size += _threadlevels[tid - 1][_numLevels];
          _threadlevels[tid][0] = size;
        }
      }

      for (COLTYPE l = 0; l < _numLevels; l++) {
        _threadlevels[tid][l + 1] += _threadlevels[tid][0];
      }

#pragma omp barrier
      COLTYPE cur = _threadlevels[tid][0];

      for (COLTYPE l = 0; l < _numLevels; l++) {
        auto [start, end] = utils::LoadBalancedPartition(
            _iperm.data() + _levels[l], _iperm.data() + _levels[l + 1], tid,
            nthreads);
        for (auto it = start; it != end; it++) {
          _threadiperm[cur++] = *it;
        }
      }
    }
    matrix_utils::permuteRow(rows, base, ai, aj, av, _threadiperm.data(),
                             _reorderedMat.ai.data(), _reorderedMat.aj.data(),
                             _reorderedMat.av.data());
  }

  void operator()(const VALTYPE *const b, VALTYPE *const x) const {
    if constexpr (!WithBarrier) {
      _bv.clearAll();
    }
#pragma omp parallel num_threads(_nthreads)
    {
      const int tid = omp_get_thread_num();
      const int nthreads = omp_get_num_threads();
      for (COLTYPE l = 0; l < _numLevels; l++) {
        const COLTYPE start = _threadlevels[tid][l];
        const COLTYPE end = _threadlevels[tid][l + 1];
        // std::cout << "tid: " << tid << " , start: " << start
        //           << " , end: " << end << std::endl;
        for (COLTYPE i = start; i < end; i++) {
          const SIZE idx = _threadiperm[i] - _reorderedMat.base;
          // std::cout << _reorderedMat.ai[i] << " " << _reorderedMat.ai[i +
          // 1]
          //           << std::endl;
          x[idx] = b[idx];
          for (auto j = _reorderedMat.ai[i] - _reorderedMat.base;
               j < _reorderedMat.ai[i + 1] - _reorderedMat.base; j++) {
            const COLTYPE j_idx = _reorderedMat.aj[j] - _reorderedMat.base;

            if constexpr (!WithBarrier) {
              while (!_bv.get(j_idx)) {
                std::this_thread::yield();
                // sleep(0);
                // std::this_thread::sleep_for(std::chrono::nanoseconds(1));
                // std::cout << "j_idx: " << j_idx << std::endl;
                // continue;
              }
            }
            x[idx] -= _reorderedMat.av[j] * x[j_idx];
          }
          if constexpr (!WithBarrier) {
            _bv.set(idx);
          }
        }
        if constexpr (WithBarrier) {
#pragma omp barrier
        }
      }
    }
  }

protected:
  int _nthreads;
  std::vector<COLTYPE> _iperm;
  std::vector<COLTYPE> _levels;

  COLTYPE _numLevels;
  std::vector<std::vector<COLTYPE>>
      _threadlevels; // level prefix for each thread
  std::vector<COLTYPE> _threadiperm;
  CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE> _reorderedMat;

  mutable utils::BitVector<COLTYPE> _bv;
};
} // namespace matrix_utils