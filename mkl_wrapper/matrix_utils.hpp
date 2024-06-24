#pragma once

#include "utils.h"
#include <execution>
#include <iostream>
#include <memory>
#include <numeric>
#include <omp.h>
#include <tuple>
#include <type_traits>

namespace matrix_utils {

/// @brief A serial compressed sparse row matrix transpose function
/// @param rows number of rows of the matrix about to be transposed
/// @param cols number of columns of the matrix about to be transposed
/// @param nnz number of nonzeros of the matrix about to be transposed
/// @param base matrix index base (0 or 1)
/// @param ai row index
/// @param aj column index
/// @param av value vector
template <typename SIZE, typename R, typename C, typename V>
auto SerialTranspose(const SIZE rows, const SIZE cols, const SIZE nnz,
                     const SIZE base, const R &ai, const C &aj, const V &av) {
  using ROWTYPE = typename std::decay<decltype(ai[0])>::type;
  using COLTYPE = typename std::decay<decltype(aj[0])>::type;
  using VALTYPE = typename std::decay<decltype(av[0])>::type;

  const SIZE cols_transpose = rows;
  const SIZE rows_transpose = cols;
  std::shared_ptr<ROWTYPE[]> ai_transpose(
      new ROWTYPE[rows_transpose + 2]); // prevent from branching
  std::shared_ptr<COLTYPE[]> aj_transpose(new COLTYPE[nnz]);
  std::shared_ptr<VALTYPE[]> av_transpose(new VALTYPE[nnz]);

  ai_transpose[0] = base;
  std::fill_n(std::execution::seq, ai_transpose.get() + 1, rows_transpose + 1,
              0);

  // assign size of row i to ai[i+1]
  for (size_t i = 0; i < nnz; i++) {
    ai_transpose[aj[i] - base + 2]++;
  }

  std::inclusive_scan(ai_transpose.get(),
                      ai_transpose.get() + rows_transpose + 2,
                      ai_transpose.get());

  for (SIZE i = 0; i < rows; i++) {
    for (COLTYPE j = ai[i] - base; j < ai[i + 1] - base; j++) {
      const COLTYPE idx = ai_transpose[aj[j] - base + 1]++ - base;
      aj_transpose[idx] = i + base;
      av_transpose[idx] = av[j];
    }
  }
  return std::make_tuple(ai_transpose, aj_transpose, av_transpose);
}

/// @brief A parallel compressed sparse row matrix transpose function
/// @param rows number of rows of the matrix about to be transposed
/// @param cols number of columns of the matrix about to be transposed
/// @param nnz number of nonzeros of the matrix about to be transposed
/// @param base matrix index base (0 or 1)
/// @param ai row index
/// @param aj column index
/// @param av value vector
template <typename SIZE, typename R, typename C, typename V>
auto ParallelTranspose(const SIZE rows, const SIZE cols, const SIZE nnz,
                       const SIZE base, const R &ai, const C &aj, const V &av) {
  using ROWTYPE = typename std::decay<decltype(ai[0])>::type;
  using COLTYPE = typename std::decay<decltype(aj[0])>::type;
  using VALTYPE = typename std::decay<decltype(av[0])>::type;
  const SIZE cols_transpose = rows;
  const SIZE rows_transpose = cols;
  std::shared_ptr<ROWTYPE[]> ai_transpose(new ROWTYPE[rows_transpose + 1]);
  std::shared_ptr<COLTYPE[]> aj_transpose(new COLTYPE[nnz]);
  std::shared_ptr<VALTYPE[]> av_transpose(new VALTYPE[nnz]);

  ai_transpose[0] = base;

  std::vector<std::unique_ptr<ROWTYPE[]>> threadPrefixSum(
      omp_get_max_threads());

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();

    auto [start, end] = utils::LoadPrefixBalancedPartition(
        ai.get(), ai.get() + rows, tid, nthreads);
    threadPrefixSum[tid].reset(new ROWTYPE[rows_transpose]());

    for (auto it = start; it < end; it++) {
      for (ROWTYPE j = *it - base; j < *(it + 1) - base; j++) {
        threadPrefixSum[tid][aj[j] - base]++;
      }
    }

#pragma omp barrier
#pragma omp for
    for (SIZE rowID = 0; rowID < rows_transpose; rowID++) {
      ai_transpose[rowID + 1] = 0;
      for (int t = 0; t < nthreads; t++) {
        ai_transpose[rowID + 1] += threadPrefixSum[t][rowID];
      }
    }

#pragma omp barrier

// may be optimized by a parallel scan
#pragma omp master
    {

      std::inclusive_scan(ai_transpose.get(),
                          ai_transpose.get() + rows_transpose + 1,
                          ai_transpose.get());
    }

#pragma omp barrier
#pragma omp for
    for (SIZE rowID = 0; rowID < rows_transpose; rowID++) {
      ROWTYPE tmp = threadPrefixSum[0][rowID];
      threadPrefixSum[0][rowID] = ai_transpose[rowID];
      for (int t = 1; t < nthreads; t++) {
        std::swap(threadPrefixSum[t][rowID], tmp);
        threadPrefixSum[t][rowID] += threadPrefixSum[t - 1][rowID];
      }
    }

#pragma omp barrier

    for (auto it = start; it < end; it++) {
      for (ROWTYPE j = *it - base; j < *(it + 1) - base; j++) {
        const SIZE rowID = it - ai.get();
        const COLTYPE idx = threadPrefixSum[tid][aj[j] - base]++ - base;
        aj_transpose[idx] = rowID + base;
        av_transpose[idx] = av[j];
      }
    }
  }
  return std::make_tuple(ai_transpose, aj_transpose, av_transpose);
}

/// @brief A parallel compressed sparse row matrix transpose function
/// @param rows number of rows of the matrix about to be transposed
/// @param cols number of columns of the matrix about to be transposed
/// @param nnz number of nonzeros of the matrix about to be transposed
/// @param base matrix index base (0 or 1)
/// @param ai row index
/// @param aj column index
/// @param av value vector
template <typename SIZE, typename R, typename C, typename V>
auto ParallelTranspose2(const SIZE rows, const SIZE cols, const SIZE nnz,
                        const SIZE base, const R &ai, const C &aj,
                        const V &av) {
  using ROWTYPE = typename std::decay<decltype(ai[0])>::type;
  using COLTYPE = typename std::decay<decltype(aj[0])>::type;
  using VALTYPE = typename std::decay<decltype(av[0])>::type;
  const SIZE cols_transpose = rows;
  const SIZE rows_transpose = cols;
  std::shared_ptr<ROWTYPE[]> ai_transpose(new ROWTYPE[rows_transpose + 1]());
  ai_transpose[0] = base;
  std::shared_ptr<COLTYPE[]> aj_transpose(new COLTYPE[nnz]);
  std::shared_ptr<VALTYPE[]> av_transpose(new VALTYPE[nnz]);
  omp_set_num_threads(3);
  std::vector<std::unique_ptr<ROWTYPE[]>> threadPrefixSum(
      omp_get_max_threads());

  std::vector<ROWTYPE> prefix(omp_get_max_threads() + 1, 0);
  prefix[0] = base;

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();

    auto [start, end] = utils::LoadPrefixBalancedPartition(
        ai.get(), ai.get() + rows, tid, nthreads);
    threadPrefixSum[tid].reset(new ROWTYPE[rows_transpose]());

    for (auto it = start; it < end; it++) {
      for (ROWTYPE j = *it - base; j < *(it + 1) - base; j++) {
        threadPrefixSum[tid][aj[j] - base]++;
      }
    }

#pragma omp barrier
    auto [startt, endt] = utils::LoadBalancedPartition(
        ai_transpose.get(), ai_transpose.get() + rows_transpose, tid, nthreads);
    for (auto it = startt; it < endt; it++) {
      const ROWTYPE rowID = it - ai_transpose.get();
      ai_transpose[rowID + 1] = (it == startt) ? 0 : ai_transpose[rowID];
      for (int t = 0; t < nthreads; t++) {
        ai_transpose[rowID + 1] += threadPrefixSum[t][rowID];
      }
    }
    prefix[tid + 1] = ai_transpose[end - ai.get()];

#pragma omp barrier

#pragma omp master
    std::inclusive_scan(prefix.begin(), prefix.end(), prefix.begin());

#pragma omp barrier
    for (auto it = startt; it < endt; it++) {
      const ROWTYPE rowID = it - ai_transpose.get();
      ai_transpose[rowID + 1] += prefix[tid];
    }

#pragma omp barrier
    for (auto it = startt; it < endt; it++) {
      const ROWTYPE rowID = it - ai_transpose.get();
      ROWTYPE tmp = threadPrefixSum[0][rowID];
      threadPrefixSum[0][rowID] = ai_transpose[rowID];
      for (int t = 1; t < nthreads; t++) {
        std::swap(threadPrefixSum[t][rowID], tmp);
        threadPrefixSum[t][rowID] += threadPrefixSum[t - 1][rowID];
      }
    }

#pragma omp barrier

    for (auto it = start; it < end; it++) {
      for (ROWTYPE j = *it - base; j < *(it + 1) - base; j++) {
        const ROWTYPE rowID = it - ai.get();
        const COLTYPE idx = threadPrefixSum[tid][aj[j] - base]++ - base;
        aj_transpose[idx] = rowID + base;
        av_transpose[idx] = av[j];
      }
    }
  }
  return std::make_tuple(ai_transpose, aj_transpose, av_transpose);
}

template <class Array>
using array_value_type = std::decay_t<decltype(std::declval<Array &>()[0])>;

/// @brief Forward-substitution algorithm for low triangular csr matrix L. Note
/// that the diagonal term must be provided
/// @tparam SIZE
/// @tparam R
/// @tparam C
/// @tparam V
/// @tparam VALTYPE
/// @param rows
/// @param cols
/// @param nnz
/// @param base
/// @param ai
/// @param aj
/// @param av
/// @param rhs
/// @param x
template <typename SIZE, typename R, typename C, typename V, typename VALTYPE>
std::enable_if_t<!std::is_same_v<array_value_type<V>, VALTYPE>, void>
ForwardSubstitution(const SIZE rows, const SIZE cols, const SIZE nnz,
                    const SIZE base, const R &ai, const C &aj, const V &av,
                    VALTYPE const *const rhs, VALTYPE *const x) {

  using ROWTYPE = typename std::decay<decltype(ai[0])>::type;
  using COLTYPE = typename std::decay<decltype(aj[0])>::type;
  ROWTYPE j;
  for (SIZE i = 0; i < rows; i++) {
    x[i] = b[i];
    for (j = ai[i] - base; j < ai[i + 1] - base - 1; j++) {
      x[i] -= av[j] * x[aj[j] - base];
    }
    x[i] /= av[j];
  }
}

template <typename SIZE, typename R, typename C, typename VEC>
void TopologicalSort(const SIZE nodes, const SIZE base, const R &ai,
                     const C &aj, VEC &iperm, VEC &prefix) {
  iperm.reserve(nodes);
  iperm.clear();
  prefix.reserve(std::max(1, nodes / 100));
  prefix.resize(1);
  prefix[0] = 0;
  std::vector<int> degrees(nodes);
  for (SIZE i = 0; i < nodes; i++) {
    prefix.push_back(prefix.back());
    degrees[i] = ai[i + 1] - ai[i];
    if (ai[i + 1] - ai[i] == 1) {
      iperm.push_back(i + base);
      prefix.back()++;
    }
  }
  size_t level = 0;
  while (iperm.size() != nodes) {
    prefix.push_back(prefix.back());
    for (size_t i = prefix[level]; i < prefix[level + 1]; i++) {
      const auto idx = iperm[i] - base;
      for (auto j = ai[idx] - base; j < ai[idx + 1] - base; j++) {
        if (--degrees[aj[j] - base] == 1) {
          iperm.push_back(aj[j]);
          prefix.back()++;
        }
      }
    }
  }
}

template <typename SIZE, typename R, typename C, typename V, typename VEC,
          typename VALTYPE>
std::enable_if_t<!std::is_same_v<array_value_type<V>, VALTYPE>, void>
LevelScheduleForwardSubstitution(const VEC &iperm, const VEC &prefix,
                                 const SIZE rows, const SIZE cols,
                                 const SIZE nnz, const SIZE base, const R &ai,
                                 const C &aj, const V &av,
                                 VALTYPE const *const rhs, VALTYPE *const x) {
  using ROWTYPE = typename std::decay<decltype(ai[0])>::type;
  using COLTYPE = typename std::decay<decltype(aj[0])>::type;

#pragma omp parallel
  {
    ROWTYPE j;
    for (int l = 0; l < prefix.size() - 1; l++) {
#pragma omp for
      for (SIZE i = prefix[l]; i < prefix[l + 1]; i++) {
        x[i] = b[i];
        for (j = ai[i] - base; j < ai[i + 1] - base - 1; j++) {
          x[i] -= av[j] * x[aj[j] - base];
        }
        x[i] /= av[j];
      }
#pragma omp barrier
    }
  }
}

} // namespace matrix_utils