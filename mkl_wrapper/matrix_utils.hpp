#pragma once

#include "utils.h"
#include <execution>
#include <iostream>
#include <memory>
#include <numeric>
#include <omp.h>
#include <tuple>

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
} // namespace matrix_utils

} // namespace matrix_utils