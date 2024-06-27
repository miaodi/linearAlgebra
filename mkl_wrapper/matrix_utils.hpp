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

template <class T> auto find_address_of(T &&p) { return p.get(); }

template <typename T> auto find_address_of(T *p) { return p; }

template <typename T> auto find_address_of(const std::vector<T> &p) {
  return p.cbegin();
}

template <typename R, typename C, typename V> struct CSRMatrix {
  using RowType = R;
  using ColType = C;
  using ValType = V;
  ColType rows;
  ColType cols;
  ColType base;
  RowType nnz;
  size_t ai_size{0};
  size_t aj_size{0};
  size_t av_size{0};
  std::shared_ptr<RowType[]> ai;
  std::shared_ptr<ColType[]> aj;
  std::shared_ptr<ValType[]> av;

  CSRMatrix() = default;
};

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
  using ROWTYPE = typename std::decay_t<decltype(ai[0])>;
  using COLTYPE = typename std::decay_t<decltype(aj[0])>;
  using VALTYPE = typename std::decay_t<decltype(av[0])>;

  const SIZE cols_transpose = rows;
  const SIZE rows_transpose = cols;
  std::shared_ptr<ROWTYPE[]> ai_transpose(
      new ROWTYPE[rows_transpose + 2]); // prevent from branching
  std::shared_ptr<COLTYPE[]> aj_transpose(new COLTYPE[nnz]);
  std::shared_ptr<VALTYPE[]> av_transpose(new VALTYPE[nnz]);

  ai_transpose[0] = base;
  std::fill_n(std::execution::seq, find_address_of(ai_transpose) + 1,
              rows_transpose + 1, 0);

  // assign size of row i to ai[i+1]
  for (size_t i = 0; i < nnz; i++) {
    ai_transpose[aj[i] - base + 2]++;
  }

  std::inclusive_scan(find_address_of(ai_transpose),
                      find_address_of(ai_transpose) + rows_transpose + 2,
                      find_address_of(ai_transpose));

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
  using ROWTYPE = typename std::decay_t<decltype(ai[0])>;
  using COLTYPE = typename std::decay_t<decltype(aj[0])>;
  using VALTYPE = typename std::decay_t<decltype(av[0])>;
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
  using ROWTYPE = typename std::decay_t<decltype(ai[0])>;
  using COLTYPE = typename std::decay_t<decltype(aj[0])>;
  using VALTYPE = typename std::decay_t<decltype(av[0])>;
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
        find_address_of(ai), find_address_of(ai) + rows, tid, nthreads);
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
    prefix[tid + 1] = ai_transpose[end - find_address_of(ai)];

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
                    VALTYPE const *const b, VALTYPE *const x) {

  using ROWTYPE = typename std::decay_t<decltype(ai[0])>;
  using COLTYPE = typename std::decay_t<decltype(aj[0])>;
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
                                 VALTYPE const *const b, VALTYPE *const x) {
  using ROWTYPE = typename std::decay_t<decltype(ai[0])>;
  using COLTYPE = typename std::decay_t<decltype(aj[0])>;

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

template <typename SIZE, typename R, typename C, typename VEC>
bool DiagonalPosition(const SIZE rows, const SIZE base, const R &ai,
                      const C &aj, VEC &diag) {
  using ROWTYPE = typename std::decay_t<decltype(ai[0])>;
  using COLTYPE = typename std::decay_t<decltype(aj[0])>;

  diag.resize(rows);
  volatile bool missing_diag = false;
#pragma omp parallel for shared(missing_diag)
  for (SIZE i = 0; i < rows; i++) {
    if (missing_diag)
      continue;
    auto mid =
        std::lower_bound(find_address_of(aj) + ai[i] - base,
                         find_address_of(aj) + ai[i + 1] - base, i + base);
    if (*mid != i + base) {
      std::cerr << "Could not find diagonal!" << std::endl;
      missing_diag = true;
    }
    diag[i] = mid - find_address_of(aj);
  }
  if (missing_diag)
    return false;
  return true;
}

template <typename SIZE, typename R, typename C, typename V>
void SplitLDU(
    const SIZE rows, const SIZE base, const R &ai, const C &aj, const V &av,
    CSRMatrix<array_value_type<R>, array_value_type<C>, array_value_type<V>> &L,
    std::vector<array_value_type<V>> &D,
    CSRMatrix<array_value_type<R>, array_value_type<C>, array_value_type<V>>
        &U) {
  using RowType = CSRMatrix<array_value_type<R>, array_value_type<C>,
                            array_value_type<V>>::RowType;
  using ColType = CSRMatrix<array_value_type<R>, array_value_type<C>,
                            array_value_type<V>>::ColType;
  using ValType = CSRMatrix<array_value_type<R>, array_value_type<C>,
                            array_value_type<V>>::ValType;

  L.rows = rows;
  L.cols = rows;
  L.base = base;
  if (L.ai_size < rows + 1) {
    L.ai.reset(new RowType[rows + 1]);
    L.ai_size = rows + 1;
  }

  U.rows = rows;
  U.cols = rows;
  U.base = base;
  if (U.ai_size < rows + 1) {
    U.ai.reset(new RowType[rows + 1]);
    U.ai_size = rows + 1;
  }

  L.ai[0] = base;
  U.ai[0] = base;
  D.resize(rows);
  std::vector<RowType> diag(rows);
  std::vector<std::pair<RowType, RowType>> LU_prefix(omp_get_max_threads() + 1);
  LU_prefix[0] = {base, base};
  volatile bool missing_diag = false;
#pragma omp parallel shared(missing_diag)
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    auto [start, end] = utils::LoadPrefixBalancedPartition(
        find_address_of(ai), find_address_of(ai) + rows, tid, nthreads);
    LU_prefix[tid + 1].first = 0;
    LU_prefix[tid + 1].second = 0;
    for (auto it = start; it < end; it++) {
      SIZE i = it - find_address_of(ai);

      auto mid =
          std::lower_bound(find_address_of(aj) + *it - base,
                           find_address_of(aj) + *(it + 1) - base, i + base);
      const bool zero_diag = *mid != i + base;
      diag[i] = mid - find_address_of(aj);
      D[i] = zero_diag ? 0 : av[diag[i]];
      const RowType L_size = mid - (find_address_of(aj) + *it - base);
      LU_prefix[tid + 1].first += L_size;
      L.ai[i + 1] = LU_prefix[tid + 1].first;
      const RowType U_size = *(it + 1) - *it - L_size - (zero_diag ? 0 : 1);
      LU_prefix[tid + 1].second += U_size;
      U.ai[i + 1] = LU_prefix[tid + 1].second;
    }
#pragma omp barrier
#pragma omp master
    {
      for (size_t i = 1; i < LU_prefix.size(); i++) {
        LU_prefix[i].first += LU_prefix[i - 1].first;
        LU_prefix[i].second += LU_prefix[i - 1].second;
      }
      L.nnz = LU_prefix[nthreads].first;
      if (L.aj_size < L.nnz) {
        L.aj.reset(new ColType[L.nnz]);
        L.aj_size = L.nnz;
      }
      if (L.av_size < L.nnz) {
        L.av.reset(new ValType[L.nnz]);
        L.av_size = L.nnz;
      }

      U.nnz = LU_prefix[nthreads].second;
      if (U.aj_size < U.nnz) {
        U.aj.reset(new ColType[U.nnz]);
        U.aj_size = U.nnz;
      }
      if (U.av_size < U.nnz) {
        U.av.reset(new ValType[U.nnz]);
        U.av_size = U.nnz;
      }
    }

#pragma omp barrier
    RowType L_pos = LU_prefix[tid].first - base;
    RowType U_pos = LU_prefix[tid].second - base;
    for (auto it = start; it < end; it++) {
      SIZE i = it - find_address_of(ai);
      const bool zero_diag = aj[diag[i]] - base != i;
      L.ai[i + 1] += LU_prefix[tid].first;
      U.ai[i + 1] += LU_prefix[tid].second;

      for (RowType j = *it - base; j < diag[i]; j++) {
        L.aj[L_pos] = aj[j];
        L.av[L_pos++] = av[j];
      }
      for (RowType j = diag[i] + (zero_diag ? 0 : 1); j < *(it + 1) - base;
           j++) {
        U.aj[U_pos] = aj[j];
        U.av[U_pos++] = av[j];
      }
    }
  }
}

} // namespace matrix_utils