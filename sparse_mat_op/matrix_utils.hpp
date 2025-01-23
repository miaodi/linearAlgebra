#pragma once

#include "sparse_mat_traits.hpp"
#include "utils.h"
#include <execution>
#include <iostream>
#include <memory>
#include <numeric>
#include <omp.h>
#include <span>
#include <tuple>
#include <type_traits>

namespace matrix_utils {

template <typename T> auto find_address_of(T &&p) { return p.get(); }

template <typename T> auto find_address_of(T *p) { return p; }

template <typename T> auto find_address_of(const std::vector<T> &p) {
  return p.cbegin();
}

template <typename T> auto find_address_of(std::vector<T> &p) {
  return p.begin();
}

template <typename T> T const *find_address_of(std::span<const T> p) {
  return p.data();
}

/// @brief only holds (does not own no need to destory) the raw pointers of the
/// CSR matrix, not resizable
/// @tparam R
/// @tparam C
/// @tparam V
template <typename R, typename C, typename V> struct CSRMatrixRawPtr {
  using ROWTYPE = R;
  using COLTYPE = C;
  using VALTYPE = V;

  COLTYPE rows;
  COLTYPE cols;
  ROWTYPE const *ai;
  COLTYPE const *aj;
  VALTYPE const *av;

  ROWTYPE Base() const { return ai[0]; }
  ROWTYPE NNZ() const { return ai[rows] - ai[0]; }

  ROWTYPE const *AI() const { return ai; }
  COLTYPE const *AJ() const { return aj; }
  VALTYPE const *AV() const { return av; }

  CSRMatrixRawPtr() = default;
};

template <typename R, typename C, typename V> struct CSRMatrix {
  using ROWTYPE = R;
  using COLTYPE = C;
  using VALTYPE = V;
  COLTYPE rows;
  COLTYPE cols;

  size_t ai_size{0};
  size_t aj_size{0};
  size_t av_size{0};
  std::shared_ptr<ROWTYPE[]> ai;
  std::shared_ptr<COLTYPE[]> aj;
  std::shared_ptr<VALTYPE[]> av;

  ROWTYPE Base() const { return ai[0]; }
  ROWTYPE NNZ() const { return ai[rows] - ai[0]; }

  ROWTYPE const *AI() const { return ai.get(); }
  COLTYPE const *AJ() const { return aj.get(); }
  VALTYPE const *AV() const { return av.get(); }

  CSRMatrix() = default;
};

template <typename R, typename C, typename V> struct CSRMatrixVec {
  using ROWTYPE = R;
  using COLTYPE = C;
  using VALTYPE = V;
  COLTYPE rows;
  COLTYPE cols;

  std::vector<ROWTYPE> ai;
  std::vector<COLTYPE> aj;
  std::vector<VALTYPE> av;

  CSRMatrixVec() = default;

  ROWTYPE Base() const { return ai[0]; }
  ROWTYPE NNZ() const { return ai[rows] - ai[0]; }

  ROWTYPE const *AI() const { return ai.data(); }
  COLTYPE const *AJ() const { return aj.data(); }
  VALTYPE const *AV() const { return av.data(); }

  template <class Archive> void serialize(Archive &ar) { ar(ai, aj, av); }
};

template <typename CSRMatrixType, bool preserve = false>
void ResizeCSRAI(CSRMatrixType &mat, const size_t size) {
  static_assert(CSRResizable<CSRMatrixType>::value,
                "CSRMatrixType must have a resizable method");
  if constexpr (std::is_same_v<
                    decltype(mat.ai),
                    std::shared_ptr<typename CSRMatrixType::ROWTYPE[]>>) {
    if (mat.ai_size < size || mat.ai == nullptr) {
      std::shared_ptr<typename CSRMatrixType::ROWTYPE[]> tmp(
          new CSRMatrixType::ROWTYPE[size]);
      if constexpr (preserve) {
        std::copy(mat.ai.get(), mat.ai.get() + mat.ai_size, tmp.get());
      }
      std::swap(mat.ai, tmp);
      mat.ai_size = size;
    }
  } else if constexpr (std::is_same_v<
                           decltype(mat.ai),
                           std::vector<typename CSRMatrixType::ROWTYPE>>) {
    mat.ai.resize(size);
  }
}

template <typename CSRMatrixType, bool preserve = false>
void ResizeCSRAJ(CSRMatrixType &mat, const size_t size) {
  static_assert(CSRResizable<CSRMatrixType>::value,
                "CSRMatrixType must have a resizable method");
  if constexpr (std::is_same_v<
                    decltype(mat.aj),
                    std::shared_ptr<typename CSRMatrixType::COLTYPE[]>>) {
    if (mat.aj_size < size || mat.aj == nullptr) {
      std::shared_ptr<typename CSRMatrixType::COLTYPE[]> tmp(
          new CSRMatrixType::COLTYPE[size]);
      if constexpr (preserve) {
        std::copy(mat.aj.get(), mat.aj.get() + mat.aj_size, tmp.get());
      }
      std::swap(mat.aj, tmp);
      mat.aj_size = size;
    }
  } else if constexpr (std::is_same_v<
                           decltype(mat.aj),
                           std::vector<typename CSRMatrixType::COLTYPE>>) {
    mat.aj.resize(size);
  }
}

template <typename CSRMatrixType, bool preserve = false>
void ResizeCSRAV(CSRMatrixType &mat, const size_t size) {
  static_assert(CSRResizable<CSRMatrixType>::value,
                "CSRMatrixType must have a resizable method");
  if constexpr (std::is_same_v<
                    decltype(mat.av),
                    std::shared_ptr<typename CSRMatrixType::VALTYPE[]>>) {
    if (mat.av_size < size || mat.av == nullptr) {
      std::shared_ptr<typename CSRMatrixType::VALTYPE[]> tmp(
          new CSRMatrixType::VALTYPE[size]);
      if constexpr (preserve) {
        std::copy(mat.av.get(), mat.av.get() + mat.av_size, tmp.get());
      }
      std::swap(mat.av, tmp);
      mat.av_size = size;
    }
  } else if constexpr (std::is_same_v<
                           decltype(mat.av),
                           std::vector<typename CSRMatrixType::VALTYPE>>) {
    mat.av.resize(size);
  }
}

template <typename ROWTYPE = int, typename COLTYPE = int,
          typename VALTYPE = double>
decltype(auto) AllocateCSRData(const COLTYPE rows, const ROWTYPE nnz) {

  std::shared_ptr<ROWTYPE[]> ai(new ROWTYPE[rows + 1]);
  std::shared_ptr<COLTYPE[]> aj(new COLTYPE[nnz]);
  std::shared_ptr<VALTYPE[]> av(new VALTYPE[nnz]);
  return std::make_tuple(ai, aj, av);
}

/// @brief A serial compressed sparse row matrix transpose function
/// @param rows number of rows of the matrix about to be transposed
/// @param cols number of columns of the matrix about to be transposed
/// @param nnz number of nonzeros of the matrix about to be transposed
/// @param base matrix index base (0 or 1)
/// @param ai row index
/// @param aj column index
/// @param av value vector
/// @param ai_transpose row index of transpose matrix
/// @param aj_transpose column index transpose matrix
/// @param av_transpose value vector transpose matrix
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void SerialTranspose(const COLTYPE rows, const COLTYPE cols, const int base,
                     ROWTYPE const *ai, COLTYPE const *aj, VALTYPE const *av,
                     ROWTYPE *ai_transpose, COLTYPE *aj_transpose,
                     VALTYPE *av_transpose) {
  const bool update_av = av_transpose != nullptr && av != nullptr;
  const COLTYPE cols_transpose = rows;
  const COLTYPE rows_transpose = cols;
  const auto nnz = ai[rows] - base;

  ai_transpose[0] = base;
  std::fill_n(std::execution::seq, ai_transpose + 1, rows_transpose, 0);

  // assign size of row i to ai[i+1]
  for (auto i = 0; i < nnz; i++) {
    if (aj[i] - base + 2 < rows_transpose + 1)
      ai_transpose[aj[i] - base + 2]++;
  }

  std::inclusive_scan(ai_transpose, ai_transpose + rows_transpose + 1,
                      ai_transpose);

  for (COLTYPE i = 0; i < rows; i++) {
    for (COLTYPE j = ai[i] - base; j < ai[i + 1] - base; j++) {
      const COLTYPE idx = ai_transpose[aj[j] - base + 1]++ - base;
      aj_transpose[idx] = i + base;
      if (update_av)
        av_transpose[idx] = av[j];
    }
  }
}

/// @brief A parallel compressed sparse row matrix transpose function
/// @param rows number of rows of the matrix about to be transposed
/// @param cols number of columns of the matrix about to be transposed
/// @param nnz number of nonzeros of the matrix about to be transposed
/// @param base matrix index base (0 or 1)
/// @param ai row index
/// @param aj column index
/// @param av value vector
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void ParallelTranspose(const COLTYPE rows, const COLTYPE cols, const int base,
                       ROWTYPE const *ai, COLTYPE const *aj, VALTYPE const *av,
                       ROWTYPE *ai_transpose, COLTYPE *aj_transpose,
                       VALTYPE *av_transpose) {
  const COLTYPE cols_transpose = rows;
  const COLTYPE rows_transpose = cols;
  const auto nnz = ai[rows] - base;
  const bool update_av = av_transpose != nullptr && av != nullptr;

  ai_transpose[0] = base;

  std::vector<std::unique_ptr<ROWTYPE[]>> threadPrefixSum(
      omp_get_max_threads());

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();

    auto [start, end] =
        utils::LoadPrefixBalancedPartition(ai, ai + rows, tid, nthreads);
    threadPrefixSum[tid].reset(new ROWTYPE[rows_transpose]());

    for (auto it = start; it < end; it++) {
      for (ROWTYPE j = *it - base; j < *(it + 1) - base; j++) {
        threadPrefixSum[tid][aj[j] - base]++;
      }
    }

#pragma omp barrier
#pragma omp for
    for (COLTYPE rowID = 0; rowID < rows_transpose; rowID++) {
      ai_transpose[rowID + 1] = 0;
      for (int t = 0; t < nthreads; t++) {
        ai_transpose[rowID + 1] += threadPrefixSum[t][rowID];
      }
    }

// may be optimized by a parallel scan
#pragma omp single
    std::inclusive_scan(ai_transpose, ai_transpose + rows_transpose + 1,
                        ai_transpose);

#pragma omp for
    for (COLTYPE rowID = 0; rowID < rows_transpose; rowID++) {
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
        const COLTYPE rowID = it - ai;
        const COLTYPE idx = threadPrefixSum[tid][aj[j] - base]++ - base;
        aj_transpose[idx] = rowID + base;
        if (update_av)
          av_transpose[idx] = av[j];
      }
    }
  }
}

/// @brief A parallel compressed sparse row matrix transpose function
/// @param rows number of rows of the matrix about to be transposed
/// @param cols number of columns of the matrix about to be transposed
/// @param nnz number of nonzeros of the matrix about to be transposed
/// @param base matrix index base (0 or 1)
/// @param ai row index
/// @param aj column index
/// @param av value vector
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void ParallelTranspose2(const COLTYPE rows, const COLTYPE cols, const int base,
                        ROWTYPE const *ai, COLTYPE const *aj, VALTYPE const *av,
                        ROWTYPE *ai_transpose, COLTYPE *aj_transpose,
                        VALTYPE *av_transpose) {
  const COLTYPE cols_transpose = rows;
  const COLTYPE rows_transpose = cols;
  const auto nnz = ai[rows] - base;
  ai_transpose[0] = base;
  const bool update_av = av_transpose != nullptr && av != nullptr;

  std::unique_ptr<ROWTYPE[]> threadPrefixSum(nullptr);

  std::vector<ROWTYPE> prefix(omp_get_max_threads() + 1, 0);
  prefix[0] = base;

  int nthreads;
  auto IdxMap = [&nthreads](const int tid, const COLTYPE rid) {
    return nthreads * rid + tid;
  };

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();

#pragma omp single
    {
      nthreads = omp_get_num_threads();
      threadPrefixSum.reset(new ROWTYPE[nthreads * rows_transpose]());
    }

    auto [start, end] =
        utils::LoadPrefixBalancedPartition(ai, ai + rows, tid, nthreads);

    for (auto it = start; it < end; it++) {
      for (ROWTYPE j = *it - base; j < *(it + 1) - base; j++) {
        threadPrefixSum[IdxMap(tid, aj[j] - base)]++;
      }
    }

#pragma omp barrier
    auto [start_row, end_row] =
        utils::LoadBalancedPartitionPos(rows_transpose, tid, nthreads);

    ROWTYPE tmp = 0;
    for (COLTYPE i = start_row; i < end_row; i++) {
      threadPrefixSum[IdxMap(0, i)] += tmp;
      for (int t = 1; t < nthreads; t++) {
        threadPrefixSum[IdxMap(t, i)] += threadPrefixSum[IdxMap(t - 1, i)];
      }
      tmp = threadPrefixSum[IdxMap(nthreads - 1, i)];
      ai_transpose[i + 1] = threadPrefixSum[IdxMap(nthreads - 1, i)];
    }
    prefix[tid + 1] = ai_transpose[end_row];

#pragma omp barrier
#pragma omp single
    std::inclusive_scan(prefix.begin(), prefix.end(), prefix.begin());

    tmp = 0;
    for (COLTYPE i = start_row; i < end_row; i++) {
      ai_transpose[i + 1] += prefix[tid];
      for (int t = 0; t < nthreads; t++) {
        std::swap(threadPrefixSum[IdxMap(t, i)], tmp);
        threadPrefixSum[IdxMap(t, i)] += prefix[tid];
      }
    }

#pragma omp barrier
    for (auto it = start; it < end; it++) {
      for (ROWTYPE j = *it - base; j < *(it + 1) - base; j++) {
        const COLTYPE rowID = it - ai;
        const COLTYPE idx = threadPrefixSum[IdxMap(tid, aj[j] - base)]++ - base;
        aj_transpose[idx] = rowID + base;
        if (update_av)
          av_transpose[idx] = av[j];
      }
    }
  }
}

template <typename ROWTYPE, typename COLTYPE>
void permutedAI(const COLTYPE rows, const int base, ROWTYPE const *ai,
                COLTYPE const *iperm, ROWTYPE *permed_ai) {
  if (iperm == nullptr) {
    std::copy(std::execution::par, ai, ai + rows + 1, permed_ai);
  }

  std::vector<MKL_INT> localNNZ(omp_get_max_threads() + 1, 0);
  permed_ai[0] = 0;
#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    auto [start, end] = utils::LoadBalancedPartition(
        permed_ai, permed_ai + rows, tid, nthreads);

    // iperm[i] = k -> pinv_{i,k} = 1 -> Aperm(i,*) = A(k, *)
    for (auto i = start; i < end; i++) {
      size_t k = iperm[i - permed_ai] - base;
      MKL_INT nz = ai[k + 1] - ai[k];
      *(i + 1) = (i == start ? 0 : *i) + nz;
      localNNZ[tid + 1] += nz;
    }
#pragma omp barrier
#pragma omp single
    {
      std::inclusive_scan(localNNZ.begin(), localNNZ.end(), localNNZ.begin(),
                          std::plus<>());
    }

    for (auto i = start + 1; i <= end; i++) {
      *i += localNNZ[tid] + base;
    }
  }
  permed_ai[0] = base;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void permute(const COLTYPE rows, const int base, ROWTYPE const *ai,
             COLTYPE const *aj, VALTYPE const *av, COLTYPE const *iperm,
             COLTYPE const *perm, ROWTYPE *permed_ai, COLTYPE *permed_aj,
             VALTYPE *permed_av) {
  permutedAI(rows, base, ai, iperm, permed_ai);
  const auto nnz = ai[rows] - base;

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    auto [start, end] = utils::LoadPrefixBalancedPartition(
        permed_ai, permed_ai + rows, tid, nthreads);

    for (auto i = start; i < end; i++) {
      // copy and convert aj and av
      size_t rowInd = iperm ? iperm[i - permed_ai] - base : (i - permed_ai);
      // permute column in each row perm[i] = k -> q_{i,k} = 1 -> new(*, k) =
      // old(*, i)
      std::transform(aj + ai[rowInd] - base, aj + ai[rowInd + 1] - base,
                     permed_aj + *i - base, [perm, base](MKL_INT ind) {
                       return perm ? perm[ind - base] : ind;
                     });

      std::copy(std::execution::seq, av + ai[rowInd] - base,
                av + ai[rowInd + 1] - base, permed_av + *i - base);

      if (perm == nullptr)
        continue;
      // intersion sort aj and av based on the column index
      auto pos = permed_aj + *(i + 1) - base - 1;
      while (pos > permed_aj + *i - base) {
        for (auto j = permed_aj + *i - base; j < pos; j++) {
          if (*j > *pos) {
            std::swap(*j, *pos);
            std::swap(permed_av[j - permed_aj], permed_av[pos - permed_aj]);
          }
        }
        pos--;
      }
    }
  }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void permuteRow(const COLTYPE rows, const int base, ROWTYPE const *ai,
                COLTYPE const *aj, VALTYPE const *av, COLTYPE const *iperm,
                ROWTYPE *permed_ai, COLTYPE *permed_aj, VALTYPE *permed_av) {
  permute(rows, base, ai, aj, av, iperm, (COLTYPE const *)nullptr, permed_ai,
          permed_aj, permed_av);
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void symPermute(const COLTYPE rows, const int base, ROWTYPE const *ai,
                COLTYPE const *aj, VALTYPE const *av, COLTYPE const *iperm,
                ROWTYPE *permed_ai, COLTYPE *permed_aj, VALTYPE *permed_av) {
  // upper triangular
  const COLTYPE n = rows;
  const auto nnz = ai[rows] - base;
  permed_ai[0] = base;
  std::vector<MKL_INT> ai_prefix(n * (omp_get_max_threads() + 1), 0);
#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();

    auto [start, end] =
        utils::LoadPrefixBalancedPartition(ai, ai + n, tid, nthreads);
    MKL_INT new_row, new_col, final_row, final_col, col;
    for (auto i = start; i != end; i++) {
      new_row = iperm ? (iperm[i - ai] - base) : (i - ai);
      for (auto j = *i; j != *(i + 1); j++) {
        if (i - ai > j - base)
          continue;
        col = aj[j - base] - base;
        new_col = iperm ? (iperm[col] - base) : col;
        final_row = std::min(new_row, new_col);
        final_col = std::max(new_row, new_col);

        ai_prefix[(tid + 1) * n + final_row]++;
      }
    }
#pragma omp barrier
#pragma omp single
    {
      for (MKL_INT i = 0; i < n; i++) {
        ai_prefix[i] = permed_ai[i] - base;
        for (int j = 0; j < nthreads; j++) {
          ai_prefix[(j + 1) * n + i] += ai_prefix[j * n + i];
        }
        permed_ai[i + 1] = ai_prefix[nthreads * n + i] + base;
      }
    }

    for (auto i = start; i != end; i++) {
      new_row = iperm ? (iperm[i - ai] - base) : (i - ai);
      for (auto j = *i; j != *(i + 1); j++) {
        if (i - ai > j - base)
          continue;
        col = aj[j - base] - base;
        new_col = iperm ? (iperm[col] - base) : col;
        final_row = std::min(new_row, new_col);
        final_col = std::max(new_row, new_col);
        // continue;
        permed_aj[ai_prefix[tid * n + final_row]] = final_col + base;
        permed_av[ai_prefix[tid * n + final_row]++] = av[j - base];
      }
    }
#pragma omp barrier
    // TODO: validate if mkl will automatically sort aj
    if (iperm) {
      auto [start_new, end_new] = utils::LoadPrefixBalancedPartition(
          permed_ai, permed_ai + n, tid, nthreads);

      for (auto i = start_new; i < end_new; i++) {
        // intersion sort aj and av based on the column index
        auto pos = permed_aj + *(i + 1) - base - 1;
        while (pos != permed_aj + *i - base) {
          for (auto j = permed_aj + *i - base; j != pos; j++) {
            if (*j > *pos) {
              MKL_INT tmp = *j;
              *j = *pos;
              *pos = tmp;
            }
          }
          pos--;
        }
      }
    }
  }
}

template <typename COLTYPE, typename VALTYPE>
void permuteVec(const COLTYPE rows, const int base, VALTYPE const *const v,
                COLTYPE const *const iperm, VALTYPE *const permed_v) {
  if (iperm) {
#pragma omp parallel for
    for (COLTYPE i = 0; i < rows; i++) {
      permed_v[i] = v[iperm[i] - base];
    }
  } else {
#pragma omp parallel for
    for (COLTYPE i = 0; i < rows; i++) {
      permed_v[i] = v[i];
    }
  }
}

template <class Array>
using array_value_type = std::decay_t<decltype(std::declval<Array &>()[0])>;

enum TriangularMatrix { L = 0, U = 1 };

template <TriangularMatrix TS = L, typename ROWTYPE, typename COLTYPE,
          typename VEC>
COLTYPE TopologicalSort(const COLTYPE nodes, const int base, ROWTYPE const *ai,
                        COLTYPE const *aj, VEC &iperm, VEC &prefix) {
  iperm.reserve(nodes);
  iperm.clear();
  std::vector<int> degrees(nodes);
  prefix.reserve(std::max(1, nodes / 100));
  prefix.resize(1);
  prefix[0] = 0;
  prefix.push_back(prefix.back());
  COLTYPE start, end, inc;
  if constexpr (TS == L) {
    start = 0;
    end = nodes;
    inc = 1;
  } else {
    start = nodes - 1;
    end = -1;
    inc = -1;
  }

  for (COLTYPE i = start; i != end; i += inc) {
    degrees[i] = ai[i + 1] - ai[i];
    if (degrees[i] == 0) {
      iperm.push_back(i + base);
      prefix.back()++;
    }
  }

  const ROWTYPE nnz = ai[nodes] - base;
  std::vector<ROWTYPE> t_ai(nodes + 1);
  std::vector<COLTYPE> t_aj(nnz);

  ParallelTranspose2(nodes, nodes, base, ai, aj, (double *)nullptr, t_ai.data(),
                     t_aj.data(), (double *)nullptr);

  COLTYPE level = 0;
  while (iperm.size() != nodes) {
    prefix.push_back(prefix.back());
    for (size_t i = prefix[level]; i < prefix[level + 1]; i++) {
      const auto idx = iperm[i] - base;
      for (auto j = t_ai[idx] - base; j < t_ai[idx + 1] - base; j++) {
        if (--degrees[t_aj[j] - base] == 0) {
          iperm.push_back(t_aj[j]);
          prefix.back()++;
        }
      }
    }
    level++;
  }
  return level;
}

template <TriangularMatrix TS = L, typename ROWTYPE, typename COLTYPE,
          typename VEC>
COLTYPE TopologicalSort2(const COLTYPE nodes, const int base, ROWTYPE const *ai,
                         COLTYPE const *aj, VEC &iperm, VEC &prefix) {
  std::vector<int> degrees(nodes, 0);
  COLTYPE start, end, inc;
  if constexpr (TS == L) {
    start = 0;
    end = nodes;
    inc = 1;
  } else {
    start = nodes - 1;
    end = -1;
    inc = -1;
  }
  COLTYPE level = 0;
  for (COLTYPE i = start; i != end; i += inc) {
    for (auto j = ai[i] - base; j < ai[i + 1] - base; j++) {
      degrees[i] = std::max(degrees[i], degrees[aj[j] - base] + 1);
    }
    level = std::max(level, degrees[i] + 1);
  }

  prefix.resize(level + 1);
  std::fill(prefix.begin(), prefix.end(), 0);

  for (COLTYPE i = 0; i < nodes; i++) {
    prefix[degrees[i] + 1]++;
  }
  std::inclusive_scan(prefix.begin(), prefix.end(), prefix.begin());

  iperm.resize(nodes);
  for (COLTYPE i = 0; i < nodes; i++) {
    iperm[prefix[degrees[i]]++] = i + base;
  }
  std::rotate(prefix.rbegin(), prefix.rbegin() + 1, prefix.rend());
  prefix[0] = 0;
  return level;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
bool Diagonal(const COLTYPE rows, const int base, ROWTYPE const *ai,
              COLTYPE const *aj, VALTYPE const *av, ROWTYPE *diagpos,
              VALTYPE *diag, const bool invert = false) {
  volatile bool missing_diag = false;
#pragma omp parallel for shared(missing_diag)
  for (COLTYPE i = 0; i < rows; i++) {
    auto mid =
        std::lower_bound(aj + ai[i] - base, aj + ai[i + 1] - base, i + base);
    if (*mid != i + base) {
      missing_diag = true;
    }
    if (diagpos)
      diagpos[i] = mid - aj + base;
    if (diag) {
      VALTYPE val = av[mid - aj];
      if (invert) {
        if (val == 0) {
          val = 1.;
        } else {
          val = 1. / val;
        }
      }
      diag[i] = val;
    }
  }
  return !missing_diag;
}

/// @brief Split a matrix into strictly lower triangular matrix L, diagonal D,
/// and strictly upper triangular matrix U
/// @tparam R
/// @tparam C
/// @tparam V
/// @param rows size of the square matrix
/// @param base matrix index base (0 or 1)
/// @param ai row index
/// @param aj column index
/// @param av value vector
/// @param L strictly lower triangular matrix
/// @param D diagonal matrix, stored as a vector. Note that zero diagonal is
/// allowed
/// @param U strictly upper triangular matrix
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE,
          typename CSRMatrixType>
void SplitLDU(const COLTYPE rows, const int base, ROWTYPE const *ai,
              COLTYPE const *aj, VALTYPE const *av, CSRMatrixType &L,
              std::vector<VALTYPE> &D, CSRMatrixType &U) {
  static_assert(
      CSRMatrixFormat<ROWTYPE, COLTYPE, VALTYPE, CSRMatrixType>::value == true);
  static_assert(CSRResizable<CSRMatrixType>::value,
                "CSRMatrixType must have a resizable method");

  ROWTYPE nnz = ai[rows] - base;
  L.rows = rows;
  L.cols = rows;
  ResizeCSRAI(L, rows + 1);

  U.rows = rows;
  U.cols = rows;
  ResizeCSRAI(U, rows + 1);

  L.ai[0] = base;
  U.ai[0] = base;
  D.resize(rows);
  std::vector<ROWTYPE> diag(rows);
  std::vector<std::pair<ROWTYPE, ROWTYPE>> LU_prefix(omp_get_max_threads() + 1);
  LU_prefix[0] = {base, base};

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    auto [start, end] =
        utils::LoadPrefixBalancedPartition(ai, ai + rows, tid, nthreads);
    LU_prefix[tid + 1].first = 0;
    LU_prefix[tid + 1].second = 0;
    for (auto it = start; it < end; it++) {
      COLTYPE i = it - ai;
      auto mid =
          std::lower_bound(aj + *it - base, aj + *(it + 1) - base, i + base);
      const bool zero_diag = (mid == aj + *(it + 1) - base || *mid != i + base);
      diag[i] = mid - aj;
      D[i] = zero_diag ? 0 : av[diag[i]];
      const ROWTYPE L_size = mid - (aj + *it - base);
      LU_prefix[tid + 1].first += L_size;
      L.ai[i + 1] = LU_prefix[tid + 1].first;
      const ROWTYPE U_size = *(it + 1) - *it - L_size - (zero_diag ? 0 : 1);
      LU_prefix[tid + 1].second += U_size;
      U.ai[i + 1] = LU_prefix[tid + 1].second;
    }
#pragma omp barrier
#pragma omp single
    {
      for (size_t i = 1; i < LU_prefix.size(); i++) {
        LU_prefix[i].first += LU_prefix[i - 1].first;
        LU_prefix[i].second += LU_prefix[i - 1].second;
      }
      const auto Lnnz = LU_prefix[nthreads].first - base;
      ResizeCSRAJ(L, Lnnz);
      ResizeCSRAV(L, Lnnz);

      const auto Unnz = LU_prefix[nthreads].second - base;
      ResizeCSRAJ(U, Unnz);
      ResizeCSRAV(U, Unnz);
    }

    ROWTYPE L_pos = LU_prefix[tid].first - base;
    ROWTYPE U_pos = LU_prefix[tid].second - base;
    for (auto it = start; it < end; it++) {
      COLTYPE i = it - ai;
      const bool zero_diag = (diag[i] == nnz || aj[diag[i]] - base != i);
      L.ai[i + 1] += LU_prefix[tid].first;
      U.ai[i + 1] += LU_prefix[tid].second;

      for (ROWTYPE j = *it - base; j < diag[i]; j++) {
        L.aj[L_pos] = aj[j];
        L.av[L_pos++] = av[j];
      }
      for (ROWTYPE j = diag[i] + (zero_diag ? 0 : 1); j < *(it + 1) - base;
           j++) {
        U.aj[U_pos] = aj[j];
        U.av[U_pos++] = av[j];
      }
    }
  }
}

template <TriangularMatrix TS = U, typename ROWTYPE, typename COLTYPE,
          typename VALTYPE, typename CSRMatrixType>
void SplitTriangle(const COLTYPE rows, const int base, ROWTYPE const *ai,
                   COLTYPE const *aj, VALTYPE const *av, CSRMatrixType &U) {
  static_assert(TS == TriangularMatrix::U);
  static_assert(
      CSRMatrixFormat<ROWTYPE, COLTYPE, VALTYPE, CSRMatrixType>::value == true);
  static_assert(CSRResizable<CSRMatrixType>::value,
                "CSRMatrixType must have a resizable method");

  U.rows = rows;
  U.cols = rows;
  ResizeCSRAI(U, rows + 1);

  U.ai[0] = base;
  std::vector<ROWTYPE> start_pos(rows);
  std::vector<ROWTYPE> prefix(omp_get_max_threads() + 1);
  prefix[0] = base;

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    auto [start, end] =
        utils::LoadPrefixBalancedPartition(ai, ai + rows, tid, nthreads);
    prefix[tid + 1] = 0;
    for (auto it = start; it < end; it++) {
      COLTYPE i = it - ai;
      auto mid =
          std::lower_bound(aj + *it - base, aj + *(it + 1) - base, i + base);
      const bool zero_diag = (mid == aj + *(it + 1) - base || *mid != i + base);
      start_pos[i] = mid - aj;
      const ROWTYPE U_size = *(it + 1) - base - start_pos[i];
      prefix[tid + 1] += U_size;
      U.ai[i + 1] = prefix[tid + 1];
    }
#pragma omp barrier
#pragma omp single
    {
      for (size_t i = 1; i < prefix.size(); i++) {
        prefix[i] += prefix[i - 1];
      }

      const auto Unnz = prefix[nthreads] - base;
      ResizeCSRAJ(U, Unnz);
      ResizeCSRAV(U, Unnz);
    }

    ROWTYPE U_pos = prefix[tid] - base;
    for (auto it = start; it < end; it++) {
      COLTYPE i = it - ai;
      U.ai[i + 1] += prefix[tid];

      for (ROWTYPE j = start_pos[i]; j < *(it + 1) - base; j++) {
        U.aj[U_pos] = aj[j];
        U.av[U_pos++] = av[j];
      }
    }
  }
}

template <TriangularMatrix TS = U, typename ROWTYPE, typename COLTYPE,
          typename VALTYPE, typename CSRMatrixType>
void TriangularToFull(const COLTYPE rows, const int base, ROWTYPE const *ai,
                      COLTYPE const *aj, VALTYPE const *av, CSRMatrixType &F) {
  static_assert(TS == TriangularMatrix::U);
  static_assert(
      CSRMatrixFormat<ROWTYPE, COLTYPE, VALTYPE, CSRMatrixType>::value == true);

  static_assert(CSRResizable<CSRMatrixType>::value,
                "CSRMatrixType must have a resizable method");

  F.rows = rows;
  F.cols = rows;
  ResizeCSRAI(F, rows + 1);

  F.ai[0] = base;

  std::unique_ptr<ROWTYPE[]> threadPrefixSum(nullptr);
  std::unique_ptr<ROWTYPE[]> prefix{nullptr};

  int nthreads;
  auto IdxMap = [&nthreads](const int tid, const COLTYPE rid) {
    return (nthreads + 1) * rid + tid;
  };

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();

#pragma omp single
    {
      nthreads = omp_get_num_threads();
      threadPrefixSum.reset(new ROWTYPE[(nthreads + 1) * rows]());
      prefix.reset(new ROWTYPE[nthreads + 1]());
      prefix[0] = base;
    }

    auto [start, end] =
        utils::LoadPrefixBalancedPartition(ai, ai + rows, tid, nthreads);

    for (auto it = start; it < end; it++) {
      COLTYPE i = it - ai;
      ROWTYPE j = aj[ai[i] - base] - base == i ? *it - base + 1 : *it - base;
      for (; j < *(it + 1) - base; j++) {
        threadPrefixSum[IdxMap(tid, aj[j] - base)]++;
      }
    }

#pragma omp barrier
    auto [start_row, end_row] =
        utils::LoadBalancedPartitionPos(rows, tid, nthreads);

    ROWTYPE tmp = 0;
    for (COLTYPE i = start_row; i < end_row; i++) {
      if (i != start_row)
        threadPrefixSum[IdxMap(0, i)] +=
            threadPrefixSum[IdxMap(nthreads, i - 1)];
      for (int t = 1; t < nthreads; t++) {
        threadPrefixSum[IdxMap(t, i)] += threadPrefixSum[IdxMap(t - 1, i)];
      }
      threadPrefixSum[IdxMap(nthreads, i)] =
          threadPrefixSum[IdxMap(nthreads - 1, i)] + ai[i + 1] - ai[i];
      F.ai[i + 1] = threadPrefixSum[IdxMap(nthreads, i)];
    }
    prefix[tid + 1] = F.ai[end_row];

#pragma omp barrier
#pragma omp single
    {
      std::inclusive_scan(prefix.get(), prefix.get() + nthreads + 1,
                          prefix.get());
      const ROWTYPE nnz = prefix[nthreads] - base;
      ResizeCSRAJ(F, nnz);
      ResizeCSRAV(F, nnz);
    }

    tmp = 0;
    for (COLTYPE i = start_row; i < end_row; i++) {
      F.ai[i + 1] += prefix[tid];
      for (int t = 0; t < nthreads + 1; t++) {
        std::swap(threadPrefixSum[IdxMap(t, i)], tmp);
        threadPrefixSum[IdxMap(t, i)] += prefix[tid];
      }
    }

#pragma omp barrier
    for (auto it = start; it < end; it++) {
      const COLTYPE i = it - ai;
      ROWTYPE j = aj[ai[i] - base] - base == i ? *it - base + 1 : *it - base;
      for (; j < *(it + 1) - base; j++) {
        const COLTYPE idx = threadPrefixSum[IdxMap(tid, aj[j] - base)]++ - base;
        F.aj[idx] = i + base;
        F.av[idx] = av[j];
      }
      std::copy(aj + ai[i] - base, aj + ai[i + 1] - base,
                find_address_of(F.aj) + threadPrefixSum[IdxMap(nthreads, i)] -
                    base);
      std::copy(av + ai[i] - base, av + ai[i + 1] - base,
                find_address_of(F.av) + threadPrefixSum[IdxMap(nthreads, i)] -
                    base);
    }
  }
}

template <typename ROWTYPE, typename COLTYPE>
bool ValidCSR(const COLTYPE rows, const COLTYPE cols, const int base,
              ROWTYPE const *ai, COLTYPE const *aj) {
  if (ai[0] != base) {
    std::cout << "ai[0] is not equal to base" << std::endl;
    return false;
  }
  for (COLTYPE i = 0; i < rows; i++) {
    if (ai[i + 1] < ai[i]) {
      std::cout << "ai is not monotonically increasing" << std::endl;
      return false;
    }
    if (!std::is_sorted(aj + ai[i] - base, aj + ai[i + 1] - base)) {
      std::cout << "Unsorted row " << i << std::endl;
      return false;
    }
    if (std::adjacent_find(aj + ai[i] - base, aj + ai[i + 1] - base) !=
        aj + ai[i + 1] - base) {
      std::cout << "Duplicate entry in row " << i << std::endl;
      return false;
    }

    if ((ai[i + 1] - ai[i] > 0) &&
        (aj[ai[i] - base] < base || aj[ai[i + 1] - base - 1] >= cols + base)) {
      std::cout << "Column index out of range in row " << i << std::endl;
      return false;
    }
  }
  return true;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
bool Symmetry(const COLTYPE size, const int base, ROWTYPE const *ai,
              COLTYPE const *aj, VALTYPE const *av) {
  const ROWTYPE nnz = ai[size] - base;
  std::vector<ROWTYPE> tai(size + 1);
  std::vector<COLTYPE> taj(nnz);
  std::vector<VALTYPE> tav(nnz);

  ParallelTranspose2(size, size, base, ai, aj, av, tai.data(), taj.data(),
                     tav.data());
  for (COLTYPE i = 0; i < size; i++) {
    if (ai[i + 1] - ai[i] != tai[i + 1] - tai[i]) {
      std::cout << "Row " << i << " has different number of nonzeros"
                << std::endl;
      return false;
    }
    for (ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; j++) {
      if (aj[j] != taj[j] || av[j] != tav[j]) {
        std::cout << "Row " << i << " is not symmetric" << std::endl;
        return false;
      }
    }
  }
  return true;
}

// alpha * diag * x + beta * y
template <typename COLTYPE, typename VALTYPE>
void DiagVecMul(const COLTYPE n, const VALTYPE alpha, VALTYPE const *diag,
                VALTYPE const *x, const VALTYPE beta, VALTYPE *y) {
  if (beta) {
#pragma omp parallel for
    for (COLTYPE i = 0; i < n; ++i) {
      y[i] = alpha * x[i] * diag[i] + beta * y[i];
    }
  } else {
#pragma omp parallel for
    for (COLTYPE i = 0; i < n; ++i) {
      y[i] = alpha * x[i] * diag[i];
    }
  }
}
} // namespace matrix_utils