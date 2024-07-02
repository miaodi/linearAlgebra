#pragma once

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

template <class T> auto find_address_of(T &&p) { return p.get(); }

template <typename T> auto find_address_of(T *p) { return p; }

template <typename T> auto find_address_of(const std::vector<T> &p) {
  return p.cbegin();
}

template <typename T> T const *find_address_of(std::span<const T> p) {
  return p.data();
}

template <typename R, typename C, typename V> struct CSRMatrix {
  using ROWTYPE = R;
  using COLTYPE = C;
  using VALTYPE = V;
  COLTYPE rows;
  COLTYPE cols;
  int base;
  ROWTYPE nnz;
  size_t ai_size{0};
  size_t aj_size{0};
  size_t av_size{0};
  std::shared_ptr<ROWTYPE[]> ai;
  std::shared_ptr<COLTYPE[]> aj;
  std::shared_ptr<VALTYPE[]> av;

  CSRMatrix() = default;
};

template <typename SIZE = int, typename ROWTYPE = int, typename COLTYPE = int,
          typename VALTYPE = double>
decltype(auto) AllocateCSRData(const SIZE rows, const ROWTYPE nnz) {

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
template <typename SIZE, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
auto SerialTranspose(const SIZE rows, const SIZE cols, const int base,
                     ROWTYPE const *ai, COLTYPE const *aj, VALTYPE const *av) {

  const SIZE cols_transpose = rows;
  const SIZE rows_transpose = cols;
  const auto nnz = ai[rows] - base;
  std::shared_ptr<ROWTYPE[]> ai_transpose(
      new ROWTYPE[rows_transpose + 2]); // prevent from branching
  std::shared_ptr<COLTYPE[]> aj_transpose(new COLTYPE[nnz]);
  std::shared_ptr<VALTYPE[]> av_transpose{nullptr};
  if (av != nullptr)
    av_transpose.reset(new VALTYPE[nnz]);

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
      if (av != nullptr)
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
template <typename SIZE, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
auto ParallelTranspose(const SIZE rows, const SIZE cols, const int base,
                       ROWTYPE const *ai, COLTYPE const *aj,
                       VALTYPE const *av) {
  const SIZE cols_transpose = rows;
  const SIZE rows_transpose = cols;
  const auto nnz = ai[rows] - base;
  std::shared_ptr<ROWTYPE[]> ai_transpose(new ROWTYPE[rows_transpose + 1]);
  std::shared_ptr<COLTYPE[]> aj_transpose(new COLTYPE[nnz]);
  std::shared_ptr<VALTYPE[]> av_transpose{nullptr};
  if (av != nullptr)
    av_transpose.reset(new VALTYPE[nnz]);

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
    for (SIZE rowID = 0; rowID < rows_transpose; rowID++) {
      ai_transpose[rowID + 1] = 0;
      for (int t = 0; t < nthreads; t++) {
        ai_transpose[rowID + 1] += threadPrefixSum[t][rowID];
      }
    }

// may be optimized by a parallel scan
#pragma omp single
    std::inclusive_scan(ai_transpose.get(),
                        ai_transpose.get() + rows_transpose + 1,
                        ai_transpose.get());

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
        const SIZE rowID = it - ai;
        const COLTYPE idx = threadPrefixSum[tid][aj[j] - base]++ - base;
        aj_transpose[idx] = rowID + base;
        if (av != nullptr)
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
template <typename SIZE, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
auto ParallelTranspose2(const SIZE rows, const SIZE cols, const int base,
                        ROWTYPE const *ai, COLTYPE const *aj,
                        VALTYPE const *av) {
  const SIZE cols_transpose = rows;
  const SIZE rows_transpose = cols;
  const auto nnz = ai[rows] - base;
  std::shared_ptr<ROWTYPE[]> ai_transpose(new ROWTYPE[rows_transpose + 1]());
  ai_transpose[0] = base;
  std::shared_ptr<COLTYPE[]> aj_transpose(new COLTYPE[nnz]);
  std::shared_ptr<VALTYPE[]> av_transpose{nullptr};
  if (av != nullptr)
    av_transpose.reset(new VALTYPE[nnz]);

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
    prefix[tid + 1] = ai_transpose[endt - find_address_of(ai_transpose)];

#pragma omp barrier

#pragma omp single
    std::inclusive_scan(prefix.begin(), prefix.end(), prefix.begin());

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
        const ROWTYPE rowID = it - ai;
        const COLTYPE idx = threadPrefixSum[tid][aj[j] - base]++ - base;
        aj_transpose[idx] = rowID + base;
        if (av != nullptr)
          av_transpose[idx] = av[j];
      }
    }
  }
  return std::make_tuple(ai_transpose, aj_transpose, av_transpose);
}

template <typename SIZE, typename ROWTYPE, typename COLTYPE>
void permutedAI(const SIZE rows, const int base, ROWTYPE const *ai,
                COLTYPE const *aj, COLTYPE const *iperm, ROWTYPE *permed_ai) {
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

    // iperm[i] = k -> pinv_{i,k} = 1 -> C(i,*) = A(k, *)
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

template <typename SIZE, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void permute(const SIZE rows, const int base, ROWTYPE const *ai,
             COLTYPE const *aj, VALTYPE const *av, COLTYPE const *iperm,
             COLTYPE const *p, ROWTYPE *permed_ai, COLTYPE *permed_aj,
             VALTYPE *permed_av) {
  permutedAI(rows, base, ai, aj, iperm, permed_ai);
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
      // permute column in each row p[i] = k -> q_{i,k} = 1 -> new(*, k) =
      // old(*, i)
      std::transform(aj + ai[rowInd] - base, aj + ai[rowInd + 1] - base,
                     permed_aj + *i - base, [p, base](MKL_INT ind) {
                       return p ? p[ind - base] : (ind - base);
                     });

      std::copy(std::execution::seq, av + ai[rowInd] - base,
                av + ai[rowInd + 1] - base, permed_av + *i - base);

      if (p == nullptr)
        continue;
      // intersion sort aj and av based on the column index
      auto pos = permed_aj + *(i + 1) - base - 1;
      while (pos != permed_aj + *i - base) {
        for (auto j = permed_aj + *i - base; j != pos; j++) {
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

template <typename SIZE, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void permuteRow(const SIZE rows, const int base, ROWTYPE const *ai,
                COLTYPE const *aj, VALTYPE const *av, COLTYPE const *iperm,
                ROWTYPE *permed_ai, COLTYPE *permed_aj, VALTYPE *permed_av) {
  permute(rows, base, ai, aj, av, iperm, nullptr, permed_ai, permed_aj,
          permed_av);
}

template <typename SIZE, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void symPermute(const SIZE rows, const int base, ROWTYPE const *ai,
                COLTYPE const *aj, VALTYPE const *av, COLTYPE const *iperm,
                ROWTYPE *permed_ai, COLTYPE *permed_aj, VALTYPE *permed_av) {
  // upper triangular
  const SIZE n = rows;
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

template <class Array>
using array_value_type = std::decay_t<decltype(std::declval<Array &>()[0])>;

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

enum TriangularSolve { L = 0, U = 1 };

template <TriangularSolve TS = L, typename SIZE, typename ROWTYPE,
          typename COLTYPE, typename VEC>
void TopologicalSort(const SIZE nodes, const int base, ROWTYPE const *ai,
                     COLTYPE const *aj, VEC &iperm, VEC &prefix) {
  iperm.reserve(nodes);
  iperm.clear();
  std::vector<int> degrees(nodes);
  prefix.reserve(std::max(1, nodes / 100));
  prefix.resize(1);
  prefix[0] = 0;
  prefix.push_back(prefix.back());
  SIZE start, end, inc;
  if constexpr (TS == L) {
    start = 0;
    end = nodes;
    inc = 1;
  } else {
    start = nodes - 1;
    end = -1;
    inc = -1;
  }

  for (SIZE i = start; i != end; i += inc) {
    degrees[i] = ai[i + 1] - ai[i];
    if (degrees[i] == 0) {
      iperm.push_back(i + base);
      prefix.back()++;
    }
  }

  auto t_csr = matrix_utils::ParallelTranspose2(
      nodes, nodes, ai[nodes] - base, base, ai, aj, (double *)nullptr);
  const auto &[t_ai, t_aj, t_av] = t_csr;
  size_t level = 0;
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
}

template <TriangularSolve TS = L, typename SIZE, typename ROWTYPE,
          typename COLTYPE, typename VEC>
void TopologicalSort2(const SIZE nodes, const int base, ROWTYPE const *ai,
                      COLTYPE const *aj, VEC &iperm, VEC &prefix) {
  std::vector<int> degrees(nodes, 0);
  SIZE start, end, inc;
  if constexpr (TS == L) {
    start = 0;
    end = nodes;
    inc = 1;
  } else {
    start = nodes - 1;
    end = -1;
    inc = -1;
  }
  SIZE level = 0;
  for (SIZE i = start; i != end; i += inc) {
    for (auto j = ai[i] - base; j < ai[i + 1] - base; j++) {
      degrees[i] = std::max(degrees[i], degrees[aj[j] - base] + 1);
    }
    level = std::max(level, degrees[i] + 1);
  }

  prefix.resize(level + 1);
  std::fill(prefix.begin(), prefix.end(), 0);

  for (SIZE i = 0; i < nodes; i++) {
    prefix[degrees[i] + 1]++;
  }
  std::inclusive_scan(prefix.begin(), prefix.end(), prefix.begin());

  iperm.reserve(nodes);
  iperm.clear();
  for (SIZE i = 0; i < nodes; i++) {
    iperm[prefix[degrees[i]]++] = i + base;
  }
  std::rotate(prefix.rbegin(), prefix.rbegin() + 1, prefix.rend());
  prefix[0] = 0;
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

template <typename SIZE, typename ROWTYPE, typename COLTYPE, typename VEC>
bool DiagonalPosition(const SIZE rows, const int base, ROWTYPE const *ai,
                      COLTYPE const *aj, VEC &diag) {

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

/// @brief Split a matrix into strictly lower triangular matrix L, diagonal D,
/// and strictly upper triangular matrix U
/// @tparam SIZE
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
template <typename SIZE, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void SplitLDU(const SIZE rows, const int base, ROWTYPE const *ai,
              COLTYPE const *aj, VALTYPE const *av,
              CSRMatrix<ROWTYPE, COLTYPE, VALTYPE> &L, std::vector<VALTYPE> &D,
              CSRMatrix<ROWTYPE, COLTYPE, VALTYPE> &U) {
  L.rows = rows;
  L.cols = rows;
  L.base = base;
  if (L.ai_size < rows + 1) {
    L.ai.reset(new ROWTYPE[rows + 1]);
    L.ai_size = rows + 1;
  }

  U.rows = rows;
  U.cols = rows;
  U.base = base;
  if (U.ai_size < rows + 1) {
    U.ai.reset(new ROWTYPE[rows + 1]);
    U.ai_size = rows + 1;
  }

  L.ai[0] = base;
  U.ai[0] = base;
  D.resize(rows);
  std::vector<ROWTYPE> diag(rows);
  std::vector<std::pair<ROWTYPE, ROWTYPE>> LU_prefix(omp_get_max_threads() + 1);
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
      const ROWTYPE L_size = mid - (find_address_of(aj) + *it - base);
      LU_prefix[tid + 1].first += L_size;
      L.ai[i + 1] = LU_prefix[tid + 1].first;
      const ROWTYPE U_size = *(it + 1) - *it - L_size - (zero_diag ? 0 : 1);
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
        L.aj.reset(new COLTYPE[L.nnz]);
        L.aj_size = L.nnz;
      }
      if (L.av_size < L.nnz) {
        L.av.reset(new VALTYPE[L.nnz]);
        L.av_size = L.nnz;
      }

      U.nnz = LU_prefix[nthreads].second;
      if (U.aj_size < U.nnz) {
        U.aj.reset(new COLTYPE[U.nnz]);
        U.aj_size = U.nnz;
      }
      if (U.av_size < U.nnz) {
        U.av.reset(new VALTYPE[U.nnz]);
        U.av_size = U.nnz;
      }
    }

#pragma omp barrier
    ROWTYPE L_pos = LU_prefix[tid].first - base;
    ROWTYPE U_pos = LU_prefix[tid].second - base;
    for (auto it = start; it < end; it++) {
      SIZE i = it - find_address_of(ai);
      const bool zero_diag = aj[diag[i]] - base != i;
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

} // namespace matrix_utils