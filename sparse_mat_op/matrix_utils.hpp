#pragma once

#include "sparse_mat_traits.hpp"
#include "utils.h"
#include <cstddef>
#include <execution>
#include <iostream>
#include <memory>
#include <numeric>
#include <omp.h>
#include <span>
#include <tuple>
#include <type_traits>

namespace matrix_utils {
template <class Array>
using array_value_type = std::decay_t<decltype(std::declval<Array &>()[0])>;

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

  ROWTYPE *AI() { return ai; }
  COLTYPE *AJ() { return aj; }
  VALTYPE *AV() { return av; }

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
  size_t diagonal_size{0};
  std::shared_ptr<ROWTYPE[]> ai;
  std::shared_ptr<COLTYPE[]> aj;
  std::shared_ptr<VALTYPE[]> av;
  std::shared_ptr<ROWTYPE[]> diagonal;

  ROWTYPE Base() const { return ai[0]; }
  ROWTYPE NNZ() const { return ai[rows] - ai[0]; }

  ROWTYPE const *AI() const { return ai ? ai.get() : nullptr; }
  COLTYPE const *AJ() const { return aj ? aj.get() : nullptr; }
  VALTYPE const *AV() const { return av ? av.get() : nullptr; }
  ROWTYPE const *Diagonal() const {
    return diagonal ? diagonal.get() : nullptr;
  }

  ROWTYPE *AI() { return ai ? ai.get() : nullptr; }
  COLTYPE *AJ() { return aj ? aj.get() : nullptr; }
  VALTYPE *AV() { return av ? av.get() : nullptr; }
  ROWTYPE *Diagonal() { return diagonal ? diagonal.get() : nullptr; }

  ROWTYPE *ResizeAI(const size_t size) {
    if (ai_size < size || ai == nullptr) {
      std::shared_ptr<ROWTYPE[]> tmp(new ROWTYPE[size]);
      if (ai != nullptr) {
        std::copy(ai.get(), ai.get() + ai_size, tmp.get());
      }
      std::swap(ai, tmp);
      ai_size = size;
    }
    return ai.get();
  }

  COLTYPE *ResizeAJ(const size_t size) {
    if (aj_size < size || aj == nullptr) {
      std::shared_ptr<COLTYPE[]> tmp(new COLTYPE[size]);
      if (aj != nullptr) {
        std::copy(aj.get(), aj.get() + aj_size, tmp.get());
      }
      std::swap(aj, tmp);
      aj_size = size;
    }
    return aj.get();
  }

  VALTYPE *ResizeAV(const size_t size) {
    if (av_size < size || av == nullptr) {
      std::shared_ptr<VALTYPE[]> tmp(new VALTYPE[size]);
      if (av != nullptr) {
        std::copy(av.get(), av.get() + av_size, tmp.get());
      }
      std::swap(av, tmp);
      av_size = size;
    }
    return av.get();
  }

  ROWTYPE *ResizeDiagonal(const size_t size) {
    if (diagonal_size < size || diagonal == nullptr) {
      std::shared_ptr<ROWTYPE[]> tmp(new ROWTYPE[size]);
      if (diagonal != nullptr) {
        std::copy(diagonal.get(), diagonal.get() + diagonal_size, tmp.get());
      }
      std::swap(diagonal, tmp);
      diagonal_size = size;
    }
    return diagonal.get();
  }

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

  ROWTYPE *AI() { return ai.data(); }
  COLTYPE *AJ() { return aj.data(); }
  VALTYPE *AV() { return av.data(); }

  ROWTYPE *ResizeAI(const size_t size) {
    if (ai.size() < size) {
      ai.resize(size);
    }
    return ai.data();
  }

  COLTYPE *ResizeAJ(const size_t size) {
    if (aj.size() < size) {
      aj.resize(size);
    }
    return aj.data();
  }

  VALTYPE *ResizeAV(const size_t size) {
    if (av.size() < size) {
      av.resize(size);
    }
    return av.data();
  }

  template <class Archive> void serialize(Archive &ar) { ar(ai, aj, av); }
};

template <typename ROWTYPE, typename COLTYPE> struct CSRStructVec {
  using ROW = ROWTYPE;
  using COL = COLTYPE;

  COLTYPE rows{};
  COLTYPE cols{};

  std::vector<ROWTYPE> ai;
  std::vector<COLTYPE> aj;

  ROWTYPE Base() const { return ai.empty() ? ROWTYPE{} : ai[0]; }
  ROWTYPE NNZ() const {
    if (ai.empty() || rows == 0) {
      return ROWTYPE{};
    }
    return ai[rows] - ai[0];
  }

  ROWTYPE const *AI() const { return ai.data(); }
  COLTYPE const *AJ() const { return aj.data(); }

  ROWTYPE *AI() { return ai.data(); }
  COLTYPE *AJ() { return aj.data(); }

  ROWTYPE *ResizeAI(const size_t size) {
    if (ai.size() < size) {
      ai.resize(size);
    }
    return ai.data();
  }

  COLTYPE *ResizeAJ(const size_t size) {
    if (aj.size() < size) {
      aj.resize(size);
    }
    return aj.data();
  }

  template <class Archive> void serialize(Archive &ar) { ar(ai, aj); }
};

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
    std::copy(std::execution::seq, ai, ai + rows + 1, permed_ai);
  }

  std::vector<ROWTYPE> localNNZ(omp_get_max_threads() + 1, 0);
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
      ROWTYPE nz = ai[k + 1] - ai[k];
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
                     permed_aj + *i - base, [perm, base](COLTYPE ind) {
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
  std::vector<COLTYPE> ai_prefix(n * (omp_get_max_threads() + 1), 0);
#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();

    auto [start, end] =
        utils::LoadPrefixBalancedPartition(ai, ai + n, tid, nthreads);
    COLTYPE new_row, new_col, final_row, final_col, col;
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
      for (COLTYPE i = 0; i < n; i++) {
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
              COLTYPE tmp = *j;
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

enum TriangularMatrix { L = 0, U = 1 };

template <typename ROWTYPE, typename COLTYPE, TriangularMatrix TS>
struct KahnSerial
{
    // Kahn's algorithm
    COLTYPE operator()( const COLTYPE nodes,
                        ROWTYPE const* ai,
                        COLTYPE const* aj,
                        COLTYPE* perm,
                        COLTYPE* prefix,
                        bool has_diagonal = false );

    std::vector<COLTYPE> _degrees;
    std::vector<ROWTYPE> _t_ai;
    std::vector<COLTYPE> _t_aj;
};

template <typename ROWTYPE, typename COLTYPE, TriangularMatrix TS>
struct KahnParallel
{
    KahnParallel( int nthreads )
        : _nthreads( nthreads ), _threads_nodes( nthreads ), _threads_prefix( nthreads + 1 )
    {
    }
    // Kahn's algorithm
    COLTYPE operator()( const COLTYPE nodes,
                        ROWTYPE const* ai,
                        COLTYPE const* aj,
                        COLTYPE* perm,
                        COLTYPE* prefix,
                        bool has_diagonal = false );

    int _nthreads;
    std::unique_ptr<std::atomic<COLTYPE>[]> _degrees{ nullptr };
    COLTYPE _degrees_size{ 0 };
    std::vector<ROWTYPE> _t_ai;
    std::vector<COLTYPE> _t_aj;
    std::vector<std::vector<COLTYPE>> _threads_nodes;
    std::vector<COLTYPE> _threads_prefix;
};

template <typename ROWTYPE, typename COLTYPE, TriangularMatrix TS>
struct TopologicalSort2
{
    // max degree
    // after sorting, the base of prefix is the same as ai[0]
    COLTYPE operator()( const COLTYPE nodes,
                        ROWTYPE const* ai,
                        COLTYPE const* aj,
                        COLTYPE* perm,
                        COLTYPE* prefix,
                        bool has_diagonal = false );
                        
    std::vector<COLTYPE> _degrees;
};

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
bool Diagonal(const COLTYPE rows, ROWTYPE const *ai, COLTYPE const *aj,
              VALTYPE const *av, ROWTYPE *diagpos, VALTYPE *diag,
              const bool invert = false) {
  volatile bool missing_diag = false;
  const auto base = ai[0];
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
          ResizableCSRMatrixType CSRMatrixType>
void SplitLDU(const COLTYPE rows, const int base, ROWTYPE const *ai,
              COLTYPE const *aj, VALTYPE const *av, CSRMatrixType &L,
              std::vector<VALTYPE> &D, CSRMatrixType &U) {
    static_assert(CSRMatrixFormat<ROWTYPE, COLTYPE, VALTYPE, CSRMatrixType>::value);

    ROWTYPE nnz = ai[rows] - base;
    L.rows = rows;
    L.cols = rows;
    L.ResizeAI(rows + 1);

    U.rows = rows;
    U.cols = rows;
    U.ResizeAI(rows + 1);

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
      L.ResizeAJ(Lnnz);
      L.ResizeAV(Lnnz);

      const auto Unnz = LU_prefix[nthreads].second - base;
      U.ResizeAJ(Unnz);
      U.ResizeAV(Unnz);
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

// Split a matrix into strictly lower triangular matrix L (assuming that the
// diagonal are 1s) and upper triangular matrix U (including the diagonal)
template <ResizableCSRMatrixType CSRMatrixType> struct SplitLU {
  using ROWTYPE = typename CSRMatrixType::ROWTYPE;
  using COLTYPE = typename CSRMatrixType::COLTYPE;
  using VALTYPE = typename CSRMatrixType::VALTYPE;
  SplitLU(int num_threads = 1)
      : num_threads(num_threads), prefixL(num_threads + 1, 0),
        prefixU(num_threads + 1, 0) {}

  /// @brief Split a matrix into strictly lower triangular matrix L (assuming
  /// that the diagonal are 1s) and upper triangular matrix U (including the
  /// diagonal)
  /// @tparam CSRMatrixType
  /// @param rows size of the square matrix
  /// @param ai row index
  /// @param diag diagonal index
  /// @param aj column index
  /// @param av value vector
  /// @param L strictly lower triangular matrix
  /// @param U upper triangular matrix
  void operator()(const COLTYPE rows, ROWTYPE const *ai, ROWTYPE const *diag,
                  COLTYPE const *aj, VALTYPE const *av, CSRMatrixType &L,
                  CSRMatrixType &U);

  int num_threads;
  std::vector<ROWTYPE> prefixL;
  std::vector<ROWTYPE> prefixU;
};

template <TriangularMatrix TS = U, typename ROWTYPE, typename COLTYPE,
          typename VALTYPE, ResizableCSRMatrixType CSRMatrixType>
void SplitTriangle(const COLTYPE rows, const int base, ROWTYPE const *ai,
                   COLTYPE const *aj, VALTYPE const *av,
                   CSRMatrixType &tri_mat) {
  static_assert(
      CSRMatrixFormat<ROWTYPE, COLTYPE, VALTYPE, CSRMatrixType>::value);

  tri_mat.rows = rows;
  tri_mat.cols = rows;
  tri_mat.ResizeAI(rows + 1);

  tri_mat.ai[0] = base;
  std::vector<ROWTYPE> mid_pos(rows);
  std::vector<ROWTYPE> prefix(omp_get_max_threads() + 1);
  prefix[0] = base;

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    auto [start, end] =
        utils::LoadPrefixBalancedPartition(ai, ai + rows, tid, nthreads);
    prefix[tid + 1] = 0;
    COLTYPE const *mid;
    ROWTYPE row_size;
    for (auto it = start; it < end; it++) {
      COLTYPE i = it - ai;
      if constexpr (TS == TriangularMatrix::U) {
        mid =
            std::lower_bound(aj + *it - base, aj + *(it + 1) - base, i + base);
        mid_pos[i] = mid - aj;
        row_size = *(it + 1) - base - mid_pos[i];
      } else {
        mid =
            std::upper_bound(aj + *it - base, aj + *(it + 1) - base, i + base);
        mid_pos[i] = mid - aj;
        row_size = mid_pos[i] - (*it - base);
      }
      prefix[tid + 1] += row_size;
      tri_mat.ai[i + 1] = prefix[tid + 1];
    }
#pragma omp barrier
#pragma omp single
    {
      for (size_t i = 1; i < prefix.size(); i++) {
        prefix[i] += prefix[i - 1];
      }

      const auto nnz = prefix[nthreads] - base;
      tri_mat.ResizeAJ(nnz);
      tri_mat.ResizeAV(nnz);
    }

    ROWTYPE pos = prefix[tid] - base;
    for (auto it = start; it < end; it++) {
      COLTYPE i = it - ai;
      tri_mat.ai[i + 1] += prefix[tid];

      if constexpr (TS == TriangularMatrix::U) {
        for (ROWTYPE j = mid_pos[i]; j < *(it + 1) - base; j++) {
          tri_mat.aj[pos] = aj[j];
          tri_mat.av[pos++] = av[j];
        }
      } else {
        for (ROWTYPE j = *it - base; j < mid_pos[i]; j++) {
          tri_mat.aj[pos] = aj[j];
          tri_mat.av[pos++] = av[j];
        }
      }
    }
  }
}

template <TriangularMatrix TS = U, typename ROWTYPE, typename COLTYPE,
          typename VALTYPE, ResizableCSRMatrixType CSRMatrixType>
void TriangularToFull(const COLTYPE rows, const int base, ROWTYPE const *ai,
                      COLTYPE const *aj, VALTYPE const *av, CSRMatrixType &F) {
  static_assert(TS == TriangularMatrix::U);
  static_assert(
      CSRMatrixFormat<ROWTYPE, COLTYPE, VALTYPE, CSRMatrixType>::value);

  F.rows = rows;
  F.cols = rows;
  F.ResizeAI(rows + 1);

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
      F.ResizeAJ(nnz);
      F.ResizeAV(nnz);
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
bool IsSymmetry( const COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj, VALTYPE const* av )
{
    const auto base = ai[0];
    const ROWTYPE nnz = ai[size] - base;
    std::vector<ROWTYPE> tai( size + 1 );
    std::vector<COLTYPE> taj( nnz );
    std::vector<VALTYPE> tav( nnz );

    ParallelTranspose2( size, size, base, ai, aj, av, tai.data(), taj.data(), tav.data() );
    for ( COLTYPE i = 0; i < size; i++ )
    {
        if ( ai[i + 1] - ai[i] != tai[i + 1] - tai[i] )
        {
            std::cout << "Row " << i << " has different number of nonzeros" << std::endl;
            return false;
        }
        for ( ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; j++ )
        {
            if ( aj[j] != taj[j] || av[j] != tav[j] )
            {
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

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE,
          ResizableCSRMatrixType CSRMatrixType>
void Block(const COLTYPE rows, const int base, ROWTYPE const *ai,
           COLTYPE const *aj, VALTYPE const *av, const COLTYPE i,
           const COLTYPE j, const COLTYPE p, const COLTYPE q,
           CSRMatrixType &subMat) {
  static_assert(
      CSRMatrixFormat<ROWTYPE, COLTYPE, VALTYPE, CSRMatrixType>::value);

  if (i + p > rows) {
    std::cerr << "Block size exceeds matrix size" << std::endl;
    return;
  }

  subMat.rows = p;
  subMat.cols = q;
  subMat.ResizeAI(p + 1);

  subMat.ai[0] = base;

  std::vector<ROWTYPE> fronts(p);
  std::vector<ROWTYPE> ends(p);

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    auto [start, end] =
        utils::LoadPrefixBalancedPartition(ai + i, ai + i + p, tid, nthreads);

    for (auto it = start; it < end; it++) {
      COLTYPE row = it - ai;
      COLTYPE block_row = row - i;
      fronts[block_row] = std::distance(
          aj, std::lower_bound(aj + ai[row] - base, aj + ai[row + 1] - base,
                               j + base));
      ends[block_row] = std::distance(
          aj, std::lower_bound(aj + ai[row] - base, aj + ai[row + 1] - base,
                               j + q + base));
      subMat.ai[block_row + 1] = ends[block_row] - fronts[block_row];
    }
#pragma omp barrier
#pragma omp single
    {
      for (size_t _i = 0; _i < p; _i++) {
        subMat.ai[_i + 1] += subMat.ai[_i];
      }

      const auto nnz = subMat.ai[p] - base;
      subMat.ResizeAJ(nnz);
      subMat.ResizeAV(nnz);
    }

    const auto nnz = subMat.ai[p] - base;

    ROWTYPE pos = subMat.ai[start - ai - i] - base;
    for (auto it = start; it < end; it++) {
      COLTYPE block_row = it - ai - i;
      for (ROWTYPE _j = fronts[block_row]; _j < ends[block_row]; _j++) {
        subMat.aj[pos] = aj[_j] - j;
        subMat.av[pos++] = av[_j];
      }
    }
  }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void Prune( const COLTYPE rows,
            ROWTYPE* ai,
            COLTYPE* aj,
            VALTYPE* av,
            const VALTYPE threshold,
            VALTYPE const* row_thresholds );

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void DiagonalScaledPrune( const COLTYPE rows,
                          ROWTYPE* ai,
                          COLTYPE* aj,
                          VALTYPE* av,
                          const VALTYPE threshold );

template <ResizableCSRMatrixType CSRMatrixType>
void RandomL(const typename CSRMatrixType::COLTYPE rows,
             const typename CSRMatrixType::COLTYPE base,
             const typename CSRMatrixType::COLTYPE nnz_per_row,
             CSRMatrixType &L) {
  L.ResizeAI(rows + 1);
  L.ResizeAJ(rows * nnz_per_row);
  L.ResizeAV(rows * nnz_per_row);
  L.rows = rows;
  L.cols = rows;
  auto ai = L.AI();
  ai[0] = base;
  auto aj = L.AJ();
  auto av = L.AV();
  utils::knuth_s random_generator;
  for (auto i = 0; i < rows; i++) {
    typename CSRMatrixType::COLTYPE row_size = std::min(nnz_per_row, i);
    ai[i + 1] = ai[i] + row_size;
  }
#pragma omp parallel for
  for (auto i = 0; i < rows; i++) {
    random_generator(ai[i + 1] - ai[i], base, i + base, aj + ai[i] - base);
  }
}

template <ResizableCSRMatrixType CSRMatrixType>
void RandomU(const typename CSRMatrixType::COLTYPE rows,
             const typename CSRMatrixType::COLTYPE base,
             const typename CSRMatrixType::COLTYPE nnz_per_row,
             CSRMatrixType &U, bool include_diagonal = false) {
  using COLTYPE = typename CSRMatrixType::COLTYPE;
  using VALTYPE = typename CSRMatrixType::VALTYPE;

  U.ResizeAI(rows + 1);
  U.ResizeAJ(rows * nnz_per_row);
  U.ResizeAV(rows * nnz_per_row);
  U.rows = rows;
  U.cols = rows;
  auto ai = U.AI();
  auto aj = U.AJ();
  auto av = U.AV();
  ai[0] = base;

  utils::knuth_s random_generator;

  for (COLTYPE i = 0; i < rows; i++) {
    const COLTYPE remaining_strict = rows - (i + 1);
    const COLTYPE max_total =
        include_diagonal ? remaining_strict + COLTYPE{1} : remaining_strict;
    const COLTYPE row_size = std::min(nnz_per_row, max_total);
    ai[i + 1] = ai[i] + row_size;
  }

#pragma omp parallel for
  for (COLTYPE i = 0; i < rows; i++) {
    const COLTYPE row_len = ai[i + 1] - ai[i];
    if (row_len == 0)
      continue;

    const COLTYPE diag_count =
        (include_diagonal && row_len > 0) ? COLTYPE{1} : COLTYPE{0};
    const COLTYPE strict_count = row_len - diag_count;

    auto row_cols = aj + ai[i] - base;
    auto row_vals = av + ai[i] - base;

    if (diag_count == COLTYPE{1}) {
      row_cols[0] = i + base;
      row_vals[0] = static_cast<VALTYPE>(1);
    }

    if (strict_count > 0) {
      random_generator(strict_count, i + 1 + base, rows + base,
                       row_cols + diag_count);
      for (COLTYPE k = 0; k < strict_count; ++k) {
        row_vals[diag_count + k] = static_cast<VALTYPE>(1);
      }
    }
  }
}

} // namespace matrix_utils
