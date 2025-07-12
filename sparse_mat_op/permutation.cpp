#include "permutation.hpp"
#include "utils.h"
#include "variadic_sort.hpp"
#include <algorithm>
#include <cassert>
#include <numeric>
#include <omp.h>
#include <vector>

namespace matrix_utils {
template <typename COLTYPE, typename VALTYPE>
void permVec(const COLTYPE rows, const COLTYPE base, VALTYPE const *const v,
             COLTYPE const *const perm, VALTYPE *const v_perm) {
  if (perm) {
#pragma omp parallel for
    for (COLTYPE i = 0; i < rows; i++) {
      v_perm[i] = v[perm[i] - base];
    }
  } else {
#pragma omp parallel for
    for (COLTYPE i = 0; i < rows; i++) {
      v_perm[i] = v[i];
    }
  }
}

template <typename COLTYPE, typename VALTYPE>
void invPermVec(const COLTYPE rows, const COLTYPE base, VALTYPE const *const v,
                COLTYPE const *const perm, VALTYPE *const v_iperm) {
  if (perm) {
#pragma omp parallel for
    for (COLTYPE i = 0; i < rows; i++) {
      v_iperm[perm[i] - base] = v[i];
    }
  } else {
#pragma omp parallel for
    for (COLTYPE i = 0; i < rows; i++) {
      v_iperm[i] = v[i];
    }
  }
}

template <typename COLTYPE>
void invPerm(const COLTYPE rows, const COLTYPE base, COLTYPE const *const perm,
             COLTYPE *const iperm) {
  assert(perm != nullptr && iperm != nullptr);
#pragma omp parallel for
  for (COLTYPE i = 0; i < rows; i++) {
    iperm[perm[i] - base] = i + base;
  }
}

template <typename COLTYPE>
bool isPermutation(const COLTYPE rows, const COLTYPE base,
                   COLTYPE const *const perm) {
  if (perm == nullptr)
    return false;
  std::vector<char> seen(rows,
                         0); // Initialize all to 0, use char for false sharing
  volatile bool flag = true;
#pragma omp parallel for shared(flag)
  for (COLTYPE i = 0; i < rows; i++) {
    if (flag == false)
      continue;
    if (perm[i] < base || perm[i] >= base + rows || seen[perm[i] - base]) {
      flag = false;
    }
    seen[perm[i] - base] = 1;
  }
  return flag;
}

template <typename COLTYPE>
void randPerm(const COLTYPE rows, const COLTYPE base, COLTYPE *const perm) {
  assert(perm != nullptr);
  std::iota(perm, perm + rows, base);
  std::random_shuffle(perm, perm + rows);
}

template <typename ROWTYPE, typename COLTYPE>
void permRowPtr(const COLTYPE rows, ROWTYPE const *ai, COLTYPE const *perm,
                ROWTYPE *perm_ai) {
  assert(ai != nullptr && perm_ai != nullptr);
  if (perm == nullptr) {
#pragma omp parallel for
    for (COLTYPE i = 0; i < rows + 1; i++) {
      perm_ai[i] = ai[i];
    }
  } else {
    const COLTYPE base = ai[0];
    std::vector<COLTYPE> localNNZ(omp_get_max_threads() + 1, 0);
    perm_ai[0] = base;
#pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      const int nthreads = omp_get_num_threads();
      auto [start, end] = utils::LoadBalancedPartitionPos(rows, tid, nthreads);
      ROWTYPE nnz = 0;
      for (auto i = start; i < end; i++) {
        COLTYPE k = perm[i] - base;
        nnz += ai[k + 1] - ai[k];
        perm_ai[i + 1] = nnz + base;
      }
      localNNZ[tid + 1] = nnz;
#pragma omp barrier
#pragma omp single
      {
        std::inclusive_scan(localNNZ.begin(), localNNZ.end(), localNNZ.begin(),
                            std::plus<>());
      } // Implicit barrier is here if nowait is not specified

      for (auto i = start + 1; i <= end; i++) {
        perm_ai[i] += localNNZ[tid];
      }
    }
  }
}

template <typename ROWTYPE, typename COLTYPE, typename... Args1,
          typename... Args2>
void permuteMat(const COLTYPE rows, const COLTYPE cols, ROWTYPE const *const ai,
                COLTYPE const *const aj, Args1... args,
                COLTYPE const *const permP, COLTYPE const *const ipermQ,
                ROWTYPE *const perm_ai, COLTYPE *const perm_aj,
                Args2... perm_args) {
  static_assert(sizeof...(Args1) == sizeof...(Args2),
                "The number of arguments must be the same");
  //   permRowPtr(rows, ai, permP, perm_ai, perm_args...);
  //   const auto base = ai[0];
  //   const auto nnz = ai[rows] - base;

  // #pragma omp parallel
  //   {
  //     const int tid = omp_get_thread_num();
  //     const int nthreads = omp_get_num_threads();
  //     auto [start, end] = utils::LoadPrefixBalancedPartitionPos(
  //         perm_ai, perm_ai + rows, tid, nthreads);

  //     for (auto perm_i = start; perm_i < end; perm_i++) {
  //       // moving pinv_i'th row to i'th row
  //       auto i = permP ? (permP[perm_i] - base) : perm_i;
  //       for (auto j = ai[i] - base; j < ai[i + 1] - base; j++) {
  //         auto perm_j = perm_ai[i] - ai[i] + j;
  //         perm_aj[perm_j] = ipermQ ? ipermQ[aj[j] - base] : aj[j];
  //         if constexpr (sizeof...(Args1) > 0) {
  //           utils::variadic_assign_uninterleave(j, perm_j, args...,
  //           perm_args...);
  //         }
  //       }
  //       if (ipermQ == nullptr)
  //         continue;
  //       utils::variadic_quick_sort(perm_ai[perm_i] - base,
  //                                  perm_aj[perm_i + 1] - base, perm_aj,
  //                                  perm_args...);
  //     }
  //   }
}

template <typename COLTYPE, typename... Args1, typename... Args2>
void test(COLTYPE const *const aj, Args1... args, COLTYPE const *const perm_aj,
          Args2... perm_args) {
  utils::variadic_assign_uninterleave(0, 1, args..., perm_args...);
}

#define INSTANTIATE_PERM_FUNCS(COLTYPE, VALTYPE)                               \
  template void permVec<COLTYPE, VALTYPE>(                                     \
      const COLTYPE, const COLTYPE, VALTYPE const *const,                      \
      COLTYPE const *const, VALTYPE *const);                                   \
  template void invPermVec<COLTYPE, VALTYPE>(                                  \
      const COLTYPE, const COLTYPE, VALTYPE const *const,                      \
      COLTYPE const *const, VALTYPE *const);

#define INSTANTIATE_INVPERM_FUNC(COLTYPE)                                      \
  template void invPerm<COLTYPE>(const COLTYPE, const COLTYPE,                 \
                                 COLTYPE const *const, COLTYPE *const);

#define INSTANTIATE_ISPERM_FUNC(COLTYPE)                                       \
  template bool isPermutation<COLTYPE>(const COLTYPE, const COLTYPE,           \
                                       COLTYPE const *const);

#define INSTANTIATE_RANDPERM_FUNC(COLTYPE)                                     \
  template void randPerm<COLTYPE>(const COLTYPE, const COLTYPE, COLTYPE *const);

#define INSTANTIATE_PERM_ROW_PTR_FUNC(COLTYPE)                                 \
  template void permRowPtr<COLTYPE>(const COLTYPE, COLTYPE const *const,       \
                                    COLTYPE const *const, COLTYPE *const);

#define INSTANTIATE_PERMUTEMAT_STRUCT_FUNC(ROWTYPE, COLTYPE)                   \
  template void permuteMat<ROWTYPE, COLTYPE>(                                  \
      const COLTYPE, const COLTYPE, ROWTYPE const *const,                      \
      COLTYPE const *const, COLTYPE const *const, COLTYPE const *const,        \
      ROWTYPE *const, COLTYPE *const);

// #define INSTANTIATE_PERMUTEMAT_FUNC(ROWTYPE, COLTYPE, VALTYPE1, VALTYPE2)      \
//   template void permuteMat<ROWTYPE, COLTYPE, VALTYPE1, VALTYPE2>(              \
//       const COLTYPE, const COLTYPE, ROWTYPE const *const,                      \
//       COLTYPE const *const, VALTYPE1, COLTYPE const *const,                    \
//       COLTYPE const *const, ROWTYPE *const, COLTYPE *const, VALTYPE2);

// Example instantiations:
INSTANTIATE_PERM_FUNCS(int, double)
INSTANTIATE_PERM_FUNCS(int, float)
INSTANTIATE_PERM_FUNCS(int, int)
INSTANTIATE_INVPERM_FUNC(int)
INSTANTIATE_ISPERM_FUNC(int)
INSTANTIATE_RANDPERM_FUNC(int)
INSTANTIATE_PERM_ROW_PTR_FUNC(int)
INSTANTIATE_PERMUTEMAT_STRUCT_FUNC(int, int)

template void test<int, int *, int *>(int const *const, int *, int const *const,
                                      int *);
} // namespace matrix_utils