#include "permutation.hpp"
#include "utils.h"
#include "variadic_sort.hpp"
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
  std::vector<std::atomic<int>> seen(rows);
  for (COLTYPE i = 0; i < rows; ++i) {
    seen[i] = 0;
  }

  bool is_perm = true;

  // #pragma omp parallel for shared(seen, is_perm)
  //   for (COLTYPE i = 0; i < rows; ++i) {
  //     if (!is_perm)
  //       continue; //  if we already found an error, skip further checks
  //     auto val = perm[i] - base;
  //     if (val < 0 || val >= rows) {
  //       is_perm = false; // value out of bounds
  //       continue;
  //     }
  //     int expected = 0;
  //     if (!seen[val].compare_exchange_strong(expected, 1)) {
  //       is_perm = false; // duplicate value found
  //     }
  //   }
#pragma omp parallel for shared(seen, is_perm)
  for (COLTYPE i = 0; i < rows; ++i) {
    if (!is_perm)
      continue; //  if we already found an error, skip further checks
    auto val = perm[i] - base;
    if (val < 0 || val >= rows) {
      is_perm = false; // value out of bounds
      continue;
    }
    if (++seen[val] > 1) {
      is_perm = false; // duplicate value found
    }
  }

  return is_perm;
}

template <typename COLTYPE>
bool isPermutationSerial(const COLTYPE rows, const COLTYPE base,
                         COLTYPE const *const perm) {
  std::vector<bool> seen(rows, false);
  for (COLTYPE i = 0; i < rows; ++i) {
    auto val = perm[i] - base;
    if (val < 0 || val >= rows) {
      return false; // value out of bounds
    }
    if (seen[val]) {
      return false; // duplicate value found
    } else {
      seen[val] = true;
    }
  }
  return true;
}

template <typename COLTYPE>
void randPerm(const COLTYPE rows, const COLTYPE base, COLTYPE *const perm) {
  assert(perm != nullptr);
  std::iota(perm, perm + rows, base);
  std::shuffle(perm, perm + rows, std::mt19937(std::random_device{}()));
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

// Internal implementation function to avoid code duplication
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
static void permuteMatImpl(const COLTYPE rows, const COLTYPE cols,
                          COLTYPE const *const permP, COLTYPE const *const ipermQ,
                          ROWTYPE const *const ai, COLTYPE const *const aj,
                          ROWTYPE *const perm_ai, COLTYPE *const perm_aj,
                          VALTYPE const *const av = nullptr, VALTYPE *const perm_av = nullptr) {
  permRowPtr(rows, ai, permP, perm_ai);
  const auto base = ai[0];
  const auto nnz = ai[rows] - base;

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    auto [start, end] = utils::LoadPrefixBalancedPartitionPos(
        perm_ai, perm_ai + rows, tid, nthreads);

    for (auto perm_i = start; perm_i < end; perm_i++) {
      // moving pinv_i'th row to i'th row
      auto i = permP ? (permP[perm_i] - base) : perm_i;
      for (auto j_idx = ai[i] - base; j_idx < ai[i + 1] - base; j_idx++) {
        auto perm_j_idx = perm_ai[perm_i] - ai[i] + j_idx;
        auto j = aj[j_idx];
        perm_aj[perm_j_idx] = ipermQ ? ipermQ[j - base] : j;
        
        // Copy values if provided
        if (av && perm_av) {
          perm_av[perm_j_idx] = av[j_idx];
        }
      }

      // Sort the column indices (and associated values) within each row
      if (ipermQ == nullptr)
        continue;

      if (av && perm_av) {
        utils::variadic_quick_sort(perm_ai[perm_i] - base, perm_ai[perm_i + 1] - base,
                                  perm_aj, perm_av);
      } else {
        utils::variadic_quick_sort(perm_ai[perm_i] - base, perm_ai[perm_i + 1] - base,
                                  perm_aj);
      }
    }
  }
}

// Structure-only permutation
template <typename ROWTYPE, typename COLTYPE>
void permuteMat(const COLTYPE rows, const COLTYPE cols,
                COLTYPE const *const permP, COLTYPE const *const ipermQ,
                ROWTYPE const *const ai, COLTYPE const *const aj,
                ROWTYPE *const perm_ai, COLTYPE *const perm_aj) {
  permuteMatImpl<ROWTYPE, COLTYPE, int>(rows, cols, permP, ipermQ, ai, aj, perm_ai, perm_aj);
}

// Structure + values permutation
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void permuteMat(const COLTYPE rows, const COLTYPE cols,
                COLTYPE const *const permP, COLTYPE const *const ipermQ,
                ROWTYPE const *const ai, COLTYPE const *const aj, VALTYPE const *const av,
                ROWTYPE *const perm_ai, COLTYPE *const perm_aj, VALTYPE *const perm_av) {
  permuteMatImpl<ROWTYPE, COLTYPE, VALTYPE>(rows, cols, permP, ipermQ, ai, aj, perm_ai, perm_aj, av, perm_av);
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
#define INSTANTIATE_ISPERM_SERIAL_FUNC(COLTYPE)                                \
  template bool isPermutationSerial<COLTYPE>(const COLTYPE, const COLTYPE,     \
                                             COLTYPE const *const);

#define INSTANTIATE_RANDPERM_FUNC(COLTYPE)                                     \
  template void randPerm<COLTYPE>(const COLTYPE, const COLTYPE, COLTYPE *const);

#define INSTANTIATE_PERM_ROW_PTR_FUNC(COLTYPE)                                 \
  template void permRowPtr<COLTYPE>(const COLTYPE, COLTYPE const *const,       \
                                    COLTYPE const *const, COLTYPE *const);

#define INSTANTIATE_PERMUTE_MAT_STRUCT_FUNC(ROWTYPE, COLTYPE)                  \
  template void permuteMat<ROWTYPE, COLTYPE>(                                  \
      const COLTYPE, const COLTYPE, COLTYPE const *const,                      \
      COLTYPE const *const, ROWTYPE const *const, COLTYPE const *const,        \
      ROWTYPE *const, COLTYPE *const);

#define INSTANTIATE_PERMUTE_MAT_VALUES_FUNC(ROWTYPE, COLTYPE, VALTYPE)        \
  template void permuteMat<ROWTYPE, COLTYPE, VALTYPE>(                        \
      const COLTYPE, const COLTYPE, COLTYPE const *const,                      \
      COLTYPE const *const, ROWTYPE const *const, COLTYPE const *const,        \
      VALTYPE const *const, ROWTYPE *const, COLTYPE *const, VALTYPE *const);

// Example instantiations:
INSTANTIATE_PERM_FUNCS(int, double)
INSTANTIATE_PERM_FUNCS(int, float)
INSTANTIATE_PERM_FUNCS(int, int)
INSTANTIATE_INVPERM_FUNC(std::int32_t)
INSTANTIATE_INVPERM_FUNC(std::int64_t)
INSTANTIATE_ISPERM_FUNC(std::int32_t)
INSTANTIATE_ISPERM_SERIAL_FUNC(std::int32_t)
INSTANTIATE_ISPERM_FUNC(std::int64_t)
INSTANTIATE_ISPERM_SERIAL_FUNC(std::int64_t)
INSTANTIATE_RANDPERM_FUNC(int)
INSTANTIATE_PERM_ROW_PTR_FUNC(int)
INSTANTIATE_PERMUTE_MAT_STRUCT_FUNC(int, int)
INSTANTIATE_PERMUTE_MAT_STRUCT_FUNC(int64_t, int64_t)
INSTANTIATE_PERMUTE_MAT_VALUES_FUNC(int, int, double)
INSTANTIATE_PERMUTE_MAT_VALUES_FUNC(int, int, float)
INSTANTIATE_PERMUTE_MAT_VALUES_FUNC(int64_t, int64_t, double)
INSTANTIATE_PERMUTE_MAT_VALUES_FUNC(int64_t, int64_t, float)
} // namespace matrix_utils