#include "Cholesky.hpp"
#include <algorithm>
#include <cstdint>
namespace factorization {

template <typename ROWTYPE, typename COLTYPE>
void EliminationTree(const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,
                     COLTYPE *parent, COLTYPE *ancestor) {
  const ROWTYPE base = ai[0];
  COLTYPE jroot;
  for (COLTYPE i = 0; i < nnodes; i++) {
    parent[i] = i + base;
    ancestor[i] = i + base;
    for (ROWTYPE j_idx = ai[i] - base, jroot = aj[j_idx] - base;
         j_idx < ai[i + 1] - base && jroot < i;
         j_idx++, jroot = aj[j_idx] - base) {
      while (ancestor[jroot] - base != jroot) {
        COLTYPE l = ancestor[jroot] - base;
        ancestor[jroot] = i + base;
        jroot = l;
      }
      if (jroot == ancestor[jroot] - base) {
        parent[jroot] = i + base;
        ancestor[jroot] = i + base;
      }
    }
  }
}

template <typename ROWTYPE, typename COLTYPE>
void RowSubtreeSize(const COLTYPE nnodes, const ROWTYPE base,
                    const COLTYPE *parent, ROWTYPE *row_size) {
  std::fill(row_size, row_size + nnodes, 1);
  for (COLTYPE i = 0; i < nnodes; i++) {
    auto k = parent[i] - base;
    if (k != i)
      row_size[k] += row_size[i];
  }
}

template <typename ROWTYPE, typename COLTYPE>
void RowCount(const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,
              const COLTYPE *parent, ROWTYPE *row_count, COLTYPE *mark) {
  const ROWTYPE base = ai[0];
  for (COLTYPE i = 0; i < nnodes; i++) {
    row_count[i] = 1;
    mark[i] = i;
    for (ROWTYPE j_idx = ai[i] - base, jroot = aj[j_idx] - base;
         j_idx < ai[i + 1] - base && jroot < i;
         j_idx++, jroot = aj[j_idx] - base) {
      while (mark[jroot] != i) {
        row_count[i] += 1;
        mark[jroot] = i;
        jroot = parent[jroot] - base;
      }
    }
  }
}

// instantiate for common types
#define INSTANTIATE_CHOLESKY(ROWTYPE, COLTYPE)                                 \
  template void EliminationTree<ROWTYPE, COLTYPE>(                             \
      const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,              \
      COLTYPE *parent, COLTYPE *ancestor);                                     \
  template void RowSubtreeSize<ROWTYPE, COLTYPE>(                              \
      const COLTYPE nnodes, const ROWTYPE base, COLTYPE const *parent,         \
      ROWTYPE *row_size);                                                      \
  template void RowCount<ROWTYPE, COLTYPE>(                                    \
      const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,              \
      const COLTYPE *parent, ROWTYPE *row_count, COLTYPE *mark);

INSTANTIATE_CHOLESKY(std::int32_t, std::int32_t)
INSTANTIATE_CHOLESKY(std::int64_t, std::int64_t)
} // namespace factorization