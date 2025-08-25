#include "Cholesky.hpp"
#include "permutation.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>

namespace factorization {

template <typename ROWTYPE, typename COLTYPE>
void EliminationTree(const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,
                     COLTYPE *parent, COLTYPE *ancestor) {
  const ROWTYPE base = ai[0];
  COLTYPE jroot;
  for (COLTYPE i = 0; i < nnodes; i++) {
    parent[i] = i + base;
    ancestor[i] = i + base;
    for (ROWTYPE j_idx = ai[i] - base; j_idx < ai[i + 1] - base; j_idx++) {
      jroot = aj[j_idx] - base;
      if (jroot >= i) // break if jroot is not in the lower triangle
        break;

      while (ancestor[jroot] - base != jroot && ancestor[jroot] - base != i) {
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

template <typename COLTYPE>
void PostOrder<COLTYPE>::BuildChildren(const COLTYPE nnodes, const COLTYPE base,
                                       const COLTYPE *parent) {
  _childrenPrefix.resize(nnodes + 1);
  _roots.clear();
  std::fill(_childrenPrefix.begin(), _childrenPrefix.end(), 0);
  for (COLTYPE i = 0; i < nnodes; i++) {
    if (parent[i] != i + base) {
      _childrenPrefix[parent[i] - base + 1]++;
    } else {
      _roots.push_back(i);
    }
  }
  std::inclusive_scan(_childrenPrefix.begin(), _childrenPrefix.end(),
                      _childrenPrefix.begin());

  _children.resize(_childrenPrefix.back());
  for (COLTYPE i = 0; i < nnodes; i++) {
    if (parent[i] != i + base) {
      _children[_childrenPrefix[parent[i] - base]++] = i;
    }
  }
  std::rotate(_childrenPrefix.rbegin(), _childrenPrefix.rbegin() + 1,
              _childrenPrefix.rend());
  _childrenPrefix[0] = 0;

  // for(COLTYPE i = 0; i < nnodes; i++){
  //   std::sort(_children.begin() + _childrenPrefix[i],
  //             _children.begin() + _childrenPrefix[i + 1]);
  // }
}

template <typename COLTYPE>
void PostOrder<COLTYPE>::DFS(const COLTYPE root, const COLTYPE base,
                             COLTYPE *&post) {
  if (_childrenPrefix[root] == _childrenPrefix[root + 1]) {
    *post = root + base;
    post++;
  } else {
    for (COLTYPE i = _childrenPrefix[root]; i < _childrenPrefix[root + 1];
         i++) {
      DFS(_children[i], base, post);
    }
    *post = root + base;
    post++;
  }
}

template <typename COLTYPE>
void PostOrder<COLTYPE>::operator()(const COLTYPE nnodes, const COLTYPE base,
                                    const COLTYPE *parent,
                                    COLTYPE *permed_parent, COLTYPE *perm,
                                    COLTYPE *iperm) {
  auto perm_cp = perm;
  BuildChildren(nnodes, base, parent);
  for (auto root : _roots) {
    DFS(root, base, perm_cp);
  }
  assert(matrix_utils::isPermutationSerial(nnodes, base, perm));
  matrix_utils::invPerm(nnodes, base, perm, iperm);
  assert(matrix_utils::isPermutationSerial(nnodes, base, iperm));

#pragma omp parallel for
  for (COLTYPE i = 0; i < nnodes; i++) {
    permed_parent[i] = iperm[parent[perm[i] - base] - base];
  }
}

template <typename COLTYPE>
void SubtreeSize(const COLTYPE nnodes, const COLTYPE base,
                 const COLTYPE *parent, COLTYPE *subtree_size) {
  std::fill(subtree_size, subtree_size + nnodes, 1);
  for (COLTYPE i = 0; i < nnodes; i++) {
    auto k = parent[i] - base;
    if (k != i)
      subtree_size[k] += subtree_size[i];
  }
}

template <typename ROWTYPE, typename COLTYPE>
void NNZCount(const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,
              const COLTYPE *parent, ROWTYPE *row_count, ROWTYPE *col_count,
              COLTYPE *mark) {
  const ROWTYPE base = ai[0];
  if (col_count != nullptr) {
    std::fill(col_count, col_count + nnodes, 1);
  }
  for (COLTYPE i = 0; i < nnodes; i++) {
    row_count[i] = 1;
    mark[i] = i;
    for (ROWTYPE j_idx = ai[i] - base; j_idx < ai[i + 1] - base; j_idx++) {
      COLTYPE jroot = aj[j_idx] - base;
      if (jroot >= i) // break if jroot is not in the lower triangle
        break;
      while (mark[jroot] != i) {
        row_count[i] += 1;
        if (col_count != nullptr) {
          col_count[jroot] += 1;
        }
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
  template void SubtreeSize<COLTYPE>(const COLTYPE nnodes, const COLTYPE base, \
                                     COLTYPE const *parent,                    \
                                     COLTYPE *subtree_size);                   \
  template void NNZCount<ROWTYPE, COLTYPE>(                                    \
      const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,              \
      const COLTYPE *parent, ROWTYPE *row_count, ROWTYPE *col_count,           \
      COLTYPE *mark);                                                          \
  template class PostOrder<COLTYPE>;

INSTANTIATE_CHOLESKY(std::int32_t, std::int32_t)
INSTANTIATE_CHOLESKY(std::int64_t, std::int64_t)

} // namespace factorization