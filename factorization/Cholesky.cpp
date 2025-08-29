#include "Cholesky.hpp"
#include "matrix_utils.hpp"
#include "permutation.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <omp.h>

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

template <matrix_utils::ResizableCSRMatrixType CSRMatrixType>
void SkeletonGraph<CSRMatrixType>::operator()(const COLTYPE nnodes,
                                              const ROWTYPE *ai,
                                              const COLTYPE *aj,
                                              const COLTYPE *parent,
                                              CSRMatrixType &leaf) {
  const ROWTYPE base = ai[0];
  _subtree_size.resize(nnodes);
  SubtreeSize(nnodes, base, parent, _subtree_size.data());
#pragma omp parallel num_threads(_nthreads)
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    auto [start, end] =
        utils::LoadPrefixBalancedPartitionPos(ai, ai + nnodes, tid, nthreads);
    _XLEAFs[tid].clear();
    _XLEAFs[tid].push_back(0);
    _LEAFs[tid].clear();
    for (auto i = start; i < end; i++) {
      COLTYPE count = 0;
      auto j = ai[i] - base;
      if (j < ai[i + 1] - base && aj[j] - base < i) {
        _LEAFs[tid].push_back(aj[j++]);
        count++;
      }

      for (; j < ai[i + 1] - base && aj[j] - base < i; j++) {
        if (aj[j - 1] + _subtree_size[aj[j] - base] - 1 < aj[j]) {
          _LEAFs[tid].push_back(aj[j]);
          count++;
        }
      }
      _XLEAFs[tid].push_back(_XLEAFs[tid].back() + count);
    }

#pragma omp barrier
#pragma omp single
    {
      _XLEAF_prefix[0] = 0;
      for (int i = 0; i < _nthreads; i++) {
        _XLEAF_prefix[i + 1] = _XLEAF_prefix[i] + _LEAFs[i].size();
      }
      leaf.ResizeAI(nnodes + 1);
      leaf.ResizeAJ(_XLEAF_prefix.back());
    }
    for (auto i = start; i <= end; i++) {
      leaf.AI()[i] = _XLEAFs[tid][i - start] + _XLEAF_prefix[tid] + base;
    }
    COLTYPE pos = _XLEAF_prefix[tid];
    for (auto i : _LEAFs[tid]) {
      leaf.AJ()[pos++] = i;
    }
  }
}

template <matrix_utils::ResizableCSRMatrixType CSRMatrixType>
void SymbolicCholesky<CSRMatrixType>::operator()(
    const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,
    const COLTYPE *parent, const COLTYPE *XLEAF, const COLTYPE *LEAF,
    CSRMatrixType &L) {
  const auto base = ai[0];
#pragma omp parallel num_threads(_nthreads)
  {
    const int tid = omp_get_thread_num();
    auto [start, end] =
        utils::LoadPrefixBalancedPartitionPos(ai, ai + nnodes, tid, _nthreads);
    _ais[tid].clear();
    _ais[tid].push_back(0);
    _ajs[tid].clear();
    for (auto i = start; i < end; i++) {
      ROWTYPE count = 0;
      for (auto j = XLEAF[i] - base; j < XLEAF[i + 1] - base; j++) {
        COLTYPE node = LEAF[j] - base;
        COLTYPE nextleaf;
        if (j + 1 == XLEAF[i + 1] - base) {
          nextleaf = i;
        } else {
          nextleaf = LEAF[j + 1] - base;
        }
        while (node < nextleaf) {
          _ajs[tid].push_back(node + base);
          count++;
          node = parent[node] - base;
        }
      }
      _ajs[tid].push_back(i + base);
      _ais[tid].push_back(_ais[tid].back() + count + 1);
    }

#pragma omp barrier
#pragma omp single
    {
      _ais_prefix[0] = 0;
      for (int i = 0; i < _nthreads; i++) {
        _ais_prefix[i + 1] = _ais_prefix[i] + _ajs[i].size();
      }
      L.ResizeAI(nnodes + 1);
      L.ResizeAJ(_ais_prefix.back());
      L.rows = nnodes;
      L.cols = nnodes;
    }
    for (auto i = start; i <= end; i++) {
      L.AI()[i] = _ais[tid][i - start] + _ais_prefix[tid] + base;
    }
    COLTYPE pos = _ais_prefix[tid];
    for (auto i : _ajs[tid]) {
      L.AJ()[pos++] = i;
    }
  }
  L.ResizeAV(_ais_prefix.back());
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

template class SkeletonGraph<
    ::matrix_utils::CSRMatrix<std::int32_t, std::int32_t, double>>;
template class SkeletonGraph<
    ::matrix_utils::CSRMatrix<std::int64_t, std::int64_t, double>>;
template class SymbolicCholesky<
    ::matrix_utils::CSRMatrix<std::int32_t, std::int32_t, double>>;
template class SymbolicCholesky<
    ::matrix_utils::CSRMatrix<std::int64_t, std::int64_t, double>>;
} // namespace factorization