#include "tree.hpp"

#include "matrix_utils.hpp"
#include "permutation.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <numeric>
#include <omp.h>

namespace graph {

template <typename ROWTYPE, typename COLTYPE>
void eliminationTree(const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,
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
COLTYPE parentToChildSibling(const COLTYPE nnodes, const COLTYPE base,
                             const COLTYPE *parent, COLTYPE *first_child,
                             COLTYPE *next_sibling, COLTYPE *roots,
                             COLTYPE *child_count) {
  std::fill(first_child, first_child + nnodes,
            std::numeric_limits<COLTYPE>::max());
  std::fill(next_sibling, next_sibling + nnodes,
            std::numeric_limits<COLTYPE>::max());
  if (child_count != nullptr) {
    std::fill(child_count, child_count + nnodes, 0);
  }

  COLTYPE nroots = 0;
  for (COLTYPE i = 0; i < nnodes; i++) {
    auto parent_i = parent[i] - base;
    if (parent_i != i) {
      if (child_count != nullptr) {
        child_count[parent_i]++;
      }
      auto parent_first_child = first_child[parent_i];
      first_child[parent_i] = i;
      next_sibling[i] = parent_first_child;
    } else {
      if (roots != nullptr) {
        roots[nroots] = i;
      }
      nroots++;
    }
  }
  return nroots;
}

template <typename COLTYPE>
COLTYPE parentTopologicalOrder(const COLTYPE nnodes, const COLTYPE base,
                               const COLTYPE *parent,
                               const COLTYPE *child_count, COLTYPE *scratch,
                               COLTYPE *perm, COLTYPE *prefix) {
  prefix[0] = base;
  if (nnodes == 0) {
    return 0;
  }

  std::copy(child_count, child_count + nnodes, scratch);

  COLTYPE processed = 0;
  COLTYPE level = 0;
  for (COLTYPE i = 0; i < nnodes; i++) {
    if (scratch[i] == 0) {
      perm[processed++] = i + base;
    }
  }
  prefix[1] = processed + base;

  while (processed != nnodes) {
    const auto level_begin = prefix[level] - base;
    const auto level_end = prefix[level + 1] - base;
    if (level_begin == level_end) {
      return level + 1;
    }

    for (auto pos = level_begin; pos < level_end; pos++) {
      const auto node = perm[pos] - base;
      const auto parent_label = parent[node];
      if (parent_label < base) {
        return level + 1;
      }
      const auto parent_node = parent_label - base;
      if (parent_node == node) {
        continue;
      }
      if (parent_node >= nnodes) {
        return level + 1;
      }
      if (--scratch[parent_node] == 0) {
        perm[processed++] = parent_label;
      }
    }

    ++level;
    prefix[level + 1] = processed + base;
  }

  return level + 1;
}

template <typename COLTYPE>
void PostOrder<COLTYPE>::buildChildren(const COLTYPE nnodes, const COLTYPE base,
                                       const COLTYPE *parent) {
  _childrenPrefix.resize(nnodes + 1);
  _roots.clear();
  _roots.reserve(nnodes);
  std::fill(_childrenPrefix.begin(), _childrenPrefix.end(), 0);

  // Count children for each parent. The count is shifted by one slot so the
  // inclusive scan below directly produces CSR-style child-list offsets.
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

  // Reuse _childrenPrefix as the insertion cursor while filling _children.
  // After this loop, each entry has advanced from the start of a child range
  // to the end of that range.
  for (COLTYPE i = 0; i < nnodes; i++) {
    if (parent[i] != i + base) {
      _children[_childrenPrefix[parent[i] - base]++] = i;
    }
  }

  // Restore CSR offsets from the advanced cursors:
  // [end(0), end(1), ..., total] -> [0, end(0), end(1), ...].
  std::rotate(_childrenPrefix.rbegin(), _childrenPrefix.rbegin() + 1,
              _childrenPrefix.rend());
  _childrenPrefix[0] = 0;

  // for(COLTYPE i = 0; i < nnodes; i++){
  //   std::sort(_children.begin() + _childrenPrefix[i],
  //             _children.begin() + _childrenPrefix[i + 1]);
  // }
}

template <typename COLTYPE>
void PostOrder<COLTYPE>::dfs(const COLTYPE root, const COLTYPE base,
                             COLTYPE *&post) {
  for (COLTYPE i = _childrenPrefix[root]; i < _childrenPrefix[root + 1]; i++) {
    dfs(_children[i], base, post);
  }
  *post = root + base;
  post++;
}

// Compute a postorder permutation for the elimination tree. The input parent
// array stores child -> parent edges using old node labels. buildChildren first
// converts that parent vector into a CSR-style parent -> children adjacency,
// then DFS emits the old nodes in postorder. The final loop rewrites the parent
// array into postorder numbering.
template <typename COLTYPE>
void PostOrder<COLTYPE>::apply(const COLTYPE nnodes, const COLTYPE base,
                               const COLTYPE *parent, COLTYPE *permed_parent,
                               COLTYPE *perm, COLTYPE *iperm) {
  auto perm_cp = perm;
  buildChildren(nnodes, base, parent);

  // DFS writes the old node ids in postorder:
  // perm[new_id] = old_id + base.
  for (auto root : _roots) {
    dfs(root, base, perm_cp);
  }
  assert(matrix_utils::isPermutationSerial(nnodes, base, perm));

  // iperm reverses the postorder map:
  // iperm[old_id] = new_id + base.
  matrix_utils::invPerm(nnodes, base, perm, iperm);
  assert(matrix_utils::isPermutationSerial(nnodes, base, iperm));

#pragma omp parallel for
  for (COLTYPE new_id = 0; new_id < nnodes; new_id++) {
    // Relabel the original parent edge old_node -> old_parent into
    // postordered numbering:
    // permed_parent[new_id] = new_parent + base.
    permed_parent[new_id] = iperm[parent[perm[new_id] - base] - base];
  }
}

template <typename COLTYPE>
void PostOrderNoRecur<COLTYPE>::apply(const COLTYPE nnodes, const COLTYPE base,
                                       const COLTYPE *parent,
                                       COLTYPE *permed_parent, COLTYPE *perm,
                                       COLTYPE *iperm) {
  _firstChild.resize(nnodes);
  _nextSibling.resize(nnodes);
  _roots.resize(nnodes);
  const auto nroots = parentToChildSibling(nnodes, base, parent,
                                          _firstChild.data(),
                                          _nextSibling.data(), _roots.data());
  _roots.resize(nroots);
  auto perm_cp = perm;

  // Iterative postorder traversal. A node is emitted only after all children
  // have been consumed from its first-child/next-sibling list. _roots starts
  // as the root list above, then serves as the explicit DFS stack.
  while (!_roots.empty()) {
    auto root = _roots.back();
    auto child = _firstChild[root];
    if (child == std::numeric_limits<COLTYPE>::max()) {
      _roots.pop_back();
      *perm_cp = root + base;
      perm_cp++;
    } else {
      _roots.push_back(child);
      _firstChild[root] = _nextSibling[child];
    }
  }
  assert(matrix_utils::isPermutationSerial(nnodes, base, perm));

  // iperm reverses the postorder map:
  // iperm[old_id] = new_id + base.
  matrix_utils::invPerm(nnodes, base, perm, iperm);
  assert(matrix_utils::isPermutationSerial(nnodes, base, iperm));

#pragma omp parallel for
  for (COLTYPE new_id = 0; new_id < nnodes; new_id++) {
    // Relabel the original parent edge old_node -> old_parent into
    // postordered numbering:
    // permed_parent[new_id] = new_parent + base.
    permed_parent[new_id] = iperm[parent[perm[new_id] - base] - base];
  }
}

template <typename COLTYPE>
void subtreeSize(const COLTYPE nnodes, const COLTYPE base,
                 const COLTYPE *parent, COLTYPE *subtree_size) {
  std::fill(subtree_size, subtree_size + nnodes, 1);
  for (COLTYPE i = 0; i < nnodes; i++) {
    auto k = parent[i] - base;
    if (k != i)
      subtree_size[k] += subtree_size[i];
  }
}

#define INSTANTIATE_TREE(ROWTYPE, COLTYPE)                                     \
  template void eliminationTree<ROWTYPE, COLTYPE>(                             \
      const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,              \
      COLTYPE *parent, COLTYPE *ancestor);                                     \
  template COLTYPE parentToChildSibling<COLTYPE>(                              \
      const COLTYPE nnodes, const COLTYPE base, const COLTYPE *parent,         \
      COLTYPE *first_child, COLTYPE *next_sibling, COLTYPE *roots,             \
      COLTYPE *child_count);                                                   \
  template COLTYPE parentTopologicalOrder<COLTYPE>(                            \
      const COLTYPE nnodes, const COLTYPE base, const COLTYPE *parent,         \
      const COLTYPE *child_count, COLTYPE *scratch, COLTYPE *perm,             \
      COLTYPE *prefix);                                                        \
  template void subtreeSize<COLTYPE>(const COLTYPE nnodes, const COLTYPE base, \
                                     const COLTYPE *parent,                    \
                                     COLTYPE *subtree_size);                   \
  template class PostOrder<COLTYPE>;                                           \
  template class PostOrderNoRecur<COLTYPE>;

INSTANTIATE_TREE(std::int32_t, std::int32_t)
INSTANTIATE_TREE(std::int64_t, std::int64_t)

} // namespace graph
