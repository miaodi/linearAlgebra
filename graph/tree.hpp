#pragma once

#include <vector>

namespace graph {

/// @brief Compute the elimination tree of a symmetric matrix. The input matrix
/// should be either the full matrix or the lower triangular part.
///
/// This follows Algorithm 4.2 in @cite scott2023algorithms. A root is encoded
/// as parent[i] == i + base.
/// @tparam ROWTYPE row index type
/// @tparam COLTYPE column index type
/// @param nnodes number of nodes
/// @param ai row index
/// @param aj column index
/// @param parent parent vector, output
/// @param ancestor ancestor vector, helper for path compression
template <typename ROWTYPE, typename COLTYPE>
void eliminationTree(const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,
                     COLTYPE *parent, COLTYPE *ancestor);

/// @brief Convert a parent array into first-child/next-sibling child lists.
///
/// Array positions are zero-based, while stored parent labels include base. A
/// root is encoded as parent[i] == i + base. The output arrays must already be
/// sized for nnodes entries. first_child[p] stores the first zero-based child of
/// p and next_sibling[c] stores the next zero-based sibling of c, or
/// numeric_limits<COLTYPE>::max() when absent. If roots is non-null, it is
/// filled with zero-based roots. If child_count is non-null, it is filled with
/// the number of children for each zero-based node.
/// @tparam COLTYPE column index type
/// @return number of roots in the tree or forest
template <typename COLTYPE>
COLTYPE parentToChildSibling(const COLTYPE nnodes, const COLTYPE base,
                             const COLTYPE *parent, COLTYPE *first_child,
                             COLTYPE *next_sibling, COLTYPE *roots = nullptr,
                             COLTYPE *child_count = nullptr);

/// @brief Compute bottom-up topological levels from a parent-array forest.
///
/// Array positions are zero-based, while stored parent labels include base. A
/// root is encoded as parent[i] == i + base. child_count must contain the
/// number of children for each zero-based node, scratch must hold nnodes
/// entries, perm receives labels in child-before-parent order, and prefix
/// receives level boundaries using the same base. The caller can detect an
/// invalid parent forest by checking prefix[levels] - base == nnodes.
/// @tparam COLTYPE column index type
/// @return number of computed levels
template <typename COLTYPE>
COLTYPE parentTopologicalOrder(const COLTYPE nnodes, const COLTYPE base,
                               const COLTYPE *parent,
                               const COLTYPE *child_count, COLTYPE *scratch,
                               COLTYPE *perm, COLTYPE *prefix);

/// @brief Compute a postorder permutation of an elimination tree.
///
/// Array positions are zero-based, while stored node labels include base.
/// perm[new_id] = old_id + base means old node old_id becomes new postorder
/// position new_id. iperm[old_id] = new_id + base is the inverse map.
/// permed_parent[new_id] stores the new parent's label, also with base.
/// @tparam COLTYPE column index type
template <typename COLTYPE> class PostOrder {
public:
  void apply(const COLTYPE nnodes, const COLTYPE base, const COLTYPE *parent,
             COLTYPE *permed_parent, COLTYPE *perm, COLTYPE *iperm);

private:
  void buildChildren(const COLTYPE nnodes, const COLTYPE base,
                     const COLTYPE *parent);

  void dfs(const COLTYPE root, const COLTYPE base, COLTYPE *&post);

  // internal data, 0-based indexing
  std::vector<COLTYPE> _childrenPrefix;
  std::vector<COLTYPE> _children;
  std::vector<COLTYPE> _roots;
};

template <typename COLTYPE> class PostOrderNoRecur {
public:
  void apply(const COLTYPE nnodes, const COLTYPE base, const COLTYPE *parent,
             COLTYPE *permed_parent, COLTYPE *perm, COLTYPE *iperm);

  // internal data, 0-based indexing
  std::vector<COLTYPE> _roots;
  std::vector<COLTYPE> _firstChild;
  std::vector<COLTYPE> _nextSibling;
};

/// @brief Compute the subtree size of each node in the elimination tree
/// (including the node itself), following Algorithm 4.5 in
/// @cite scott2023algorithms.
///
/// A standard elimination-tree parent array has each non-root parent numbered
/// after its child, so this forward accumulation visits children before
/// parents. Preserve that property after any relabeling.
/// @tparam COLTYPE column index type
/// @param nnodes number of nodes
/// @param base base index of the matrix (usually 0 or 1)
/// @param parent parent vector from the elimination tree
/// @param subtree_size output vector containing the subtree size of each node
/// (including the node itself)
template <typename COLTYPE>
void subtreeSize(const COLTYPE nnodes, const COLTYPE base,
                 const COLTYPE *parent, COLTYPE *subtree_size);

} // namespace graph
