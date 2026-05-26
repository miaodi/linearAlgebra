#pragma once

#include <vector>

namespace graph
{

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
void eliminationTree( const COLTYPE nnodes, const ROWTYPE* ai, const COLTYPE* aj, COLTYPE* parent, COLTYPE* ancestor );

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
template <bool FillRoots, typename COLTYPE>
COLTYPE parentToChildSibling( const COLTYPE nnodes,
                              const COLTYPE base,
                              const COLTYPE* parent,
                              COLTYPE* first_child,
                              COLTYPE* next_sibling,
                              COLTYPE* roots = nullptr,
                              COLTYPE* child_count = nullptr );

template <typename COLTYPE>
COLTYPE parentToChildSibling( const COLTYPE nnodes,
                              const COLTYPE base,
                              const COLTYPE* parent,
                              COLTYPE* first_child,
                              COLTYPE* next_sibling,
                              COLTYPE* roots = nullptr,
                              COLTYPE* child_count = nullptr );

/// @brief Convert a parent array into grouped child adjacency in CSR form.
///
/// Array positions are zero-based, while stored parent labels include base. A
/// root is encoded as parent[i] == i + base. child_offsets must hold nnodes + 1
/// entries, and children must hold at least nnodes minus the number of roots.
/// The children of node i occupy children[child_offsets[i]:child_offsets[i+1]).
/// If FillRoots is true, roots is filled with zero-based roots.
/// @tparam FillRoots whether to fill the roots output array
/// @tparam COLTYPE column index type
/// @return number of roots in the tree or forest
template <bool FillRoots, typename COLTYPE>
COLTYPE parentToChildCSR( const COLTYPE nnodes,
                          const COLTYPE base,
                          const COLTYPE* parent,
                          COLTYPE* child_offsets,
                          COLTYPE* children,
                          COLTYPE* roots = nullptr );

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
COLTYPE parentTopologicalOrder( const COLTYPE nnodes,
                                const COLTYPE base,
                                const COLTYPE* parent,
                                const COLTYPE* child_count,
                                COLTYPE* scratch,
                                COLTYPE* perm,
                                COLTYPE* prefix );

/// @brief Elimination-tree analysis and scheduling data for sparse Cholesky.
///
/// `compute()` expects a square symmetric matrix pattern stored as full CSR or
/// lower-triangular CSR. Row column indices must be sorted in ascending order
/// because the underlying elimination-tree routine scans each row until it
/// reaches the diagonal/upper triangle. Upper-triangular-only CSR is not valid
/// input for computing the tree with this routine.
///
/// Stored labels use the same base as `ai[0]`. A root is encoded as
/// `parent[i] == i + base`.
template <typename COLTYPE>
class EliminationTree
{
public:
    COLTYPE nnodes() const { return _nnodes; }
    COLTYPE base() const { return _base; }
    COLTYPE nroots() const { return _nroots; }
    COLTYPE topologicalLevels() const { return _topologicalLevels; }

    COLTYPE const* parent() const { return _parent.data(); }
    COLTYPE const* roots() const { return _roots.data(); }
    COLTYPE const* childOffsets() const { return _childOffsets.data(); }
    COLTYPE const* children() const { return _children.data(); }
    COLTYPE const* childCounts() const { return _childCounts.data(); }
    COLTYPE const* topologicalOrder() const { return _topologicalOrder.data(); }
    COLTYPE const* topologicalPrefix() const { return _topologicalPrefix.data(); }

    /// @brief Compute parent from matrix structure, then analyze the tree.
    template <typename ROWTYPE>
    bool compute( const COLTYPE nnodes_in, const ROWTYPE* ai, const COLTYPE* aj )
    {
        if ( ai == nullptr || aj == nullptr )
        {
            return false;
        }

        std::vector<COLTYPE> ancestor( nnodes_in );
        _parent.resize( nnodes_in );
        eliminationTree( nnodes_in, ai, aj, _parent.data(), ancestor.data() );
        return analyze( nnodes_in, static_cast<COLTYPE>( ai[0] ), _parent.data() );
    }

    /// @brief Build child and schedule data from an existing parent array.
    ///
    /// `parent_in` must contain `nnodes` base-labeled entries. Every parent label
    /// must be in `[base, base + nnodes)`. Roots must satisfy
    /// `parent_in[i] == i + base`. Cycles or invalid forests are rejected when the
    /// child-before-parent topological order cannot cover all nodes.
    bool analyze( const COLTYPE nnodes_in, const COLTYPE base_in, const COLTYPE* parent_in );

private:
    COLTYPE _nnodes{};
    COLTYPE _base{};
    COLTYPE _nroots{};
    COLTYPE _topologicalLevels{};
    std::vector<COLTYPE> _parent;
    std::vector<COLTYPE> _roots;
    std::vector<COLTYPE> _childOffsets;
    std::vector<COLTYPE> _children;
    std::vector<COLTYPE> _childCounts;
    std::vector<COLTYPE> _topologicalOrder;
    std::vector<COLTYPE> _topologicalPrefix;
    std::vector<COLTYPE> _scratch;
};

/// @brief Compute a postorder permutation of an elimination tree.
///
/// Array positions are zero-based, while stored node labels include base.
/// perm[new_id] = old_id + base means old node old_id becomes new postorder
/// position new_id. iperm[old_id] = new_id + base is the inverse map.
/// permed_parent[new_id] stores the new parent's label, also with base.
/// @tparam COLTYPE column index type
template <typename COLTYPE>
class PostOrder
{
public:
    void apply( const COLTYPE nnodes,
                const COLTYPE base,
                const COLTYPE* parent,
                COLTYPE* permed_parent,
                COLTYPE* perm,
                COLTYPE* iperm );

private:
    void buildChildren( const COLTYPE nnodes, const COLTYPE base, const COLTYPE* parent );

    void dfs( const COLTYPE root, const COLTYPE base, COLTYPE*& post );

    // internal data, 0-based indexing
    std::vector<COLTYPE> _childrenPrefix;
    std::vector<COLTYPE> _children;
    std::vector<COLTYPE> _roots;
};

template <typename COLTYPE>
class PostOrderNoRecur
{
public:
    void apply( const COLTYPE nnodes,
                const COLTYPE base,
                const COLTYPE* parent,
                COLTYPE* permed_parent,
                COLTYPE* perm,
                COLTYPE* iperm );

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
void subtreeSize( const COLTYPE nnodes, const COLTYPE base, const COLTYPE* parent, COLTYPE* subtree_size );

} // namespace graph
