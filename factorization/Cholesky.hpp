#pragma once
#include <vector>

namespace factorization {

/// @brief Compute the elimination tree of a symmetric matrix. Note that the
/// input matrix should be either full matrix or lower triangular matrix.
/// Algorithm 4.2 in @cite scott2023algorithms
/// Note that the root is defined as parent[i] == i + base
/// @tparam ROWTYPE row index type
/// @tparam COLTYPE column index type
/// @param nnodes number of nodes
/// @param ai row index
/// @param aj column index
/// @param parent parent vector, output
/// @param ancestor ancestor vector, helper for path compression
template <typename ROWTYPE, typename COLTYPE>
void EliminationTree(const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,
                     COLTYPE *parent, COLTYPE *ancestor);

/// @brief Compute the post order of the elimination tree
/// @brief post[i] = j means node j is the i-th node in the post order
/// @tparam COLTYPE column index type
template <typename COLTYPE> class PostOrder {
public:
  void operator()(const COLTYPE nnodes, const COLTYPE base,
                  const COLTYPE *parent, COLTYPE *permed_parent, COLTYPE *perm,
                  COLTYPE *iperm);

private:
  void BuildChildren(const COLTYPE nnodes, const COLTYPE base,
                     const COLTYPE *parent);

  void DFS(const COLTYPE root, const COLTYPE base, COLTYPE *&post);

  // internal data, 0-based indexing
  std::vector<COLTYPE> _childrenPrefix;
  std::vector<COLTYPE> _children;
  std::vector<COLTYPE> _roots;
};

/// @brief Compute the subtree size of each node in the elimination tree
/// @brief Note that the elimination tree must be postordered
/// @tparam COLTYPE column index type
/// @param nnodes number of nodes
/// @param base base index of the matrix (usually 0 or 1)
/// @param parent parent vector from the elimination tree
/// @param subtree_size output vector containing the subtree size of each node
/// (including the node itself)
template <typename COLTYPE>
void SubtreeSize(const COLTYPE nnodes, const COLTYPE base,
                 const COLTYPE *parent, COLTYPE *subtree_size);

// @brief Compute the row count of each row in L of the Cholesky factorization
/// @tparam ROWTYPE row index type
/// @tparam COLTYPE column index type
/// @param nnodes number of nodes
/// @param base base index of the matrix (usually 0 or 1)
/// @param parent parent vector from the elimination tree
/// @param row_count output vector containing the count of rows in L
/// @param col_count output vector containing the count of columns in L
/// @param mark mark vector, helper for path compression
template <typename ROWTYPE, typename COLTYPE>
void NNZCount(const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,
              const COLTYPE *parent, ROWTYPE *row_count, ROWTYPE *col_count,
              COLTYPE *mark);
} // namespace factorization