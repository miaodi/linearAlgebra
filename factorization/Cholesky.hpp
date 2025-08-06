#pragma once

namespace factorization {

/// @brief Compute the elimination tree of a symmetric matrix. Note that the
/// input matrix should be either full matrix or lower triangular matrix.
/// Algorithm 4.2 in
/// @cite scott2023algorithms
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

template <typename ROWTYPE, typename COLTYPE>
void RowSubtreeSize(const COLTYPE nnodes, const ROWTYPE base,
                    const COLTYPE *parent, ROWTYPE *row_size);

// @brief Compute the row count of each row in L of the Cholesky factorization
/// @tparam ROWTYPE row index type
/// @tparam COLTYPE column index type
/// @param nnodes number of nodes
/// @param base base index of the matrix (usually 0 or 1)
/// @param parent parent vector from the elimination tree
/// @param row_count output vector containing the count of rows in L
template <typename ROWTYPE, typename COLTYPE>
void RowCount(const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,
              const COLTYPE *parent, ROWTYPE *row_count, COLTYPE *mark);
} // namespace factorization