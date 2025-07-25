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
} // namespace factorization