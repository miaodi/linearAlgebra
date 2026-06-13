#pragma once

#include <limits>

namespace reordering
{

/// @brief Find a maximum-cardinality matching for a square bipartite graph in CSR format.
/// @tparam ROWTYPE Type for row pointer indices.
/// @tparam COLTYPE Type for row and column indices.
/// @param rows Number of rows and columns in the square graph.
/// @param ai Row pointer array, size rows + 1. ai[0] determines the index base.
/// @param aj Column indices array, with the same index base as ai.
/// @param matching_row Output array, size rows. If row i is matched to column j,
/// matching_row[i] = j + base. Unmatched rows are set to std::numeric_limits<COLTYPE>::max().
/// @param matching_col Output array, size rows. If column j is matched to row i,
/// matching_col[j] = i + base. Unmatched columns are set to std::numeric_limits<COLTYPE>::max().
/// @return Number of matched row/column pairs.
template <typename ROWTYPE, typename COLTYPE>
COLTYPE MaximumMatching( const COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj, COLTYPE* matching_row, COLTYPE* matching_col );

} // namespace reordering
