#pragma once
#include <functional>
#include <limits>
#include <vector>
namespace reordering {

/// @brief Find maximum matching for a bipartite graph represented in CSR format
/// @tparam ROWTYPE Type for row indices
/// @tparam COLTYPE Type for column indices
/// @param rows Number of rows (and columns) in the square matrix
/// @param ai Row pointer array (size rows + 1)
/// @param aj Column indices array (size nnz)
/// @param matching Output array to store the matching (size rows). matching[j]
/// = i means column j is matched to row i. Unmatched columns will have
/// matching[j] = std::numeric_limits<COLTYPE>::max()
template <typename ROWTYPE, typename COLTYPE>
void MaximumMatching(const COLTYPE rows, ROWTYPE const *ai, COLTYPE const *aj,
                     COLTYPE *matching_row, COLTYPE *matching_col) {
  const COLTYPE INVALID = std::numeric_limits<COLTYPE>::max();
  const ROWTYPE base = ai[0];
  std::vector<bool> visited(rows, false);
  std::vector<COLTYPE> parent(rows, INVALID);
  std::fill_n(matching_row, rows, INVALID);
  std::fill_n(matching_col, rows, INVALID);

  auto augment_path = [&](COLTYPE v, COLTYPE curr) {
    while (curr != INVALID) {
      COLTYPE next_v = matching_row[curr];
      matching_row[curr] = v + base;
      matching_col[v] = curr + base;
      v = next_v;
      if (v != INVALID)
        curr = parent[curr];
      else
        curr = INVALID;
    }
  };

  std::function<bool(COLTYPE)> bpm = [&](COLTYPE u) {
    std::fill(visited.begin(), visited.end(), false);
    std::fill(parent.begin(), parent.end(), INVALID);

    COLTYPE curr = u;
    while (curr != INVALID) {
      bool found_augmenting_path = false;
      for (ROWTYPE i = ai[curr - base] - base; i < ai[curr - base + 1] - base;
           i++) {
        COLTYPE v = aj[i] - base;
        if (matching_col[v] == INVALID) {
          // Found an unmatched vertex, augment the path
          augment_path(v, curr);
          return true;
        } else if (!visited[v]) {
          visited[v] = true;
          // Continue searching
          auto curr_1 = matching_col[v] - base;
          found_augmenting_path = true;
          parent[curr_1] = curr;
          curr = curr_1;
          break;
        }
      }
      if (!found_augmenting_path) {
        curr = parent[curr];
      }
    }
    return false;
  };

  for (COLTYPE u = 0; u < rows; u++) {
    if (matching_row[u] == INVALID)
      bpm(u);
  }
}
} // namespace reordering