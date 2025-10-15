#pragma once
#include <deque>
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
/// @param matching_row Output array for matching from rows to columns (size
/// rows). For example, if row i is matched to column j, matching_row[i] = j
/// @param matching_col Output array for matching from columns to rows (size
/// rows). For example, if column j is matched to row i, matching_col[j] = i
/// Unmatched columns will have matching_col[j] =
/// std::numeric_limits<COLTYPE>::max()
template <typename ROWTYPE, typename COLTYPE>
void MaximumMatching(const COLTYPE rows, ROWTYPE const *ai, COLTYPE const *aj,
                     COLTYPE *matching_row, COLTYPE *matching_col);

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
class HungarianAlgorithm {
public:
  HungarianAlgorithm() = default;
  void operator()(const COLTYPE n, ROWTYPE const *ai, COLTYPE const *aj,
                  VALTYPE const *av, COLTYPE *matching_row,
                  COLTYPE *matching_col, VALTYPE *potential_row,
                  VALTYPE *potential_col);

private:
  void initialize();
  void initialize_row(const COLTYPE row);
  void augment(const COLTYPE t);
  bool update_potentials();
  void prep_row(const COLTYPE row);
  bool match_row(const COLTYPE row);

private:
  std::vector<COLTYPE> parent;
  std::vector<int> S; // rows in the alternating tree Z
  std::vector<int> T; // columns in the alternating tree Z
  std::vector<VALTYPE> min_slack;
  // std::vector<VALTYPE> min_slack_cpy;
  std::deque<COLTYPE> Q;
  static constexpr COLTYPE INVALID = std::numeric_limits<COLTYPE>::max();

  // Data that does not owned by this class
  VALTYPE *potential_row;
  VALTYPE *potential_col;
  COLTYPE *matching_row;
  COLTYPE *matching_col;

  COLTYPE n;
  ROWTYPE const *ai;
  COLTYPE const *aj;
  VALTYPE const *av;
  ROWTYPE base;
};
} // namespace reordering