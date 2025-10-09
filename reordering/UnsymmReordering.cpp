#include "UnsymmReordering.hpp"

namespace reordering {

// MaximumMatching function definition
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
      matching_row[curr] = v;
      matching_col[v] = curr;
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
          found_augmenting_path = true;
          augment_path(v, curr);
          return true;
        } else if (!visited[v]) {
          visited[v] = true;
          // Continue searching
          auto curr_1 = matching_col[v];
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

// HungarianAlgorithm member function definitions
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void HungarianAlgorithm<ROWTYPE, COLTYPE, VALTYPE>::operator()(
    const COLTYPE n, ROWTYPE const *ai, COLTYPE const *aj, VALTYPE const *av,
    COLTYPE *matching_row, COLTYPE *matching_col, VALTYPE *potential_row,
    VALTYPE *potential_col) {
  this->n = n;
  this->ai = ai;
  this->aj = aj;
  this->av = av;
  this->matching_row = matching_row;
  this->matching_col = matching_col;
  this->potential_row = potential_row;
  this->potential_col = potential_col;

  initialize();

  for (COLTYPE i = 0; i < n; i++) {
    initialize_row();
    match_row(i);
  }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void HungarianAlgorithm<ROWTYPE, COLTYPE, VALTYPE>::initialize() {
  parent.resize(n, INVALID);
  S.resize(n, false);
  T.resize(n, false);
  min_slack.resize(n, std::numeric_limits<VALTYPE>::max());

  std::fill_n(matching_row, n, INVALID);
  std::fill_n(matching_col, n, INVALID);
  const ROWTYPE base = ai[0];
  for (COLTYPE j = 0; j < n; j++) {
    VALTYPE min_cost = std::numeric_limits<VALTYPE>::max();
    for (ROWTYPE i = ai[0] - base; i < ai[n] - base; i++) {
      min_cost = std::min(min_cost, av[i]);
    }
    potential_row[j] =
        min_cost == std::numeric_limits<VALTYPE>::max() ? 0 : min_cost;
  }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void HungarianAlgorithm<ROWTYPE, COLTYPE, VALTYPE>::initialize_row() {
  std::fill(parent.begin(), parent.end(), INVALID);
  std::fill(S.begin(), S.end(), false);
  std::fill(T.begin(), T.end(), false);
  std::fill(min_slack.begin(), min_slack.end(),
            std::numeric_limits<VALTYPE>::max());
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void HungarianAlgorithm<ROWTYPE, COLTYPE, VALTYPE>::augment(const COLTYPE t,
                                                            const COLTYPE s) {
  COLTYPE curT = t;
  COLTYPE curS = s;
  const ROWTYPE base = ai[0];
  while (true) {
    const COLTYPE nextT = matching_row[curS] - base;
    matching_row[curS] = curT + base;
    matching_col[curT] = curS + base;
    if (nextT == INVALID)
      break;
    curS = parent[nextT];
    curT = nextT;
  }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void HungarianAlgorithm<ROWTYPE, COLTYPE, VALTYPE>::update_potentials() {
  VALTYPE delta = std::numeric_limits<VALTYPE>::max();
  for (COLTYPE j = 0; j < n; j++) {
    if (!T[j]) {
      delta = std::min(delta, min_slack[j]);
    }
  }
  for (COLTYPE i = 0; i < n; i++) {
    if (S[i]) {
      potential_row[i] += delta;
    }
  }
  for (COLTYPE j = 0; j < n; j++) {
    if (T[j]) {
      potential_col[j] -= delta;
    } else {
      min_slack[j] -= delta;
    }
  }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void HungarianAlgorithm<ROWTYPE, COLTYPE, VALTYPE>::visit_matched_col(
    const COLTYPE col) {
  const ROWTYPE base = ai[0];
  const auto nextS = matching_col[col] - base;
  // TODO: maybe not needed
  if (S[nextS])
    return;
  S[nextS] = true;
  Q.push_back(nextS);
  for (ROWTYPE i = ai[nextS] - base; i < ai[nextS + 1] - base; i++) {
    const COLTYPE j = aj[i] - base;
    if (!T[j]) {
      const VALTYPE cost = av[i] - potential_row[nextS] - potential_col[j];
      if (min_slack[j] > cost) {
        min_slack[j] = cost;
        parent[j] = nextS;
      }
    }
  }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void HungarianAlgorithm<ROWTYPE, COLTYPE, VALTYPE>::match_row(
    const COLTYPE row) {
  if (matching_row[row] != INVALID)
    return;
  S[row] = true;
  Q.clear();
  Q.push_back(row);
  const ROWTYPE base = ai[0];

  // initialize min_slack
  for (ROWTYPE i = ai[row] - base; i < ai[row + 1] - base; i++) {
    const COLTYPE j = aj[i] - base;
    min_slack[j] = av[i] - potential_row[row] - potential_col[j];
    parent[j] = row;
  }

  while (true) {
    while (Q.size()) {
      const auto curS = Q.front();
      Q.pop_front();
      for (ROWTYPE i = ai[curS] - base; i < ai[curS + 1] - base; i++) {
        const COLTYPE curT = aj[i] - base;
        const VALTYPE cost = av[i] - potential_row[curS] - potential_col[curT];

        if (cost <= std::numeric_limits<VALTYPE>::epsilon()) {
          // If cost is zero, consider this edge for the matching
          if (!T[curT]) {
            // if curT is not seen in the alternating tree rooted at row
            T[curT] = true;
            if (matching_col[curT] == INVALID) {
              // If curT is not matched, we found an augmenting path
              augment(curT, row);
              return;
            } else {
              // Otherwise, add the matched row to the tree
              visit_matched_col(curT);
            }
          }
        } else if (!T[curT] && min_slack[curT] > cost) {
          min_slack[curT] = cost;
          parent[curT] = curS;
        }
      }
    }

    update_potentials();

    // Add edges with zero slack to the tree
    for (COLTYPE j = 0; j < n; j++) {
      if (!T[j] && min_slack[j] <= std::numeric_limits<VALTYPE>::epsilon()) {
        T[j] = true;
        if (matching_col[j] == INVALID) {
          // Found an augmenting path
          augment(j, row);
          return;
        } else {
          visit_matched_col(j);
        }
      }
    }
  }
}
template void MaximumMatching<int, int>(const int rows, int const *ai,
                                        int const *aj, int *matching_row,
                                        int *matching_col);

template class HungarianAlgorithm<int, int, double>;

} // namespace reordering
