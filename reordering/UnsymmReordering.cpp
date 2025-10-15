#include "UnsymmReordering.hpp"
#include <iostream>
#include <queue>
namespace reordering {
template <typename ROWTYPE, typename COLTYPE>
void MaximumMatching(const COLTYPE rows, ROWTYPE const *ai, COLTYPE const *aj,
                     COLTYPE *matching_row, COLTYPE *matching_col) {
  const COLTYPE INVALID = std::numeric_limits<COLTYPE>::max();
  const ROWTYPE base = ai[0];
  std::vector<bool> visited(rows, false);
  std::vector<COLTYPE> parent(rows, INVALID);
  std::fill_n(matching_row, rows, INVALID);
  std::fill_n(matching_col, rows, INVALID);

  auto augment_path = [&](COLTYPE t) {
    while (true) {
      COLTYPE s = parent[t];
      COLTYPE next_t = matching_row[s];
      matching_row[s] = t + base;
      matching_col[t] = s + base;
      if (next_t == INVALID)
        break;
      t = next_t - base;
    }
  };

  std::function<bool(COLTYPE)> bpm = [&](COLTYPE u) {
    std::fill(visited.begin(), visited.end(), false);
    std::fill(parent.begin(), parent.end(), INVALID);

    std::queue<COLTYPE> q;
    q.push(u);
    visited[u] = true;

    COLTYPE end_t = INVALID;
    while (!q.empty()) {
      COLTYPE s = q.front();
      q.pop();
      for (ROWTYPE i = ai[s] - base; i < ai[s + 1] - base; i++) {
        COLTYPE t = aj[i] - base;
        if (!visited[t]) {
          visited[t] = true;
          parent[t] = s;
          if (matching_col[t] == INVALID) {
            // Found an augmenting path
            end_t = t;
            break;
          } else {
            q.push(matching_col[t] - base);
          }
        }
      }
      if (end_t != INVALID)
        break;
    }
    if (end_t != INVALID) {
      augment_path(end_t);
      return true;
    }
    return false;
  };

  for (COLTYPE u = 0; u < rows; u++) {
    if (matching_row[u] == INVALID)
      if (!bpm(u)) {
        matching_row[u] = u + base;
        matching_col[u] = u + base;
        std::cerr << "Warning: failed to find augmenting path for row " << u
                  << std::endl;
      }
  }
}

// HungarianAlgorithm member function definitions
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void HungarianAlgorithm<ROWTYPE, COLTYPE, VALTYPE>::operator()(
    const COLTYPE n, ROWTYPE const *ai, COLTYPE const *aj, VALTYPE const *av,
    COLTYPE *matching_row, COLTYPE *matching_col, VALTYPE *potential_row,
    VALTYPE *potential_col) {
  // Store the input data
  this->n = n;
  this->ai = ai;
  this->base = ai[0];
  this->aj = aj;
  this->av = av;
  this->matching_row = matching_row;
  this->matching_col = matching_col;
  this->potential_row = potential_row;
  this->potential_col = potential_col;

  initialize();

  for (COLTYPE i = 0; i < n; i++) {
    if (matching_row[i] != INVALID)
      return;
    initialize_row(i);
    if (!match_row(i)) {
      this->matching_row[i] = i + base;
      this->matching_col[i] = i + base;
      std::cerr << "Warning: failed to find augmenting path for row " << i
                << std::endl;
    }
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
  for (COLTYPE i = 0; i < n; i++) {
    VALTYPE min_cost = std::numeric_limits<VALTYPE>::max();
    for (ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; j++) {
      min_cost = std::min(min_cost, av[j]);
    }
    potential_row[i] =
        min_cost == std::numeric_limits<VALTYPE>::max() ? 0 : min_cost;
  }
  std::fill_n(potential_col, n, static_cast<VALTYPE>(0));
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void HungarianAlgorithm<ROWTYPE, COLTYPE, VALTYPE>::initialize_row(
    const COLTYPE row) {
  std::fill(parent.begin(), parent.end(), INVALID);
  std::fill(S.begin(), S.end(), false);
  std::fill(T.begin(), T.end(), false);
  std::fill(min_slack.begin(), min_slack.end(),
            std::numeric_limits<VALTYPE>::max());
  Q.clear();
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void HungarianAlgorithm<ROWTYPE, COLTYPE, VALTYPE>::augment(COLTYPE t) {
  const ROWTYPE base = ai[0];
  while (true) {
    COLTYPE s = parent[t];
    COLTYPE next_t =
        matching_row[s] == INVALID ? INVALID : matching_row[s] - base;
    matching_row[s] = t + base;
    matching_col[t] = s + base;
    if (next_t == INVALID)
      break;
    t = next_t;
  }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
bool HungarianAlgorithm<ROWTYPE, COLTYPE, VALTYPE>::update_potentials() {
  VALTYPE delta = std::numeric_limits<VALTYPE>::max();

  //  check T\Z
  for (COLTYPE j = 0; j < n; j++) {
    if (!T[j]) {
      delta = std::min(delta, min_slack[j]);
    }
  }
  if (delta == 0) {
    std::cerr << "Warning: delta is zero in update_potentials!" << std::endl;
  }
  if (delta == std::numeric_limits<VALTYPE>::max()) {
    std::cerr << "Warning: delta is infinity in update_potentials!"
              << std::endl;
    return false;
  }
  for (COLTYPE i = 0; i < n; i++) {
    if (S[i]) {
      // increase potential by delta for rows in S
      potential_row[i] += delta;
    }
  }
  COLTYPE count = 0;
  for (COLTYPE j = 0; j < n; j++) {
    if (T[j]) {
      // decrease potential by delta for columns in T
      potential_col[j] -= delta;
    } else {
      // decrease min_slack by delta for columns not in T
      min_slack[j] -= delta;
      if (min_slack[j] <= std::numeric_limits<VALTYPE>::epsilon()) {
        count++;
      }
    }
  }

  return count != 0;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void HungarianAlgorithm<ROWTYPE, COLTYPE, VALTYPE>::prep_row(
    const COLTYPE row) {
  const ROWTYPE base = ai[0];
  // // TODO: maybe not needed
  if (S[row]) {
    std::cerr << "Warning: visiting an already visited row!" << std::endl;
  }
  S[row] = true;
  Q.push_back(row);
  for (ROWTYPE i = ai[row] - base; i < ai[row + 1] - base; i++) {
    const COLTYPE j = aj[i] - base;
    if (!T[j]) {
      const VALTYPE cost = av[i] - (potential_col[j] + potential_row[row]);
      if (min_slack[j] > cost) {
        min_slack[j] = cost;
        parent[j] = row;
      }
    }
  }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
bool HungarianAlgorithm<ROWTYPE, COLTYPE, VALTYPE>::match_row(
    const COLTYPE row) {
  const ROWTYPE base = ai[0];
  prep_row(row);
  while (true) {
    while (Q.size()) {
      const auto curS = Q.front();
      Q.pop_front();
      for (ROWTYPE i = ai[curS] - base; i < ai[curS + 1] - base; i++) {
        const COLTYPE curT = aj[i] - base;
        const VALTYPE cost =
            av[i] - (potential_col[curT] + potential_row[curS]);
        if (cost <= std::numeric_limits<VALTYPE>::epsilon()) {
          // If cost is zero, consider this edge for the matching
          if (!T[curT]) {
            // if curT is not seen in the alternating tree rooted at row
            T[curT] = true;
            if (matching_col[curT] == INVALID) {
              // If curT is not matched, we found an augmenting path
              augment(curT);
              return true;
            } else {
              // Otherwise, add the matched row to the tree
              const auto nextS = matching_col[curT] - base;
              prep_row(nextS);
            }
          }
        } else if (!T[curT] && min_slack[curT] > cost) {
          min_slack[curT] = cost;
          parent[curT] = curS;
        }
      }
    }
    if (!update_potentials()) {
      std::cerr << "Error: failed to update potentials!" << std::endl;
      return false;
    }

    // min_slack_cpy = min_slack;
    for (COLTYPE j = 0; j < n; j++) {
      if (!T[j] && min_slack[j] <= std::numeric_limits<VALTYPE>::epsilon()) {
        if (parent[j] == INVALID) {
          std::cerr << "Error: visiting a column with invalid parent!"
                    << std::endl;
          return false;
        }
        T[j] = true;
        if (matching_col[j] == INVALID) {
          // If j is not matched, we found an augmenting path
          augment(j);
          return true;
        } else {
          // Otherwise, add the matched row to the tree
          const auto nextS = matching_col[j] - base;
          prep_row(nextS);
        }
      }
    }
  }
  return false;
}
template void MaximumMatching<int, int>(const int rows, int const *ai,
                                        int const *aj, int *matching_row,
                                        int *matching_col);

template class HungarianAlgorithm<int, int, double>;

} // namespace reordering
