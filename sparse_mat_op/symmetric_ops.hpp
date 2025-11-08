#pragma once

#include "matrix_utils.hpp"
#include "sparse_mat_traits.hpp"
#include "utils.h"
#include <numeric>
#include <vector>

namespace matrix_utils {

template <typename ROWTYPE, typename COLTYPE, bool KEEPDIAG = true>
void AATSymbolic(const COLTYPE size, ROWTYPE const *ai, COLTYPE const *aj,
                 ROWTYPE *ai_AAT) {
  const int base = ai[0];
  ai_AAT[0] = base;

  for (COLTYPE i = 0; i < size; i++) {
    ai_AAT[i + 1] = 0;
  }
  std::vector<ROWTYPE> start_pos(size);
  ROWTYPE j, k;
  for (COLTYPE i = 0; i < size; i++) {
    start_pos[i] = ai[i] - base;
    for (j = ai[i] - base; j < ai[i + 1] - base; j++) {
      COLTYPE col = aj[j] - base;
      if (col > i) {
        break; // skip upper triangle
      } else if (col == i) {
        if constexpr (KEEPDIAG)
          ai_AAT[i + 1]++;
        j++;
        break;
      }

      ai_AAT[col + 1]++; // increment the row size for AAT[col, i]
      ai_AAT[i + 1]++;   // increment the row size for AAT[i, col]
      for (k = start_pos[col]; k < ai[col + 1] - base; k++) {
        COLTYPE col2 = aj[k] - base;
        if (col2 == i) {
          k++;
          break;
        } else if (col2 > i) {
          break;
        }
        ai_AAT[col2 + 1]++; // increment the row size for AAT[col2, col]
        ai_AAT[col + 1]++;  // increment the row size for AAT[col, col2]
      }
      start_pos[col] = k;
    }
    start_pos[i] = j;
  }
  for (COLTYPE i = 0; i < size; i++) {
    for (ROWTYPE j = start_pos[i]; j < ai[i + 1] - base; j++) {
      COLTYPE col = aj[j] - base;
      ai_AAT[col + 1]++; // increment the row size for AAT[col, i]
      ai_AAT[i + 1]++;   // increment the row size for AAT[i, col]
    }
  }

  std::inclusive_scan(ai_AAT, ai_AAT + size + 1, ai_AAT);
}

template <typename ROWTYPE, typename COLTYPE, bool KEEPDIAG = true>
void AATNumeric(const COLTYPE size, ROWTYPE const *ai, COLTYPE const *aj,
                ROWTYPE const *ai_AAT, COLTYPE *aj_AAT) {
  const ROWTYPE base = ai[0];

  std::vector<ROWTYPE> start_pos(size);
  std::vector<ROWTYPE> start_pos_AAT(size);
  ROWTYPE j, k;
  for (COLTYPE i = 0; i < size; i++) {
    start_pos[i] = ai[i] - base;
    start_pos_AAT[i] = ai_AAT[i] - base;
    for (j = ai[i] - base; j < ai[i + 1] - base; j++) {
      COLTYPE col = aj[j] - base;
      if (col > i) {
        break; // skip upper triangle
      } else if (col == i) {
        if constexpr (KEEPDIAG) {
          aj_AAT[start_pos_AAT[i]++] = i + base;
        }
        j++;
        break;
      }
      aj_AAT[start_pos_AAT[col]++] = i + base; // AAT[col, i]
      aj_AAT[start_pos_AAT[i]++] = col + base; // AAT[i, col]
      for (k = start_pos[col]; k < ai[col + 1] - base; k++) {
        COLTYPE col2 = aj[k] - base;
        if (col2 == i) {
          k++;
          break;
        } else if (col2 > i) {
          break;
        }
        aj_AAT[start_pos_AAT[col2]++] = col + base; // AAT[col2, col]
        aj_AAT[start_pos_AAT[col]++] = col2 + base; // AAT[col, col2]
      }
      start_pos[col] = k;
    }
    start_pos[i] = j;
  }
  for (COLTYPE i = 0; i < size; i++) {
    for (ROWTYPE j = start_pos[i]; j < ai[i + 1] - base; j++) {
      COLTYPE col = aj[j] - base;
      aj_AAT[start_pos_AAT[col]++] = i + base; // AAT[col, i]
      aj_AAT[start_pos_AAT[i]++] = col + base; // AAT[i, col]
    }
    std::sort(aj_AAT + ai_AAT[i] - base,
              aj_AAT + ai_AAT[i + 1] - base); // sort the row
  }
}
} // namespace matrix_utils