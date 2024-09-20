#pragma once

#include "matrix_utils.hpp"
#include <forward_list>
#include <omp.h>
#include <type_traits>
#include <vector>

namespace matrix_utils {

// NOTE: for level 0 ICC with Symmetric matrix input
template <typename ROWTYPE, typename COLTYPE, typename CSRMatrixType>
void ICCLevel0SymSymbolic(const COLTYPE size, const int base, ROWTYPE const *ai,
                          COLTYPE const *aj, CSRMatrixType &icc) {
  static_assert(
      CSRMatrixFormat<ROWTYPE, COLTYPE, typename CSRMatrixType::VALTYPE,
                      CSRMatrixType>::value == true);
  static_assert(CSRResizable<CSRMatrixType>::value,
                "CSRMatrixType must have a resizable method");
  icc.rows = size;
  icc.cols = size;
  ResizeCSRAI(icc, size + 1);
  const ROWTYPE nnz = ai[size] - base;

  ResizeCSRAJ(icc, nnz);
  ResizeCSRAV(icc, nnz);
#pragma omp parallel for
  for (COLTYPE i = 0; i < size + 1; i++) {
    icc.ai[i] = ai[i];
  }
#pragma omp parallel for
  for (ROWTYPE i = 0; i < nnz; i++) {
    icc.aj[i] = aj[i];
  }
}

template <typename ROWTYPE, typename COLTYPE, typename CSRMatrixType>
void ICCLevelSymbolic(const COLTYPE size, const int base, ROWTYPE const *ai,
                      COLTYPE const *aj, COLTYPE const *diag_pos, const int lvl,
                      CSRMatrixType &icc) {
  static_assert(
      CSRMatrixFormat<ROWTYPE, COLTYPE, typename CSRMatrixType::VALTYPE,
                      CSRMatrixType>::value == true);
  static_assert(CSRResizable<CSRMatrixType>::value,
                "CSRMatrixType must have a resizable method");
  icc.rows = size;
  icc.cols = size;
  ResizeCSRAI(icc, size + 1);
  const ROWTYPE nnz = ai[size] - base;
  const COLTYPE NONE = std::numeric_limits<COLTYPE>::max();
  std::vector<COLTYPE> llist(size, NONE);
  std::vector<ROWTYPE> cur_diag_pos(diag_pos, diag_pos + size);
  ROWTYPE nnz_icc = nnz;
  std::vector<COLTYPE> av_lvls(nnz_icc);
  ResizeCSRAJ(icc, nnz_icc);
  std::forward_list<std::pair<COLTYPE, COLTYPE>> current_row; // <col, lvl>

  COLTYPE list_size;
  ROWTYPE i_idx, i_idx_end;
  COLTYPE k, i, lik, next_i, level;
  icc.ai[0] = base;
  for (COLTYPE j = 0; j < size; j++) {
    i_idx = diag_pos[j];
    i_idx_end = ai[j + 1];
    auto cur_row_it = current_row.before_begin();
    list_size = i_idx_end - i_idx;

    // initialize the current row's nonzeros
    for (; i_idx != i_idx_end; i_idx++) {
      cur_row_it = current_row.insert_after(
          cur_row_it, std::make_pair(aj[i_idx - base], 0));
    }

    // use max as the list end to prevent from branch prediction
    current_row.insert_after(cur_row_it, std::make_pair(NONE, 0));
    // iterate for k from 0 to j-1
    k = llist[j];
    while (k < j) {
      i_idx = cur_diag_pos[k];
      i_idx_end = ai[k + 1];
      //   update llist if necessary
      if (i_idx + 1 != i_idx_end) {
        llist[k] = llist[aj[i_idx + 1 - base] - base];
        llist[aj[i_idx + 1 - base] - base] = k;
        cur_diag_pos[k]++;
      }

      lik = av_lvls[i_idx - base];
      cur_row_it = current_row.begin();
      next_i = std::next(cur_row_it)->first;
      for (; i_idx < i_idx_end; i_idx++) {
        i = aj[i_idx - base];
        level = lik + av_lvls[i_idx] + 1;

        while (next_i <= i) {
          cur_row_it = std::next(cur_row_it);
          next_i = std::next(cur_row_it)->first;
        }
        if (level <= lvl) {
          if (cur_row_it->first == i) {
            cur_row_it->second = std::min(cur_row_it->second, level);
          } else {
            cur_row_it =
                current_row.insert_after(cur_row_it, std::make_pair(i, level));
            next_i = std::next(cur_row_it)->first;
            list_size++;
          }
        }
      }

      icc.ai[j + 1] = icc.ai[j] + list_size;
      //   resize if needed
      if (icc.ai[j + 1] - base > nnz_icc) {
        // estimate the new size
        if (2 * (j - base) >= size)
          nnz_icc *= 2;
        else
          nnz_icc = nnz_icc * std::ceil(size * 1. / (j - base));
        av_lvls.resize(nnz_icc);
        ResizeCSRAJ<CSRMatrixType, true>(icc, nnz_icc);
      }

      //   copy the current row to icc
      cur_row_it = current_row.begin();

      i_idx = icc.ai[j];
      while (cur_row_it != current_row.end()) {
        icc.aj[i_idx - base] = cur_row_it->first;
        av_lvls[i_idx - base] = cur_row_it->second;
        i_idx++;
        cur_row_it++;
      }
      current_row.clear();
    }
    ResizeCSRAV(icc, icc.ai[size] - base);
  }
}

} // namespace matrix_utils