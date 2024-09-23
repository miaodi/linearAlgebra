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
  std::vector<ROWTYPE> jk(size);
  ROWTYPE nnz_icc = nnz;
  std::vector<COLTYPE> av_lvls(nnz_icc);
  ResizeCSRAJ(icc, nnz_icc);
  std::forward_list<std::pair<COLTYPE, COLTYPE>> current_row; // <col, lvl>
  typename std::forward_list<std::pair<COLTYPE, COLTYPE>>::iterator cur_row_it;

  COLTYPE list_size, lidx, k, i, lik, next_i, level, llist_next;
  ROWTYPE i_idx, i_idx_end;
  icc.ai[0] = base;
  for (COLTYPE j = 0; j < size; j++) {
    i_idx = diag_pos[j];
    i_idx_end = ai[j + 1];
    cur_row_it = current_row.before_begin();
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
      i_idx = jk[k]++;
      i_idx_end = icc.ai[k + 1];

      llist_next = llist[k];
      //   update llist if necessary
      if (i_idx + 1 < i_idx_end) {
        llist[k] = llist[icc.aj[i_idx + 1 - base] - base];
        llist[icc.aj[i_idx + 1 - base] - base] = k;
      }
      k = llist_next;

      lik = av_lvls[i_idx - base];
      cur_row_it = current_row.begin();
      next_i = std::next(cur_row_it)->first;
      // merge row k to row j
      for (; i_idx < i_idx_end; i_idx++) {
        i = icc.aj[i_idx - base];

        while (next_i <= i) {
          cur_row_it = std::next(cur_row_it);
          next_i = std::next(cur_row_it)->first;
        }
        level = lik + av_lvls[i_idx - base] + 1;
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

    cur_row_it = current_row.begin();
    i_idx = icc.ai[j];

    //   copy the current row to icc
    for (lidx = 0; lidx < list_size; lidx++) {
      icc.aj[i_idx - base] = cur_row_it->first;
      av_lvls[i_idx - base] = cur_row_it->second;
      i_idx++;
      cur_row_it++;
    }

    //   update llist if necessary
    if (icc.ai[j] + 1 < icc.ai[j + 1]) {
      llist[j] = llist[icc.aj[icc.ai[j] + 1 - base] - base];
      llist[icc.aj[icc.ai[j] + 1 - base] - base] = j;
      jk[j] = icc.ai[j] + 1;
    }
    current_row.clear();
  }
  ResizeCSRAV(icc, icc.ai[size] - base);
}

template <typename ROWTYPE, typename COLTYPE, typename CSRMatrixType>
void ICCLevelVecSymbolic(const COLTYPE size, const int base, ROWTYPE const *ai,
                         COLTYPE const *aj, COLTYPE const *diag_pos,
                         const int lvl, CSRMatrixType &icc) {
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
  std::vector<ROWTYPE> jk(size);
  ROWTYPE nnz_icc = nnz;
  std::vector<COLTYPE> av_lvls(nnz_icc);
  ResizeCSRAJ(icc, nnz_icc);
  std::vector<std::pair<COLTYPE, COLTYPE>> current_row; // <col, lvl>
  COLTYPE current_row_size_before, current_row_size_after, current_row_pos;
  typename std::vector<std::pair<COLTYPE, COLTYPE>>::iterator cur_row_it;
  current_row.reserve(size * .5);

  COLTYPE lidx, k, i, lik, next_i, level, llist_next;
  ROWTYPE i_idx, i_idx_end;
  icc.ai[0] = base;
  for (COLTYPE j = 0; j < size; j++) {
    i_idx = diag_pos[j];
    i_idx_end = ai[j + 1];
    current_row_size_before = current_row_size_after = i_idx_end - i_idx;

    // initialize the current row's nonzeros
    for (; i_idx != i_idx_end; i_idx++) {
      current_row.emplace_back(std::make_pair(aj[i_idx - base], 0));
    }

    // iterate for k from 0 to j-1
    k = llist[j];
    while (k < j) {
      i_idx = jk[k]++;
      i_idx_end = icc.ai[k + 1];

      llist_next = llist[k];
      //   update llist if necessary
      if (i_idx + 1 < i_idx_end) {
        llist[k] = llist[icc.aj[i_idx + 1 - base] - base];
        llist[icc.aj[i_idx + 1 - base] - base] = k;
      }
      k = llist_next;

      lik = av_lvls[i_idx - base];
      current_row_pos = 0;
      next_i = current_row_pos + 1 == current_row_size_before
                   ? NONE
                   : current_row[current_row_pos + 1].first;
      // merge row k to row j
      for (; i_idx < i_idx_end; i_idx++) {
        i = icc.aj[i_idx - base];

        while (next_i <= i) {
          current_row_pos += 1;
          next_i = current_row_pos + 1 == current_row_size_before
                       ? NONE
                       : current_row[current_row_pos + 1].first;
        }
        level = lik + av_lvls[i_idx - base] + 1;
        if (level <= lvl) {
          if (current_row[current_row_pos].first == i) {
            current_row[current_row_pos].second =
                std::min(current_row[current_row_pos].second, level);
          } else {
            current_row.emplace_back(std::make_pair(i, level));
            current_row_size_after++;
          }
        }
      }
      std::inplace_merge(current_row.begin(),
                         current_row.begin() + current_row_size_before,
                         current_row.begin() + current_row_size_after);
      current_row_size_before = current_row_size_after;
    }
    icc.ai[j + 1] = icc.ai[j] + current_row_size_before;

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

    //   update llist if necessary
    if (current_row_size_before > 1) {
      llist[j] = llist[current_row[1].first - base];
      llist[current_row[1].first - base] = j;
      jk[j] = icc.ai[j] + 1;
    }

    i_idx = icc.ai[j];
    //   copy the current row to icc
    for (const auto &p : current_row) {
      icc.aj[i_idx - base] = p.first;
      av_lvls[i_idx - base] = p.second;
      i_idx++;
    }

    current_row.clear();
  }
  ResizeCSRAV(icc, icc.ai[size] - base);
}

} // namespace matrix_utils