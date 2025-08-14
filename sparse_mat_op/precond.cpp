#include "precond.hpp"
#include "matrix_utils.hpp"

namespace matrix_utils {
template <ResizableDiagonalType CSRMatrixType>
bool ILULevelSymbolic<CSRMatrixType>::operator()(
    const typename CSRMatrixType::COLTYPE size,
    typename CSRMatrixType::ROWTYPE const *ai,
    typename CSRMatrixType::COLTYPE const *aj, const int lvl,
    CSRMatrixType &ilu) {
  if (lvl < 0) {
    return false;
  } else if (lvl == 0) {
    return ILULevel0Symbolic(size, ai, aj, ilu);
  }
  ilu.rows = size;
  ilu.cols = size;
  ilu.ResizeAI(size + 1);
  ilu.ResizeDiagonal(size);
  const auto base = ai[0];
  typename CSRMatrixType::ROWTYPE nnz = ai[size] - base, cur_nnz;
  const auto NONE = std::numeric_limits<typename CSRMatrixType::COLTYPE>::max();

  ilu.ResizeAJ(nnz);
  ilu.ResizeAV(nnz);
  _levels.resize(nnz);
  typename CSRMatrixType::ROWTYPE i_idx, i_idx_end, k_idx, k_idx2, j_idx,
      j_idx_end, k_idx_end;
  typename CSRMatrixType::COLTYPE j, k;

  int lvl_ik, level;

  auto *ilu_ai = ilu.AI();
  auto *ilu_aj = ilu.AJ();
  auto *ilu_av = ilu.AV();
  auto *ilu_diag = ilu.Diagonal();
  ilu_ai[0] = base;
  for (typename CSRMatrixType::COLTYPE i = 0; i < size; i++) {
    _current_row.clear();
    i_idx = ai[i] - base;
    i_idx_end = ai[i + 1] - base;

    // initialize the current row's nonzeros
    for (; i_idx < i_idx_end; i_idx++) {
      _current_row.emplace_back(aj[i_idx] - base, 0);
    }
    _current_row.emplace_back(NONE, 0); // max as the end

    for (k_idx = 0; k_idx < _current_row.size(); k_idx++) {
      k = _current_row[k_idx].first;
      if (k >= i) {
        break;
      }
      lvl_ik = _current_row[k_idx].second;

      j_idx = ilu_diag[k] - base + 1;
      j_idx_end = ilu_ai[k + 1] - base;
      k_idx_end = _current_row.size();
      k_idx2 = k_idx;
      for (; j_idx < j_idx_end; j_idx++) {
        level = lvl_ik + _levels[j_idx] + 1;
        if (level > lvl) {
          continue; // skip this element
        }
        while (_current_row[k_idx2].first < ilu_aj[j_idx] - base) {
          k_idx2++;
        }
        if (_current_row[k_idx2].first > ilu_aj[j_idx] - base) {
          // insert new element
          _current_row.emplace_back(ilu_aj[j_idx] - base, level);
        } else {
          // update the level
          _current_row[k_idx2].second =
              std::min(_current_row[k_idx2].second, level);
        }
      }
      std::merge(
          _current_row.begin(), _current_row.begin() + k_idx_end,
          _current_row.begin() + k_idx_end, _current_row.end(),
          std::back_inserter(_current_row2),
          [](const auto &a, const auto &b) { return a.first < b.first; });
      std::swap(_current_row, _current_row2);
      _current_row2.clear();
    }
    ilu_ai[i + 1] = ilu_ai[i] + _current_row.size() - 1;
    cur_nnz = ilu_ai[i + 1] - base;
    if (_current_row[k_idx].first != i)
      return false;
    ilu_diag[i] = ilu_ai[i] + k_idx;

    // copy to ilu aj and _levels
    if (cur_nnz > nnz) {
      // estimate the new size
      if (2 * (i - base) >= size)
        nnz *= 2;
      else
        nnz = nnz * std::ceil(size * 1. / (i - base));
      nnz = std::max(nnz, cur_nnz);
      ilu_aj = ilu.ResizeAJ(nnz);
      _levels.resize(nnz);
    }

    for (i_idx = ilu_ai[i] - base, k_idx2 = 0; i_idx < ilu_ai[i + 1] - base;
         i_idx++, k_idx2++) {
      ilu_aj[i_idx] = _current_row[k_idx2].first;
      _levels[i_idx] = _current_row[k_idx2].second;
    }
  }
  ilu.ResizeAV(ilu_ai[size] - base);
  return true;
}

template void ICCLevel0SymSymbolic<int, int, CSRMatrix<int, int, double>>(
    const int rows, int const *ai, int const *aj,
    CSRMatrix<int, int, double> &icc);

template void ICCLevelSymbolic0<int, int, CSRMatrix<int, int, double>>(
    const int rows, int const *ai, int const *aj, int const *diag_pos,
    const int lvl, CSRMatrix<int, int, double> &icc);

template void ICCLevelSymbolic1<int, int, CSRMatrix<int, int, double>>(
    const int rows, int const *ai, int const *aj, int const *diag_pos,
    const int lvl, CSRMatrix<int, int, double> &icc);

template void ICCLevelSymbolic2<int, int, CSRMatrix<int, int, double>>(
    const int rows, int const *ai, int const *aj, int const *diag_pos,
    const int lvl, CSRMatrix<int, int, double> &icc);

template void ICCLevelSymbolic3<int, int, CSRMatrix<int, int, double>>(
    const int rows, int const *ai, int const *aj, int const *diag_pos,
    const int lvl, CSRMatrix<int, int, double> &icc);

template bool ICCLevelNumeric<int, int, double>(
    const int rows, int const *ai, int const *aj, double const *av,
    int const *diag_pos, const int lvl, const double omega, int const *icc_ai,
    int const *icc_aj, double *icc_av);

template <ResizableDiagonalType CSRMatrixType>
bool ILULevelNumeric(const typename CSRMatrixType::COLTYPE size,
                     typename CSRMatrixType::ROWTYPE const *ai,
                     typename CSRMatrixType::COLTYPE const *aj,
                     typename CSRMatrixType::VALTYPE const *av, const int lvl,
                     CSRMatrixType &ilu) {
  const auto base = ai[0];
  typename CSRMatrixType::ROWTYPE i_idx, ilu_i_idx, k_idx, j_idx2, j_idx;
  typename CSRMatrixType::COLTYPE j, k;

  auto const *ilu_ai = ilu.AI();
  auto const *ilu_aj = ilu.AJ();
  auto *ilu_av = ilu.AV();
  auto const *ilu_diag = ilu.Diagonal();
  typename CSRMatrixType::VALTYPE akk, aik;

  for (typename CSRMatrixType::COLTYPE i = 0; i < size; i++) {
    // std::cout << "i: " << i << std::endl;
    // initialize the current row's nonzeros
    i_idx = ai[i] - base;
    for (ilu_i_idx = ilu_ai[i] - base; ilu_i_idx < ilu_ai[i + 1] - base;
         ilu_i_idx++) {
      if (i_idx == ai[i + 1] - base || aj[i_idx] != ilu_aj[ilu_i_idx]) {
        ilu_av[ilu_i_idx] = 0; // initialize to zero
      } else {
        ilu_av[ilu_i_idx] = av[i_idx++]; // copy the value
      }
    }
    k_idx = ilu_ai[i] - base;
    while (true) {
      k = ilu_aj[k_idx] - base;
      if (k >= i) {
        break;
      }
      akk = ilu_av[ilu_diag[k] - base];
      if (akk == 0) {
        // akk = ilu_av[ilu_diag[k] - base] = 1e-16;
        return false;
      }
      ilu_av[k_idx] /= akk; // a_{ik} = a_{ik} / a_{kk}
      aik = ilu_av[k_idx];

      j_idx2 = k_idx; // j_idx2 is for ith row, j_idx is for kth row
      for (j_idx = ilu_diag[k] - base + 1; j_idx < ilu_ai[k + 1] - base;) {
        if (ilu_aj[j_idx] == ilu_aj[j_idx2]) {
          ilu_av[j_idx++] -= aik * ilu_av[j_idx2++];
        } else if (ilu_aj[j_idx] < ilu_aj[j_idx2]) {
          j_idx++;
        } else {
          j_idx2++;
        }
      }
      k_idx++;
    }
  }
  return true;
}

template struct ILULevelSymbolic<matrix_utils::CSRMatrix<int, int, double>>;
template bool ILULevelNumeric<matrix_utils::CSRMatrix<int, int, double>>(
    const int size, int const *ai, int const *aj, double const *av,
    const int lvl, matrix_utils::CSRMatrix<int, int, double> &ilu);
} // namespace matrix_utils