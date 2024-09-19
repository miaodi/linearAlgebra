#pragma once

namespace matrix_utils {
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void WeightedJacobiUpdate(const COLTYPE rows, const COLTYPE cols,
                          const int base, ROWTYPE const *ai, COLTYPE const *aj,
                          VALTYPE const *av, const VALTYPE omega,
                          VALTYPE const *diag, VALTYPE const *rhs, , VALTYPE *x,
                          VALTYPE *tmp) {
                            
                          }

template <typename CSRMatrixType> struct WeightedJacobi {
  using ROWTYPE = typename CSRMatrixType::ROWTYPE;
  using COLTYPE = typename CSRMatrixType::COLTYPE;
  using VALTYPE = typename CSRMatrixType::VALTYPE;
};
} // namespace matrix_utils