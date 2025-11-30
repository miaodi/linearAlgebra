#include "Transformation.hpp"
#include "matrix_utils.hpp"

namespace solver {

// Use common matrix types from matrix_utils
using matrix_utils::CSRMatrix;
using matrix_utils::CSRMatrixVec;

// Explicit template instantiations for common types
// Using int for row/column types and double for values

// ============================================================================
// RowPermutation instantiations
// ============================================================================

template class RowPermutation<CSRMatrix<int, int, double>, std::vector<double>>;
template class RowPermutation<CSRMatrixVec<int, int, double>, std::vector<double>>;

// ============================================================================
// ColumnPermutation instantiations
// ============================================================================

template class ColumnPermutation<CSRMatrix<int, int, double>, std::vector<double>>;
template class ColumnPermutation<CSRMatrixVec<int, int, double>, std::vector<double>>;

// ============================================================================
// RowColPermutation instantiations
// ============================================================================

template class RowColPermutation<CSRMatrix<int, int, double>, std::vector<double>>;
template class RowColPermutation<CSRMatrixVec<int, int, double>, std::vector<double>>;

// ============================================================================
// RowScaling instantiations
// ============================================================================

template class RowScaling<CSRMatrix<int, int, double>, std::vector<double>>;
template class RowScaling<CSRMatrixVec<int, int, double>, std::vector<double>>;

// ============================================================================
// ColumnScaling instantiations
// ============================================================================

template class ColumnScaling<CSRMatrix<int, int, double>, std::vector<double>>;
template class ColumnScaling<CSRMatrixVec<int, int, double>, std::vector<double>>;

// ============================================================================
// RowColScaling instantiations
// ============================================================================

template class RowColScaling<CSRMatrix<int, int, double>, std::vector<double>>;
template class RowColScaling<CSRMatrixVec<int, int, double>, std::vector<double>>;

// ============================================================================
// IdentityTransformation instantiations
// ============================================================================

template class IdentityTransformation<CSRMatrix<int, int, double>, std::vector<double>>;
template class IdentityTransformation<CSRMatrixVec<int, int, double>, std::vector<double>>;

// ============================================================================
// Additional instantiations with float
// ============================================================================

template class RowPermutation<CSRMatrix<int, int, float>, std::vector<float>>;
template class RowPermutation<CSRMatrixVec<int, int, float>, std::vector<float>>;

template class ColumnPermutation<CSRMatrix<int, int, float>, std::vector<float>>;
template class ColumnPermutation<CSRMatrixVec<int, int, float>, std::vector<float>>;

template class RowColPermutation<CSRMatrix<int, int, float>, std::vector<float>>;
template class RowColPermutation<CSRMatrixVec<int, int, float>, std::vector<float>>;

template class RowScaling<CSRMatrix<int, int, float>, std::vector<float>>;
template class RowScaling<CSRMatrixVec<int, int, float>, std::vector<float>>;

template class ColumnScaling<CSRMatrix<int, int, float>, std::vector<float>>;
template class ColumnScaling<CSRMatrixVec<int, int, float>, std::vector<float>>;

template class RowColScaling<CSRMatrix<int, int, float>, std::vector<float>>;
template class RowColScaling<CSRMatrixVec<int, int, float>, std::vector<float>>;

template class IdentityTransformation<CSRMatrix<int, int, float>, std::vector<float>>;
template class IdentityTransformation<CSRMatrixVec<int, int, float>, std::vector<float>>;

} // namespace solver
