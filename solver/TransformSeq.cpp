#include "TransformSeq.hpp"
#include "matrix_utils.hpp"

namespace solver {

// Use common matrix types from matrix_utils
using matrix_utils::CSRMatrix;
using matrix_utils::CSRMatrixVec;

// Explicit template instantiations for common types
// Using int for row/column types and double for values

// ============================================================================
// TransformSeq instantiations with double
// ============================================================================

template class TransformSeq<CSRMatrix<int, int, double>, std::vector<double>>;
template class TransformSeq<CSRMatrixVec<int, int, double>, std::vector<double>>;

// ============================================================================
// TransformSeq instantiations with float
// ============================================================================

template class TransformSeq<CSRMatrix<int, int, float>, std::vector<float>>;
template class TransformSeq<CSRMatrixVec<int, int, float>, std::vector<float>>;

} // namespace solver
