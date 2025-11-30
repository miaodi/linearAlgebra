#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace matrix_utils
{

/// @brief Scale a vector by element-wise multiplication with optional parallelization
/// @details Computes x[i] = x[i] * s[i] for all i in parallel using OpenMP.
///          Use nthreads > 1 for large vectors to leverage multi-core performance.
/// @tparam COLTYPE Index type (e.g., int, int64_t)
/// @tparam VALTYPE Value type (e.g., float, double)
/// @param size Vector size
/// @param x Vector to scale (modified in-place)
/// @param s Scaling factors (size: size)
/// @param nthreads Number of OpenMP threads (default: 1 for serial execution)
template <typename COLTYPE, typename VALTYPE>
void ScaleVec(const COLTYPE size, VALTYPE* x, VALTYPE const* s, int nthreads = 1);

/// @brief Scale a vector by element-wise division (inverse scaling) with optional parallelization
/// @details Computes x[i] = x[i] / s[i] for all i in parallel using OpenMP.
///          Use nthreads > 1 for large vectors to leverage multi-core performance.
/// @tparam COLTYPE Index type (e.g., int, int64_t)
/// @tparam VALTYPE Value type (e.g., float, double)
/// @param size Vector size
/// @param x Vector to scale (modified in-place)
/// @param s Scaling factors (size: size)
/// @param nthreads Number of OpenMP threads (default: 1 for serial execution)
template <typename COLTYPE, typename VALTYPE>
void InvScaleVec(const COLTYPE size, VALTYPE* x, VALTYPE const* s, int nthreads = 1);

/// @brief Apply row and/or column scaling to a CSR matrix with optional parallelization
/// @details Flexible scaling function supporting three modes based on nullptr arguments:
///          1. Column-only scaling (dr == nullptr): A[i,j] ← A[i,j] * dc[j]
///          2. Row-only scaling (dc == nullptr): A[i,j] ← dr[i] * A[i,j]
///          3. Row and column scaling (both non-null): A[i,j] ← dr[i] * A[i,j] * dc[j]
///          
///          The function uses compile-time template specialization to generate three
///          optimized code paths with zero runtime branching overhead. Parallelization
///          via OpenMP distributes row iterations across threads for improved performance
///          on large matrices.
/// 
/// @tparam ROWTYPE Row pointer type (e.g., int, int64_t)
/// @tparam COLTYPE Column index type (e.g., int, int64_t)
/// @tparam VALTYPE Value type (e.g., float, double)
/// @param rows Number of matrix rows
/// @param ai CSR row pointer array (size: rows+1, ai[0] indicates base index)
/// @param aj CSR column index array (size: nnz)
/// @param av CSR value array (size: nnz), modified in-place
/// @param dr Row scaling factors (size: rows), or nullptr to skip row scaling
/// @param dc Column scaling factors (size: cols), or nullptr to skip column scaling
/// @param nthreads Number of OpenMP threads (default: 1 for serial execution)
/// @return true if scaling was successfully applied, false if both dr and dc are nullptr
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
bool ScaleMat(const COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj, VALTYPE* av,
              VALTYPE const* dr, VALTYPE const* dc, int nthreads = 1);

} // namespace matrix_utils
