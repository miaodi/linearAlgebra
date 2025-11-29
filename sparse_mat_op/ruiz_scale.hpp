#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <omp.h>
#include <vector>

namespace scaling
{
/// @brief Norm type used for Ruiz scaling
enum class RuizScalingNormType : std::uint8_t
{
    MaxNorm,  ///< Use infinity norm (max absolute value)
    L2Norm        ///< Use Euclidean (L2) norm
};

/// @brief Iterative matrix equilibration using Ruiz scaling algorithm
/// 
/// @details Implements the Ruiz scaling algorithm for symmetric matrix equilibration.
/// The algorithm iteratively scales rows and columns to have unit norm, improving
/// the numerical conditioning of the matrix. The scaled matrix is: D_r * A * D_c
/// where D_r and D_c are diagonal matrices accumulated over iterations.
/// 
/// The algorithm is suitable for preconditioning before iterative solvers or
/// direct factorizations, as it reduces the condition number and numerical errors.
/// 
/// @reference D. Ruiz, "A scaling algorithm to equilibrate both rows and columns norms in matrices",
///            Technical Report RT/APO/01/4, ENSEEIHT-IRIT, 2001.
///            http://www.enseeiht.fr/lima/apo/SCALE/
/// 
/// @tparam ROWTYPE Type for row pointers (e.g., int, int64_t)
/// @tparam COLTYPE Type for column indices (e.g., int, int64_t)
/// @tparam VALTYPE Type for matrix values (e.g., float, double)
/// @tparam NORM Norm type to use (MaxNorm or L2Norm)
/// 
/// @param rows Number of rows in the matrix
/// @param cols Number of columns in the matrix
/// @param ai CSR row pointer array (size: rows+1)
/// @param aj CSR column index array (size: nnz)
/// @param av CSR value array (size: nnz), modified in-place
/// @param dr Row scaling factors (size: rows), accumulated over iterations
/// @param dc Column scaling factors (size: cols), accumulated over iterations
/// @param max_iters Maximum number of iterations (default: 20)
/// @param tol Convergence tolerance based on maximum relative change (default: 1e-3)
/// 
/// @return true if converged within max_iters, false otherwise
/// 
/// @note The matrix values (av) are modified in-place to contain the scaled matrix.
///       To recover the original matrix, apply: A_orig[i,j] = av[i,j] / (dr[i] * dc[j])
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, RuizScalingNormType NORM = RuizScalingNormType::MaxNorm>
bool RuizScaleSerial(const COLTYPE rows, const COLTYPE cols, ROWTYPE const* ai, COLTYPE const* aj,
                     VALTYPE* av, VALTYPE* dr, VALTYPE* dc, const int max_iters = 20,
                     const VALTYPE tol = static_cast<VALTYPE>(1e-3));

} // namespace scaling