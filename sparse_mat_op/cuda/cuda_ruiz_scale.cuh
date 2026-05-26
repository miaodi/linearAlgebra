#pragma once

#include "cuda_tiled_sparse_mat.cuh"
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>

namespace matrix_utils::sparse_cuda
{
/// @brief Norm type used for CUDA Ruiz scaling
enum class CudaRuizScalingNormType : uint8_t
{
    MaxNorm, ///< Use infinity norm (max absolute value)
    L2Norm   ///< Use Euclidean (L2) norm
};

/// @brief CUDA implementation of Ruiz scaling algorithm for matrix equilibration
///
/// @details Implements the Ruiz scaling algorithm using CUDA for parallel computation.
/// The algorithm iteratively scales rows and columns to have unit norm, improving
/// the numerical conditioning of the matrix. The scaled matrix is: D_r * A * D_c
/// where D_r and D_c are diagonal matrices accumulated over iterations.
///
/// This CUDA version provides significant speedup for large sparse matrices
/// by computing row and column norms in parallel.
///
/// @reference D. Ruiz, "A scaling algorithm to equilibrate both rows and columns norms in matrices",
///            Technical Report RT/APO/01/4, ENSEEIHT-IRIT, 2001.
///
/// @tparam ROWTYPE Type for row pointers (e.g., int, int64_t)
/// @tparam COLTYPE Type for column indices (e.g., int, int64_t)
/// @tparam VALTYPE Type for matrix values (e.g., float, double)
/// @tparam NORM Norm type to use (MaxNorm or L2Norm)
///
/// @param rows Number of rows in the matrix
/// @param cols Number of columns in the matrix
/// @param d_ai CSR row pointer array on device (size: rows+1)
/// @param d_aj CSR column index array on device (size: nnz)
/// @param d_av CSR value array on device (size: nnz), modified in-place
/// @param d_dr Row scaling factors on device (size: rows), accumulated over iterations
/// @param d_dc Column scaling factors on device (size: cols), accumulated over iterations
/// @param max_iters Maximum number of iterations (default: 20)
///
/// @return true on successful completion
///
/// @note The matrix values (d_av) are modified in-place to contain the scaled matrix.
///       To recover the original matrix, apply: A_orig[i,j] = d_av[i,j] / (d_dr[i] * d_dc[j])
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, CudaRuizScalingNormType NORM = CudaRuizScalingNormType::MaxNorm>
bool RuizScaleCuda( const COLTYPE rows,
                    const COLTYPE cols,
                    const ROWTYPE* d_ai,
                    const COLTYPE* d_aj,
                    VALTYPE* d_av,
                    VALTYPE* d_dr,
                    VALTYPE* d_dc,
                    const int max_iters = 20 );

/// @brief CUDA Ruiz scaling using tile-COO matrix layout.
///
/// @details Norm computation is performed tile-by-tile with one warp processing one tile.
/// Tile-local row/column norm accumulators live in shared memory, then are atomically
/// merged into global row/column norm arrays.
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, CudaRuizScalingNormType NORM = CudaRuizScalingNormType::MaxNorm>
bool RuizScaleCuda( DeviceTileCOOMatrix<ROWTYPE, COLTYPE, VALTYPE>& tile_mat,
                    VALTYPE* d_dr,
                    VALTYPE* d_dc,
                    const int max_iters = 20 );

} // namespace matrix_utils::sparse_cuda

#include "cuda_ruiz_scale_impl.cuh"
