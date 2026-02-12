#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "cuda_memory.cuh"

namespace matrix_utils::sparse_cuda
{
/**
 * @brief Device CSR matrix structure using DeviceArray for automatic memory management
 */
 template <typename ROWTYPE, typename COLTYPE>
 struct DeviceCSRMatrix
 {
     COLTYPE n_rows = 0;
     ROWTYPE base = 0;
     DeviceArray<ROWTYPE> ai; // row pointers (size n_rows + 1)
     DeviceArray<COLTYPE> aj; // column indices (size nnz)
 };

//==============================================================================
// Public API declarations
//==============================================================================

/// @brief Find the position and value of the diagonal entry for each row of a device CSR matrix.
/// @param rows Number of rows
/// @param d_ai CSR row pointers (device, size rows+1)
/// @param d_aj CSR column indices (device)
/// @param d_av CSR values (device, may be nullptr if only position is needed)
/// @param d_diag_pos Output: position (index into AJ/AV) of diagonal for each row, or -1 if not found (device, size rows)
/// @param d_diag_val Output: diagonal value for each row, or 0 if not found (device, size rows, may be nullptr)
/// @param stream CUDA stream
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void CSRFindDiagonalDevice(
    COLTYPE rows,
    const ROWTYPE* d_ai,
    const COLTYPE* d_aj,
    const VALTYPE* d_av,
    ROWTYPE* d_diag_pos,
    VALTYPE* d_diag_val,
    cudaStream_t stream = nullptr);

/// @brief Compress a device CSR matrix using a keep/drop mask on every entry.
///
/// For each entry, if d_keep_mask[k] != 0, the entry is kept; otherwise it is dropped.
/// Supports both 0-based and 1-based CSR (base is preserved in output).
///
/// @param rows Number of rows
/// @param d_ai_in Input CSR row pointers (device, size rows+1)
/// @param d_aj_in Input CSR column indices (device)
/// @param d_av_in Input CSR values (device, may be nullptr for pattern-only)
/// @param d_keep_mask Per-entry keep mask (device, size nnz)
/// @param d_ai_out Output CSR row pointers (device, size rows+1)
/// @param d_aj_out Output CSR column indices (device)
/// @param d_av_out Output CSR values (device, may be nullptr for pattern-only)
/// @param stream CUDA stream
/// @return Number of entries removed (nnz_before - nnz_after).
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, typename FLAGTYPE = int>
ROWTYPE CSRSelectByMaskDevice(
    COLTYPE rows,
    const ROWTYPE* d_ai_in,
    const COLTYPE* d_aj_in,
    const VALTYPE* d_av_in,
    const FLAGTYPE* d_keep_mask,
    ROWTYPE* d_ai_out,
    COLTYPE* d_aj_out,
    VALTYPE* d_av_out,
    cudaStream_t stream = nullptr);

/// @brief Generate a keep/drop mask for CSR entries using diagonal scaled pruning.
///
/// Removes entries where |a_{i,j}| < |a_{i,i}| * |a_{j,j}| * threshold.
/// Diagonal entries are always kept (mask = 1).
///
/// @param rows Number of rows
/// @param d_ai CSR row pointers (device, size rows+1)
/// @param d_aj CSR column indices (device)
/// @param d_av CSR values (device)
/// @param threshold Pruning threshold
/// @param d_mask Output: keep mask (device, size nnz, 1=keep, 0=drop)
/// @param stream CUDA stream
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, typename FLAGTYPE = int>
void CSRGenDiagScaledPruneMask(
    COLTYPE rows,
    const ROWTYPE* d_ai,
    const COLTYPE* d_aj,
    const VALTYPE* d_av,
    VALTYPE threshold,
    FLAGTYPE* d_mask,
    cudaStream_t stream = nullptr);

//==============================================================================
// Explicit template instantiation declarations
//==============================================================================

extern template void CSRFindDiagonalDevice<int, int, float>(int, const int*, const int*, const float*, int*, float*, cudaStream_t);
extern template void CSRFindDiagonalDevice<int, int, double>(int, const int*, const int*, const double*, int*, double*, cudaStream_t);
extern template void CSRFindDiagonalDevice<std::int64_t, int, float>(int, const std::int64_t*, const int*, const float*, std::int64_t*, float*, cudaStream_t);
extern template void CSRFindDiagonalDevice<std::int64_t, int, double>(int, const std::int64_t*, const int*, const double*, std::int64_t*, double*, cudaStream_t);

extern template int CSRSelectByMaskDevice<int, int, float, int>(int, const int*, const int*, const float*, const int*, int*, int*, float*, cudaStream_t);
extern template int CSRSelectByMaskDevice<int, int, double, int>(int, const int*, const int*, const double*, const int*, int*, int*, double*, cudaStream_t);
extern template std::int64_t CSRSelectByMaskDevice<std::int64_t, int, float, int>(int, const std::int64_t*, const int*, const float*, const int*, std::int64_t*, int*, float*, cudaStream_t);
extern template std::int64_t CSRSelectByMaskDevice<std::int64_t, int, double, int>(int, const std::int64_t*, const int*, const double*, const int*, std::int64_t*, int*, double*, cudaStream_t);

extern template void CSRGenDiagScaledPruneMask<int, int, float, int>(int, const int*, const int*, const float*, float, int*, cudaStream_t);
extern template void CSRGenDiagScaledPruneMask<int, int, double, int>(int, const int*, const int*, const double*, double, int*, cudaStream_t);
extern template void CSRGenDiagScaledPruneMask<std::int64_t, int, float, int>(int, const std::int64_t*, const int*, const float*, float, int*, cudaStream_t);
extern template void CSRGenDiagScaledPruneMask<std::int64_t, int, double, int>(int, const std::int64_t*, const int*, const double*, double, int*, cudaStream_t);

} // namespace matrix_utils::sparse_cuda
