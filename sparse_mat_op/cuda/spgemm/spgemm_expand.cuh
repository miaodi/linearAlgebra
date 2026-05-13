#pragma once

#include "cuda_memory.cuh"

#include <array>
#include <cstdint>
#include <cuda_runtime.h>

namespace matrix_utils::sparse_cuda
{

enum class SpGEMMRowClass : int
{
    Thread = 0,
    Warp = 1,
    CTA = 2,
    Global = 3
};

struct SpGEMMSymbolicOptions
{
    std::int64_t thread_threshold = 32;
    std::int64_t warp_threshold = 736;
    std::int64_t cta_threshold = 6144;
};

template <typename ROWTYPE, typename COLTYPE>
struct SpGEMMSymbolicResult
{
    COLTYPE n_rows = 0;
    ROWTYPE base = 0;
    ROWTYPE total_expanded_nnz = 0;

    // Sorted-row partition offsets:
    // [0, thread), [thread, warp), [warp, cta), [cta, global_end).
    std::array<COLTYPE, 5> row_class_offsets{};

    // expanded_nnz[i] = sum_{k in cols(A_i)} nnz(B_k), in original row order.
    DeviceArray<ROWTYPE> expanded_nnz;

    // Prefix sum over expanded_nnz. Size n_rows + 1 and starts at base.
    DeviceArray<ROWTYPE> expanded_row_ptr;

    // Original row ids sorted by expanded_nnz. Row ids preserve the CSR base.
    DeviceArray<COLTYPE> row_perm;

    // expanded_nnz sorted in the same order as row_perm.
    DeviceArray<ROWTYPE> sorted_expanded_nnz;

    COLTYPE classBegin(SpGEMMRowClass row_class) const
    {
        return row_class_offsets[static_cast<int>(row_class)];
    }

    COLTYPE classEnd(SpGEMMRowClass row_class) const
    {
        return row_class_offsets[static_cast<int>(row_class) + 1];
    }
};

template <typename COLTYPE, typename VALTYPE>
struct SpGEMMExpandedProducts
{
    // Raw expanded products in CSR-row order. Duplicate columns are preserved.
    DeviceArray<COLTYPE> col_ind;
    DeviceArray<VALTYPE> values;
};

/**
 * @brief Symbolic expansion analysis for row-wise CSR SpGEMM C = A * B.
 *
 * This follows the expansion-oriented symbolic phase used by GPU SpGEMM
 * algorithms: it counts raw intermediate products per output row before
 * duplicate output columns are merged.
 */
template <typename ROWTYPE, typename COLTYPE>
bool SpGEMMSymbolicAnalyzeCSR(
    COLTYPE A_rows,
    COLTYPE A_cols,
    const ROWTYPE* d_A_row_ptr,
    const COLTYPE* d_A_col_ind,
    COLTYPE B_rows,
    const ROWTYPE* d_B_row_ptr,
    ROWTYPE base,
    SpGEMMSymbolicResult<ROWTYPE, COLTYPE>& result,
    SpGEMMSymbolicOptions options = SpGEMMSymbolicOptions{},
    cudaStream_t stream = nullptr);

/**
 * @brief Expand CSR SpGEMM products into row-wise temporary arrays.
 *
 * For every row i and every A(i,k), this emits all raw products
 * A(i,k) * B(k,j) into the slice
 * [symbolic.expanded_row_ptr[i], symbolic.expanded_row_ptr[i + 1]).
 * Duplicate j columns are intentionally preserved for the later sort/reduce
 * phase. Rows are dispatched through the row classes computed by the symbolic
 * phase: one thread, one warp, one CTA, or multiple global CTAs per row.
 */
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
bool SpGEMMExpandCSR(
    COLTYPE A_rows,
    COLTYPE A_cols,
    const ROWTYPE* d_A_row_ptr,
    const COLTYPE* d_A_col_ind,
    const VALTYPE* d_A_values,
    COLTYPE B_rows,
    const ROWTYPE* d_B_row_ptr,
    const COLTYPE* d_B_col_ind,
    const VALTYPE* d_B_values,
    ROWTYPE base,
    const SpGEMMSymbolicResult<ROWTYPE, COLTYPE>& symbolic,
    SpGEMMExpandedProducts<COLTYPE, VALTYPE>& expanded,
    cudaStream_t stream = nullptr);

extern template bool SpGEMMSymbolicAnalyzeCSR<int, int>(
    int,
    int,
    const int*,
    const int*,
    int,
    const int*,
    int,
    SpGEMMSymbolicResult<int, int>&,
    SpGEMMSymbolicOptions,
    cudaStream_t);

extern template bool SpGEMMSymbolicAnalyzeCSR<std::int64_t, int>(
    int,
    int,
    const std::int64_t*,
    const int*,
    int,
    const std::int64_t*,
    std::int64_t,
    SpGEMMSymbolicResult<std::int64_t, int>&,
    SpGEMMSymbolicOptions,
    cudaStream_t);

extern template bool SpGEMMExpandCSR<int, int, float>(
    int,
    int,
    const int*,
    const int*,
    const float*,
    int,
    const int*,
    const int*,
    const float*,
    int,
    const SpGEMMSymbolicResult<int, int>&,
    SpGEMMExpandedProducts<int, float>&,
    cudaStream_t);

extern template bool SpGEMMExpandCSR<int, int, double>(
    int,
    int,
    const int*,
    const int*,
    const double*,
    int,
    const int*,
    const int*,
    const double*,
    int,
    const SpGEMMSymbolicResult<int, int>&,
    SpGEMMExpandedProducts<int, double>&,
    cudaStream_t);

extern template bool SpGEMMExpandCSR<std::int64_t, int, float>(
    int,
    int,
    const std::int64_t*,
    const int*,
    const float*,
    int,
    const std::int64_t*,
    const int*,
    const float*,
    std::int64_t,
    const SpGEMMSymbolicResult<std::int64_t, int>&,
    SpGEMMExpandedProducts<int, float>&,
    cudaStream_t);

extern template bool SpGEMMExpandCSR<std::int64_t, int, double>(
    int,
    int,
    const std::int64_t*,
    const int*,
    const double*,
    int,
    const std::int64_t*,
    const int*,
    const double*,
    std::int64_t,
    const SpGEMMSymbolicResult<std::int64_t, int>&,
    SpGEMMExpandedProducts<int, double>&,
    cudaStream_t);

} // namespace matrix_utils::sparse_cuda
