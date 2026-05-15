#pragma once

#include "cuda_memory.cuh"

#include <array>
#include <cstdint>
#include <cuda_runtime.h>

namespace matrix_utils::sparse_cuda
{

// Fixed-width type for indexing into expanded product arrays.
// Expanded product counts routinely exceed INT_MAX for large SpGEMM,
// so this is always 64-bit regardless of the input CSR index type.
using ExpandedIndex = std::int64_t;

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
    COLTYPE n_cols = 0; // max column dimension of product (0 = unknown, sort uses all bits)
    ROWTYPE base = 0;
    ExpandedIndex total_expanded_nnz = 0;

    // Sorted-row partition offsets:
    // [0, thread), [thread, warp), [warp, cta), [cta, global_end).
    std::array<COLTYPE, 5> row_class_offsets{};

    // expanded_nnz[i] = sum_{k in cols(A_i)} nnz(B_k), in original row order.
    DeviceArray<ExpandedIndex> expanded_nnz;

    // Prefix sum over expanded_nnz. Size n_rows + 1 and starts at base.
    DeviceArray<ExpandedIndex> expanded_row_ptr;

    // Original row ids sorted by expanded_nnz. Row ids preserve the CSR base.
    DeviceArray<COLTYPE> row_perm;

    // expanded_nnz sorted in the same order as row_perm.
    DeviceArray<ExpandedIndex> sorted_expanded_nnz;

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
 * Input CSR row pointers use ROWTYPE. Expanded product counts use
 * ExpandedIndex (int64_t) internally, so products beyond INT_MAX are handled.
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
    int, int, const int*, const int*, int, const int*, int,
    SpGEMMSymbolicResult<int, int>&, SpGEMMSymbolicOptions, cudaStream_t);

extern template bool SpGEMMExpandCSR<int, int, float>(
    int, int, const int*, const int*, const float*,
    int, const int*, const int*, const float*, int,
    const SpGEMMSymbolicResult<int, int>&, SpGEMMExpandedProducts<int, float>&, cudaStream_t);

extern template bool SpGEMMExpandCSR<int, int, double>(
    int, int, const int*, const int*, const double*,
    int, const int*, const int*, const double*, int,
    const SpGEMMSymbolicResult<int, int>&, SpGEMMExpandedProducts<int, double>&, cudaStream_t);

} // namespace matrix_utils::sparse_cuda
