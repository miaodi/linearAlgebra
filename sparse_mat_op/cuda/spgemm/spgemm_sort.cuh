#pragma once

#include "spgemm/spgemm_expand.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace matrix_utils::sparse_cuda
{

/**
 * @brief Sort expanded SpGEMM products by column inside each output row.
 *
 * This is a baseline segmented radix sort over the expanded row slices described
 * by symbolic.expanded_row_ptr. It preserves duplicate columns and moves values
 * together with their column keys, preparing the data for a later row-wise
 * reduction/contraction phase. The sorted output must not alias the expanded
 * input buffers.
 */
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
bool SpGEMMSortExpandedProductsByColumn( const SpGEMMSymbolicResult<ROWTYPE, COLTYPE>& symbolic,
                                         const SpGEMMExpandedProducts<COLTYPE, VALTYPE>& expanded,
                                         SpGEMMExpandedProducts<COLTYPE, VALTYPE>& sorted,
                                         cudaStream_t stream = nullptr );

extern template bool SpGEMMSortExpandedProductsByColumn<int, int, float>(
    const SpGEMMSymbolicResult<int, int>&,
    const SpGEMMExpandedProducts<int, float>&,
    SpGEMMExpandedProducts<int, float>&,
    cudaStream_t );

extern template bool SpGEMMSortExpandedProductsByColumn<int, int, double>(
    const SpGEMMSymbolicResult<int, int>&,
    const SpGEMMExpandedProducts<int, double>&,
    SpGEMMExpandedProducts<int, double>&,
    cudaStream_t );

extern template bool SpGEMMSortExpandedProductsByColumn<std::int64_t, int, float>(
    const SpGEMMSymbolicResult<std::int64_t, int>&,
    const SpGEMMExpandedProducts<int, float>&,
    SpGEMMExpandedProducts<int, float>&,
    cudaStream_t );

extern template bool SpGEMMSortExpandedProductsByColumn<std::int64_t, int, double>(
    const SpGEMMSymbolicResult<std::int64_t, int>&,
    const SpGEMMExpandedProducts<int, double>&,
    SpGEMMExpandedProducts<int, double>&,
    cudaStream_t );

} // namespace matrix_utils::sparse_cuda
