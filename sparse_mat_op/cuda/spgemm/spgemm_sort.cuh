#pragma once

#include "spgemm/spgemm_expand.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace matrix_utils::sparse_cuda
{

/**
 * @brief Sort expanded SpGEMM products by column inside each output row.
 *
 * Uses cub::DoubleBuffer to ping-pong between expanded and sorted buffers,
 * eliminating CUB's internal temporary copy.  The expanded input is clobbered.
 * After return, sorted always holds the result; if the final radix pass left
 * the data in expanded's buffer the arrays are swapped.
 */
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
bool SpGEMMSortExpandedProductsByColumn( const SpGEMMSymbolicResult<ROWTYPE, COLTYPE>& symbolic,
                                         SpGEMMExpandedProducts<COLTYPE, VALTYPE>& expanded,
                                         SpGEMMExpandedProducts<COLTYPE, VALTYPE>& sorted,
                                         cudaStream_t stream = nullptr );

extern template bool SpGEMMSortExpandedProductsByColumn<int, int, float>( const SpGEMMSymbolicResult<int, int>&,
                                                                          SpGEMMExpandedProducts<int, float>&,
                                                                          SpGEMMExpandedProducts<int, float>&,
                                                                          cudaStream_t );

extern template bool SpGEMMSortExpandedProductsByColumn<int, int, double>( const SpGEMMSymbolicResult<int, int>&,
                                                                           SpGEMMExpandedProducts<int, double>&,
                                                                           SpGEMMExpandedProducts<int, double>&,
                                                                           cudaStream_t );

} // namespace matrix_utils::sparse_cuda
