#pragma once

#include "cuda_csr_utils.cuh"
#include "spgemm/spgemm_expand.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace matrix_utils::sparse_cuda
{

template <typename VALTYPE>
struct SpGEMMReducedProducts
{
    ExpandedIndex nnz = 0;
    DeviceArray<std::uint64_t> row_col_keys;
    DeviceArray<VALTYPE> values;
};

/**
 * @brief Contract sorted expanded SpGEMM products (packed-key reduce-by-key path).
 */
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
bool SpGEMMContractSortedProducts( const SpGEMMSymbolicResult<ROWTYPE, COLTYPE>& symbolic,
                                   const SpGEMMExpandedProducts<COLTYPE, VALTYPE>& sorted,
                                   SpGEMMReducedProducts<VALTYPE>& reduced,
                                   cudaStream_t stream = nullptr );

/**
 * @brief Construct CSR output from reduced SpGEMM products.
 */
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
bool SpGEMMConstructCSR( const SpGEMMSymbolicResult<ROWTYPE, COLTYPE>& symbolic,
                         const SpGEMMReducedProducts<VALTYPE>& reduced,
                         DeviceCSRMatrix<ROWTYPE, COLTYPE>& output,
                         DeviceArray<VALTYPE>& output_values,
                         cudaStream_t stream = nullptr );

/**
 * @brief Direct segmented contraction: sorted expanded products → CSR output.
 */
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
bool SpGEMMContractSortedProducts( const SpGEMMSymbolicResult<ROWTYPE, COLTYPE>& symbolic,
                                   const SpGEMMExpandedProducts<COLTYPE, VALTYPE>& sorted,
                                   DeviceCSRMatrix<ROWTYPE, COLTYPE>& output,
                                   DeviceArray<VALTYPE>& output_values,
                                   cudaStream_t stream = nullptr );

extern template bool SpGEMMContractSortedProducts<int, int, float>( const SpGEMMSymbolicResult<int, int>&,
                                                                    const SpGEMMExpandedProducts<int, float>&,
                                                                    SpGEMMReducedProducts<float>&,
                                                                    cudaStream_t );

extern template bool SpGEMMContractSortedProducts<int, int, double>( const SpGEMMSymbolicResult<int, int>&,
                                                                     const SpGEMMExpandedProducts<int, double>&,
                                                                     SpGEMMReducedProducts<double>&,
                                                                     cudaStream_t );

extern template bool SpGEMMConstructCSR<int, int, float>( const SpGEMMSymbolicResult<int, int>&,
                                                          const SpGEMMReducedProducts<float>&,
                                                          DeviceCSRMatrix<int, int>&,
                                                          DeviceArray<float>&,
                                                          cudaStream_t );

extern template bool SpGEMMConstructCSR<int, int, double>( const SpGEMMSymbolicResult<int, int>&,
                                                           const SpGEMMReducedProducts<double>&,
                                                           DeviceCSRMatrix<int, int>&,
                                                           DeviceArray<double>&,
                                                           cudaStream_t );

extern template bool SpGEMMContractSortedProducts<int, int, float>( const SpGEMMSymbolicResult<int, int>&,
                                                                    const SpGEMMExpandedProducts<int, float>&,
                                                                    DeviceCSRMatrix<int, int>&,
                                                                    DeviceArray<float>&,
                                                                    cudaStream_t );

extern template bool SpGEMMContractSortedProducts<int, int, double>( const SpGEMMSymbolicResult<int, int>&,
                                                                     const SpGEMMExpandedProducts<int, double>&,
                                                                     DeviceCSRMatrix<int, int>&,
                                                                     DeviceArray<double>&,
                                                                     cudaStream_t );

} // namespace matrix_utils::sparse_cuda
