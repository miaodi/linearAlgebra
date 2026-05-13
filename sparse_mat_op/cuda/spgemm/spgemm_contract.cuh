#pragma once

#include "cuda_csr_utils.cuh"
#include "spgemm/spgemm_expand.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace matrix_utils::sparse_cuda
{

template <typename ROWTYPE, typename VALTYPE>
struct SpGEMMReducedProducts
{
    ROWTYPE nnz = 0;
    DeviceArray<std::uint64_t> row_col_keys;
    DeviceArray<VALTYPE> values;
};

/**
 * @brief Contract sorted expanded SpGEMM products by reducing duplicate (row, column) products.
 *
 * The input must be sorted by column within each row. Adjacent duplicates with
 * the same (row, column) are summed. This baseline implementation uses a global
 * packed-key reduce-by-key path, which balances work over all expanded products
 * instead of assigning contraction work one row at a time.
 */
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
bool SpGEMMContractSortedProducts( const SpGEMMSymbolicResult<ROWTYPE, COLTYPE>& symbolic,
                                   const SpGEMMExpandedProducts<COLTYPE, VALTYPE>& sorted,
                                   SpGEMMReducedProducts<ROWTYPE, VALTYPE>& reduced,
                                   cudaStream_t stream = nullptr );

/**
 * @brief Construct CSR output from reduced SpGEMM products.
 *
 * This matches the paper's construct phase: unpack reduced (row, column) keys,
 * copy reduced values, and build the CSR row pointer by counting unique columns
 * per row.
 */
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
bool SpGEMMConstructCSR( const SpGEMMSymbolicResult<ROWTYPE, COLTYPE>& symbolic,
                         const SpGEMMReducedProducts<ROWTYPE, VALTYPE>& reduced,
                         DeviceCSRMatrix<ROWTYPE, COLTYPE>& output,
                         DeviceArray<VALTYPE>& output_values,
                         cudaStream_t stream = nullptr );

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
bool SpGEMMContractSortedProducts( const SpGEMMSymbolicResult<ROWTYPE, COLTYPE>& symbolic,
                                   const SpGEMMExpandedProducts<COLTYPE, VALTYPE>& sorted,
                                   DeviceCSRMatrix<ROWTYPE, COLTYPE>& output,
                                   DeviceArray<VALTYPE>& output_values,
                                   cudaStream_t stream = nullptr );

extern template bool SpGEMMContractSortedProducts<int, int, float>( const SpGEMMSymbolicResult<int, int>&,
                                                                    const SpGEMMExpandedProducts<int, float>&,
                                                                    SpGEMMReducedProducts<int, float>&,
                                                                    cudaStream_t );

extern template bool SpGEMMContractSortedProducts<int, int, double>( const SpGEMMSymbolicResult<int, int>&,
                                                                     const SpGEMMExpandedProducts<int, double>&,
                                                                     SpGEMMReducedProducts<int, double>&,
                                                                     cudaStream_t );

extern template bool SpGEMMContractSortedProducts<std::int64_t, int, float>(
    const SpGEMMSymbolicResult<std::int64_t, int>&,
    const SpGEMMExpandedProducts<int, float>&,
    SpGEMMReducedProducts<std::int64_t, float>&,
    cudaStream_t );

extern template bool SpGEMMContractSortedProducts<std::int64_t, int, double>(
    const SpGEMMSymbolicResult<std::int64_t, int>&,
    const SpGEMMExpandedProducts<int, double>&,
    SpGEMMReducedProducts<std::int64_t, double>&,
    cudaStream_t );

extern template bool SpGEMMConstructCSR<int, int, float>( const SpGEMMSymbolicResult<int, int>&,
                                                          const SpGEMMReducedProducts<int, float>&,
                                                          DeviceCSRMatrix<int, int>&,
                                                          DeviceArray<float>&,
                                                          cudaStream_t );

extern template bool SpGEMMConstructCSR<int, int, double>( const SpGEMMSymbolicResult<int, int>&,
                                                           const SpGEMMReducedProducts<int, double>&,
                                                           DeviceCSRMatrix<int, int>&,
                                                           DeviceArray<double>&,
                                                           cudaStream_t );

extern template bool SpGEMMConstructCSR<std::int64_t, int, float>( const SpGEMMSymbolicResult<std::int64_t, int>&,
                                                                   const SpGEMMReducedProducts<std::int64_t, float>&,
                                                                   DeviceCSRMatrix<std::int64_t, int>&,
                                                                   DeviceArray<float>&,
                                                                   cudaStream_t );

extern template bool SpGEMMConstructCSR<std::int64_t, int, double>(
    const SpGEMMSymbolicResult<std::int64_t, int>&,
    const SpGEMMReducedProducts<std::int64_t, double>&,
    DeviceCSRMatrix<std::int64_t, int>&,
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

extern template bool SpGEMMContractSortedProducts<std::int64_t, int, float>(
    const SpGEMMSymbolicResult<std::int64_t, int>&,
    const SpGEMMExpandedProducts<int, float>&,
    DeviceCSRMatrix<std::int64_t, int>&,
    DeviceArray<float>&,
    cudaStream_t );

extern template bool SpGEMMContractSortedProducts<std::int64_t, int, double>(
    const SpGEMMSymbolicResult<std::int64_t, int>&,
    const SpGEMMExpandedProducts<int, double>&,
    DeviceCSRMatrix<std::int64_t, int>&,
    DeviceArray<double>&,
    cudaStream_t );

} // namespace matrix_utils::sparse_cuda
