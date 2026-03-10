#pragma once

#include "cuda_csr_utils.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace matrix_utils::sparse_cuda
{

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
struct DeviceTileCSRMatrix
{
    COLTYPE n_rows = 0;
    COLTYPE n_cols = 0;
    ROWTYPE base = 0;
    int tile_k = 0;
    DeviceArray<uint64_t> tile_keys; // one key per nonzero, sorted by tile
    DeviceArray<COLTYPE> row_ind;    // COO row indices in tile order
    DeviceArray<COLTYPE> col_ind;    // COO col indices in tile order
    DeviceArray<VALTYPE> values;     // values in tile order
};

template <typename NnzType, typename IndexType>
__global__ void BuildTileKeys(
    NnzType nnz,
    int k,
    const IndexType* __restrict__ coo_row,
    const IndexType* __restrict__ coo_col,
    uint64_t* __restrict__ keys,
    int col_bits,
    IndexType base);

template <typename NnzType, typename IndexType>
void LaunchBuildTileKeys(
    NnzType nnz,
    int k,
    const IndexType* d_coo_row,
    const IndexType* d_coo_col,
    uint64_t* d_keys,
    int col_bits,
    IndexType base,
    cudaStream_t stream = nullptr);

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void CSRToTileCSR(
    COLTYPE rows,
    COLTYPE cols,
    const ROWTYPE* d_ai,
    const COLTYPE* d_aj,
    const VALTYPE* d_av,
    int k,
    DeviceTileCSRMatrix<ROWTYPE, COLTYPE, VALTYPE>& out,
    cudaStream_t stream = nullptr);

extern template void CSRToTileCSR<int, int, float>(int, int, const int*, const int*, const float*, int, DeviceTileCSRMatrix<int, int, float>&, cudaStream_t);
extern template void CSRToTileCSR<int, int, double>(int, int, const int*, const int*, const double*, int, DeviceTileCSRMatrix<int, int, double>&, cudaStream_t);
extern template void CSRToTileCSR<std::int64_t, int, float>(int, int, const std::int64_t*, const int*, const float*, int, DeviceTileCSRMatrix<std::int64_t, int, float>&, cudaStream_t);
extern template void CSRToTileCSR<std::int64_t, int, double>(int, int, const std::int64_t*, const int*, const double*, int, DeviceTileCSRMatrix<std::int64_t, int, double>&, cudaStream_t);

} // namespace matrix_utils::sparse_cuda
