#pragma once

#include "cuda_csr_utils.cuh"
#include <cstdint>
#include <cuda_runtime.h>

namespace matrix_utils::sparse_cuda
{
inline __host__ __device__ void DecodeTileKey(uint64_t key, int col_bits, uint64_t& tile_row, uint64_t& tile_col)
{
    const uint64_t col_mask = (col_bits >= 64) ? ~uint64_t{0} : ((uint64_t{1} << col_bits) - 1);
    tile_col = key & col_mask;
    tile_row = key >> col_bits;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
struct DeviceTileCOOMatrix
{
    COLTYPE n_rows = 0;
    COLTYPE n_cols = 0;
    ROWTYPE base = 0;
    int tile_k = 0;
    int tile_col_bits = 0;
    COLTYPE n_tile_rows = 0;
    COLTYPE n_tile_cols = 0;
    COLTYPE n_tiles = 0;

    DeviceArray<ROWTYPE> permutation;       // sorted-index -> original nnz index

    DeviceArray<COLTYPE> tile_row_ind;      // decoded tile row index per non-empty tile
    DeviceArray<COLTYPE> tile_col_ind;      // decoded tile col index per non-empty tile
    DeviceArray<ROWTYPE> tile_nnz_prefix;   // tile nnz prefix sum, size n_tiles + 1

    DeviceArray<COLTYPE> row_ind;           // COO row indices in tile order
    DeviceArray<COLTYPE> col_ind;           // COO col indices in tile order
    DeviceArray<VALTYPE> values;            // COO values in tile order

    const COLTYPE* tileRowIndData() const { return tile_row_ind.data(); }
    const COLTYPE* tileColIndData() const { return tile_col_ind.data(); }
    const ROWTYPE* tileNnzPrefixData() const { return tile_nnz_prefix.data(); }
    const COLTYPE* rowIndData() const { return row_ind.data(); }
    const COLTYPE* colIndData() const { return col_ind.data(); }
    VALTYPE* valuesData() { return values.data(); }
    const VALTYPE* valuesData() const { return values.data(); }
    size_t nnz() const { return values.size(); }
};

template <typename NnzType, typename IndexType>
__global__ void BuildTileKeys(NnzType nnz, int k, const IndexType* __restrict__ coo_row,
                        const IndexType* __restrict__ coo_col, int col_bits,
                        IndexType base, uint64_t* __restrict__ keys);

template <typename NnzType, typename IndexType>
void LaunchBuildTileKeys(NnzType nnz, int k, const IndexType* d_coo_row, const IndexType* d_coo_col,
                    int col_bits, IndexType base, uint64_t* d_keys, cudaStream_t stream = nullptr);

template <typename NnzType>
NnzType CountUniqueTileKeys(NnzType n, const uint64_t* d_keys, cudaStream_t stream = nullptr);

/// @brief Build tile-level COO metadata from sorted tile keys.
/// @details Produces one entry per non-empty tile: unique key, tile nnz count,
/// and decoded tile row/column indices. The input keys must already be sorted.
template <typename NnzType, typename CountType, typename IndexType>
void TileKeysToCOOMeta(NnzType n, NnzType n_tiles, const uint64_t* d_keys, int col_bits,
                       uint64_t* d_unique_keys, CountType* d_tile_nnz, IndexType* d_tile_rows,
                       IndexType* d_tile_cols, cudaStream_t stream = nullptr);

/// @brief Convert CSR to tile-grouped COO layout on device.
/// @tparam SORT_BY_TILE_NNZ If true, non-empty tiles are reordered by nnz
/// in ascending order (small to large) before final COO gather. If false,
/// tile order follows key order (tile row/col lexicographic order).
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, bool SORT_BY_TILE_NNZ = false>
void CSRToTileCOO(COLTYPE rows, COLTYPE cols, const ROWTYPE* d_ai, const COLTYPE* d_aj,
          const VALTYPE* d_av, int k, DeviceTileCOOMatrix<ROWTYPE, COLTYPE, VALTYPE>& out,
          cudaStream_t stream = nullptr);

extern template void CSRToTileCOO<int, int, float, false>(int, int, const int*, const int*, const float*, int,
                           DeviceTileCOOMatrix<int, int, float>&, cudaStream_t);
extern template void CSRToTileCOO<int, int, float, true>(int, int, const int*, const int*, const float*, int,
                           DeviceTileCOOMatrix<int, int, float>&, cudaStream_t);
extern template void CSRToTileCOO<int, int, double, false>(int, int, const int*, const int*, const double*, int,
                            DeviceTileCOOMatrix<int, int, double>&, cudaStream_t);
extern template void CSRToTileCOO<int, int, double, true>(int, int, const int*, const int*, const double*, int,
                            DeviceTileCOOMatrix<int, int, double>&, cudaStream_t);
extern template void CSRToTileCOO<std::int64_t, int, float, false>(int, int, const std::int64_t*,
                                                            const int*, const float*, int,
                                DeviceTileCOOMatrix<std::int64_t, int, float>&,
                                                            cudaStream_t);
extern template void CSRToTileCOO<std::int64_t, int, float, true>(int, int, const std::int64_t*,
                                                            const int*, const float*, int,
                                DeviceTileCOOMatrix<std::int64_t, int, float>&,
                                                            cudaStream_t);
extern template void CSRToTileCOO<std::int64_t, int, double, false>(int, int, const std::int64_t*,
                                                             const int*, const double*, int,
                                 DeviceTileCOOMatrix<std::int64_t, int, double>&,
                                                             cudaStream_t);
extern template void CSRToTileCOO<std::int64_t, int, double, true>(int, int, const std::int64_t*,
                                                             const int*, const double*, int,
                                 DeviceTileCOOMatrix<std::int64_t, int, double>&,
                                                             cudaStream_t);

} // namespace matrix_utils::sparse_cuda
