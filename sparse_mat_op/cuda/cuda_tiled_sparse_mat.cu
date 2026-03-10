#include "cuda_tiled_sparse_mat.cuh"
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

#include <stdexcept>
#include <string>
namespace matrix_utils::sparse_cuda
{

namespace detail
{

inline void check_cuda(cudaError_t status, const char* msg)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(status));
    }
}

inline int CeilLog2U64(uint64_t value)
{
    int bits = 0;
    uint64_t x = 1;
    while (x < value)
    {
        x <<= 1;
        ++bits;
    }
    return bits;
}

} // namespace detail

template <typename NnzType, typename IndexType>
__global__ void BuildTileKeys(
    NnzType nnz,
    int k,
    const IndexType* __restrict__ coo_row,
    const IndexType* __restrict__ coo_col,
    int col_bits,
    IndexType base,
    uint64_t* __restrict__ keys)
{
    NnzType i = static_cast<NnzType>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= nnz) return;

    const uint64_t row = static_cast<uint64_t>(coo_row[i] - base);
    const uint64_t col = static_cast<uint64_t>(coo_col[i] - base);
    const uint64_t tile_row = row >> k;
    const uint64_t tile_col = col >> k;

    keys[i] = (tile_row << col_bits) | tile_col;
}

template <typename NnzType, typename IndexType>
__global__ void DecodeTileKeys(
    NnzType n,
    const uint64_t* __restrict__ keys,
    int col_bits,
    IndexType* __restrict__ tile_rows,
    IndexType* __restrict__ tile_cols)
{
    NnzType i = static_cast<NnzType>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint64_t row = 0;
    uint64_t col = 0;
    DecodeTileKey(keys[i], col_bits, row, col);
    tile_rows[i] = static_cast<IndexType>(row);
    tile_cols[i] = static_cast<IndexType>(col);
}

template <typename NnzType, typename IndexType>
void LaunchDecodeTileKeys(
    NnzType n,
    const uint64_t* d_keys,
    int col_bits,
    IndexType* d_tile_rows,
    IndexType* d_tile_cols,
    cudaStream_t stream)
{
    if (n <= 0) return;

    constexpr int block = 256;
    const int grid = static_cast<int>((n + block - 1) / block);
    DecodeTileKeys<NnzType, IndexType><<<grid, block, 0, stream>>>(
        n, d_keys, col_bits, d_tile_rows, d_tile_cols);
}

template <typename NnzType>
NnzType CountUniqueTileKeys(
    NnzType n,
    const uint64_t* d_keys,
    cudaStream_t stream)
{
    if (n <= 0) return 0;
    if (!d_keys)
    {
        throw std::invalid_argument("CountUniqueTileKeys received null pointer");
    }

    auto exec = thrust::cuda::par.on(stream);
    auto keys_begin = thrust::device_pointer_cast(d_keys);
    const NnzType transitions = thrust::inner_product(
        exec,
        keys_begin + 1,
        keys_begin + static_cast<size_t>(n),
        keys_begin,
        NnzType{0},
        thrust::plus<NnzType>(),
        thrust::not_equal_to<uint64_t>());
    return transitions + NnzType{1};
}

template <typename NnzType, typename CountType, typename IndexType>
void TileKeysToCOOMeta(
    NnzType n,
    NnzType n_tiles,
    const uint64_t* d_keys,
    int col_bits,
    uint64_t* d_unique_keys,
    CountType* d_tile_nnz,
    IndexType* d_tile_rows,
    IndexType* d_tile_cols,
    cudaStream_t stream)
{
    if (n <= 0) return;
    if (!d_keys || !d_unique_keys || !d_tile_nnz || !d_tile_rows || !d_tile_cols)
    {
        throw std::invalid_argument("TileKeysToCOOMeta received null pointer");
    }
    if (col_bits < 0 || col_bits >= 64)
    {
        throw std::invalid_argument("TileKeysToCOOMeta requires 0 <= col_bits < 64");
    }
    if (n_tiles <= 0)
    {
        throw std::invalid_argument("TileKeysToCOOMeta requires n_tiles > 0 when n > 0");
    }

    auto exec = thrust::cuda::par.on(stream);
    auto keys_begin = thrust::device_pointer_cast(d_keys);
    auto unique_begin = thrust::device_pointer_cast(d_unique_keys);
    auto tile_nnz_begin = thrust::device_pointer_cast(d_tile_nnz);

    const auto reduce_result = thrust::reduce_by_key(
        exec,
        keys_begin,
        keys_begin + static_cast<size_t>(n),
        thrust::make_constant_iterator(CountType{1}),
        unique_begin,
        tile_nnz_begin);

    const NnzType produced_tiles = static_cast<NnzType>(reduce_result.first - unique_begin);
    if (produced_tiles != n_tiles)
    {
        throw std::runtime_error("TileKeysToCOOMeta n_tiles does not match key run count");
    }
    LaunchDecodeTileKeys(n_tiles, d_unique_keys, col_bits, d_tile_rows, d_tile_cols, stream);
    detail::check_cuda(cudaGetLastError(), "DecodeTileKeys kernel launch");
}

template <typename NnzType, typename IndexType>
void LaunchBuildTileKeys(
    NnzType nnz,
    int k,
    const IndexType* d_coo_row,
    const IndexType* d_coo_col,
    int col_bits,
    IndexType base,
    uint64_t* d_keys,
    cudaStream_t stream)
{
    if (nnz <= 0) return;

    constexpr int block = 256;
    const int grid = static_cast<int>((nnz + block - 1) / block);
    BuildTileKeys<NnzType, IndexType><<<grid, block, 0, stream>>>(
        nnz, k, d_coo_row, d_coo_col, col_bits, base, d_keys);
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void CSRToTileCOO(
    COLTYPE rows,
    COLTYPE cols,
    const ROWTYPE* d_ai,
    const COLTYPE* d_aj,
    const VALTYPE* d_av,
    int k,
    DeviceTileCOOMatrix<ROWTYPE, COLTYPE, VALTYPE>& out,
    cudaStream_t stream)
{
    if (!d_ai || !d_aj || !d_av)
    {
        throw std::invalid_argument("CSRToTileCOO received null pointer");
    }
    if (k < 0 || k >= 63)
    {
        throw std::invalid_argument("CSRToTileCOO requires 0 <= k < 63");
    }

    ROWTYPE base{}, last{};
    detail::check_cuda(cudaMemcpy(&base, d_ai, sizeof(ROWTYPE), cudaMemcpyDeviceToHost), "load CSR base");
    detail::check_cuda(cudaMemcpy(&last, d_ai + rows, sizeof(ROWTYPE), cudaMemcpyDeviceToHost), "load CSR nnz bound");
    const ROWTYPE nnz = last - base;
    const uint64_t tile_size = uint64_t{1} << k;
    const uint64_t num_tile_rows_u64 =
        (static_cast<uint64_t>(rows) + tile_size - 1) / tile_size;
    const uint64_t num_tile_cols_u64 =
        (static_cast<uint64_t>(cols) + tile_size - 1) / tile_size;
    const int row_bits = detail::CeilLog2U64(num_tile_rows_u64 == 0 ? 1 : num_tile_rows_u64);
    const int col_bits = detail::CeilLog2U64(num_tile_cols_u64 == 0 ? 1 : num_tile_cols_u64);
    const int sort_end_bit = row_bits + col_bits;
    if (sort_end_bit > 64)
    {
        throw std::invalid_argument("CSRToTileCOO key width exceeds 64 bits");
    }
    const int effective_end_bit = sort_end_bit > 0 ? sort_end_bit : 1;

    out.n_rows = rows;
    out.n_cols = cols;
    out.base = base;
    out.tile_k = k;
    out.tile_col_bits = col_bits;
    out.n_tile_rows = static_cast<COLTYPE>(num_tile_rows_u64);
    out.n_tile_cols = static_cast<COLTYPE>(num_tile_cols_u64);
    out.n_tiles = 0;
    out.permutation.resize(static_cast<size_t>(nnz));
    out.row_ind.resize(static_cast<size_t>(nnz));
    out.col_ind.resize(static_cast<size_t>(nnz));
    out.values.resize(static_cast<size_t>(nnz));

    if (rows <= 0 || nnz <= 0)
    {
        out.permutation.resize(0);
        out.tile_nnz_prefix.resize(1);
        detail::check_cuda(cudaMemsetAsync(out.tile_nnz_prefix.data(), 0, sizeof(ROWTYPE), stream),
                           "reset tile_nnz_prefix");
        out.tile_row_ind.resize(0);
        out.tile_col_ind.resize(0);
        return;
    }

    DeviceArray<COLTYPE> coo_rows;
    coo_rows.resize(static_cast<size_t>(nnz));
    CSRPtrToCOORowDevice(rows, d_ai, coo_rows.data(), stream);

    DeviceArray<uint64_t> keys_in;
    DeviceArray<uint64_t> keys_sorted;
    keys_in.resize(static_cast<size_t>(nnz));
    keys_sorted.resize(static_cast<size_t>(nnz));
    DeviceArray<uint64_t> unique_keys;
    unique_keys.resize(static_cast<size_t>(nnz));

    DeviceArray<ROWTYPE> perm_in;
    perm_in.resize(static_cast<size_t>(nnz));

    LaunchBuildTileKeys<ROWTYPE, COLTYPE>(
        nnz, k, coo_rows.data(), d_aj, col_bits, static_cast<COLTYPE>(base), keys_in.data(), stream);
    detail::check_cuda(cudaGetLastError(), "BuildTileKeys kernel launch");

    thrust::sequence(thrust::cuda::par.on(stream), perm_in.data(), perm_in.data() + static_cast<size_t>(nnz));

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage,
        temp_storage_bytes,
        keys_in.data(),
        keys_sorted.data(),
        perm_in.data(),
        out.permutation.data(),
        nnz,
        0,
        effective_end_bit,
        stream);

    DeviceArray<std::uint8_t> temp_storage;
    temp_storage.resize(temp_storage_bytes == 0 ? 1 : temp_storage_bytes);

    cub::DeviceRadixSort::SortPairs(
        temp_storage.data(),
        temp_storage_bytes,
        keys_in.data(),
        keys_sorted.data(),
        perm_in.data(),
        out.permutation.data(),
        nnz,
        0,
        effective_end_bit,
        stream);

    auto exec = thrust::cuda::par.on(stream);
    const auto nnz_count = static_cast<size_t>(nnz);
    auto perm_begin = thrust::device_pointer_cast(out.permutation.data());
    thrust::gather(
        exec,
        perm_begin,
        perm_begin + nnz_count,
        thrust::device_pointer_cast(coo_rows.data()),
        thrust::device_pointer_cast(out.row_ind.data()));
    thrust::gather(
        exec,
        perm_begin,
        perm_begin + nnz_count,
        thrust::device_pointer_cast(d_aj),
        thrust::device_pointer_cast(out.col_ind.data()));
    thrust::gather(
        exec,
        perm_begin,
        perm_begin + nnz_count,
        thrust::device_pointer_cast(d_av),
        thrust::device_pointer_cast(out.values.data()));

    const size_t n_tiles = static_cast<size_t>(CountUniqueTileKeys<size_t>(
        nnz_count,
        keys_sorted.data(),
        stream));

    DeviceArray<ROWTYPE> tile_nnz;
    tile_nnz.resize(n_tiles);
    out.tile_row_ind.resize(n_tiles);
    out.tile_col_ind.resize(n_tiles);
    out.tile_nnz_prefix.resize(n_tiles + 1);

    TileKeysToCOOMeta<size_t, ROWTYPE, COLTYPE>(
        nnz_count,
        n_tiles,
        keys_sorted.data(),
        col_bits,
        unique_keys.data(),
        tile_nnz.data(),
        out.tile_row_ind.data(),
        out.tile_col_ind.data(),
        stream);

    thrust::exclusive_scan(thrust::cuda::par.on(stream),
                           tile_nnz.data(),
                           tile_nnz.data() + n_tiles,
                           out.tile_nnz_prefix.data());
    const ROWTYPE nnz_total = nnz;
    detail::check_cuda(cudaMemcpyAsync(out.tile_nnz_prefix.data() + n_tiles,
                                       &nnz_total,
                                       sizeof(ROWTYPE),
                                       cudaMemcpyHostToDevice,
                                       stream),
                       "write tile_nnz_prefix end");

    out.n_tiles = static_cast<COLTYPE>(n_tiles);
}

template __global__ void BuildTileKeys<int, int>(
    int, int, const int*, const int*, int, int, uint64_t*);
template __global__ void BuildTileKeys<std::int64_t, int>(
    std::int64_t, int, const int*, const int*, int, int, uint64_t*);
template void LaunchBuildTileKeys<int, int>(
    int, int, const int*, const int*, int, int, uint64_t*, cudaStream_t);
template void LaunchBuildTileKeys<std::int64_t, int>(
    std::int64_t, int, const int*, const int*, int, int, uint64_t*, cudaStream_t);
template int CountUniqueTileKeys<int>(int, const uint64_t*, cudaStream_t);
template std::int64_t CountUniqueTileKeys<std::int64_t>(std::int64_t, const uint64_t*, cudaStream_t);
template size_t CountUniqueTileKeys<size_t>(size_t, const uint64_t*, cudaStream_t);
template void TileKeysToCOOMeta<int, int, int>(
    int, int, const uint64_t*, int, uint64_t*, int*, int*, int*, cudaStream_t);
template void TileKeysToCOOMeta<std::int64_t, std::int64_t, int>(
    std::int64_t, std::int64_t, const uint64_t*, int, uint64_t*, std::int64_t*, int*, int*, cudaStream_t);
template void TileKeysToCOOMeta<size_t, int, int>(
    size_t, size_t, const uint64_t*, int, uint64_t*, int*, int*, int*, cudaStream_t);
template void CSRToTileCOO<int, int, float>(int, int, const int*, const int*, const float*, int, DeviceTileCOOMatrix<int, int, float>&, cudaStream_t);
template void CSRToTileCOO<int, int, double>(int, int, const int*, const int*, const double*, int, DeviceTileCOOMatrix<int, int, double>&, cudaStream_t);
template void CSRToTileCOO<std::int64_t, int, float>(int, int, const std::int64_t*, const int*, const float*, int, DeviceTileCOOMatrix<std::int64_t, int, float>&, cudaStream_t);
template void CSRToTileCOO<std::int64_t, int, double>(int, int, const std::int64_t*, const int*, const double*, int, DeviceTileCOOMatrix<std::int64_t, int, double>&, cudaStream_t);

} // namespace matrix_utils::sparse_cuda
