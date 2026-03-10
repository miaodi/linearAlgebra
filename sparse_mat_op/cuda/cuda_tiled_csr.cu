#include "cuda_tiled_csr.cuh"
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
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
    uint64_t* __restrict__ keys,
    int col_bits,
    IndexType base)
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
void LaunchBuildTileKeys(
    NnzType nnz,
    int k,
    const IndexType* d_coo_row,
    const IndexType* d_coo_col,
    uint64_t* d_keys,
    int col_bits,
    IndexType base,
    cudaStream_t stream)
{
    if (nnz <= 0) return;

    constexpr int block = 256;
    const int grid = static_cast<int>((nnz + block - 1) / block);
    BuildTileKeys<NnzType, IndexType><<<grid, block, 0, stream>>>(
        nnz, k, d_coo_row, d_coo_col, d_keys, col_bits, base);
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void CSRToTileCSR(
    COLTYPE rows,
    COLTYPE cols,
    const ROWTYPE* d_ai,
    const COLTYPE* d_aj,
    const VALTYPE* d_av,
    int k,
    DeviceTileCSRMatrix<ROWTYPE, COLTYPE, VALTYPE>& out,
    cudaStream_t stream)
{
    if (!d_ai || !d_aj || !d_av)
    {
        throw std::invalid_argument("CSRToTileCSR received null pointer");
    }
    if (k < 0 || k >= 63)
    {
        throw std::invalid_argument("CSRToTileCSR requires 0 <= k < 63");
    }

    ROWTYPE base{}, last{};
    detail::check_cuda(cudaMemcpy(&base, d_ai, sizeof(ROWTYPE), cudaMemcpyDeviceToHost), "load CSR base");
    detail::check_cuda(cudaMemcpy(&last, d_ai + rows, sizeof(ROWTYPE), cudaMemcpyDeviceToHost), "load CSR nnz bound");
    const ROWTYPE nnz = last - base;

    out.n_rows = rows;
    out.n_cols = cols;
    out.base = base;
    out.tile_k = k;
    out.tile_keys.resize(static_cast<size_t>(nnz));
    out.row_ind.resize(static_cast<size_t>(nnz));
    out.col_ind.resize(static_cast<size_t>(nnz));
    out.values.resize(static_cast<size_t>(nnz));

    if (rows <= 0 || nnz <= 0)
    {
        return;
    }

    DeviceArray<COLTYPE> coo_rows;
    coo_rows.resize(static_cast<size_t>(nnz));
    CSRPtrToCOORowDevice(rows, d_ai, coo_rows.data(), stream);

    DeviceArray<uint64_t> keys_in;
    keys_in.resize(static_cast<size_t>(nnz));

    DeviceArray<ROWTYPE> perm_in;
    DeviceArray<ROWTYPE> perm_out;
    perm_in.resize(static_cast<size_t>(nnz));
    perm_out.resize(static_cast<size_t>(nnz));

    const uint64_t tile_size = uint64_t{1} << k;
    const uint64_t num_tile_cols =
        (static_cast<uint64_t>(cols) + tile_size - 1) / tile_size;
    const int col_bits = detail::CeilLog2U64(num_tile_cols == 0 ? 1 : num_tile_cols);

    LaunchBuildTileKeys<ROWTYPE, COLTYPE>(
        nnz, k, coo_rows.data(), d_aj, keys_in.data(), col_bits, static_cast<COLTYPE>(base), stream);
    detail::check_cuda(cudaGetLastError(), "BuildTileKeys kernel launch");

    thrust::sequence(thrust::cuda::par.on(stream), perm_in.data(), perm_in.data() + static_cast<size_t>(nnz));

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage,
        temp_storage_bytes,
        keys_in.data(),
        out.tile_keys.data(),
        perm_in.data(),
        perm_out.data(),
        nnz,
        0,
        64,
        stream);

    DeviceArray<std::uint8_t> temp_storage;
    temp_storage.resize(temp_storage_bytes == 0 ? 1 : temp_storage_bytes);

    cub::DeviceRadixSort::SortPairs(
        temp_storage.data(),
        temp_storage_bytes,
        keys_in.data(),
        out.tile_keys.data(),
        perm_in.data(),
        perm_out.data(),
        nnz,
        0,
        64,
        stream);

    auto exec = thrust::cuda::par.on(stream);
    const auto nnz_count = static_cast<size_t>(nnz);
    auto perm_begin = thrust::device_pointer_cast(perm_out.data());
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
}

template __global__ void BuildTileKeys<int, int>(
    int, int, const int*, const int*, uint64_t*, int, int);
template __global__ void BuildTileKeys<std::int64_t, int>(
    std::int64_t, int, const int*, const int*, uint64_t*, int, int);
template void LaunchBuildTileKeys<int, int>(
    int, int, const int*, const int*, uint64_t*, int, int, cudaStream_t);
template void LaunchBuildTileKeys<std::int64_t, int>(
    std::int64_t, int, const int*, const int*, uint64_t*, int, int, cudaStream_t);
template void CSRToTileCSR<int, int, float>(int, int, const int*, const int*, const float*, int, DeviceTileCSRMatrix<int, int, float>&, cudaStream_t);
template void CSRToTileCSR<int, int, double>(int, int, const int*, const int*, const double*, int, DeviceTileCSRMatrix<int, int, double>&, cudaStream_t);
template void CSRToTileCSR<std::int64_t, int, float>(int, int, const std::int64_t*, const int*, const float*, int, DeviceTileCSRMatrix<std::int64_t, int, float>&, cudaStream_t);
template void CSRToTileCSR<std::int64_t, int, double>(int, int, const std::int64_t*, const int*, const double*, int, DeviceTileCSRMatrix<std::int64_t, int, double>&, cudaStream_t);

} // namespace matrix_utils::sparse_cuda
