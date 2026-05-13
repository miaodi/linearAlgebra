#include "spgemm/spgemm_expand.cuh"

#include <cub/cub.cuh>
#include <cuda/std/functional>
#include <limits>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace matrix_utils::sparse_cuda
{
namespace
{

template <typename ROWTYPE>
bool thresholdFits(std::int64_t value)
{
    return value >= 0 && value <= static_cast<std::int64_t>(std::numeric_limits<ROWTYPE>::max());
}

bool thresholdsAreOrdered(const SpGEMMSymbolicOptions& options)
{
    return options.thread_threshold <= options.warp_threshold &&
           options.warp_threshold <= options.cta_threshold;
}

template <typename COLTYPE>
bool rowClassOffsetsAreValid(const std::array<COLTYPE, 5>& offsets, COLTYPE rows)
{
    return offsets[0] == 0 && offsets[4] == rows &&
           offsets[0] <= offsets[1] && offsets[1] <= offsets[2] &&
           offsets[2] <= offsets[3] && offsets[3] <= offsets[4];
}

template <typename ROWTYPE, typename COLTYPE>
__global__ void compute_expanded_nnz_kernel(
    COLTYPE rows,
    const ROWTYPE* A_row_ptr,
    const COLTYPE* A_col_ind,
    const ROWTYPE* B_row_ptr,
    ROWTYPE base,
    ROWTYPE* expanded_nnz,
    ROWTYPE* scan_input)
{
    COLTYPE row = static_cast<COLTYPE>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows)
    {
        return;
    }

    ROWTYPE count = 0;
    const ROWTYPE a_begin = A_row_ptr[row] - base;
    const ROWTYPE a_end = A_row_ptr[row + 1] - base;

    for (ROWTYPE p = a_begin; p < a_end; ++p)
    {
        const COLTYPE b_row = static_cast<COLTYPE>(A_col_ind[p] - base);
        count += B_row_ptr[b_row + 1] - B_row_ptr[b_row];
    }

    expanded_nnz[row] = count;
    scan_input[row] = count;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
__device__ void expand_row_sequential(
    COLTYPE row,
    const ROWTYPE* A_row_ptr,
    const COLTYPE* A_col_ind,
    const VALTYPE* A_values,
    const ROWTYPE* B_row_ptr,
    const COLTYPE* B_col_ind,
    const VALTYPE* B_values,
    ROWTYPE base,
    const ROWTYPE* expanded_row_ptr,
    COLTYPE* expanded_col_ind,
    VALTYPE* expanded_values)
{
    ROWTYPE out = expanded_row_ptr[row] - base;
    const ROWTYPE a_begin = A_row_ptr[row] - base;
    const ROWTYPE a_end = A_row_ptr[row + 1] - base;

    for (ROWTYPE p = a_begin; p < a_end; ++p)
    {
        const COLTYPE b_row = static_cast<COLTYPE>(A_col_ind[p] - base);
        const VALTYPE a_value = A_values[p];
        const ROWTYPE b_begin = B_row_ptr[b_row] - base;
        const ROWTYPE b_end = B_row_ptr[b_row + 1] - base;

        for (ROWTYPE q = b_begin; q < b_end; ++q)
        {
            expanded_col_ind[out] = B_col_ind[q];
            expanded_values[out] = a_value * B_values[q];
            ++out;
        }
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
__device__ void expand_one_product_by_local_index(
    COLTYPE row,
    ROWTYPE local_product,
    const ROWTYPE* A_row_ptr,
    const COLTYPE* A_col_ind,
    const VALTYPE* A_values,
    const ROWTYPE* B_row_ptr,
    const COLTYPE* B_col_ind,
    const VALTYPE* B_values,
    ROWTYPE base,
    const ROWTYPE* expanded_row_ptr,
    COLTYPE* expanded_col_ind,
    VALTYPE* expanded_values)
{
    ROWTYPE remaining = local_product;
    const ROWTYPE a_begin = A_row_ptr[row] - base;
    const ROWTYPE a_end = A_row_ptr[row + 1] - base;

    for (ROWTYPE p = a_begin; p < a_end; ++p)
    {
        const COLTYPE b_row = static_cast<COLTYPE>(A_col_ind[p] - base);
        const ROWTYPE b_begin = B_row_ptr[b_row] - base;
        const ROWTYPE b_nnz = B_row_ptr[b_row + 1] - B_row_ptr[b_row];
        if (remaining < b_nnz)
        {
            const ROWTYPE q = b_begin + remaining;
            const ROWTYPE out = expanded_row_ptr[row] - base + local_product;
            expanded_col_ind[out] = B_col_ind[q];
            expanded_values[out] = A_values[p] * B_values[q];
            return;
        }
        remaining -= b_nnz;
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
__global__ void expand_thread_rows_kernel(
    COLTYPE slot_begin,
    COLTYPE slot_end,
    const COLTYPE* row_perm,
    const ROWTYPE* A_row_ptr,
    const COLTYPE* A_col_ind,
    const VALTYPE* A_values,
    const ROWTYPE* B_row_ptr,
    const COLTYPE* B_col_ind,
    const VALTYPE* B_values,
    ROWTYPE base,
    const ROWTYPE* expanded_row_ptr,
    COLTYPE* expanded_col_ind,
    VALTYPE* expanded_values)
{
    COLTYPE slot = slot_begin + static_cast<COLTYPE>(blockIdx.x * blockDim.x + threadIdx.x);
    if (slot >= slot_end)
    {
        return;
    }

    const COLTYPE row = static_cast<COLTYPE>(row_perm[slot] - base);
    expand_row_sequential(
        row,
        A_row_ptr,
        A_col_ind,
        A_values,
        B_row_ptr,
        B_col_ind,
        B_values,
        base,
        expanded_row_ptr,
        expanded_col_ind,
        expanded_values);
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
__global__ void expand_warp_rows_kernel(
    COLTYPE slot_begin,
    COLTYPE slot_end,
    const COLTYPE* row_perm,
    const ROWTYPE* A_row_ptr,
    const COLTYPE* A_col_ind,
    const VALTYPE* A_values,
    const ROWTYPE* B_row_ptr,
    const COLTYPE* B_col_ind,
    const VALTYPE* B_values,
    ROWTYPE base,
    const ROWTYPE* expanded_row_ptr,
    COLTYPE* expanded_col_ind,
    VALTYPE* expanded_values)
{
    constexpr int warp_size = 32;
    const int lane = threadIdx.x & (warp_size - 1);
    const int warp_in_block = threadIdx.x / warp_size;
    const int warps_per_block = blockDim.x / warp_size;
    const COLTYPE slot = slot_begin + static_cast<COLTYPE>(blockIdx.x * warps_per_block + warp_in_block);
    if (slot >= slot_end)
    {
        return;
    }

    const COLTYPE row = static_cast<COLTYPE>(row_perm[slot] - base);
    const ROWTYPE row_nnz = expanded_row_ptr[row + 1] - expanded_row_ptr[row];
    for (ROWTYPE local = static_cast<ROWTYPE>(lane); local < row_nnz; local += warp_size)
    {
        expand_one_product_by_local_index(
            row,
            local,
            A_row_ptr,
            A_col_ind,
            A_values,
            B_row_ptr,
            B_col_ind,
            B_values,
            base,
            expanded_row_ptr,
            expanded_col_ind,
            expanded_values);
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
__global__ void expand_cta_rows_kernel(
    COLTYPE slot_begin,
    COLTYPE slot_end,
    const COLTYPE* row_perm,
    const ROWTYPE* A_row_ptr,
    const COLTYPE* A_col_ind,
    const VALTYPE* A_values,
    const ROWTYPE* B_row_ptr,
    const COLTYPE* B_col_ind,
    const VALTYPE* B_values,
    ROWTYPE base,
    const ROWTYPE* expanded_row_ptr,
    COLTYPE* expanded_col_ind,
    VALTYPE* expanded_values)
{
    const COLTYPE slot = slot_begin + static_cast<COLTYPE>(blockIdx.x);
    if (slot >= slot_end)
    {
        return;
    }

    const COLTYPE row = static_cast<COLTYPE>(row_perm[slot] - base);
    const ROWTYPE row_nnz = expanded_row_ptr[row + 1] - expanded_row_ptr[row];
    for (ROWTYPE local = static_cast<ROWTYPE>(threadIdx.x); local < row_nnz; local += blockDim.x)
    {
        expand_one_product_by_local_index(
            row,
            local,
            A_row_ptr,
            A_col_ind,
            A_values,
            B_row_ptr,
            B_col_ind,
            B_values,
            base,
            expanded_row_ptr,
            expanded_col_ind,
            expanded_values);
    }
}

template <typename ROWTYPE, typename COLTYPE>
__global__ void compute_global_tile_counts_kernel(
    COLTYPE global_slot_begin,
    COLTYPE global_rows,
    const COLTYPE* row_perm,
    const ROWTYPE* expanded_row_ptr,
    ROWTYPE base,
    ROWTYPE tile_size,
    ROWTYPE* tile_scan_input)
{
    const COLTYPE local_global_row = static_cast<COLTYPE>(blockIdx.x * blockDim.x + threadIdx.x);
    if (local_global_row >= global_rows)
    {
        return;
    }

    const COLTYPE slot = global_slot_begin + local_global_row;
    const COLTYPE row = static_cast<COLTYPE>(row_perm[slot] - base);
    const ROWTYPE row_nnz = expanded_row_ptr[row + 1] - expanded_row_ptr[row];
    tile_scan_input[local_global_row] = (row_nnz + tile_size - 1) / tile_size;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
__global__ void expand_global_rows_kernel(
    COLTYPE global_slot_begin,
    COLTYPE global_rows,
    const ROWTYPE* global_tile_offsets,
    const COLTYPE* row_perm,
    const ROWTYPE* A_row_ptr,
    const COLTYPE* A_col_ind,
    const VALTYPE* A_values,
    const ROWTYPE* B_row_ptr,
    const COLTYPE* B_col_ind,
    const VALTYPE* B_values,
    ROWTYPE base,
    const ROWTYPE* expanded_row_ptr,
    COLTYPE* expanded_col_ind,
    VALTYPE* expanded_values)
{
    const ROWTYPE tile = static_cast<ROWTYPE>(blockIdx.x);

    COLTYPE low = 0;
    COLTYPE high = global_rows;
    while (low < high)
    {
        const COLTYPE mid = static_cast<COLTYPE>((low + high) / 2);
        if (global_tile_offsets[mid] <= tile)
        {
            low = static_cast<COLTYPE>(mid + 1);
        }
        else
        {
            high = mid;
        }
    }

    const COLTYPE local_global_row = static_cast<COLTYPE>(low - 1);
    const COLTYPE slot = global_slot_begin + local_global_row;
    const COLTYPE row = static_cast<COLTYPE>(row_perm[slot] - base);
    const ROWTYPE tile_in_row = tile - global_tile_offsets[local_global_row];
    const ROWTYPE row_nnz = expanded_row_ptr[row + 1] - expanded_row_ptr[row];
    const ROWTYPE local_begin = tile_in_row * static_cast<ROWTYPE>(blockDim.x);

    for (ROWTYPE local = local_begin + static_cast<ROWTYPE>(threadIdx.x);
         local < row_nnz && local < local_begin + static_cast<ROWTYPE>(blockDim.x);
         local += blockDim.x)
    {
        expand_one_product_by_local_index(
            row,
            local,
            A_row_ptr,
            A_col_ind,
            A_values,
            B_row_ptr,
            B_col_ind,
            B_values,
            base,
            expanded_row_ptr,
            expanded_col_ind,
            expanded_values);
    }
}

} // namespace

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
    SpGEMMSymbolicOptions options,
    cudaStream_t stream)
{
    if (A_rows < 0 || A_cols < 0 || B_rows < 0 || A_cols != B_rows)
    {
        return false;
    }
    if (!thresholdsAreOrdered(options) ||
        !thresholdFits<ROWTYPE>(options.thread_threshold) ||
        !thresholdFits<ROWTYPE>(options.warp_threshold) ||
        !thresholdFits<ROWTYPE>(options.cta_threshold))
    {
        return false;
    }
    if ((A_rows > 0 || A_cols > 0) && (d_A_row_ptr == nullptr || d_B_row_ptr == nullptr))
    {
        return false;
    }
    if (A_rows > 0 && d_A_col_ind == nullptr)
    {
        return false;
    }

    result.n_rows = A_rows;
    result.base = base;
    result.total_expanded_nnz = 0;
    result.row_class_offsets = {0, 0, 0, 0, 0};
    result.expanded_nnz.resize(static_cast<size_t>(A_rows));
    result.expanded_row_ptr.resize(static_cast<size_t>(A_rows + 1));
    result.row_perm.resize(static_cast<size_t>(A_rows));
    result.sorted_expanded_nnz.resize(static_cast<size_t>(A_rows));

    DeviceArray<ROWTYPE> scan_input;
    scan_input.resize(static_cast<size_t>(A_rows + 1));
    checkCudaError(cudaMemsetAsync(scan_input.data(), 0, static_cast<size_t>(A_rows + 1) * sizeof(ROWTYPE), stream),
                   "initialize SpGEMM symbolic scan input");

    if (A_rows > 0)
    {
        const int threads = 256;
        const int blocks = (static_cast<int>(A_rows) + threads - 1) / threads;
        compute_expanded_nnz_kernel<ROWTYPE, COLTYPE><<<blocks, threads, 0, stream>>>(
            A_rows,
            d_A_row_ptr,
            d_A_col_ind,
            d_B_row_ptr,
            base,
            result.expanded_nnz.data(),
            scan_input.data());
        checkCudaError(cudaGetLastError(), "launch SpGEMM expanded row-count kernel");
    }

    void* temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveScan(
        temp_storage,
        temp_storage_bytes,
        scan_input.data(),
        result.expanded_row_ptr.data(),
        ::cuda::std::plus<ROWTYPE>(),
        base,
        static_cast<int>(A_rows + 1),
        stream);

    DeviceArray<std::uint8_t> scan_storage;
    scan_storage.resize(temp_storage_bytes);
    cub::DeviceScan::ExclusiveScan(
        scan_storage.data(),
        temp_storage_bytes,
        scan_input.data(),
        result.expanded_row_ptr.data(),
        ::cuda::std::plus<ROWTYPE>(),
        base,
        static_cast<int>(A_rows + 1),
        stream);

    if (A_rows > 0)
    {
        checkCudaError(cudaMemcpyAsync(
                           result.sorted_expanded_nnz.data(),
                           result.expanded_nnz.data(),
                           static_cast<size_t>(A_rows) * sizeof(ROWTYPE),
                           cudaMemcpyDeviceToDevice,
                           stream),
                       "copy SpGEMM expanded row counts");

        auto policy = thrust::cuda::par.on(stream);
        auto row_begin = thrust::device_pointer_cast(result.row_perm.data());
        auto row_end = row_begin + A_rows;
        auto nnz_begin = thrust::device_pointer_cast(result.sorted_expanded_nnz.data());
        auto nnz_end = nnz_begin + A_rows;

        thrust::sequence(policy, row_begin, row_end, static_cast<COLTYPE>(base));
        thrust::stable_sort_by_key(policy, nnz_begin, nnz_end, row_begin);

        const ROWTYPE thread_threshold = static_cast<ROWTYPE>(options.thread_threshold);
        const ROWTYPE warp_threshold = static_cast<ROWTYPE>(options.warp_threshold);
        const ROWTYPE cta_threshold = static_cast<ROWTYPE>(options.cta_threshold);

        auto thread_end = thrust::upper_bound(policy, nnz_begin, nnz_end, thread_threshold);
        auto warp_end = thrust::upper_bound(policy, nnz_begin, nnz_end, warp_threshold);
        auto cta_end = thrust::upper_bound(policy, nnz_begin, nnz_end, cta_threshold);

        result.row_class_offsets = {
            0,
            static_cast<COLTYPE>(thread_end - nnz_begin),
            static_cast<COLTYPE>(warp_end - nnz_begin),
            static_cast<COLTYPE>(cta_end - nnz_begin),
            A_rows};
    }
    else
    {
        result.row_class_offsets = {0, 0, 0, 0, 0};
    }

    ROWTYPE expanded_end = base;
    checkCudaError(cudaMemcpyAsync(
                       &expanded_end,
                       result.expanded_row_ptr.data() + A_rows,
                       sizeof(ROWTYPE),
                       cudaMemcpyDeviceToHost,
                       stream),
                   "copy SpGEMM total expanded nnz");
    checkCudaError(cudaStreamSynchronize(stream), "synchronize SpGEMM symbolic analysis");
    result.total_expanded_nnz = expanded_end - base;

    return true;
}

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
    cudaStream_t stream)
{
    if (A_rows < 0 || A_cols < 0 || B_rows < 0 || A_cols != B_rows)
    {
        return false;
    }
    if (symbolic.n_rows != A_rows || symbolic.base != base || symbolic.total_expanded_nnz < 0)
    {
        return false;
    }
    if (!rowClassOffsetsAreValid(symbolic.row_class_offsets, A_rows))
    {
        return false;
    }
    if ((A_rows > 0 || A_cols > 0) && (d_A_row_ptr == nullptr || d_B_row_ptr == nullptr))
    {
        return false;
    }

    const ROWTYPE total = symbolic.total_expanded_nnz;
    expanded.col_ind.resize(static_cast<size_t>(total));
    expanded.values.resize(static_cast<size_t>(total));
    if (total == 0)
    {
        return true;
    }
    if (d_A_col_ind == nullptr || d_A_values == nullptr || d_B_col_ind == nullptr ||
        d_B_values == nullptr || symbolic.expanded_row_ptr.data() == nullptr ||
        symbolic.row_perm.data() == nullptr)
    {
        return false;
    }

    constexpr int threads_per_block = 256;
    const COLTYPE thread_begin = symbolic.classBegin(SpGEMMRowClass::Thread);
    const COLTYPE thread_end = symbolic.classEnd(SpGEMMRowClass::Thread);
    const COLTYPE warp_begin = symbolic.classBegin(SpGEMMRowClass::Warp);
    const COLTYPE warp_end = symbolic.classEnd(SpGEMMRowClass::Warp);
    const COLTYPE cta_begin = symbolic.classBegin(SpGEMMRowClass::CTA);
    const COLTYPE cta_end = symbolic.classEnd(SpGEMMRowClass::CTA);
    const COLTYPE global_begin = symbolic.classBegin(SpGEMMRowClass::Global);
    const COLTYPE global_end = symbolic.classEnd(SpGEMMRowClass::Global);

    const COLTYPE thread_rows = thread_end - thread_begin;
    if (thread_rows > 0)
    {
        const int blocks = (static_cast<int>(thread_rows) + threads_per_block - 1) / threads_per_block;
        expand_thread_rows_kernel<ROWTYPE, COLTYPE, VALTYPE><<<blocks, threads_per_block, 0, stream>>>(
            thread_begin,
            thread_end,
            symbolic.row_perm.data(),
            d_A_row_ptr,
            d_A_col_ind,
            d_A_values,
            d_B_row_ptr,
            d_B_col_ind,
            d_B_values,
            base,
            symbolic.expanded_row_ptr.data(),
            expanded.col_ind.data(),
            expanded.values.data());
        checkCudaError(cudaGetLastError(), "launch SpGEMM thread-row expansion kernel");
    }

    const COLTYPE warp_rows = warp_end - warp_begin;
    if (warp_rows > 0)
    {
        constexpr int warps_per_block = threads_per_block / 32;
        const int blocks = (static_cast<int>(warp_rows) + warps_per_block - 1) / warps_per_block;
        expand_warp_rows_kernel<ROWTYPE, COLTYPE, VALTYPE><<<blocks, threads_per_block, 0, stream>>>(
            warp_begin,
            warp_end,
            symbolic.row_perm.data(),
            d_A_row_ptr,
            d_A_col_ind,
            d_A_values,
            d_B_row_ptr,
            d_B_col_ind,
            d_B_values,
            base,
            symbolic.expanded_row_ptr.data(),
            expanded.col_ind.data(),
            expanded.values.data());
        checkCudaError(cudaGetLastError(), "launch SpGEMM warp-row expansion kernel");
    }

    const COLTYPE cta_rows = cta_end - cta_begin;
    if (cta_rows > 0)
    {
        expand_cta_rows_kernel<ROWTYPE, COLTYPE, VALTYPE><<<static_cast<unsigned int>(cta_rows), threads_per_block, 0, stream>>>(
            cta_begin,
            cta_end,
            symbolic.row_perm.data(),
            d_A_row_ptr,
            d_A_col_ind,
            d_A_values,
            d_B_row_ptr,
            d_B_col_ind,
            d_B_values,
            base,
            symbolic.expanded_row_ptr.data(),
            expanded.col_ind.data(),
            expanded.values.data());
        checkCudaError(cudaGetLastError(), "launch SpGEMM CTA-row expansion kernel");
    }

    const COLTYPE global_rows = global_end - global_begin;
    if (global_rows > 0)
    {
        DeviceArray<ROWTYPE> global_tile_scan_input;
        DeviceArray<ROWTYPE> global_tile_offsets;
        global_tile_scan_input.resize(static_cast<size_t>(global_rows + 1));
        global_tile_offsets.resize(static_cast<size_t>(global_rows + 1));
        checkCudaError(
            cudaMemsetAsync(global_tile_scan_input.data(), 0, static_cast<size_t>(global_rows + 1) * sizeof(ROWTYPE), stream),
            "initialize SpGEMM global tile counts");

        const int count_blocks = (static_cast<int>(global_rows) + threads_per_block - 1) / threads_per_block;
        compute_global_tile_counts_kernel<ROWTYPE, COLTYPE><<<count_blocks, threads_per_block, 0, stream>>>(
            global_begin,
            global_rows,
            symbolic.row_perm.data(),
            symbolic.expanded_row_ptr.data(),
            base,
            static_cast<ROWTYPE>(threads_per_block),
            global_tile_scan_input.data());
        checkCudaError(cudaGetLastError(), "launch SpGEMM global tile-count kernel");

        void* temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveScan(
            temp_storage,
            temp_storage_bytes,
            global_tile_scan_input.data(),
            global_tile_offsets.data(),
            ::cuda::std::plus<ROWTYPE>(),
            ROWTYPE{0},
            static_cast<int>(global_rows + 1),
            stream);

        DeviceArray<std::uint8_t> scan_storage;
        scan_storage.resize(temp_storage_bytes);
        cub::DeviceScan::ExclusiveScan(
            scan_storage.data(),
            temp_storage_bytes,
            global_tile_scan_input.data(),
            global_tile_offsets.data(),
            ::cuda::std::plus<ROWTYPE>(),
            ROWTYPE{0},
            static_cast<int>(global_rows + 1),
            stream);

        ROWTYPE total_tiles = 0;
        checkCudaError(
            cudaMemcpyAsync(
                &total_tiles,
                global_tile_offsets.data() + global_rows,
                sizeof(ROWTYPE),
                cudaMemcpyDeviceToHost,
                stream),
            "copy SpGEMM global tile count");
        checkCudaError(cudaStreamSynchronize(stream), "synchronize SpGEMM global tile count");
        if (total_tiles > static_cast<ROWTYPE>(std::numeric_limits<int>::max()))
        {
            return false;
        }
        if (total_tiles > 0)
        {
            expand_global_rows_kernel<ROWTYPE, COLTYPE, VALTYPE><<<static_cast<unsigned int>(total_tiles), threads_per_block, 0, stream>>>(
                global_begin,
                global_rows,
                global_tile_offsets.data(),
                symbolic.row_perm.data(),
                d_A_row_ptr,
                d_A_col_ind,
                d_A_values,
                d_B_row_ptr,
                d_B_col_ind,
                d_B_values,
                base,
                symbolic.expanded_row_ptr.data(),
                expanded.col_ind.data(),
                expanded.values.data());
            checkCudaError(cudaGetLastError(), "launch SpGEMM global-row expansion kernel");
        }
    }

    checkCudaError(cudaStreamSynchronize(stream), "synchronize SpGEMM expansion");

    return true;
}

template bool SpGEMMSymbolicAnalyzeCSR<int, int>(
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

template bool SpGEMMSymbolicAnalyzeCSR<std::int64_t, int>(
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

template bool SpGEMMExpandCSR<int, int, float>(
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

template bool SpGEMMExpandCSR<int, int, double>(
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

template bool SpGEMMExpandCSR<std::int64_t, int, float>(
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

template bool SpGEMMExpandCSR<std::int64_t, int, double>(
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
