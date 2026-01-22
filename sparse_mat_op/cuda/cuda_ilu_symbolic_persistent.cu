#include "cuda_ilu_symbolic.cuh"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/scatter.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include <limits>
#include <vector>

namespace cuda_iterative_solver
{

template <typename COLTYPE>
struct Pair
{
    COLTYPE src;
    COLTYPE cur;

    __host__ __device__ bool operator==(const Pair& other) const
    {
        return src == other.src && cur == other.cur;
    }

    __host__ __device__ bool operator<(const Pair& other) const
    {
        return src < other.src || (src == other.src && cur < other.cur);
    }
};

template <typename COLTYPE>
struct PairSrc
{
    __host__ __device__ COLTYPE operator()(const Pair<COLTYPE>& p) const
    {
        return p.src;
    }
};

template <typename COLTYPE>
struct PairCurPlusBase
{
    COLTYPE base;
    __host__ __device__ explicit PairCurPlusBase(COLTYPE base_in) : base(base_in) {}
    __host__ __device__ COLTYPE operator()(const Pair<COLTYPE>& p) const
    {
        return p.cur + base;
    }
};

__device__ int g_ilu_row_counter;

template <typename COLTYPE>
__device__ __forceinline__ unsigned int hash_node(COLTYPE v)
{
    unsigned int x = static_cast<unsigned int>(v);
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

template <typename COLTYPE>
__device__ __forceinline__ bool hash_insert_if_absent(COLTYPE* table, int size, COLTYPE key)
{
    const unsigned int mask = static_cast<unsigned int>(size - 1);
    unsigned int pos = hash_node(key) & mask;
    for (int probe = 0; probe < size; ++probe)
    {
        COLTYPE val = table[pos];
        if (val == key)
        {
            return false;
        }
        if (val == static_cast<COLTYPE>(-1))
        {
            table[pos] = key;
            return true;
        }
        pos = (pos + 1) & mask;
    }
    return false;
}

template <typename COLTYPE>
__device__ __forceinline__ bool hash_contains(const COLTYPE* table, int size, COLTYPE key)
{
    const unsigned int mask = static_cast<unsigned int>(size - 1);
    unsigned int pos = hash_node(key) & mask;
    for (int probe = 0; probe < size; ++probe)
    {
        COLTYPE val = table[pos];
        if (val == key)
        {
            return true;
        }
        if (val == static_cast<COLTYPE>(-1))
        {
            return false;
        }
        pos = (pos + 1) & mask;
    }
    return false;
}

template <typename T>
__device__ __forceinline__ void insertion_sort(T* data, int count)
{
    for (int i = 1; i < count; ++i)
    {
        T key = data[i];
        int j = i - 1;
        while (j >= 0 && data[j] > key)
        {
            data[j + 1] = data[j];
            --j;
        }
        data[j + 1] = key;
    }
}

template <typename T>
__device__ __forceinline__ int unique_in_place(T* data, int count)
{
    if (count == 0)
    {
        return 0;
    }
    int out = 1;
    for (int i = 1; i < count; ++i)
    {
        if (data[i] != data[out - 1])
        {
            data[out++] = data[i];
        }
    }
    return out;
}

template <typename ROWTYPE, typename COLTYPE>
__device__ void ProcessNode(
    COLTYPE row,
    const ROWTYPE* d_ai,
    const COLTYPE* d_aj,
    int lvl,
    COLTYPE base,
    bool keepdiag,
    COLTYPE* frontier,
    COLTYPE* next_frontier,
    COLTYPE* temp,
    int frontier_cap,
    int temp_cap,
    COLTYPE* hash,
    int hash_cap,
    Pair<COLTYPE>* out_pairs,
    unsigned long long* out_count,
    unsigned long long out_cap,
    int lane,
    int* overflow_flag)
{
    constexpr int warp_size = 32;
    const unsigned int mask = 0xffffffffu;

    int overflow = 0;
    if (lane == 0)
    {
        overflow = *overflow_flag;
    }
    overflow = __shfl_sync(mask, overflow, 0);
    if (overflow != 0)
    {
        return;
    }

    for (int idx = lane; idx < hash_cap; idx += warp_size)
    {
        hash[idx] = static_cast<COLTYPE>(-1);
    }
    __syncwarp(mask);

    int frontier_size = 0;
    int temp_count = 0;
    if (lane == 0)
    {
        frontier[0] = row;
        frontier_size = 1;
        if (!hash_insert_if_absent(hash, hash_cap, row))
        {
            atomicExch(overflow_flag, 1);
        }
        if (keepdiag)
        {
            if (temp_count < temp_cap)
            {
                temp[temp_count++] = row;
            }
            else
            {
                atomicExch(overflow_flag, 1);
            }
        }
    }

    frontier_size = __shfl_sync(mask, frontier_size, 0);
    temp_count = __shfl_sync(mask, temp_count, 0);
    if (lane == 0)
    {
        overflow = *overflow_flag;
    }
    overflow = __shfl_sync(mask, overflow, 0);
    if (overflow != 0)
    {
        return;
    }

    for (int level = 0; level <= lvl; ++level)
    {
        if (frontier_size == 0)
        {
            break;
        }

        int adj_count = 0;
        if (lane == 0)
        {
            for (int f = 0; f < frontier_size; ++f)
            {
                COLTYPE node = frontier[f];
                ROWTYPE row_start = d_ai[node] - base;
                ROWTYPE row_end = d_ai[node + 1] - base;
                for (ROWTYPE jj = row_start; jj < row_end; ++jj)
                {
                    COLTYPE neighbor = d_aj[jj] - base;
                    if (adj_count < frontier_cap)
                    {
                        next_frontier[adj_count++] = neighbor;
                    }
                    else
                    {
                        atomicExch(overflow_flag, 1);
                        break;
                    }
                }
                if (*overflow_flag != 0)
                {
                    break;
                }
            }

            if (*overflow_flag == 0)
            {
                insertion_sort(next_frontier, adj_count);
                int unique_count = unique_in_place(next_frontier, adj_count);

                int new_frontier_size = 0;
                for (int idx = 0; idx < unique_count; ++idx)
                {
                    COLTYPE v = next_frontier[idx];
                    if (!hash_contains(hash, hash_cap, v))
                    {
                        if (!hash_insert_if_absent(hash, hash_cap, v))
                        {
                            atomicExch(overflow_flag, 1);
                            break;
                        }
                        if (v < row)
                        {
                            if (new_frontier_size < frontier_cap)
                            {
                                frontier[new_frontier_size++] = v;
                            }
                            else
                            {
                                atomicExch(overflow_flag, 1);
                                break;
                            }
                        }
                        else
                        {
                            if (temp_count < temp_cap)
                            {
                                temp[temp_count++] = v;
                            }
                            else
                            {
                                atomicExch(overflow_flag, 1);
                                break;
                            }
                        }
                    }
                }
                frontier_size = new_frontier_size;
            }
        }

        frontier_size = __shfl_sync(mask, frontier_size, 0);
        temp_count = __shfl_sync(mask, temp_count, 0);
        if (lane == 0)
        {
            overflow = *overflow_flag;
        }
        overflow = __shfl_sync(mask, overflow, 0);
        if (overflow != 0)
        {
            return;
        }
    }

    unsigned long long base_offset = 0;
    if (lane == 0)
    {
        base_offset = atomicAdd(out_count, static_cast<unsigned long long>(temp_count));
        if (base_offset + static_cast<unsigned long long>(temp_count) > out_cap)
        {
            atomicExch(overflow_flag, 1);
        }
    }
    base_offset = __shfl_sync(mask, base_offset, 0);
    if (lane == 0)
    {
        overflow = *overflow_flag;
    }
    overflow = __shfl_sync(mask, overflow, 0);
    if (overflow != 0)
    {
        return;
    }

    for (int idx = lane; idx < temp_count; idx += warp_size)
    {
        out_pairs[base_offset + idx].src = row;
        out_pairs[base_offset + idx].cur = temp[idx];
    }
}

template <typename ROWTYPE, typename COLTYPE>
__global__ void ilu_symbolic_u_persistent_kernel(
    COLTYPE n,
    const ROWTYPE* d_ai,
    const COLTYPE* d_aj,
    int lvl,
    COLTYPE base,
    bool keepdiag,
    COLTYPE* frontier,
    COLTYPE* next_frontier,
    COLTYPE* temp,
    COLTYPE* hash,
    int frontier_cap,
    int temp_cap,
    int hash_cap,
    Pair<COLTYPE>* out_pairs,
    unsigned long long out_cap,
    unsigned long long* out_count,
    int* overflow_flag,
    int total_warps)
{
    constexpr int warp_size = 32;
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = global_thread / warp_size;
    const int lane = threadIdx.x & (warp_size - 1);

    if (warp_id >= total_warps)
    {
        return;
    }

    COLTYPE* warp_frontier = frontier + static_cast<size_t>(warp_id) * frontier_cap;
    COLTYPE* warp_next_frontier = next_frontier + static_cast<size_t>(warp_id) * frontier_cap;
    COLTYPE* warp_temp = temp + static_cast<size_t>(warp_id) * temp_cap;
    COLTYPE* warp_hash = hash + static_cast<size_t>(warp_id) * hash_cap;

    while (true)
    {
        COLTYPE row = 0;
        if (lane == 0)
        {
            int next = atomicAdd(&g_ilu_row_counter, 1);
            row = static_cast<COLTYPE>(next);
        }
        row = __shfl_sync(0xffffffffu, row, 0);
        if (row >= n)
        {
            return;
        }

        ProcessNode(
            row,
            d_ai,
            d_aj,
            lvl,
            base,
            keepdiag,
            warp_frontier,
            warp_next_frontier,
            warp_temp,
            frontier_cap,
            temp_cap,
            warp_hash,
            hash_cap,
            out_pairs,
            out_count,
            out_cap,
            lane,
            overflow_flag);

        int overflow = 0;
        if (lane == 0)
        {
            overflow = *overflow_flag;
        }
        overflow = __shfl_sync(0xffffffffu, overflow, 0);
        if (overflow != 0)
        {
            return;
        }
    }
}

static int next_pow2(int v)
{
    int p = 1;
    while (p < v)
    {
        p <<= 1;
    }
    return p;
}

template <typename ROWTYPE, typename COLTYPE>
bool ILUSymbolicU_CUDA_Persistent(
    COLTYPE n,
    const ROWTYPE* d_ai,
    const COLTYPE* d_aj,
    int lvl,
    COLTYPE base,
    bool keepdiag,
    ROWTYPE* d_u_ai,
    COLTYPE** d_u_aj,
    ROWTYPE* u_nnz)
{
    if (n <= 0 || lvl < 0)
    {
        return false;
    }

    ROWTYPE nnz = 0;
    cudaMemcpy(&nnz, &d_ai[n], sizeof(ROWTYPE), cudaMemcpyDeviceToHost);
    nnz -= base;

    std::vector<ROWTYPE> h_ai(static_cast<size_t>(n) + 1);
    cudaMemcpy(h_ai.data(), d_ai, (static_cast<size_t>(n) + 1) * sizeof(ROWTYPE),
               cudaMemcpyDeviceToHost);

    ROWTYPE max_degree = 0;
    for (COLTYPE i = 0; i < n; ++i)
    {
        ROWTYPE deg = h_ai[static_cast<size_t>(i) + 1] - h_ai[static_cast<size_t>(i)];
        if (deg > max_degree)
        {
            max_degree = deg;
        }
    }

    constexpr int threads_per_block = 256;
    constexpr int warp_size = 32;
    const int warps_per_block = threads_per_block / warp_size;

    cudaDeviceProp prop{};
    int device = 0;
    cudaGetDevice(&device);
    int max_warps = static_cast<int>(n);
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess)
    {
        const int warps_per_sm = 4;
        max_warps = prop.multiProcessorCount * warps_per_sm;
    }
    const int total_warps = std::max(1, std::min(static_cast<int>(n), max_warps));
    const int num_blocks = (total_warps + warps_per_block - 1) / warps_per_block;

    const int avg_degree = (n > 0) ? static_cast<int>(nnz / n) : 0;
    const int base_degree = std::max(1, std::max(static_cast<int>(max_degree), avg_degree));

    int frontier_per_warp = std::max(64, std::min(1024, base_degree * 2));
    int temp_per_warp = std::max(64, std::min(4096, base_degree * (lvl + 1) + 1));
    int hash_per_warp = next_pow2(frontier_per_warp * 1024);
    if (hash_per_warp < 32)
    {
        hash_per_warp = 32;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (static_cast<size_t>(total_warps) > max_size / static_cast<size_t>(frontier_per_warp) ||
        static_cast<size_t>(total_warps) > max_size / static_cast<size_t>(temp_per_warp) ||
        static_cast<size_t>(total_warps) > max_size / static_cast<size_t>(hash_per_warp))
    {
        return false;
    }

    const size_t total_frontier = static_cast<size_t>(total_warps) * frontier_per_warp;
    const size_t total_temp = static_cast<size_t>(total_warps) * temp_per_warp;
    const size_t total_hash = static_cast<size_t>(total_warps) * hash_per_warp;

    COLTYPE* d_frontier = nullptr;
    COLTYPE* d_next_frontier = nullptr;
    COLTYPE* d_temp = nullptr;
    COLTYPE* d_hash = nullptr;
    Pair<COLTYPE>* d_pairs = nullptr;
    unsigned long long* d_pair_count = nullptr;
    int* d_overflow = nullptr;

    if (static_cast<size_t>(n) > std::numeric_limits<size_t>::max() / static_cast<size_t>(temp_per_warp))
    {
        return false;
    }
    const size_t max_pairs = static_cast<size_t>(n) * static_cast<size_t>(temp_per_warp);
    const unsigned long long out_cap = static_cast<unsigned long long>(max_pairs);

    if (cudaMalloc(&d_frontier, total_frontier * sizeof(COLTYPE)) != cudaSuccess ||
        cudaMalloc(&d_next_frontier, total_frontier * sizeof(COLTYPE)) != cudaSuccess ||
        cudaMalloc(&d_temp, total_temp * sizeof(COLTYPE)) != cudaSuccess ||
        cudaMalloc(&d_hash, total_hash * sizeof(COLTYPE)) != cudaSuccess ||
        cudaMalloc(&d_pairs, max_pairs * sizeof(Pair<COLTYPE>)) != cudaSuccess ||
        cudaMalloc(&d_pair_count, sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_overflow, sizeof(int)) != cudaSuccess)
    {
        if (d_frontier)
            cudaFree(d_frontier);
        if (d_next_frontier)
            cudaFree(d_next_frontier);
        if (d_temp)
            cudaFree(d_temp);
        if (d_hash)
            cudaFree(d_hash);
        if (d_pairs)
            cudaFree(d_pairs);
        if (d_pair_count)
            cudaFree(d_pair_count);
        if (d_overflow)
            cudaFree(d_overflow);
        return false;
    }

    unsigned long long zero_u64 = 0;
    int zero_i32 = 0;
    cudaMemcpy(d_pair_count, &zero_u64, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_overflow, &zero_i32, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(g_ilu_row_counter, &zero_i32, sizeof(int));

    ilu_symbolic_u_persistent_kernel<<<num_blocks, threads_per_block>>>(
        n,
        d_ai,
        d_aj,
        lvl,
        base,
        keepdiag,
        d_frontier,
        d_next_frontier,
        d_temp,
        d_hash,
        frontier_per_warp,
        temp_per_warp,
        hash_per_warp,
        d_pairs,
        out_cap,
        d_pair_count,
        d_overflow,
        total_warps);

    int overflow = 0;
    cudaMemcpy(&overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost);
    if (overflow != 0)
    {
        cudaFree(d_frontier);
        cudaFree(d_next_frontier);
        cudaFree(d_temp);
        cudaFree(d_hash);
        cudaFree(d_pairs);
        cudaFree(d_pair_count);
        cudaFree(d_overflow);
        return false;
    }

    unsigned long long pair_count_u64 = 0;
    cudaMemcpy(&pair_count_u64, d_pair_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    if (pair_count_u64 > static_cast<unsigned long long>(std::numeric_limits<size_t>::max()))
    {
        cudaFree(d_frontier);
        cudaFree(d_next_frontier);
        cudaFree(d_temp);
        cudaFree(d_hash);
        cudaFree(d_pairs);
        cudaFree(d_pair_count);
        cudaFree(d_overflow);
        return false;
    }

    size_t pair_count = static_cast<size_t>(pair_count_u64);
    thrust::device_ptr<Pair<COLTYPE>> pairs_begin(d_pairs);
    thrust::device_ptr<Pair<COLTYPE>> pairs_end = pairs_begin + pair_count;

    if (pair_count == 0)
    {
        thrust::device_vector<ROWTYPE> u_ai_dev(static_cast<size_t>(n) + 1, ROWTYPE(base));
        cudaMemcpy(d_u_ai, thrust::raw_pointer_cast(u_ai_dev.data()),
                   (static_cast<size_t>(n) + 1) * sizeof(ROWTYPE), cudaMemcpyDeviceToDevice);
        *u_nnz = 0;
        *d_u_aj = nullptr;
        cudaFree(d_frontier);
        cudaFree(d_next_frontier);
        cudaFree(d_temp);
        cudaFree(d_hash);
        cudaFree(d_pairs);
        cudaFree(d_pair_count);
        cudaFree(d_overflow);
        return true;
    }

    thrust::sort(pairs_begin, pairs_end);
    auto unique_end = thrust::unique(pairs_begin, pairs_end);
    pair_count = static_cast<size_t>(unique_end - pairs_begin);

    thrust::device_vector<ROWTYPE> u_ai_dev(static_cast<size_t>(n) + 1, ROWTYPE(0));
    thrust::device_vector<ROWTYPE> row_counts(n, ROWTYPE(0));

    auto row_it = thrust::make_transform_iterator(pairs_begin, PairSrc<COLTYPE>());
    auto row_it_end = thrust::make_transform_iterator(pairs_begin + pair_count, PairSrc<COLTYPE>());
    thrust::device_vector<COLTYPE> unique_rows(pair_count);
    thrust::device_vector<ROWTYPE> unique_counts(pair_count);

    auto reduce_end = thrust::reduce_by_key(
        row_it, row_it_end,
        thrust::make_constant_iterator<ROWTYPE>(1),
        unique_rows.begin(), unique_counts.begin());
    size_t unique_size = static_cast<size_t>(reduce_end.first - unique_rows.begin());
    unique_rows.resize(unique_size);
    unique_counts.resize(unique_size);

    thrust::scatter(unique_counts.begin(), unique_counts.end(), unique_rows.begin(), row_counts.begin());
    thrust::inclusive_scan(row_counts.begin(), row_counts.end(), u_ai_dev.begin() + 1);

    if (base != 0)
    {
        thrust::transform(u_ai_dev.begin() + 1, u_ai_dev.end(),
                          thrust::make_constant_iterator(ROWTYPE(base)),
                          u_ai_dev.begin() + 1, thrust::plus<ROWTYPE>());
    }
    u_ai_dev[0] = ROWTYPE(base);

    cudaMemcpy(d_u_ai, thrust::raw_pointer_cast(u_ai_dev.data()),
               (static_cast<size_t>(n) + 1) * sizeof(ROWTYPE), cudaMemcpyDeviceToDevice);

    if (pair_count > static_cast<size_t>(std::numeric_limits<ROWTYPE>::max()))
    {
        cudaFree(d_frontier);
        cudaFree(d_next_frontier);
        cudaFree(d_temp);
        cudaFree(d_hash);
        cudaFree(d_pairs);
        cudaFree(d_pair_count);
        cudaFree(d_overflow);
        return false;
    }

    *u_nnz = static_cast<ROWTYPE>(pair_count);
    if (*u_nnz > 0)
    {
        cudaMalloc(d_u_aj, static_cast<size_t>(*u_nnz) * sizeof(COLTYPE));
        auto u_aj_ptr = thrust::device_pointer_cast(*d_u_aj);
        thrust::transform(pairs_begin, pairs_begin + pair_count, u_aj_ptr,
                          PairCurPlusBase<COLTYPE>(base));
    }
    else
    {
        *d_u_aj = nullptr;
    }

    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_temp);
    cudaFree(d_hash);
    cudaFree(d_pairs);
    cudaFree(d_pair_count);
    cudaFree(d_overflow);

    return true;
}

template bool ILUSymbolicU_CUDA_Persistent<int, int>(
    int n, const int* d_ai, const int* d_aj, int lvl, int base, bool keepdiag,
    int* d_u_ai, int** d_u_aj, int* u_nnz);

template bool ILUSymbolicU_CUDA_Persistent<int64_t, int>(
    int n, const int64_t* d_ai, const int* d_aj, int lvl, int base, bool keepdiag,
    int64_t* d_u_ai, int** d_u_aj, int64_t* u_nnz);

} // namespace cuda_iterative_solver
