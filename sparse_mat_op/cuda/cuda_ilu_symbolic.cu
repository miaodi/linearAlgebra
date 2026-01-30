#include "cuda_ilu_symbolic.cuh"
#include "cuda_memory.cuh"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/scatter.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>
#include <cub/cub.cuh>
#include <cuda/std/iterator>
#include <cuco/pair.cuh>
#include <cuco/static_set.cuh>
#include <cuco/static_set_ref.cuh>
#include <cstdint>
#include <iostream>
#include <limits>

namespace cuda_iterative_solver
{

// Pair structure for BFS frontier: (source_node, current_node)
template <typename COLTYPE>
using Pair = cuco::pair<COLTYPE, COLTYPE>;

template <typename COLTYPE>
struct PairLess
{
    __host__ __device__ bool operator()(const Pair<COLTYPE>& a, const Pair<COLTYPE>& b) const
    {
        return (a.first < b.first) || (a.first == b.first && a.second < b.second);
    }
};

template <typename COLTYPE>
struct PairSrc
{
    __host__ __device__ COLTYPE operator()(const Pair<COLTYPE>& p) const
    {
        return p.first;
    }
};

template <typename COLTYPE>
struct PairCurPlusBase
{
    COLTYPE base;
    __host__ __device__ explicit PairCurPlusBase(COLTYPE base_in) : base(base_in) {}
    __host__ __device__ COLTYPE operator()(const Pair<COLTYPE>& p) const
    {
        return p.second + base;
    }
};

// Kernel: Initialize frontier with (i, i) for all i
template <typename COLTYPE>
__global__ void init_frontier_kernel(Pair<COLTYPE>* frontier, COLTYPE n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        frontier[idx].first = idx;
        frontier[idx].second = idx;
    }
}

// Kernel: Compute degree for all nodes (pre-computation)
template <typename ROWTYPE, typename COLTYPE>
__global__ void compute_all_degrees_kernel(
    const ROWTYPE* d_ai,
    COLTYPE n,
    COLTYPE base,
    int* degrees)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    
    ROWTYPE row_start = d_ai[idx] - base;
    ROWTYPE row_end = d_ai[idx + 1] - base;
    degrees[idx] = row_end - row_start;
}

// Kernel: Look up degrees for each node in current frontier
template <typename COLTYPE>
__global__ void lookup_frontier_degrees_kernel(
    const Pair<COLTYPE>* current_frontier,
    int current_size,
    const int* all_degrees,
    int* frontier_degrees)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= current_size)
        return;
    
    Pair<COLTYPE> p = current_frontier[idx];
    COLTYPE j = p.second;
    frontier_degrees[idx] = all_degrees[j];
}

// Kernel: Try to insert each element into the set and record success
// Returns true if the element was newly inserted (not a duplicate or already visited)
template <typename COLTYPE, typename Ref>
__global__ void insert_and_check_kernel(
    const Pair<COLTYPE>* pairs,
    int n,
    Ref ref,
    bool* inserted)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    
    // insert() returns true if the element was inserted (new), false if already present
    inserted[idx] = ref.insert(pairs[idx]);
}



// Kernel: Build next frontier using inclusive scan (Step 2)
// Each thread handles one element in the next frontier
template <typename ROWTYPE, typename COLTYPE>
__global__ void build_next_frontier_kernel(
    const Pair<COLTYPE>* current_frontier,
    int current_size,
    const int* inclusive_sum,
    const ROWTYPE* d_ai,
    const COLTYPE* d_aj,
    COLTYPE base,
    int next_frontier_size,
    Pair<COLTYPE>* next_frontier)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= next_frontier_size)
        return;
    
    // Use thrust::upper_bound to find which frontier node this output belongs to
    const int* upper = thrust::upper_bound(thrust::seq, inclusive_sum, inclusive_sum + current_size, idx);
    int frontier_idx = upper - inclusive_sum;
    
    // Local index within this frontier node's neighbors
    // For inclusive scan: need to subtract the previous cumulative sum
    int prev_sum = (frontier_idx > 0) ? inclusive_sum[frontier_idx - 1] : 0;
    int local_idx = idx - prev_sum;
    
    Pair<COLTYPE> p = current_frontier[frontier_idx];
    COLTYPE i = p.first;
    COLTYPE j = p.second;
    
    // Get the neighbor at local_idx
    ROWTYPE row_start = d_ai[j] - base;
    COLTYPE neighbor = d_aj[row_start + local_idx] - base;
    
    // Write to next frontier
    next_frontier[idx].first = i;
    next_frontier[idx].second = neighbor;
}

// Functor: Filter pairs where i < j (for U pattern without diagonal)
template <typename COLTYPE>
struct FilterU_lt {
    __device__ bool operator()(const Pair<COLTYPE>& p) const {
        return p.first < p.second;
    }
};

// Functor: Filter pairs where i <= j (for U pattern with diagonal)
template <typename COLTYPE>
struct FilterU_lte {
    __device__ bool operator()(const Pair<COLTYPE>& p) const {
        return p.first <= p.second;
    }
};

// Functor: Filter pairs where i > j (for next frontier)
template <typename COLTYPE>
struct FilterNext {
    __device__ bool operator()(const Pair<COLTYPE>& p) const {
        return p.first > p.second;
    }
};

// Kernel: Extract U pattern by atomically appending (i,j) where i <= j
template <typename ROWTYPE, typename COLTYPE>
__global__ void extract_u_pattern_kernel(
    const Pair<COLTYPE>* pairs,
    int n,
    const bool* mask,
    COLTYPE base,
    bool keepdiag,
    ROWTYPE* d_row_counts,  // atomic counters for each row
    COLTYPE* d_u_cols_temp) // temporary storage for columns
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || !mask[idx])
        return;
    
    Pair<COLTYPE> p = pairs[idx];
    COLTYPE i = p.first;
    COLTYPE j = p.second;
    
    // Keep only i <= j
    if (i > j)
        return;
    
    // Skip diagonal if not keeping it
    if (i == j && !keepdiag)
        return;
    
    // Atomically increment row counter and get position
    ROWTYPE pos = atomicAdd(&d_row_counts[i], ROWTYPE(1));
    
    // Store column index (will be sorted later per row)
    // Note: This is a simplified version; real implementation needs
    // proper global indexing based on prefix sum of row_counts
}

// Main function implementing ILU(k) U-row symbolic factorization
template <typename ROWTYPE, typename COLTYPE>
bool ILUSymbolicU_CUDA(
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
        return false;
    
    // Get NNZ from input matrix
    ROWTYPE nnz;
    cudaMemcpy(&nnz, &d_ai[n], sizeof(ROWTYPE), cudaMemcpyDeviceToHost);
    nnz -= base;
    
    // Allocate device memory for frontiers
    thrust::device_vector<Pair<COLTYPE>> current_frontier(n);
    thrust::device_vector<Pair<COLTYPE>> next_frontier;
    
    // Initialize frontier with (i, i) for all i
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    init_frontier_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(current_frontier.data()), n);
    cudaDeviceSynchronize();

    // cuco::static_set for visited tracking
    size_t visited_capacity = static_cast<size_t>(n) * 1024u * 32u;
    if (visited_capacity > static_cast<size_t>(1e9))
        visited_capacity = static_cast<size_t>(1e9); // Cap to avoid excessive memory
    if (visited_capacity == 0)
        visited_capacity = 1;

    // Use Pair directly as key - it's bitwise comparable
    using visited_key_t = Pair<COLTYPE>;
    constexpr visited_key_t empty_key{std::numeric_limits<COLTYPE>::max(), std::numeric_limits<COLTYPE>::max()};
    // Use linear_probing with cg_size=1 for single-thread insert operations
    using probing_scheme_t = cuco::linear_probing<1, cuco::default_hash_function<visited_key_t>>;
    using visited_set_t = cuco::static_set<visited_key_t, cuco::extent<std::size_t>, cuda::thread_scope_device,
                                           cuda::std::equal_to<visited_key_t>, probing_scheme_t>;
    visited_set_t visited_set(visited_capacity, cuco::empty_key<visited_key_t>{empty_key});

    // Mark initial frontier (i, i) as visited using host-side bulk insert
    visited_set.insert(current_frontier.begin(), current_frontier.end());

    // Storage for U pattern accumulation
    thrust::device_vector<Pair<COLTYPE>> u_pattern;

    // Add initial diagonal pairs to U pattern if keeping diagonal
    if (keepdiag)
    {
        u_pattern.insert(u_pattern.end(), current_frontier.begin(), current_frontier.end());
    }

    // Pre-compute degrees for all nodes before BFS
    thrust::device_vector<int> all_degrees(n);
    blocks = (n + threads - 1) / threads;
    compute_all_degrees_kernel<<<blocks, threads>>>(d_ai, n, base,
                                                    thrust::raw_pointer_cast(all_degrees.data()));
    cudaDeviceSynchronize();

    // BFS for k levels
    for (int level = 0; level <= lvl; ++level)
    {
        int current_size = current_frontier.size();
        if (current_size == 0)
            break;

        // Step 1: Look up degrees for nodes in current frontier
        thrust::device_vector<int> degrees(current_size);
        blocks = (current_size + threads - 1) / threads;
        lookup_frontier_degrees_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(current_frontier.data()), current_size,
            thrust::raw_pointer_cast(all_degrees.data()), thrust::raw_pointer_cast(degrees.data()));
        cudaDeviceSynchronize();

        // Step 2: Compute inclusive scan to get cumulative sums
        thrust::device_vector<int> inclusive_sum(current_size);
        thrust::inclusive_scan(degrees.begin(), degrees.end(), inclusive_sum.begin());

        // Get total size from last element of inclusive scan
        int actual_next_size = 0;
        if (current_size > 0)
        {
            cudaMemcpy(&actual_next_size, thrust::raw_pointer_cast(inclusive_sum.data()) + current_size - 1,
                       sizeof(int), cudaMemcpyDeviceToHost);
        }

        if (actual_next_size == 0)
            break;

        // Allocate next frontier
        next_frontier.resize(actual_next_size);

        // Step 3: Build next frontier using inclusive scan
        blocks = (actual_next_size + threads - 1) / threads;
        build_next_frontier_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(current_frontier.data()), current_size,
            thrust::raw_pointer_cast(inclusive_sum.data()), d_ai, d_aj, base, actual_next_size,
            thrust::raw_pointer_cast(next_frontier.data()));
        cudaDeviceSynchronize();

        // Insert all elements into visited_set and record which insertions succeeded
        // This replaces: sort + unique + contains + filter + insert
        // Elements that are duplicates (within next_frontier) or already visited will fail to insert
        thrust::device_vector<bool> is_new(actual_next_size);
        auto insert_ref = visited_set.ref(cuco::op::insert);
        blocks = (actual_next_size + threads - 1) / threads;
        insert_and_check_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(next_frontier.data()),
            actual_next_size,
            insert_ref,
            thrust::raw_pointer_cast(is_new.data()));
        cudaDeviceSynchronize();

        // Filter to get only newly inserted (unvisited, unique) elements
        thrust::device_vector<Pair<COLTYPE>> unvisited(actual_next_size);
        auto new_end_unvisited =
            thrust::copy_if(next_frontier.begin(), next_frontier.end(), is_new.begin(),
                            unvisited.begin(), [] __device__(bool b) { return b; });
        unvisited.resize(new_end_unvisited - unvisited.begin());

        int unvisited_size = unvisited.size();
        if (unvisited_size == 0)
            break;

        // Extract pairs where i <= j to U pattern (upper triangular)
        thrust::device_vector<Pair<COLTYPE>> u_pairs(unvisited_size);
        auto new_end_u = keepdiag ? thrust::copy_if(unvisited.begin(), unvisited.end(),
                                                    u_pairs.begin(), FilterU_lte<COLTYPE>())
                                  : thrust::copy_if(unvisited.begin(), unvisited.end(),
                                                    u_pairs.begin(), FilterU_lt<COLTYPE>());
        u_pairs.resize(cuda::std::distance(u_pairs.begin(), new_end_u));

        // Append to global U pattern
        u_pattern.insert(u_pattern.end(), u_pairs.begin(), u_pairs.end());

        // Keep only i > j for next iteration
        thrust::device_vector<Pair<COLTYPE>> next_frontier_filtered(unvisited_size);
        auto new_end_next = thrust::copy_if(unvisited.begin(), unvisited.end(),
                                            next_frontier_filtered.begin(), FilterNext<COLTYPE>());
        next_frontier_filtered.resize(cuda::std::distance(next_frontier_filtered.begin(), new_end_next));

        current_frontier = next_frontier_filtered;
    }

    // Now build CSR structure from collected pairs
    // Sort by row then column
    thrust::sort(u_pattern.begin(), u_pattern.end(), PairLess<COLTYPE>());

    // Remove any duplicates that might have accumulated
    auto new_end = thrust::unique(u_pattern.begin(), u_pattern.end());
    u_pattern.erase(new_end, u_pattern.end());

    // Build row pointers
    thrust::device_vector<ROWTYPE> u_ai_dev(n + 1, ROWTYPE(0));
    thrust::device_vector<ROWTYPE> row_counts(n, ROWTYPE(0));

    auto row_it = thrust::make_transform_iterator(u_pattern.begin(), PairSrc<COLTYPE>());
    auto row_it_end = thrust::make_transform_iterator(u_pattern.end(), PairSrc<COLTYPE>());
    thrust::device_vector<COLTYPE> unique_rows(u_pattern.size());
    thrust::device_vector<ROWTYPE> unique_counts(u_pattern.size());

    auto reduce_end = thrust::reduce_by_key(row_it, row_it_end, thrust::make_constant_iterator<ROWTYPE>(1),
                                            unique_rows.begin(), unique_counts.begin());
    size_t unique_size = reduce_end.first - unique_rows.begin();
    unique_rows.resize(unique_size);
    unique_counts.resize(unique_size);

    thrust::scatter(unique_counts.begin(), unique_counts.end(), unique_rows.begin(), row_counts.begin());

    thrust::inclusive_scan(row_counts.begin(), row_counts.end(), u_ai_dev.begin() + 1);
    if (base != 0)
    {
        thrust::transform(u_ai_dev.begin() + 1, u_ai_dev.end(), thrust::make_constant_iterator(ROWTYPE(base)),
                          u_ai_dev.begin() + 1, thrust::plus<ROWTYPE>());
    }
    u_ai_dev[0] = ROWTYPE(base);

    cudaMemcpy(d_u_ai, thrust::raw_pointer_cast(u_ai_dev.data()), (n + 1) * sizeof(ROWTYPE), cudaMemcpyDeviceToDevice);

    // Build column indices
    *u_nnz = u_pattern.size();
    cudaMalloc(d_u_aj, *u_nnz * sizeof(COLTYPE));

    auto u_aj_ptr = thrust::device_pointer_cast(*d_u_aj);
    thrust::transform(u_pattern.begin(), u_pattern.end(), u_aj_ptr, PairCurPlusBase<COLTYPE>(base));

    return true;
}

// Explicit template instantiations
template bool ILUSymbolicU_CUDA<int, int>(
    int n, const int* d_ai, const int* d_aj, int lvl, int base, bool keepdiag,
    int* d_u_ai, int** d_u_aj, int* u_nnz);

template bool ILUSymbolicU_CUDA<int64_t, int>(
    int n, const int64_t* d_ai, const int* d_aj, int lvl, int base, bool keepdiag,
    int64_t* d_u_ai, int** d_u_aj, int64_t* u_nnz);

} // namespace cuda_iterative_solver
