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
#include <cuco/static_map.cuh>
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

//==============================================================================
// ILUSymbolic_CUDA: Full L+U symbolic factorization using (src, cur, max_cur)
//==============================================================================

// Triplet type for full ILU BFS: ((src, cur), max_cur)
// Using cuco::pair for compatibility with cuco hash maps
template <typename COLTYPE>
using Triplet = cuco::pair<Pair<COLTYPE>, COLTYPE>;

// Helper functions to access Triplet fields
template <typename COLTYPE>
__host__ __device__ inline COLTYPE triplet_src(const Triplet<COLTYPE>& t) { return t.first.first; }

template <typename COLTYPE>
__host__ __device__ inline COLTYPE triplet_cur(const Triplet<COLTYPE>& t) { return t.first.second; }

template <typename COLTYPE>
__host__ __device__ inline COLTYPE triplet_path_max(const Triplet<COLTYPE>& t) { return t.second; }

template <typename COLTYPE>
__host__ __device__ inline Triplet<COLTYPE> make_triplet(COLTYPE src, COLTYPE cur, COLTYPE path_max)
{
    return Triplet<COLTYPE>{Pair<COLTYPE>{src, cur}, path_max};
}

template <typename COLTYPE>
struct TripletLess
{
    __host__ __device__ bool operator()(const Triplet<COLTYPE>& a, const Triplet<COLTYPE>& b) const
    {
        if (triplet_src(a) != triplet_src(b)) return triplet_src(a) < triplet_src(b);
        return triplet_cur(a) < triplet_cur(b);
    }
};

template <typename COLTYPE>
struct TripletSrc
{
    __host__ __device__ COLTYPE operator()(const Triplet<COLTYPE>& t) const
    {
        return triplet_src(t);
    }
};

template <typename COLTYPE>
struct TripletKey
{
    __host__ __device__ Pair<COLTYPE> operator()(const Triplet<COLTYPE>& t) const
    {
        return t.first;
    }
};

template <typename COLTYPE>
struct TripletToPair
{
    COLTYPE base;
    __host__ __device__ explicit TripletToPair(COLTYPE base_in) : base(base_in) {}
    __host__ __device__ Pair<COLTYPE> operator()(const Triplet<COLTYPE>& t) const
    {
        return Pair<COLTYPE>{triplet_src(t), triplet_cur(t) + base};
    }
};

struct Count2
{
    int keep;
    int fill;
};

__host__ __device__ inline Count2 operator+(const Count2& a, const Count2& b)
{
    return Count2{a.keep + b.keep, a.fill + b.fill};
}

struct Count2Plus
{
    __host__ __device__ Count2 operator()(const Count2& a, const Count2& b) const
    {
        return a + b;
    }
};

// Kernel: Initialize frontier with ((i, i), 0) for all i
template <typename COLTYPE>
__global__ void init_triplet_frontier_kernel(Triplet<COLTYPE>* frontier, COLTYPE n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        frontier[idx] = make_triplet<COLTYPE>(idx, idx, 0);  // Initial path_max is 0 (no intermediate nodes yet)
    }
}

// Kernel: Look up degrees for each node in triplet frontier
template <typename COLTYPE>
__global__ void lookup_triplet_frontier_degrees_kernel(
    const Triplet<COLTYPE>* current_frontier,
    int current_size,
    const int* all_degrees,
    int* frontier_degrees)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= current_size)
        return;
    
    COLTYPE j = triplet_cur(current_frontier[idx]);
    frontier_degrees[idx] = all_degrees[j];
}

// Kernel: Build next triplet frontier using inclusive scan
template <typename ROWTYPE, typename COLTYPE>
__global__ void build_next_triplet_frontier_kernel(
    const Triplet<COLTYPE>* current_frontier,
    int current_size,
    const int* inclusive_sum,
    const ROWTYPE* d_ai,
    const COLTYPE* d_aj,
    COLTYPE base,
    int next_frontier_size,
    Triplet<COLTYPE>* next_frontier)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= next_frontier_size)
        return;
    
    // Find which frontier node this output belongs to
    const int* upper = thrust::upper_bound(thrust::seq, inclusive_sum, inclusive_sum + current_size, idx);
    int frontier_idx = upper - inclusive_sum;
    
    int prev_sum = (frontier_idx > 0) ? inclusive_sum[frontier_idx - 1] : 0;
    int local_idx = idx - prev_sum;
    
    Triplet<COLTYPE> t = current_frontier[frontier_idx];
    COLTYPE src = triplet_src(t);
    COLTYPE cur = triplet_cur(t);
    COLTYPE path_max = triplet_path_max(t);
    
    // Get the neighbor at local_idx
    ROWTYPE row_start = d_ai[cur] - base;
    COLTYPE neighbor = d_aj[row_start + local_idx] - base;
    
    // Write to next frontier: ((src, neighbor), max(path_max, neighbor))
    COLTYPE new_path_max = (neighbor > path_max) ? neighbor : path_max;
    next_frontier[idx] = make_triplet(src, neighbor, new_path_max);
}

// Kernel: Try to insert/update in map and determine which elements to keep
template <typename COLTYPE, typename MapRef>
__global__ void check_and_insert_triplet_kernel(
    const Triplet<COLTYPE>* triplets,
    int n,
    MapRef map_ref,
    unsigned char* flags) // bit 0: keep, bit 1: add_to_fill
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    
    const Triplet<COLTYPE>& triplet = triplets[idx];
    COLTYPE src = triplet_src(triplet);
    COLTYPE cur = triplet_cur(triplet);
    COLTYPE path_max = triplet_path_max(triplet);
    
    // Early exit: cur >= src means we won't keep this for next frontier
    // and it can't be a valid fill entry (fill requires cur > path_max which implies cur contributes)
    bool dominated = (cur >= src);
    
    unsigned char out = 0;
    
    // Try to insert using insert_and_find which returns (iterator, bool)
    auto [slot, is_new_key] = map_ref.insert_and_find(triplet);
    
    // Determine if this is a valid fill entry: cur == path_max means cur is the largest
    // node on the path from src, making (src, cur) a fill-in entry
    bool is_fill = (cur == path_max);
    
    if (is_new_key)
    {
        // New key: add to fill if condition met, keep if cur < src
        if (is_fill) out |= 2u;
        if (!dominated) out |= 1u;
    }
    else if (!dominated)
    {
        const COLTYPE stored_max = slot->second;
        if ( path_max < stored_max )
        {
            // Key exists and cur < src: try to improve stored path_max
            if (path_max < atomicMin(const_cast<COLTYPE*>(&(slot->second)), path_max))
            {
                out |= 1u;
                // If we improved and cur == path_max, this becomes a fill entry
                if (is_fill) out |= 2u;
            }
        }
    }
    flags[idx] = out;
}

// Functor: Filter triplets where cur != src (skip diagonal for next frontier)
template <typename COLTYPE>
struct FilterTripletNonDiag {
    __device__ bool operator()(const Triplet<COLTYPE>& t) const {
        return triplet_cur(t) != triplet_src(t);
    }
};

template <typename COLTYPE, int BLOCK>
__global__ void scatter_keep_fill_kernel(
    const Triplet<COLTYPE>* next_frontier,
    const unsigned char* flags,
    int n,
    int* counts,
    Triplet<COLTYPE>* keep_out,
    Pair<COLTYPE>* fill_out)
{
    using BlockScan = cub::BlockScan<Count2, BLOCK>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    __shared__ Count2 block_base;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned char f = 0;
    if (idx < n)
        f = flags[idx];
    Count2 item{(f & 1u) ? 1 : 0, (f & 2u) ? 1 : 0};
    Count2 prefix{};
    Count2 block_total{};

    BlockScan(temp_storage).ExclusiveSum(item, prefix, block_total);
    if (threadIdx.x == 0)
    {
        block_base.keep = atomicAdd(&counts[0], block_total.keep);
        block_base.fill = atomicAdd(&counts[1], block_total.fill);
    }
    __syncthreads();

    if (idx < n)
    {
        if (f & 1u)
            keep_out[block_base.keep + prefix.keep] = next_frontier[idx];
        if (f & 2u)
            fill_out[block_base.fill + prefix.fill] = next_frontier[idx].first;
    }
}

template <int BLOCK>
__global__ void count_flags_kernel(const unsigned char* flags, int n, int* counts)
{
    using BlockReduce = cub::BlockReduce<Count2, BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned char f = 0;
    if (idx < n)
        f = flags[idx];
    Count2 item{(f & 1u) ? 1 : 0, (f & 2u) ? 1 : 0};
    Count2 block_sum = BlockReduce(temp_storage).Reduce(item, Count2Plus());
    if (threadIdx.x == 0)
    {
        atomicAdd(&counts[0], block_sum.keep);
        atomicAdd(&counts[1], block_sum.fill);
    }
}

// Main function implementing full ILU(k) symbolic factorization (L + U)
template <typename ROWTYPE, typename COLTYPE>
bool ILUSymbolic_CUDA(
    COLTYPE n,
    const ROWTYPE* d_ai,
    const COLTYPE* d_aj,
    int lvl,
    COLTYPE base,
    ROWTYPE* d_lu_ai,
    COLTYPE** d_lu_aj,
    ROWTYPE* lu_nnz)
{
    if (n <= 0 || lvl < 0)
        return false;
    
    // Get NNZ from input matrix
    ROWTYPE nnz;
    cudaMemcpy(&nnz, &d_ai[n], sizeof(ROWTYPE), cudaMemcpyDeviceToHost);
    nnz -= base;
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // cuco::static_map for visited tracking with path_max values
    size_t map_capacity = static_cast<size_t>(n) * 1024u * 16u;
    if (map_capacity > static_cast<size_t>(1e10))
        map_capacity = static_cast<size_t>(1e10);
    if (map_capacity == 0)
        map_capacity = 1;

    using map_key_t = Pair<COLTYPE>;
    using map_value_t = COLTYPE;
    constexpr map_key_t empty_key{std::numeric_limits<COLTYPE>::max(), std::numeric_limits<COLTYPE>::max()};
    constexpr map_value_t empty_value = std::numeric_limits<COLTYPE>::max();
    
    using probing_scheme_t = cuco::linear_probing<1, cuco::default_hash_function<map_key_t>>;
    using visited_map_t = cuco::static_map<map_key_t, map_value_t, cuco::extent<std::size_t>, 
                                            cuda::thread_scope_device,
                                            cuda::std::equal_to<map_key_t>, probing_scheme_t>;
    visited_map_t visited_map(map_capacity, 
                               cuco::empty_key<map_key_t>{empty_key},
                               cuco::empty_value<map_value_t>{empty_value});

    // Storage for combined LU pattern accumulation
    thrust::device_vector<Pair<COLTYPE>> lu_pattern;
    lu_pattern.reserve(nnz * (lvl + 2));  // Reserve estimated capacity

    // Always include diagonal in pattern
    lu_pattern.resize(n);
    thrust::tabulate(lu_pattern.begin(), lu_pattern.end(),
        [=] __device__ (int i) { return Pair<COLTYPE>{i, i}; });

    // Insert initial diagonal entries into map with path_max = 0
    {
        thrust::device_vector<cuco::pair<map_key_t, map_value_t>> init_pairs(n);
        thrust::tabulate(init_pairs.begin(), init_pairs.end(),
            [=] __device__ (int i) { 
                return cuco::pair<map_key_t, map_value_t>{Pair<COLTYPE>{i, i}, COLTYPE(0)}; 
            });
        visited_map.insert(init_pairs.begin(), init_pairs.end());
    }

    // Pre-compute degrees for all nodes
    thrust::device_vector<int> all_degrees(n);
    blocks = (n + threads - 1) / threads;
    compute_all_degrees_kernel<<<blocks, threads>>>(d_ai, n, base,
                                                    thrust::raw_pointer_cast(all_degrees.data()));

    // Allocate frontier buffers once and reuse
    thrust::device_vector<Triplet<COLTYPE>> current_frontier(n);
    thrust::device_vector<Triplet<COLTYPE>> next_frontier;
    thrust::device_vector<int> degrees;
    thrust::device_vector<int> inclusive_sum;
    thrust::device_vector<unsigned char> flags;
    thrust::device_vector<int> counts(2);
    thrust::device_vector<Triplet<COLTYPE>> temp_triplets;

    // Initialize frontier with ((i, i), 0) for all i
    blocks = (n + threads - 1) / threads;
    init_triplet_frontier_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(current_frontier.data()), n);
    cudaDeviceSynchronize();

    // Timing infrastructure
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float time_degree_lookup = 0.0f, total_degree_lookup = 0.0f;
    float time_scan = 0.0f, total_scan = 0.0f;
    float time_build_frontier = 0.0f, total_build_frontier = 0.0f;
    float time_hash_insert = 0.0f, total_hash_insert = 0.0f;
    float time_extract_fill = 0.0f, total_extract_fill = 0.0f;

    // BFS for k levels
    for (int level = 0; level <= lvl; ++level)
    {
        int current_size = current_frontier.size();
        if (current_size == 0)
            break;

        // Step 1: Look up degrees for nodes in current frontier
        cudaEventRecord(start);
        degrees.resize(current_size);
        blocks = (current_size + threads - 1) / threads;
        lookup_triplet_frontier_degrees_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(current_frontier.data()), current_size,
            thrust::raw_pointer_cast(all_degrees.data()), thrust::raw_pointer_cast(degrees.data()));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_degree_lookup, start, stop);
        total_degree_lookup += time_degree_lookup;

        // Step 2: Compute inclusive scan
        cudaEventRecord(start);
        inclusive_sum.resize(current_size);
        thrust::inclusive_scan(degrees.begin(), degrees.end(), inclusive_sum.begin());

        int actual_next_size = 0;
        if (current_size > 0)
        {
            cudaMemcpy(&actual_next_size, thrust::raw_pointer_cast(inclusive_sum.data()) + current_size - 1,
                       sizeof(int), cudaMemcpyDeviceToHost);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_scan, start, stop);
        total_scan += time_scan;

        if (actual_next_size == 0)
            break;

        // Step 3: Build next frontier
        cudaEventRecord(start);
        next_frontier.resize(actual_next_size);
        blocks = (actual_next_size + threads - 1) / threads;
        build_next_triplet_frontier_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(current_frontier.data()), current_size,
            thrust::raw_pointer_cast(inclusive_sum.data()), d_ai, d_aj, base, actual_next_size,
            thrust::raw_pointer_cast(next_frontier.data()));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_build_frontier, start, stop);
        total_build_frontier += time_build_frontier;

        // Step 4: Check and insert into map, determine which to keep and add to fill
        cudaEventRecord(start);
        flags.resize(actual_next_size);
        
        auto map_ref = visited_map.ref(cuco::op::insert_and_find);
        blocks = (actual_next_size + threads - 1) / threads;
        check_and_insert_triplet_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(next_frontier.data()),
            actual_next_size,
            map_ref,
            thrust::raw_pointer_cast(flags.data()));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_hash_insert, start, stop);
        total_hash_insert += time_hash_insert;

        // Step 5: Scan flags and scatter keep/fill in one pass
        cudaEventRecord(start);
        cudaMemset(thrust::raw_pointer_cast(counts.data()), 0, 2 * sizeof(int));

        blocks = (actual_next_size + threads - 1) / threads;
        count_flags_kernel<256><<<blocks, threads>>>(
            thrust::raw_pointer_cast(flags.data()), actual_next_size,
            thrust::raw_pointer_cast(counts.data()));
        int keep_count = 0;
        int fill_count = 0;
        if (actual_next_size > 0)
        {
            int h_counts[2] = {0, 0};
            cudaMemcpy(h_counts, thrust::raw_pointer_cast(counts.data()), sizeof(h_counts), cudaMemcpyDeviceToHost);
            keep_count = h_counts[0];
            fill_count = h_counts[1];
        }

        size_t old_size = lu_pattern.size();
        size_t needed = old_size + static_cast<size_t>(fill_count);
        if (lu_pattern.capacity() < needed)
        {
            size_t new_cap = lu_pattern.capacity();
            if (new_cap == 0)
                new_cap = 1;
            while (new_cap < needed)
                new_cap *= 2;
            lu_pattern.reserve(new_cap);
        }
        lu_pattern.resize(needed);
        temp_triplets.resize(keep_count);

        cudaMemset(thrust::raw_pointer_cast(counts.data()), 0, 2 * sizeof(int));
        scatter_keep_fill_kernel<COLTYPE, 256><<<blocks, threads>>>(
            thrust::raw_pointer_cast(next_frontier.data()),
            thrust::raw_pointer_cast(flags.data()),
            actual_next_size,
            thrust::raw_pointer_cast(counts.data()),
            thrust::raw_pointer_cast(temp_triplets.data()),
            thrust::raw_pointer_cast(lu_pattern.data()) + old_size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_extract_fill, start, stop);
        total_extract_fill += time_extract_fill;

        // Step 6: Swap to current_frontier for next iteration
        current_frontier.swap(temp_triplets);

        // Print per-level timing
        std::cout << "Level " << level 
                  << " | frontier=" << current_size 
                  << " next_size=" << actual_next_size
                  << " | degree=" << time_degree_lookup << "ms"
                  << " scan=" << time_scan << "ms"
                  << " build=" << time_build_frontier << "ms"
                  << " hash=" << time_hash_insert << "ms"
                  << " scatter=" << time_extract_fill << "ms"
                  << std::endl;
    }

    // Print accumulated totals
    std::cout << "\n=== BFS Timing Totals ===" << std::endl;
    std::cout << "Degree lookup:   " << total_degree_lookup << " ms" << std::endl;
    std::cout << "Scan + memcpy:   " << total_scan << " ms" << std::endl;
    std::cout << "Build frontier:  " << total_build_frontier << " ms" << std::endl;
    std::cout << "Hash insert:     " << total_hash_insert << " ms" << std::endl;
    std::cout << "Scatter:         " << total_extract_fill << " ms" << std::endl;
    std::cout << "Total BFS time:  " << (total_degree_lookup + total_scan + total_build_frontier + 
                                          total_hash_insert + total_extract_fill) << " ms" << std::endl;
    std::cout << "=========================\n" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Build combined LU CSR structure
    thrust::sort(lu_pattern.begin(), lu_pattern.end(), PairLess<COLTYPE>());
    auto lu_unique_end = thrust::unique(lu_pattern.begin(), lu_pattern.end());
    lu_pattern.erase(lu_unique_end, lu_pattern.end());

    // Build LU row pointers using lower_bound for efficiency
    thrust::device_vector<ROWTYPE> lu_ai_dev(n + 1);
    
    // Count elements per row using adjacent_difference on sorted data
    auto row_it = thrust::make_transform_iterator(lu_pattern.begin(), PairSrc<COLTYPE>());
    
    // Use lower_bound to find row boundaries directly
    thrust::device_vector<COLTYPE> row_indices(n);
    thrust::sequence(row_indices.begin(), row_indices.end());
    
    thrust::lower_bound(row_it, row_it + lu_pattern.size(),
                        row_indices.begin(), row_indices.end(),
                        lu_ai_dev.begin());
    
    // Last element is total size
    lu_ai_dev[n] = lu_pattern.size();
    
    // Add base if needed
    if (base != 0)
    {
        thrust::transform(lu_ai_dev.begin(), lu_ai_dev.end(), 
                          thrust::make_constant_iterator(ROWTYPE(base)),
                          lu_ai_dev.begin(), thrust::plus<ROWTYPE>());
    }

    cudaMemcpy(d_lu_ai, thrust::raw_pointer_cast(lu_ai_dev.data()), 
               (n + 1) * sizeof(ROWTYPE), cudaMemcpyDeviceToDevice);

    *lu_nnz = lu_pattern.size();
    cudaMalloc(d_lu_aj, *lu_nnz * sizeof(COLTYPE));

    auto lu_aj_ptr = thrust::device_pointer_cast(*d_lu_aj);
    thrust::transform(lu_pattern.begin(), lu_pattern.end(), lu_aj_ptr, PairCurPlusBase<COLTYPE>(base));

    return true;
}

// Explicit template instantiations
template bool ILUSymbolicU_CUDA<int, int>(
    int n, const int* d_ai, const int* d_aj, int lvl, int base, bool keepdiag,
    int* d_u_ai, int** d_u_aj, int* u_nnz);

template bool ILUSymbolicU_CUDA<int64_t, int>(
    int n, const int64_t* d_ai, const int* d_aj, int lvl, int base, bool keepdiag,
    int64_t* d_u_ai, int** d_u_aj, int64_t* u_nnz);

template bool ILUSymbolic_CUDA<int, int>(
    int n, const int* d_ai, const int* d_aj, int lvl, int base,
    int* d_lu_ai, int** d_lu_aj, int* lu_nnz);

template bool ILUSymbolic_CUDA<int64_t, int>(
    int n, const int64_t* d_ai, const int* d_aj, int lvl, int base,
    int64_t* d_lu_ai, int** d_lu_aj, int64_t* lu_nnz);

} // namespace cuda_iterative_solver
