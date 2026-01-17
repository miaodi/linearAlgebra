#include "cuda_ilu_symbolic.h"
#include "cuda_memory.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
#include <iostream>
#include <limits>

namespace cuda_iterative_solver
{

// Pair structure for BFS frontier: (source_node, current_node)
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

// Hash function for pair (source, current)
template <typename COLTYPE>
__device__ __host__ unsigned long long hash_pair(COLTYPE src, COLTYPE cur)
{
    // Simple hash combining two integers
    return (static_cast<unsigned long long>(src) << 32) | static_cast<unsigned long long>(cur);
}

// Kernel: Initialize frontier with (i, i) for all i
template <typename COLTYPE>
__global__ void init_frontier_kernel(Pair<COLTYPE>* frontier, COLTYPE n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        frontier[idx].src = idx;
        frontier[idx].cur = idx;
    }
}

// Kernel: Expand frontier by exploring neighbors
// For each (i, j) in current frontier, add (i, adj[j]) to next frontier
template <typename ROWTYPE, typename COLTYPE>
__global__ void expand_frontier_kernel(
    const Pair<COLTYPE>* current_frontier,
    int current_size,
    const ROWTYPE* d_ai,
    const COLTYPE* d_aj,
    COLTYPE base,
    Pair<COLTYPE>* next_frontier,
    int* next_frontier_sizes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= current_size)
        return;
    
    Pair<COLTYPE> p = current_frontier[idx];
    COLTYPE j = p.cur;
    COLTYPE i = p.src;
    
    ROWTYPE row_start = d_ai[j] - base;
    ROWTYPE row_end = d_ai[j + 1] - base;
    int degree = row_end - row_start;
    
    // Allocate space in next frontier for this node's neighbors
    int write_pos = atomicAdd(next_frontier_sizes, degree);
    
    // Write all (i, adj[j]) pairs
    for (ROWTYPE k = row_start; k < row_end; ++k)
    {
        COLTYPE neighbor = d_aj[k] - base;
        next_frontier[write_pos].src = i;
        next_frontier[write_pos].cur = neighbor;
        write_pos++;
    }
}

// Kernel: Remove duplicate pairs (assumes sorted input)
template <typename COLTYPE>
__global__ void mark_unique_kernel(
    const Pair<COLTYPE>* pairs,
    int n,
    bool* is_unique)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    
    if (idx == 0)
    {
        is_unique[0] = true;
    }
    else
    {
        is_unique[idx] = !(pairs[idx] == pairs[idx - 1]);
    }
}

// Kernel: Check if pair (i,j) is visited using hash table
template <typename COLTYPE>
__global__ void check_visited_kernel(
    const Pair<COLTYPE>* pairs,
    int n,
    unsigned long long* hash_table,
    unsigned long long hash_table_size,
    bool* is_visited)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    
    Pair<COLTYPE> p = pairs[idx];
    unsigned long long hash = hash_pair(p.src, p.cur);
    unsigned long long pos = hash % hash_table_size;
    
    // Linear probing
    const int max_probe = 128;
    for (int probe = 0; probe < max_probe; ++probe)
    {
        unsigned long long stored = hash_table[pos];
        if (stored == hash)
        {
            // Already visited
            is_visited[idx] = true;
            return;
        }
        if (stored == ULLONG_MAX)
        {
            // Not visited
            is_visited[idx] = false;
            return;
        }
        pos = (pos + 1) % hash_table_size;
    }
    // If we reach here, hash table is too full
    is_visited[idx] = false;
}

// Kernel: Mark pairs as visited in hash table
template <typename COLTYPE>
__global__ void mark_visited_kernel(
    const Pair<COLTYPE>* pairs,
    int n,
    unsigned long long* hash_table,
    unsigned long long hash_table_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    
    Pair<COLTYPE> p = pairs[idx];
    unsigned long long hash = hash_pair(p.src, p.cur);
    unsigned long long pos = hash % hash_table_size;
    
    // Linear probing with atomic operations
    const int max_probe = 128;
    for (int probe = 0; probe < max_probe; ++probe)
    {
        unsigned long long old = atomicCAS(&hash_table[pos], ULLONG_MAX, hash);
        if (old == ULLONG_MAX || old == hash)
        {
            // Successfully inserted or already present
            return;
        }
        pos = (pos + 1) % hash_table_size;
    }
    // Hash table full - should not happen with proper sizing
}

// Functor: Filter pairs where i > j (for U pattern without diagonal)
template <typename COLTYPE>
struct FilterU_gt {
    __device__ bool operator()(const Pair<COLTYPE>& p) const {
        return p.src > p.cur;
    }
};

// Functor: Filter pairs where i >= j (for U pattern with diagonal)
template <typename COLTYPE>
struct FilterU_gte {
    __device__ bool operator()(const Pair<COLTYPE>& p) const {
        return p.src >= p.cur;
    }
};

// Functor: Filter pairs where i < j (for next frontier)
template <typename COLTYPE>
struct FilterNext {
    __device__ bool operator()(const Pair<COLTYPE>& p) const {
        return p.src < p.cur;
    }
};

// Kernel: Filter pairs based on condition (i >= j for output, i < j for next frontier)
template <typename COLTYPE>
__global__ void filter_pairs_kernel(
    const Pair<COLTYPE>* pairs,
    int n,
    const bool* keep_mask,
    bool filter_gte,  // true: keep i >= j, false: keep i < j
    bool* result_mask)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    
    if (!keep_mask[idx])
    {
        result_mask[idx] = false;
        return;
    }
    
    Pair<COLTYPE> p = pairs[idx];
    if (filter_gte)
    {
        result_mask[idx] = (p.src >= p.cur);
    }
    else
    {
        result_mask[idx] = (p.src < p.cur);
    }
}

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
    COLTYPE i = p.src;
    COLTYPE j = p.cur;
    
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
    
    // Hash table for visited tracking (load factor ~0.5 for good performance)
    size_t hash_table_size = size_t(n) * n / 2 * 2; // Adjust based on expected fill
    if (hash_table_size > 1e9)
        hash_table_size = 1e9; // Cap to avoid excessive memory
    
    thrust::device_vector<unsigned long long> hash_table(hash_table_size, ULLONG_MAX);
    
    // Temporary storage for row counts
    thrust::device_vector<ROWTYPE> row_counts(n, ROWTYPE(0));
    
    // Storage for U pattern accumulation
    thrust::device_vector<Pair<COLTYPE>> u_pattern;
    
    // BFS for k levels
    for (int level = 0; level <= lvl; ++level)
    {
        int current_size = current_frontier.size();
        if (current_size == 0)
            break;
        
        if (level < lvl)
        {
            // Use NNZ as upper bound for next frontier size
            next_frontier.resize(nnz);
            
            thrust::device_vector<int> next_frontier_size_dev(1, 0);
            
            // Expand frontier
            blocks = (current_size + threads - 1) / threads;
            expand_frontier_kernel<<<blocks, threads>>>(
                thrust::raw_pointer_cast(current_frontier.data()),
                current_size,
                d_ai,
                d_aj,
                base,
                thrust::raw_pointer_cast(next_frontier.data()),
                thrust::raw_pointer_cast(next_frontier_size_dev.data()));
            cudaDeviceSynchronize();
            
            // Get actual size
            int actual_next_size;
            cudaMemcpy(&actual_next_size, 
                      thrust::raw_pointer_cast(next_frontier_size_dev.data()),
                      sizeof(int), cudaMemcpyDeviceToHost);
            next_frontier.resize(actual_next_size);
            
            if (actual_next_size == 0)
                break;
            
            // Sort pairs
            thrust::sort(next_frontier.begin(), next_frontier.end());
            
            // Remove duplicates
            auto new_end = thrust::unique(next_frontier.begin(), next_frontier.end());
            next_frontier.erase(new_end, next_frontier.end());
            
            int unique_size = next_frontier.size();
            
            // Check which pairs are already visited
            thrust::device_vector<bool> is_visited(unique_size);
            blocks = (unique_size + threads - 1) / threads;
            check_visited_kernel<<<blocks, threads>>>(
                thrust::raw_pointer_cast(next_frontier.data()),
                unique_size,
                thrust::raw_pointer_cast(hash_table.data()),
                hash_table_size,
                thrust::raw_pointer_cast(is_visited.data()));
            cudaDeviceSynchronize();
            
            // Filter unvisited pairs using stencil
            thrust::device_vector<Pair<COLTYPE>> unvisited(unique_size);
            auto new_end_unvisited = thrust::copy_if(next_frontier.begin(), next_frontier.end(),
                           is_visited.begin(),
                           unvisited.begin(),
                           thrust::logical_not<bool>());
            unvisited.resize(thrust::distance(unvisited.begin(), new_end_unvisited));
            
            int unvisited_size = unvisited.size();
            if (unvisited_size == 0)
                break;
            
            // Mark as visited
            blocks = (unvisited_size + threads - 1) / threads;
            mark_visited_kernel<<<blocks, threads>>>(
                thrust::raw_pointer_cast(unvisited.data()),
                unvisited_size,
                thrust::raw_pointer_cast(hash_table.data()),
                hash_table_size);
            cudaDeviceSynchronize();
            
            // Extract pairs where i >= j to U pattern
            thrust::device_vector<Pair<COLTYPE>> u_pairs(unvisited_size);
            auto new_end_u = keepdiag ? 
                thrust::copy_if(unvisited.begin(), unvisited.end(),
                               u_pairs.begin(),
                               FilterU_gte<COLTYPE>()) :
                thrust::copy_if(unvisited.begin(), unvisited.end(),
                               u_pairs.begin(),
                               FilterU_gt<COLTYPE>());
            u_pairs.resize(thrust::distance(u_pairs.begin(), new_end_u));
            
            // Append to global U pattern
            u_pattern.insert(u_pattern.end(), u_pairs.begin(), u_pairs.end());
            
            // Keep only i < j for next iteration
            thrust::device_vector<Pair<COLTYPE>> next_frontier_filtered(unvisited_size);
            auto new_end_next = thrust::copy_if(unvisited.begin(), unvisited.end(),
                           next_frontier_filtered.begin(),
                           FilterNext<COLTYPE>());
            next_frontier_filtered.resize(thrust::distance(next_frontier_filtered.begin(), new_end_next));
            
            current_frontier = next_frontier_filtered;
        }
        else
        {
            // Last level: extract all remaining pairs where i >= j
            int current_sz = current_frontier.size();
            thrust::device_vector<Pair<COLTYPE>> u_pairs(current_sz);
            auto new_end_final = keepdiag ? 
                thrust::copy_if(current_frontier.begin(), current_frontier.end(),
                               u_pairs.begin(),
                               FilterU_gte<COLTYPE>()) :
                thrust::copy_if(current_frontier.begin(), current_frontier.end(),
                               u_pairs.begin(),
                               FilterU_gt<COLTYPE>());
            u_pairs.resize(thrust::distance(u_pairs.begin(), new_end_final));
            u_pattern.insert(u_pattern.end(), u_pairs.begin(), u_pairs.end());
        }
    }
    
    // Now build CSR structure from collected pairs
    // Sort by row then column
    thrust::sort(u_pattern.begin(), u_pattern.end());
    
    // Remove any duplicates that might have accumulated
    auto new_end = thrust::unique(u_pattern.begin(), u_pattern.end());
    u_pattern.erase(new_end, u_pattern.end());
    
    // Build row pointers
    thrust::device_vector<ROWTYPE> u_ai_dev(n + 1, ROWTYPE(0));
    
    // Copy pattern to host for row counting (simpler than device-side)
    std::vector<Pair<COLTYPE>> u_pattern_host(u_pattern.size());
    thrust::copy(u_pattern.begin(), u_pattern.end(), u_pattern_host.begin());
    
    // Count entries per row on host
    std::vector<ROWTYPE> u_ai_host(n + 1, ROWTYPE(0));
    for (size_t i = 0; i < u_pattern_host.size(); ++i)
    {
        COLTYPE row = u_pattern_host[i].src;
        u_ai_host[row + 1]++;
    }
    
    // Prefix sum to get row pointers on host
    for (int i = 0; i < n; ++i)
    {
        u_ai_host[i + 1] += u_ai_host[i];
    }
    
    // Adjust for base indexing
    for (int i = 0; i <= n; ++i)
    {
        u_ai_host[i] += base;
    }
    
    // Copy to output
    cudaMemcpy(d_u_ai, u_ai_host.data(),
              (n + 1) * sizeof(ROWTYPE), cudaMemcpyHostToDevice);
    
    // Build column indices
    *u_nnz = u_pattern_host.size();
    cudaMalloc(d_u_aj, *u_nnz * sizeof(COLTYPE));
    
    // Extract columns and adjust for base on host
    std::vector<COLTYPE> u_aj_host(*u_nnz);
    for (size_t i = 0; i < u_pattern_host.size(); ++i)
    {
        u_aj_host[i] = u_pattern_host[i].cur + base;
    }
    
    cudaMemcpy(*d_u_aj, u_aj_host.data(),
              *u_nnz * sizeof(COLTYPE), cudaMemcpyHostToDevice);
    
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
