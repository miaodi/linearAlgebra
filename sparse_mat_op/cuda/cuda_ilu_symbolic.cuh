#pragma once

#include <cstddef>
#include <cstdint>

namespace cuda_iterative_solver
{

/**
 * @brief CUDA implementation of ILU(k) U-row symbolic factorization
 * 
 * This algorithm computes the sparsity pattern of the upper triangular part
 * of ILU(k) factorization using a BFS-based frontier expansion approach.
 * 
 * Algorithm Overview:
 * - For each node i, find all j (i <= j) such that there exists a path i -> j
 *   where all intermediate nodes are smaller than i
 * - Uses k-hop BFS with pairs (source, current)
 * - Maintains frontiers: (0,0), (1,1), ..., (n-1,n-1) initially
 * - For each level (0 to k):
 *   1. Expand frontier: for (i,j), add all (i, adj(j))
 *   2. Radix sort and remove duplicates
 *   3. Filter unvisited pairs using hash table
 *   4. Mark as visited
 *   5. Extract pairs where i <= j (U pattern)
 *   6. Keep only i < j for next iteration
 * 
 * @tparam ROWTYPE Row pointer type (int or int64_t)
 * @tparam COLTYPE Column index type
 * 
 * @param n Matrix size
 * @param d_ai Device row pointers (CSR format)
 * @param d_aj Device column indices (CSR format)
 * @param lvl Fill level k
 * @param base Index base (0 or 1)
 * @param keepdiag If true, include diagonal in U pattern
 * @param d_u_ai Output: U row pointers (allocated by caller, size n+1)
 * @param d_u_aj Output: U column indices (allocated and returned)
 * @param u_nnz Output: Number of nonzeros in U
 * 
 * @return True if successful, false on error
 */
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
    ROWTYPE* u_nnz
);

// Explicit instantiations
extern template bool ILUSymbolicU_CUDA<int, int>(
    int n, const int* d_ai, const int* d_aj, int lvl, int base, bool keepdiag,
    int* d_u_ai, int** d_u_aj, int* u_nnz);

extern template bool ILUSymbolicU_CUDA<int64_t, int>(
    int n, const int64_t* d_ai, const int* d_aj, int lvl, int base, bool keepdiag,
    int64_t* d_u_ai, int** d_u_aj, int64_t* u_nnz);

} // namespace cuda_iterative_solver
