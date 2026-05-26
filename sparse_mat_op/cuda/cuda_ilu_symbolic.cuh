#pragma once

#include <cstddef>
#include <cstdint>
#include "cuda_csr_utils.cuh"

namespace matrix_utils::sparse_cuda
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
bool ILUSymbolicU_CUDA( COLTYPE n,
                        const ROWTYPE* d_ai,
                        const COLTYPE* d_aj,
                        int lvl,
                        COLTYPE base,
                        bool keepdiag,
                        ROWTYPE* d_u_ai,
                        COLTYPE** d_u_aj,
                        ROWTYPE* u_nnz );

/**
 * @brief CUDA implementation of ILU(k) U-row symbolic factorization
 *
 * This algorithm computes the sparsity pattern of the upper triangular part
 * of ILU(k) factorization using a SpMM-based approach.
 *
 * @tparam ROWTYPE Row pointer type (int or int64_t)
 * @tparam COLTYPE Column index type
 *
 * @param n Matrix size
 * @param d_ai Device row pointers (CSR format)
 * @param d_aj Device column indices (CSR format)
 * @param lvl Fill level k
 * @param base Index base (0 or 1)
 * @param d_u_ai Output: U row pointers (allocated by caller, size n+1)
 * @param d_u_aj Output: U column indices (allocated and returned)
 * @param u_nnz Output: Number of nonzeros in U
 *
 * @return True if successful, false on error
 */
template <typename ROWTYPE, typename COLTYPE>
bool ILUSymbolicU_SpMM_CUDA( COLTYPE n,
                             const ROWTYPE* d_ai,
                             const COLTYPE* d_aj,
                             int lvl,
                             COLTYPE base,
                             DeviceCSRMatrix<ROWTYPE, COLTYPE>& U_matrix );

/**
 * @brief Warp-persistent CUDA implementation of ILU(k) U-row symbolic factorization
 */
template <typename ROWTYPE, typename COLTYPE>
bool ILUSymbolicU_CUDA_Persistent( COLTYPE n,
                                   const ROWTYPE* d_ai,
                                   const COLTYPE* d_aj,
                                   int lvl,
                                   COLTYPE base,
                                   bool keepdiag,
                                   ROWTYPE* d_u_ai,
                                   COLTYPE** d_u_aj,
                                   ROWTYPE* u_nnz );

// Explicit instantiations
extern template bool ILUSymbolicU_CUDA<int, int>( int n,
                                                  const int* d_ai,
                                                  const int* d_aj,
                                                  int lvl,
                                                  int base,
                                                  bool keepdiag,
                                                  int* d_u_ai,
                                                  int** d_u_aj,
                                                  int* u_nnz );

extern template bool ILUSymbolicU_CUDA<int64_t, int>( int n,
                                                      const int64_t* d_ai,
                                                      const int* d_aj,
                                                      int lvl,
                                                      int base,
                                                      bool keepdiag,
                                                      int64_t* d_u_ai,
                                                      int** d_u_aj,
                                                      int64_t* u_nnz );

extern template bool ILUSymbolicU_CUDA_Persistent<int, int>( int n,
                                                             const int* d_ai,
                                                             const int* d_aj,
                                                             int lvl,
                                                             int base,
                                                             bool keepdiag,
                                                             int* d_u_ai,
                                                             int** d_u_aj,
                                                             int* u_nnz );

extern template bool ILUSymbolicU_CUDA_Persistent<int64_t, int>( int n,
                                                                 const int64_t* d_ai,
                                                                 const int* d_aj,
                                                                 int lvl,
                                                                 int base,
                                                                 bool keepdiag,
                                                                 int64_t* d_u_ai,
                                                                 int** d_u_aj,
                                                                 int64_t* u_nnz );

extern template bool ILUSymbolicU_SpMM_CUDA<int, int>( int n,
                                                       const int* d_ai,
                                                       const int* d_aj,
                                                       int lvl,
                                                       int base,
                                                       DeviceCSRMatrix<int, int>& U_matrix );
extern template bool ILUSymbolicU_SpMM_CUDA<std::int64_t, int>( int n,
                                                                const std::int64_t* d_ai,
                                                                const int* d_aj,
                                                                int lvl,
                                                                int base,
                                                                DeviceCSRMatrix<std::int64_t, int>& U_matrix );

/**
 * @brief CUDA implementation of full ILU(k) symbolic factorization (combined LU)
 *
 * This algorithm computes the combined LU sparsity pattern using a modified BFS
 * approach with triplets ((src, cur), max_cur).
 *
 * Algorithm Overview:
 * - Maintains triplets ((src, cur), max_cur) where max_cur tracks the maximum
 *   intermediate node visited on the path from src to cur
 * - Frontier expansion: ((src, cur), max_cur) -> ((src, neighbor), max(max_cur, neighbor))
 * - A triplet is kept if max_cur < src (valid ILU path condition)
 * - An entry (src, cur) is added to factorization if cur > max_cur
 * - Uses static_map to track visited (src, cur) pairs with their max_cur values
 *
 * @tparam ROWTYPE Row pointer type (int or int64_t)
 * @tparam COLTYPE Column index type
 *
 * @param n Matrix size
 * @param d_ai Device row pointers (CSR format)
 * @param d_aj Device column indices (CSR format)
 * @param lvl Fill level k
 * @param base Index base (0 or 1)
 * @note Diagonal is always included in LU pattern
 * @param d_lu_ai Output: LU row pointers (allocated by caller, size n+1)
 * @param d_lu_aj Output: LU column indices (allocated and returned)
 * @param lu_nnz Output: Number of nonzeros in LU
 *
 * @return True if successful, false on error
 */
template <typename ROWTYPE, typename COLTYPE>
bool ILUSymbolic_CUDA( COLTYPE n,
                       const ROWTYPE* d_ai,
                       const COLTYPE* d_aj,
                       int lvl,
                       COLTYPE base,
                       ROWTYPE* d_lu_ai,
                       COLTYPE** d_lu_aj,
                       ROWTYPE* lu_nnz );

extern template bool ILUSymbolic_CUDA<int, int>( int n,
                                                 const int* d_ai,
                                                 const int* d_aj,
                                                 int lvl,
                                                 int base,
                                                 int* d_lu_ai,
                                                 int** d_lu_aj,
                                                 int* lu_nnz );

extern template bool ILUSymbolic_CUDA<int64_t, int>( int n,
                                                     const int64_t* d_ai,
                                                     const int* d_aj,
                                                     int lvl,
                                                     int base,
                                                     int64_t* d_lu_ai,
                                                     int** d_lu_aj,
                                                     int64_t* lu_nnz );

} // namespace matrix_utils::sparse_cuda
