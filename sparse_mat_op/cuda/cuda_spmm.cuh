#pragma once

#include <cstddef>
#include <cstdint>

namespace cuda_iterative_solver
{
/**
 * @brief Step 1: Compute workload prefix sum and memory requirements for outer products
 *
 * This function computes the workload for each row (nnz_A[i] * nnz_B[i]) and performs
 * a prefix sum to determine total work and memory requirements. The user should call
 * this first, then allocate memory based on total_pairs, then call Step 2.
 *
 * @tparam ROWTYPE Type for row pointers (int or int64_t)
 * @tparam COLTYPE Type for column indices (int)
 *
 * @param n_rows Number of rows in matrices A and B
 * @param d_ai_AT Device pointer to row pointers of matrix AT
 * @param d_ai_B Device pointer to row pointers of matrix B
 * @param base Index base (0 or 1)
 * @param d_workload_prefix [Output] Device pointer to allocated prefix sum array (n_rows elements)
 *                          Caller is responsible for freeing this with cudaFree()
 *
 * @return true if successful, false otherwise
 *
 * @note The d_workload_prefix array must be freed by the caller using cudaFree()
 * @note This function allocates device memory for the prefix sum array
 */
template <typename ROWTYPE, typename COLTYPE>
bool SpMMAnalyze(COLTYPE n_rows, const ROWTYPE* d_ai_AT, const ROWTYPE* d_ai_B, ROWTYPE base,
                 ROWTYPE* d_workload_prefix);

} // namespace cuda_iterative_solver
