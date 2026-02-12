#pragma once

#include "cuda_memory.cuh"
#include <cstddef>
#include <cstdint>
#include <cuda/std/utility>

namespace cuda_iterative_solver
{
/**
 * @brief Device CSR matrix structure using DeviceArray for automatic memory management
 */
template <typename ROWTYPE, typename COLTYPE>
struct DeviceCSRMatrix
{
    COLTYPE n_rows = 0;
    ROWTYPE base = 0;
    DeviceArray<ROWTYPE> ai; // row pointers (size n_rows + 1)
    DeviceArray<COLTYPE> aj; // column indices (size nnz)
};
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
 * @param d_ai_A Device pointer to row pointers of matrix A
 * @param d_ai_B Device pointer to row pointers of matrix B
 * @param base Index base (0 or 1)
 * @param d_workload_prefix [Output] Device pointer to allocated prefix sum array (n_rows + 1 elements)
 * @param required_array_size [Output] Host pointer to total required array size
 *                            (= d_workload_prefix[n_rows] - base)
 *
 * @return true if successful, false otherwise
 *
 * @note d_workload_prefix memory must be allocated/freed by the caller
 */
template <typename ROWTYPE, typename COLTYPE>
bool SpMMAnalyze(COLTYPE n_rows, const ROWTYPE* d_ai_A, const ROWTYPE* d_ai_B, ROWTYPE base,
                 ROWTYPE* d_workload_prefix, ROWTYPE* required_array_size);

/**
 * @brief Step 2: Build packed COO sparsity pattern for C = A * B from outer products, 
 * where A is in CSC and B is in CSR and are both n x n
 *
 * This function forms outer products of A_{*i} (CSC column i) and B_{i*} (CSR row i),
 * sorts the (row, col) pairs, and removes duplicates. Output is in packed COO format
 * with uint64_t keys (row in upper 32 bits, col in lower 32 bits).
 *
 * @tparam ROWTYPE Type for pointer arrays (int or int64_t)
 * @tparam COLTYPE Type for indices (int), must be <= 32 bits
 *
 * @param n Shared dimension (A columns = B rows)
 * @param d_ai_A Device pointer to CSC column pointers of A (size n + 1)
 * @param d_aj_A Device pointer to CSC row indices of A
 * @param d_ai_B Device pointer to CSR row pointers of B (size n + 1)
 * @param d_aj_B Device pointer to CSR column indices of B
 * @param base Index base (0 or 1)
 * @param packed_coo [Output] DeviceArray to receive packed COO format (uint64_t keys)
 *
 * @return true if successful, false otherwise
 */
template <typename ROWTYPE, typename COLTYPE>
bool SpMMStruct(COLTYPE n, const ROWTYPE* d_ai_A, const COLTYPE* d_aj_A, const ROWTYPE* d_ai_B,
                const COLTYPE* d_aj_B, ROWTYPE base, DeviceArray<uint64_t>& packed_coo);

/**
 * @brief Convert packed COO format to CSR format
 *
 * Takes sorted and deduplicated uint64_t keys (row in upper 32 bits, col in lower 32 bits)
 * and converts to CSR format using run-length encoding and scan operations.
 *
 * @tparam ROWTYPE Type for row pointers (int or int64_t)
 * @tparam COLTYPE Type for column indices (int), must be <= 32 bits
 *
 * @param d_keys Device pointer to packed uint64_t keys (row||col), must be sorted and unique
 * @param unique_nnz Number of unique non-zero entries
 * @param n_rows Number of rows in the matrix
 * @param base Index base (0 or 1)
 * @param output [Output] DeviceCSRMatrix to hold output CSR structure
 *
 * @return true if successful, false otherwise
 */
template <typename ROWTYPE, typename COLTYPE>
bool PackedCOOtoCSR(const uint64_t* d_keys, ROWTYPE unique_nnz, COLTYPE n_rows, ROWTYPE base,
                    DeviceCSRMatrix<ROWTYPE, COLTYPE>& output);

} // namespace cuda_iterative_solver
