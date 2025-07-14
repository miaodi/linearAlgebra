#pragma once

namespace matrix_utils {

/**
 * These functions are for permuting a vector based on a permutation vector
 *  ---------------------------------------------------------------
 */

/** @brief Permute a vector based on a permutation vector
 *  @param rows Number of rows in the vector
 *  @param base Base index of the matrix (0 or 1)
 *  @param v Input vector to be permuted
 *  @param perm Permutation vector, where perm[i] = j means P(i, j) = 1
 *  @param v_perm Output vector after permutation
 *  @tparam COLTYPE Type of the column indices (e.g., int, long)
 *  @tparam VALTYPE Type of the values in the vector (e.g., float, double)
 *  v_perm = P * v
 */
template <typename COLTYPE, typename VALTYPE>
void permVec(const COLTYPE rows, const COLTYPE base, VALTYPE const *const v,
             COLTYPE const *const perm, VALTYPE *const v_perm);

/** @brief Inverse permute a vector based on a permutation vector
 *  @param rows Number of rows in the vector
 *  @param base Base index of the matrix (0 or 1)
 *  @param v Input vector to be inverse permuted
 *  @param perm Permutation vector, where perm[i] = j means P(i, j) = 1
 *  @param v_iperm Output vector after inverse permutation
 *  @tparam COLTYPE Type of the column indices (e.g., int, long)
 *  @tparam VALTYPE Type of the values in the vector (e.g., float, double)
 *  v_iperm = P^T * v
 */
template <typename COLTYPE, typename VALTYPE>
void invPermVec(const COLTYPE rows, const COLTYPE base, VALTYPE const *const v,
                COLTYPE const *const perm, VALTYPE *const v_iperm);

/** @brief Inverse permutation of a permutation vector iperm = P^{-1} * perm
 *  @param rows Number of rows in the permutation vector
 *  @param base Base index of the matrix (0 or 1)
 *  @param perm Permutation vector, where perm[i] = j means P(i, j) = 1
 *  @param iperm Output inverse permutation vector, where iperm[j] = i means
 *  perm[i] = j
 *  @tparam COLTYPE Type of the column indices (e.g., int, long)
 */
template <typename COLTYPE>
void invPerm(const COLTYPE rows, const COLTYPE base, COLTYPE const *const perm,
             COLTYPE *const iperm);

/** @brief Check if a given vector is a valid permutation
 *  @param rows Number of rows in the permutation vector
 *  @param base Base index of the matrix (0 or 1)
 *  @param perm Permutation vector to be checked
 *  @return true if the vector is a valid permutation, false otherwise
 *  @tparam COLTYPE Type of the column indices (e.g., int, long)
 */
template <typename COLTYPE>
bool isPermutation(const COLTYPE rows, const COLTYPE base,
                   COLTYPE const *const perm);
                   
template <typename COLTYPE>
bool isPermutationSerial(const COLTYPE rows, const COLTYPE base,
                         COLTYPE const *const perm);

/** @brief Generate a random permutation
 *  @param rows Number of rows in the permutation vector
 *  @param base Base index of the matrix (0 or 1)
 *  @param perm Output permutation vector, where perm[i] = j means P(i, j) = 1
 *  @tparam COLTYPE Type of the column indices (e.g., int, long)
 */
template <typename COLTYPE>
void randPerm(const COLTYPE rows, const COLTYPE base, COLTYPE *const perm);

/**
 *  ---------------------------------------------------------------
 */

/**
 * These functions are for permuting a matrix based on a permutation vector
 *  ---------------------------------------------------------------
 */

/** @brief Permute the row pointer of a matrix based on a permutation vector
 *  @param rows Number of rows in the matrix
 *  @param ai Row pointer of the matrix
 *  @param perm Permutation vector, where perm[i] = j means P(i, j) = 1
 *  @param permed_ai Output row pointer after permutation
 *  @tparam ROWTYPE Type of the row indices (e.g., int, long)
 *  @tparam COLTYPE Type of the column indices (e.g., int, long)
 */
template <typename ROWTYPE, typename COLTYPE>
void permRowPtr(const COLTYPE rows, ROWTYPE const *ai, COLTYPE const *perm,
                ROWTYPE *perm_ai);

/** @brief Permute the matrix pA = P * A * Q^T based on a permutation vector
 *  @param rows Number of rows in the matrix
 *  @param cols Number of columns in the matrix
 *  @param ai Row pointer of the matrix
 *  @param args Additional arguments for the permutation
 *  @param permP Permutation vector for rows p[i] = j means P(i, j) = 1
 *  @param ipermQ Inverse permutation vector for columns q[j] = i means Q(i, j)
 * = 1
 *  @param perm_ai Output row pointer after permutation
 *  @tparam ROWTYPE Type of the row indices (e.g., int, long)
 *  @tparam COLTYPE Type of the column indices (e.g., int, long)
 */
template <typename ROWTYPE, typename COLTYPE, typename... Args>
void permuteMat(const COLTYPE rows, const COLTYPE cols,
                COLTYPE const *const permP, COLTYPE const *const ipermQ,
                ROWTYPE const *const ai, COLTYPE const *const aj,
                ROWTYPE *const perm_ai, COLTYPE *const perm_aj, Args *...args);

} // namespace matrix_utils