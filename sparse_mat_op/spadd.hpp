#pragma once

#include "sparse_mat_traits.hpp"
#include <omp.h>

namespace matrix_utils
{

/**
 * @brief Sparse matrix addition C = alpha * A + beta * B
 *
 * Two-phase algorithm:
 * 1. Analysis phase: Determines sparsity pattern of C and allocates memory
 * 2. Numerical phase: Computes actual values
 *
 * @tparam CSRMatrixType Type satisfying ResizableCSR concept
 */
template <ResizableCSR CSRMatrixType>
struct SpADD
{
    using ROWTYPE = typename CSRMatrixType::ROWTYPE;
    using COLTYPE = typename CSRMatrixType::COLTYPE;
    using VALTYPE = typename CSRMatrixType::VALTYPE;

    int _nthreads;

    /**
     * @brief Constructor
     * @param num_threads Number of OpenMP threads to use
     */
    explicit SpADD( int num_threads = omp_get_max_threads() ) : _nthreads( num_threads ) {}

    /**
     * @brief Analysis phase: Determine sparsity pattern of C = alpha * A + beta * B
     *
     * Computes the row pointers for the result matrix C without computing values.
     * This allows for efficient memory allocation before the numerical phase.
     *
     * @param A_rows Number of rows in matrix A
     * @param A_cols Number of columns in matrix A (must equal B_cols)
     * @param A_ai Row pointers of matrix A
     * @param A_aj Column indices of matrix A
     * @param B_rows Number of rows in matrix B (must equal A_rows)
     * @param B_cols Number of columns in matrix B (must equal A_cols)
     * @param B_ai Row pointers of matrix B
     * @param B_aj Column indices of matrix B
     * @param C Output matrix C (dimensions and row pointers will be set)
     */
    void analysis( const COLTYPE A_rows,
                   const COLTYPE A_cols,
                   const ROWTYPE* A_ai,
                   const COLTYPE* A_aj,
                   const COLTYPE B_rows,
                   const COLTYPE B_cols,
                   const ROWTYPE* B_ai,
                   const COLTYPE* B_aj,
                   CSRMatrixType& C );

    /**
     * @brief Numerical phase: Compute C = alpha * A + beta * B
     *
     * Computes the actual values of the sparse matrix sum.
     * Must call analysis() first to determine the sparsity pattern.
     *
     * @param A_rows Number of rows in matrix A
     * @param A_cols Number of columns in matrix A
     * @param A_ai Row pointers of matrix A
     * @param A_aj Column indices of matrix A
     * @param A_av Values of matrix A
     * @param alpha Scalar multiplier for matrix A
     * @param B_rows Number of rows in matrix B
     * @param B_cols Number of columns in matrix B
     * @param B_ai Row pointers of matrix B
     * @param B_aj Column indices of matrix B
     * @param B_av Values of matrix B
     * @param beta Scalar multiplier for matrix B
     * @param C Output matrix C (column indices and values will be computed)
     */
    void operator()( const COLTYPE A_rows,
                     const COLTYPE A_cols,
                     const ROWTYPE* A_ai,
                     const COLTYPE* A_aj,
                     const VALTYPE* A_av,
                     const VALTYPE alpha,
                     const COLTYPE B_rows,
                     const COLTYPE B_cols,
                     const ROWTYPE* B_ai,
                     const COLTYPE* B_aj,
                     const VALTYPE* B_av,
                     const VALTYPE beta,
                     CSRMatrixType& C );

    /**
     * @brief Set number of threads
     * @param num_threads Number of OpenMP threads
     */
    void setNumThreads( int num_threads ) { _nthreads = num_threads; }
};

} // namespace matrix_utils
