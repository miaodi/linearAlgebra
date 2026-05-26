#pragma once

#include <cusparse.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace matrix_utils::sparse_cuda
{

/**
 * @brief Triangular matrix type for SpTRSV
 */
enum class TriangularType
{
    Lower, ///< Lower triangular matrix
    Upper  ///< Upper triangular matrix
};

/**
 * @brief Unit diagonal assumption for SpTRSV
 */
enum class DiagonalType
{
    NonUnit, ///< Diagonal elements are explicitly stored
    Unit     ///< Diagonal elements are assumed to be 1.0
};

/**
 * @brief Sparse Triangular Solve (SpTRSV) operator: Lx = b or Ux = b
 *
 * This implementation uses a synchronization-free approach:
 * - Each thread is responsible for one row
 * - Solution vector is initialized to NaN
 * - Threads spin-wait when encountering NaN values (dependencies)
 *
 * @tparam ROWTYPE Row pointer type (e.g., int, int64_t)
 * @tparam COLTYPE Column index type (e.g., int, int64_t)
 * @tparam VALTYPE Value type (e.g., double, float)
 */
template <typename ROWTYPE = int, typename COLTYPE = int, typename VALTYPE = double>
class SpTRSVOperator
{
public:
    /**
     * @brief Constructor
     *
     * @param n Matrix dimension (n x n)
     * @param d_ia Device row pointers (length n+1)
     * @param d_ja Device column indices
     * @param d_av Device values
     * @param tri_type Triangular type (Lower or Upper)
     * @param diag_type Diagonal type (NonUnit or Unit)
     */
    SpTRSVOperator( COLTYPE n,
                    const ROWTYPE* d_ia,
                    const COLTYPE* d_ja,
                    const VALTYPE* d_av,
                    TriangularType tri_type = TriangularType::Lower,
                    DiagonalType diag_type = DiagonalType::NonUnit );

    ~SpTRSVOperator() = default;

    /**
     * @brief Solve Lx = b or Ux = b
     *
     * @param d_b Input right-hand side vector (device memory, length n)
     * @param d_x Output solution vector (device memory, length n)
     *            Note: d_x will be initialized to NaN internally
     */
    void solve( const VALTYPE* d_b, VALTYPE* d_x );

    /**
     * @brief Get matrix size
     * @return Matrix dimension
     */
    COLTYPE size() const { return _n; }

private:
    COLTYPE _n;               ///< Matrix dimension
    const ROWTYPE* _d_ia;     ///< Device row pointers
    const COLTYPE* _d_ja;     ///< Device column indices
    const VALTYPE* _d_av;     ///< Device values
    TriangularType _tri_type; ///< Lower or Upper triangular
    DiagonalType _diag_type;  ///< Unit or NonUnit diagonal
};

/**
 * @brief Convenience function to solve Lx = b with lower triangular L
 *
 * @param n Matrix dimension
 * @param d_ia Device row pointers
 * @param d_ja Device column indices
 * @param d_av Device values
 * @param d_b Device right-hand side
 * @param d_x Device solution vector (output)
 * @param diag_type Diagonal type (default: NonUnit)
 */
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void sptrsv_lower( COLTYPE n,
                   const ROWTYPE* d_ia,
                   const COLTYPE* d_ja,
                   const VALTYPE* d_av,
                   const VALTYPE* d_b,
                   VALTYPE* d_x,
                   DiagonalType diag_type = DiagonalType::NonUnit );

/**
 * @brief Convenience function to solve Ux = b with upper triangular U
 *
 * @param n Matrix dimension
 * @param d_ia Device row pointers
 * @param d_ja Device column indices
 * @param d_av Device values
 * @param d_b Device right-hand side
 * @param d_x Device solution vector (output)
 * @param diag_type Diagonal type (default: NonUnit)
 */
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void sptrsv_upper( COLTYPE n,
                   const ROWTYPE* d_ia,
                   const COLTYPE* d_ja,
                   const VALTYPE* d_av,
                   const VALTYPE* d_b,
                   VALTYPE* d_x,
                   DiagonalType diag_type = DiagonalType::NonUnit );

} // namespace matrix_utils::sparse_cuda
