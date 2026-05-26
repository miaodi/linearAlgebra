#pragma once

#include "config.h"

#ifdef USE_MKL
#include <mkl_spblas.h>
#include <mkl_types.h>
#include <stdexcept>
#include <type_traits>
#include <string>

namespace matrix_utils
{

/**
 * @brief MKL SpMV wrapper - uses Intel MKL's optimized sparse matrix-vector multiplication
 *
 * This class provides an interface compatible with the matrix_utils::SPMV wrapper,
 * using Intel MKL's highly optimized sparse BLAS routines.
 *
 * Key features:
 * - Uses MKL's mkl_sparse_d_mv and mkl_sparse_s_mv for optimal CPU performance
 * - Automatic matrix optimization with MKL hints
 * - Support for both 0-based and 1-based indexing
 * - Support for alpha/beta scaling (y = alpha * A * x + beta * y)
 * - Move semantics to prevent resource duplication
 * - Type safety: Only works with MKL_INT for row/column indices
 *
 * Note: ROWTYPE and COLTYPE must be MKL_INT. Other integer types will result in
 *       a runtime error with a clear message.
 *
 * Usage:
 *   MKLSPMV<MKL_INT, MKL_INT, double> spmv;
 *   spmv.preprocess(n, ia, ja, av);
 *   spmv(b, x, alpha, beta);
 */
template <typename ROWTYPE = int, typename COLTYPE = int, typename VALTYPE = double>
class MKLSPMV
{
public:
    MKLSPMV()
    {
        // Check type compatibility at construction time
        if constexpr ( !std::is_same_v<ROWTYPE, MKL_INT> || !std::is_same_v<COLTYPE, MKL_INT> )
        {
            _incompatible_types = true;
        }
    }

    /**
     * @brief Preprocess the matrix structure for subsequent SpMV operations
     *
     * This creates the MKL sparse matrix handle and optimizes it for SpMV operations.
     * Must be called before the first SpMV operation.
     *
     * @param size Matrix dimension (number of rows)
     * @param ai Row pointers (size size+1)
     * @param aj Column indices
     * @param av Matrix values
     *
     * @throws std::runtime_error if ROWTYPE or COLTYPE is not MKL_INT
     */
    void preprocess( COLTYPE size, ROWTYPE const* __restrict ai, COLTYPE const* __restrict aj, VALTYPE const* __restrict av )
    {
        // Check type compatibility
        if constexpr ( !std::is_same_v<ROWTYPE, MKL_INT> || !std::is_same_v<COLTYPE, MKL_INT> )
        {
            throw std::runtime_error(
                "MKLSPMV requires ROWTYPE and COLTYPE to be MKL_INT. "
                "Current types: ROWTYPE=" +
                std::string( typeid( ROWTYPE ).name() ) +
                ", COLTYPE=" + std::string( typeid( COLTYPE ).name() ) +
                ", MKL_INT=" + std::string( typeid( MKL_INT ).name() ) +
                ". "
                "Please use MKLSPMV<MKL_INT, MKL_INT, VALTYPE> or choose a different SpMV "
                "implementation." );
        }

        if constexpr ( !std::is_same_v<VALTYPE, double> && !std::is_same_v<VALTYPE, float> )
        {
            throw std::runtime_error(
                "MKLSPMV only supports float and double value types. "
                "Current type: VALTYPE=" +
                std::string( typeid( VALTYPE ).name() ) );
        }

        _size = size;
        _ai = ai;
        _aj = aj;
        _av = av;
        _base = static_cast<int>( ai ? ai[0] : 0 );

        // Create MKL sparse matrix handle
        if ( _mkl_mat )
        {
            mkl_sparse_destroy( _mkl_mat );
            _mkl_mat = nullptr;
        }

        sparse_index_base_t mkl_base = ( _base == 0 ) ? SPARSE_INDEX_BASE_ZERO : SPARSE_INDEX_BASE_ONE;
        COLTYPE nnz = ai[size] - ai[0];

        // Create MKL CSR matrix
        sparse_status_t status;
        if constexpr ( std::is_same_v<VALTYPE, double> )
        {
            status = mkl_sparse_d_create_csr( &_mkl_mat, mkl_base, size, size,
                                              const_cast<ROWTYPE*>( ai ), const_cast<ROWTYPE*>( ai + 1 ),
                                              const_cast<COLTYPE*>( aj ), const_cast<VALTYPE*>( av ) );
        }
        else if constexpr ( std::is_same_v<VALTYPE, float> )
        {
            status = mkl_sparse_s_create_csr( &_mkl_mat, mkl_base, size, size,
                                              const_cast<ROWTYPE*>( ai ), const_cast<ROWTYPE*>( ai + 1 ),
                                              const_cast<COLTYPE*>( aj ), const_cast<VALTYPE*>( av ) );
        }

        if ( status != SPARSE_STATUS_SUCCESS )
        {
            throw std::runtime_error( "MKL sparse matrix creation failed" );
        }

        // Optimize for SpMV operations
        _descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        _descr.mode = SPARSE_FILL_MODE_FULL;
        _descr.diag = SPARSE_DIAG_NON_UNIT;

        mkl_sparse_set_mv_hint( _mkl_mat, SPARSE_OPERATION_NON_TRANSPOSE, _descr, 1000 );
        mkl_sparse_set_memory_hint( _mkl_mat, SPARSE_MEMORY_AGGRESSIVE );
        mkl_sparse_optimize( _mkl_mat );
    }

    ~MKLSPMV()
    {
        if ( _mkl_mat )
        {
            mkl_sparse_destroy( _mkl_mat );
        }
    }

    // Delete copy constructor and assignment to prevent double-free
    MKLSPMV( const MKLSPMV& ) = delete;
    MKLSPMV& operator=( const MKLSPMV& ) = delete;

    // Move constructor
    MKLSPMV( MKLSPMV&& other ) noexcept
        : _size( other._size ),
          _ai( other._ai ),
          _aj( other._aj ),
          _av( other._av ),
          _base( other._base ),
          _mkl_mat( other._mkl_mat ),
          _descr( other._descr )
    {
        other._mkl_mat = nullptr;
    }

    // Move assignment
    MKLSPMV& operator=( MKLSPMV&& other ) noexcept
    {
        if ( this != &other )
        {
            if ( _mkl_mat )
            {
                mkl_sparse_destroy( _mkl_mat );
            }
            _size = other._size;
            _ai = other._ai;
            _aj = other._aj;
            _av = other._av;
            _base = other._base;
            _mkl_mat = other._mkl_mat;
            _descr = other._descr;
            other._mkl_mat = nullptr;
        }
        return *this;
    }

    /**
     * @brief Perform SpMV: x = alpha * A * b + beta * x
     *
     * @param b Input vector (size n)
     * @param x Output vector (size n)
     * @param alpha Scalar multiplier for A*b
     * @param beta Scalar multiplier for x
     *
     * @throws std::runtime_error if types are incompatible or MKL operation fails
     */
    void operator()( VALTYPE const* __restrict const b,
                     VALTYPE* __restrict const x,
                     const VALTYPE alpha = 1.0,
                     const VALTYPE beta = 0.0 ) const
    {
        // Check type compatibility
        if constexpr ( !std::is_same_v<ROWTYPE, MKL_INT> || !std::is_same_v<COLTYPE, MKL_INT> )
        {
            throw std::runtime_error(
                "MKLSPMV requires ROWTYPE and COLTYPE to be MKL_INT. "
                "Preprocess was not called or types are incompatible." );
        }

        sparse_status_t status;
        if constexpr ( std::is_same_v<VALTYPE, double> )
        {
            status = mkl_sparse_d_mv( SPARSE_OPERATION_NON_TRANSPOSE, alpha, _mkl_mat, _descr, b, beta, x );
        }
        else if constexpr ( std::is_same_v<VALTYPE, float> )
        {
            status = mkl_sparse_s_mv( SPARSE_OPERATION_NON_TRANSPOSE, alpha, _mkl_mat, _descr, b, beta, x );
        }
        else
        {
            throw std::runtime_error( "MKLSPMV only supports float and double value types." );
        }

        if ( status != SPARSE_STATUS_SUCCESS )
        {
            throw std::runtime_error( "MKL SpMV operation failed with status: " + std::to_string( status ) );
        }
    }

    COLTYPE size() const { return _size; }
    using VALTYPE_ALIAS = VALTYPE;

    /**
     * @brief Check if the current template instantiation is compatible with MKL
     */
    static constexpr bool is_compatible()
    {
        return std::is_same_v<ROWTYPE, MKL_INT> && std::is_same_v<COLTYPE, MKL_INT> &&
               ( std::is_same_v<VALTYPE, double> || std::is_same_v<VALTYPE, float> );
    }

private:
    COLTYPE _size = 0;
    ROWTYPE const* _ai = nullptr;
    COLTYPE const* _aj = nullptr;
    VALTYPE const* _av = nullptr;
    int _base = 0;
    sparse_matrix_t _mkl_mat = nullptr;
    matrix_descr _descr;
    bool _incompatible_types = false;
};

} // namespace matrix_utils

#endif // USE_MKL
