#pragma once

#include "matrix_utils.hpp"
#include "sparse_mat_traits.hpp"
#include <atomic>
#include <numeric>
#include <vector>
#include <omp.h>

namespace matrix_utils
{

template <typename ROWTYPE, typename COLTYPE, bool KEEPDIAG = true>
void APlusATPrefix( const COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj, ROWTYPE* ai_AAT );

template <typename ROWTYPE, typename COLTYPE, bool KEEPDIAG = true>
void APlusATFill( const COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj, ROWTYPE const* ai_AAT, COLTYPE* aj_AAT );

template <typename ROWTYPE, typename COLTYPE, bool KEEPDIAG = true>
void APlusATSerial( const COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj, ROWTYPE* ai_out, COLTYPE* aj_out );

/// @brief Functor for computing A+A^T with memory reuse
/// @details Encapsulates both symbolic and numeric phases of A+A^T computation.
///          Reuses internal memory across multiple invocations for efficiency.
///          Returns CSRStructVec to avoid manual deduplication.
template <typename ROWTYPE, typename COLTYPE, bool KEEPDIAG = true>
struct APlusATStruct
{
    APlusATStruct( const int num_threads = omp_get_max_threads() ) : _nthreads{ num_threads }
    {
        if ( _nthreads < 1 )
            throw std::runtime_error( "Number of threads must be at least 1." );
        _thread_sums.resize( _nthreads + 1, 0 );
    }

    /// @brief Compute A+A^T and return as CSRStructVec (handles deduplication automatically)
    /// @return CSRStructVec containing the deduplicated structure of A+A^T
    void operator()( COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj, ROWTYPE* ai_APlusAT, COLTYPE* aj_APlusAT );

    void setNumThreads( const int num_threads )
    {
        _nthreads = num_threads;
        _thread_sums.resize( _nthreads + 1, 0 );
    }

    // Expose phases individually for benchmarking
    void prefixOnly( COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj, ROWTYPE* ai_APlusAT );
    void fillAndCompactOnly( COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj, ROWTYPE* ai_APlusAT, COLTYPE* aj_APlusAT );

private:
    void prefix( const COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj );
    void fillAndCompact( const COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj, ROWTYPE* ai_APlusAT, COLTYPE* aj_APlusAT );

    // Reusable memory buffers
    int _nthreads;
    std::vector<ROWTYPE> _thread_sums;
    std::unique_ptr<std::atomic<ROWTYPE>[]> _row_pos;
    size_t _row_pos_capacity = 0;
    CSRStructVec<ROWTYPE, COLTYPE> _APAT;
};

template <ResizableCSR CSRMatrixType>
void Block( const typename CSRMatrixType::COLTYPE rows,
            const typename CSRMatrixType::ROWTYPE base,
            typename CSRMatrixType::ROWTYPE const* ai,
            typename CSRMatrixType::COLTYPE const* aj,
            typename CSRMatrixType::VALTYPE const* av,
            const typename CSRMatrixType::COLTYPE i,
            const typename CSRMatrixType::COLTYPE j,
            const typename CSRMatrixType::COLTYPE p,
            const typename CSRMatrixType::COLTYPE q,
            CSRMatrixType& subMat );

template <ResizableCSR CSRMatrixType>
void partitionCSR1x2( const typename CSRMatrixType::COLTYPE rows,
                      const typename CSRMatrixType::COLTYPE cols,
                      typename CSRMatrixType::ROWTYPE const* ai,
                      typename CSRMatrixType::COLTYPE const* aj,
                      typename CSRMatrixType::VALTYPE const* av,
                      const typename CSRMatrixType::COLTYPE col_split,
                      const typename CSRMatrixType::ROWTYPE base,
                      CSRMatrixType& A1,
                      CSRMatrixType& A2,
                      const int nthreads = omp_get_max_threads() );

template <ResizableCSR CSRMatrixType>
void partitionCSR2x2( const typename CSRMatrixType::COLTYPE rows,
                      const typename CSRMatrixType::COLTYPE cols,
                      typename CSRMatrixType::ROWTYPE const* ai,
                      typename CSRMatrixType::COLTYPE const* aj,
                      typename CSRMatrixType::VALTYPE const* av,
                      const typename CSRMatrixType::COLTYPE row_split,
                      const typename CSRMatrixType::COLTYPE col_split,
                      CSRMatrixType& A11,
                      CSRMatrixType& A12,
                      CSRMatrixType& A21,
                      CSRMatrixType& A22,
                      const int nthreads = omp_get_max_threads() );

template <ResizableCSR CSRMatrixType>
void partitionCSR1xN( const typename CSRMatrixType::COLTYPE rows,
                      const typename CSRMatrixType::COLTYPE cols,
                      typename CSRMatrixType::ROWTYPE const* ai,
                      typename CSRMatrixType::COLTYPE const* aj,
                      typename CSRMatrixType::VALTYPE const* av,
                      const int N,
                      typename CSRMatrixType::COLTYPE const* col_splits,
                      const typename CSRMatrixType::ROWTYPE base,
                      CSRMatrixType* blocks,
                      const int nthreads = omp_get_max_threads() );

template <ResizableCSR CSRMatrixType>
void partitionCSRMxN( const typename CSRMatrixType::COLTYPE rows,
                      const typename CSRMatrixType::COLTYPE cols,
                      typename CSRMatrixType::ROWTYPE const* ai,
                      typename CSRMatrixType::COLTYPE const* aj,
                      typename CSRMatrixType::VALTYPE const* av,
                      const int M,
                      typename CSRMatrixType::COLTYPE const* row_splits,
                      const int N,
                      typename CSRMatrixType::COLTYPE const* col_splits,
                      CSRMatrixType* blocks,
                      const int nthreads = omp_get_max_threads() );
} // namespace matrix_utils
