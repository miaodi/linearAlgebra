#pragma once

#include "BitVector.hpp"
#include "matrix_utils.hpp"
#include "spmv_load.hpp"
#include <concepts>
#include <cstring>
#include <immintrin.h>
#include <omp.h>
#include "dot_kernel.hpp"
#include "mkl_spmv.hpp"
namespace matrix_utils
{
enum class WorkloadMode
{
    ALBUS,
    CAMLB
};

// compute x = alpha * A * b + beta * x

struct SerialSPMV
{
    SerialSPMV() = default;

    template <BetaMode Mode, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
    void compute( const COLTYPE size,
                  const int base,
                  ROWTYPE const* __restrict ai,
                  COLTYPE const* __restrict aj,
                  VALTYPE const* __restrict av,
                  VALTYPE const* __restrict const b,
                  VALTYPE* __restrict const x,
                  const VALTYPE alpha,
                  const VALTYPE beta ) const
    {
        for ( COLTYPE i = 0; i < size; i++ )
        {
            VALTYPE val = 0;
#pragma unroll
            for ( ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; j++ )
            {
                val += av[j] * b[aj[j] - base];
            }
            const VALTYPE ax = alpha * val;
            x[i] = apply_beta<Mode>( ax, beta, x[i] );
        }
    }

    template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
    void operator()( const COLTYPE size,
                     const int base,
                     ROWTYPE const* __restrict ai,
                     COLTYPE const* __restrict aj,
                     VALTYPE const* __restrict av,
                     VALTYPE const* __restrict const b,
                     VALTYPE* __restrict const x,
                     const VALTYPE alpha,
                     const VALTYPE beta ) const
    {
        if ( beta == static_cast<VALTYPE>( 0 ) )
        {
            compute<BetaMode::Zero>( size, base, ai, aj, av, b, x, alpha, beta );
        }
        else if ( beta == static_cast<VALTYPE>( 1 ) )
        {
            compute<BetaMode::One>( size, base, ai, aj, av, b, x, alpha, beta );
        }
        else
        {
            compute<BetaMode::Generic>( size, base, ai, aj, av, b, x, alpha, beta );
        }
    }
};

// Plain OpenMP parallel SPMV (simple row distribution)
template <typename ROWTYPE = int, typename COLTYPE = int, typename VALTYPE = double>
struct ParallelSPMV
{
    ParallelSPMV( const int num_threads = omp_get_max_threads() ) : _nthreads{ num_threads } {}

    void setNumThreads( const int num_threads ) { _nthreads = num_threads; }

    template <BetaMode Mode>
    void compute( const COLTYPE size,
                  const int base,
                  ROWTYPE const* __restrict ai,
                  COLTYPE const* __restrict aj,
                  VALTYPE const* __restrict av,
                  VALTYPE const* __restrict const b,
                  VALTYPE* __restrict const x,
                  const VALTYPE alpha,
                  const VALTYPE beta ) const
    {
#pragma omp parallel for num_threads( _nthreads )
        for ( COLTYPE i = 0; i < size; i++ )
        {
            VALTYPE val = 0;
            for ( ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; j++ )
            {
                val += av[j] * b[aj[j] - base];
            }
            const VALTYPE ax = alpha * val;
            x[i] = apply_beta<Mode>( ax, beta, x[i] );
        }
    }

    void operator()( const COLTYPE size,
                     const int base,
                     ROWTYPE const* __restrict ai,
                     COLTYPE const* __restrict aj,
                     VALTYPE const* __restrict av,
                     VALTYPE const* __restrict const b,
                     VALTYPE* __restrict const x,
                     const VALTYPE alpha,
                     const VALTYPE beta ) const
    {
        if ( beta == static_cast<VALTYPE>( 0 ) )
        {
            compute<BetaMode::Zero>( size, base, ai, aj, av, b, x, alpha, beta );
        }
        else if ( beta == static_cast<VALTYPE>( 1 ) )
        {
            compute<BetaMode::One>( size, base, ai, aj, av, b, x, alpha, beta );
        }
        else
        {
            compute<BetaMode::Generic>( size, base, ai, aj, av, b, x, alpha, beta );
        }
    }

private:
    int _nthreads;
};

template <typename ROWTYPE = int, typename COLTYPE = int, typename VALTYPE = double, RowDotKernel Kernel = RowDotKernel::Scalar>
struct RowBalancedParallelSPMV
{
    RowBalancedParallelSPMV( const int num_threads = omp_get_max_threads() )
        : _nthreads{ num_threads }
    {
    }

    void setNumThreads( const int num_threads ) { _nthreads = num_threads; }

    template <BetaMode Mode, int Base>
    void compute( const COLTYPE size,
                  ROWTYPE const* __restrict ai,
                  COLTYPE const* __restrict aj,
                  VALTYPE const* __restrict av,
                  VALTYPE const* __restrict const b,
                  VALTYPE* __restrict const x,
                  const VALTYPE alpha,
                  const VALTYPE beta ) const
    {
#pragma omp parallel num_threads( _nthreads )
        {
            const int tid = omp_get_thread_num();
            const int nthreads = omp_get_num_threads();

            auto [start, end] = utils::LoadPrefixBalancedPartitionPos( ai, ai + size, tid, nthreads );

            for ( COLTYPE i = start; i < end; i++ )
            {
                VALTYPE val = DotRangeSIMD<Kernel, Base>( ai[i] - Base, ai[i + 1] - Base, aj, av, b );
                const VALTYPE ax = alpha * val;
                x[i] = apply_beta<Mode>( ax, beta, x[i] );
            }
        }
    }

    void operator()( const COLTYPE size,
                     const int base,
                     ROWTYPE const* __restrict ai,
                     COLTYPE const* __restrict aj,
                     VALTYPE const* __restrict av,
                     VALTYPE const* __restrict const b,
                     VALTYPE* __restrict const x,
                     const VALTYPE alpha,
                     const VALTYPE beta ) const
    {
        if ( base == 0 )
        {
            if ( beta == static_cast<VALTYPE>( 0 ) )
            {
                compute<BetaMode::Zero, 0>( size, ai, aj, av, b, x, alpha, beta );
            }
            else if ( beta == static_cast<VALTYPE>( 1 ) )
            {
                compute<BetaMode::One, 0>( size, ai, aj, av, b, x, alpha, beta );
            }
            else
            {
                compute<BetaMode::Generic, 0>( size, ai, aj, av, b, x, alpha, beta );
            }
        }
        else
        {
            if ( beta == static_cast<VALTYPE>( 0 ) )
            {
                compute<BetaMode::Zero, 1>( size, ai, aj, av, b, x, alpha, beta );
            }
            else if ( beta == static_cast<VALTYPE>( 1 ) )
            {
                compute<BetaMode::One, 1>( size, ai, aj, av, b, x, alpha, beta );
            }
            else
            {
                compute<BetaMode::Generic, 1>( size, ai, aj, av, b, x, alpha, beta );
            }
        }
    }

private:
    int _nthreads;
};

/// @brief Advanced Load-Balanced SPMV with workload-aware partitioning
/// @details Supports two workload partitioning modes:
///          - ALBUS: Simple nnz-based partitioning with element-level granularity
///          - CAMLB: Cache-aware workload partitioning based on memory access costs
///
/// @reference ALBUS mode:
///            "ALBUS: A Method for Load-Balancing of Sparse Matrix Vector Multiplication on GPUs"
///            Hartwig Anzt, et al.
///            Parallel Computing, 2020
///
/// @reference CAMLB mode:
///            "CAMLB-SpMV: A Cache-Aware Memory Load Balance Strategy for SpMV on Many-Core
///            Architectures" Xin He, Miao Wang, Haipeng Jia, Yunquan Zhang IEEE Transactions on
///            Parallel and Distributed Systems (TPDS), 2019 DOI: 10.1109/TPDS.2018.2878777
///
/// @tparam ROWTYPE Integer type for row pointers (e.g., int, int64_t)
/// @tparam COLTYPE Integer type for column indices (e.g., int, int64_t)
/// @tparam VALTYPE Value type for matrix elements (e.g., double, float)
/// @tparam Kernel Row dot product kernel: Scalar or Simd
/// @tparam WMode Workload partitioning mode: ALBUS or CAMLB
template <typename ROWTYPE = int, typename COLTYPE = int, typename VALTYPE = double, RowDotKernel Kernel = RowDotKernel::Scalar, WorkloadMode WMode = WorkloadMode::ALBUS>
class ALBUSSPMV
{
public:
    ALBUSSPMV( const int num_threads = omp_get_num_threads() ) : _nthreads{ num_threads } {}

    void setNumThreads( const int num_threads ) { _nthreads = num_threads; }

    void preprocess( const COLTYPE size, ROWTYPE const* __restrict ai, COLTYPE const* __restrict aj, VALTYPE const* __restrict av )
    {
        const int base = static_cast<int>( ai ? ai[0] : 0 );
        const ROWTYPE nnz = ai[size] - base;

        if ( nnz <= 0 )
        {
            _nthreads = 1;
        }
        else
        {
            if ( _nthreads <= 0 )
                _nthreads = 1;
            if ( _nthreads > nnz )
                _nthreads = static_cast<int>( nnz );
        }

        _threadBlockSizePrefix.assign( _nthreads + 1, ROWTYPE( 0 ) );
        _threadStartRow.assign( _nthreads + 1, COLTYPE( 0 ) );

        if constexpr ( WMode == WorkloadMode::CAMLB )
        {
            // Cache-aware workload partitioning
            std::vector<std::size_t> workload_prefix( nnz + 1 );

            // Compute cache-line based workload costs
            // Parameters: cache line = 64 bytes, L1 cache = 32KB -> 512 lines
            constexpr std::size_t cache_line_bytes = 64;
            constexpr std::size_t cache_lines = 512;

            compute_element_workload_prefix_hw<ROWTYPE, COLTYPE, VALTYPE>(
                size, ai, aj, av, nullptr, nullptr, cache_line_bytes, cache_lines, workload_prefix.data() );

            // Partition by workload cost instead of nnz
            const std::size_t total_work = workload_prefix[nnz];

            // Find partition points based on workload (can be done in parallel)
#pragma omp parallel for num_threads( _nthreads )
            for ( int i = 1; i <= _nthreads; i++ )
            {
                // Compute target work for this partition boundary
                // Use careful calculation to distribute work evenly, including remainder
                const std::size_t target_work = ( static_cast<std::size_t>( i ) * total_work ) / _nthreads;

                // Binary search to find element index with cumulative work >= target_work
                auto it = std::lower_bound( workload_prefix.begin(), workload_prefix.end(), target_work );
                ROWTYPE elem_idx = static_cast<ROWTYPE>( std::distance( workload_prefix.begin(), it ) );

                if ( elem_idx > nnz )
                    elem_idx = nnz;
                _threadBlockSizePrefix[i] = elem_idx;

                // Find row containing this element
                const ROWTYPE target = elem_idx + base;
                auto row_it = std::upper_bound( ai, ai + size + 1, target );
                ROWTYPE row = static_cast<ROWTYPE>( std::distance( ai, row_it ) ) - 1;

                if ( row < 0 )
                    row = 0;
                if ( row > static_cast<ROWTYPE>( size ) )
                    row = size;
                _threadStartRow[i] = static_cast<COLTYPE>( row );
            }

            // Ensure the last thread partition ends at exactly nnz
            _threadBlockSizePrefix[_nthreads] = nnz;
            _threadStartRow[_nthreads] = size;
        }
        else
        {
            // ALBUS: Simple nnz-based partitioning with element-level granularity
            // Distributes non-zero elements evenly across threads, ignoring memory access patterns
            // This provides better load balance than row-based partitioning for irregular matrices
            // but doesn't account for cache effects

            // Compute partition boundaries in parallel
#pragma omp parallel for num_threads( _nthreads )
            for ( int i = 1; i <= _nthreads; i++ )
            {
                // Use same even distribution formula as CAMLB: (i * total_work) / nthreads
                // This naturally distributes remainder across threads, max difference = 1
                const ROWTYPE target_nnz = ( static_cast<ROWTYPE>( i ) * nnz ) / _nthreads;
                _threadBlockSizePrefix[i] = target_nnz;

                // Find which row contains this element boundary
                const ROWTYPE target = target_nnz + base;
                auto it = std::upper_bound( ai, ai + size + 1, target );
                ROWTYPE row = static_cast<ROWTYPE>( std::distance( ai, it ) ) - 1;

                if ( row > static_cast<ROWTYPE>( size ) )
                    row = size;
                _threadStartRow[i] = static_cast<COLTYPE>( row );
            }
        }

        _threadBoundaryValue.assign( 2 * _nthreads, static_cast<VALTYPE>( 0 ) );
    }

    void operator()( const COLTYPE size,
                     ROWTYPE const* __restrict ai,
                     COLTYPE const* __restrict aj,
                     VALTYPE const* __restrict av,
                     const VALTYPE* __restrict const b,
                     VALTYPE* __restrict const x,
                     const VALTYPE alpha,
                     const VALTYPE beta ) const
    {
        const int base = static_cast<int>( ai ? ai[0] : 0 );
        if ( base == 0 )
        {
            if ( beta == static_cast<VALTYPE>( 0 ) )
            {
                compute<BetaMode::Zero, 0>( size, ai, aj, av, b, x, alpha, beta );
            }
            else if ( beta == static_cast<VALTYPE>( 1 ) )
            {
                compute<BetaMode::One, 0>( size, ai, aj, av, b, x, alpha, beta );
            }
            else
            {
                compute<BetaMode::Generic, 0>( size, ai, aj, av, b, x, alpha, beta );
            }
        }
        else
        {
            if ( beta == static_cast<VALTYPE>( 0 ) )
            {
                compute<BetaMode::Zero, 1>( size, ai, aj, av, b, x, alpha, beta );
            }
            else if ( beta == static_cast<VALTYPE>( 1 ) )
            {
                compute<BetaMode::One, 1>( size, ai, aj, av, b, x, alpha, beta );
            }
            else
            {
                compute<BetaMode::Generic, 1>( size, ai, aj, av, b, x, alpha, beta );
            }
        }
    }

private:
    template <BetaMode Mode, int Base>
    void compute( const COLTYPE size,
                  ROWTYPE const* __restrict ai,
                  COLTYPE const* __restrict aj,
                  VALTYPE const* __restrict av,
                  const VALTYPE* __restrict const b,
                  VALTYPE* __restrict const x,
                  const VALTYPE alpha,
                  const VALTYPE beta ) const
    {
        const ROWTYPE nnz = ai[size] - Base;
        if ( nnz == 0 )
        {
            if constexpr ( Mode == BetaMode::Zero )
            {
                std::fill( x, x + size, VALTYPE( 0 ) );
            }
            else if constexpr ( Mode == BetaMode::One )
            {
                // leave x untouched
            }
            else
            {
                if ( beta == VALTYPE( 0 ) )
                {
                    std::fill( x, x + size, VALTYPE( 0 ) );
                }
                else if ( beta != VALTYPE( 1 ) )
                {
#pragma omp simd
                    for ( COLTYPE i = 0; i < size; ++i )
                        x[i] *= beta;
                }
            }
            return;
        }

        std::memset( _threadBoundaryValue.data(), 0, sizeof( VALTYPE ) * 2 * _nthreads );
#pragma omp parallel num_threads( _nthreads )
        {
            const int tid = omp_get_thread_num();
            const int tidBegin = tid << 1;
            const int tidEnd = tidBegin | 1;
            const COLTYPE startRow = _threadStartRow[tid];
            const COLTYPE endRow = _threadStartRow[tid + 1];
            const ROWTYPE nzStart = _threadBlockSizePrefix[tid];
            const ROWTYPE nzEnd = _threadBlockSizePrefix[tid + 1];
            VALTYPE val;
            if ( startRow < endRow )
            {
                COLTYPE row = startRow;
                val = DotRangeSIMD<Kernel, Base>( nzStart, ai[startRow + 1] - Base, aj, av, b );
                _threadBoundaryValue[tidBegin] = alpha * val;

#pragma unroll( 32 )
                for ( row = startRow + 1; row < endRow; row++ )
                {
                    val = DotRangeSIMD<Kernel, Base>( ai[row] - Base, ai[row + 1] - Base, aj, av, b );
                    const VALTYPE ax = alpha * val;
                    x[row] = apply_beta<Mode>( ax, beta, x[row] );
                }

                val = DotRangeSIMD<Kernel, Base>( ai[endRow] - Base, nzEnd, aj, av, b );
                _threadBoundaryValue[tidEnd] = alpha * val;
            }
            else
            {
                val = DotRangeSIMD<Kernel, Base>( nzStart, nzEnd, aj, av, b );
                _threadBoundaryValue[tidBegin] = alpha * val;
            }
        }

        COLTYPE idx = std::numeric_limits<COLTYPE>::max();
#pragma unroll( 32 )
        for ( int tid = 0; tid < _nthreads; tid++ )
        {
            if ( _threadStartRow[tid] != idx )
            {
                idx = _threadStartRow[tid];
                x[idx] = apply_beta<Mode>( _threadBoundaryValue[2 * tid], beta, x[idx] );
            }
            else
            {
                x[idx] += _threadBoundaryValue[2 * tid];
            }
            if ( tid == _nthreads - 1 )
                break;
            if ( _threadStartRow[tid + 1] != idx )
            {
                idx = _threadStartRow[tid + 1];
                x[idx] = apply_beta<Mode>( _threadBoundaryValue[2 * tid + 1], beta, x[idx] );
            }
            else
            {
                x[idx] += _threadBoundaryValue[2 * tid + 1];
            }
        }
    }

    int _nthreads;
    mutable std::vector<ROWTYPE> _threadBlockSizePrefix;
    mutable std::vector<COLTYPE> _threadStartRow;
    mutable std::vector<VALTYPE> _threadBoundaryValue;
};
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, typename T>
constexpr bool spmv_has_preprocess = requires( const COLTYPE size,
                                               ROWTYPE const* __restrict ai,
                                               COLTYPE const* __restrict aj,
                                               VALTYPE const* __restrict av,
                                               T& t ) { t.preprocess( size, ai, aj, av ); };

/**
 * Sparse Matrix-Vector Multiplication (SPMV) operator
 * After preprocess(), the operator() can be called to perform SPMV
 * Usage:
 *  matrix_utils::SPMV<CSRMatrixType, SPMVType> spmv;
 *  spmv.setMatrix(&csr_matrix);
 *  spmv.preprocess();
 *  spmv(b, x, alpha, beta); // x = alpha * A * b + beta * x
 */
template <typename CSRMatrixType, typename SPMVType>
struct SPMV
{
    using ROWTYPE = typename CSRMatrixType::ROWTYPE;
    using COLTYPE = typename CSRMatrixType::COLTYPE;
    using VALTYPE = typename CSRMatrixType::VALTYPE;

    SPMV() : _matrix{ nullptr } {}

    void setMatrix( CSRMatrixType const* matrix ) { _matrix = matrix; }

    void preprocess()
    {
        if constexpr ( spmv_has_preprocess<ROWTYPE, COLTYPE, VALTYPE, SPMVType> )
        {
            _spmv.preprocess( _matrix->rows, _matrix->AI(), _matrix->AJ(), _matrix->AV() );
        }
    }
    COLTYPE size() const
    {
        if ( _matrix )
        {
            return _matrix->rows;
        }
        else
        {
            return 0;
        }
    }

    void operator()( const VALTYPE* __restrict const b,
                     VALTYPE* __restrict const x,
                     const VALTYPE alpha = 1.,
                     const VALTYPE beta = 0. ) const
    {
        if constexpr ( requires {
                           _spmv( _matrix->rows, _matrix->Base(), _matrix->AI(), _matrix->AJ(),
                                  _matrix->AV(), b, x, alpha, beta );
                       } )
        {
            _spmv( _matrix->rows, _matrix->Base(), _matrix->AI(), _matrix->AJ(), _matrix->AV(), b,
                   x, alpha, beta );
        }
        else if constexpr ( requires {
                                _spmv( _matrix->rows, _matrix->AI(), _matrix->AJ(), _matrix->AV(),
                                       b, x, alpha, beta );
                            } )
        {
            _spmv( _matrix->rows, _matrix->AI(), _matrix->AJ(), _matrix->AV(), b, x, alpha, beta );
        }
        else if constexpr ( requires { _spmv( b, x, alpha, beta ); } )
        {
            _spmv( b, x, alpha, beta );
        }
        else
        {
            static_assert( sizeof( SPMVType ) == 0, "SPMVType operator() signature not supported" );
        }
    }

    CSRMatrixType const* _matrix;
    SPMVType _spmv;
};

} // namespace matrix_utils
