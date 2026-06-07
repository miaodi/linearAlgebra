#pragma once

#include "circularbuffer.hpp"
#include "matrix_utils.hpp"
#include "sparse_mat_traits.hpp"
#include <deque>
#include <forward_list>
#include <limits>
#include <map>
#include <omp.h>
#include <ranges>
#include <unordered_map>
#include <vector>

// for all preconditioner operators, assuming the diagonal entries are filled.
// In other words, a zero value should be provided for the diagonal entries if
// it is a void entry in A.
namespace matrix_utils
{

// ICC helper declarations (definitions in precond.cpp)
template <ResizableCSR CSRMatrixType>
void ICCLevel0SymSymbolic( const typename CSRMatrixType::COLTYPE size,
                           typename CSRMatrixType::ROWTYPE const* ai,
                           typename CSRMatrixType::COLTYPE const* aj,
                           CSRMatrixType& icc );

template <typename CSRMatrixType>
void ICCLevelSymbolic0( const typename CSRMatrixType::COLTYPE size,
                        typename CSRMatrixType::ROWTYPE const* ai,
                        typename CSRMatrixType::COLTYPE const* aj,
                        typename CSRMatrixType::COLTYPE const* diag_pos,
                        const int lvl,
                        CSRMatrixType& icc );

template <ResizableCSR CSRMatrixType>
void ICCLevelSymbolic1( const typename CSRMatrixType::COLTYPE size,
                        typename CSRMatrixType::ROWTYPE const* ai,
                        typename CSRMatrixType::COLTYPE const* aj,
                        typename CSRMatrixType::COLTYPE const* diag_pos,
                        const int lvl,
                        CSRMatrixType& icc );

template <ResizableCSR CSRMatrixType>
void ICCLevelSymbolic2( const typename CSRMatrixType::COLTYPE size,
                        typename CSRMatrixType::ROWTYPE const* ai,
                        typename CSRMatrixType::COLTYPE const* aj,
                        typename CSRMatrixType::COLTYPE const* diag_pos,
                        const int lvl,
                        CSRMatrixType& icc );

template <ResizableCSR CSRMatrixType>
void ICCLevelSymbolic3( const typename CSRMatrixType::COLTYPE size,
                        typename CSRMatrixType::ROWTYPE const* ai,
                        typename CSRMatrixType::COLTYPE const* aj,
                        typename CSRMatrixType::COLTYPE const* diag_pos,
                        const int lvl,
                        CSRMatrixType& icc );

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
bool ICCLevelNumeric( const COLTYPE size,
                      ROWTYPE const* ai,
                      COLTYPE const* aj,
                      VALTYPE const* av,
                      COLTYPE const* diag_pos,
                      const int lvl,
                      const VALTYPE omega,
                      ROWTYPE const* icc_ai,
                      COLTYPE const* icc_aj,
                      VALTYPE* icc_av );

template <ResizableDiagonal CSRMatrixType>
class ICCLevelSymbolicParallel
{
public:
    using COLTYPE = typename CSRMatrixType::COLTYPE;
    using ROWTYPE = typename CSRMatrixType::ROWTYPE;

    ICCLevelSymbolicParallel( const int num_threads )
        : _num_threads( num_threads ),
          _Li_path_max( num_threads ),
          _visited( num_threads ),
          _L( num_threads ),
          _Q( num_threads ),
          _Q_next( num_threads )
    {
    }

    bool operator()( const COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj, const int lvl, CSRMatrixType& L );

private:
    int _num_threads;
    std::vector<std::vector<COLTYPE>> _Li_path_max; //
    std::vector<std::vector<COLTYPE>> _visited;
    std::vector<std::vector<COLTYPE>> _L;
    std::vector<std::unordered_map<COLTYPE, COLTYPE>> _Q;
    std::vector<std::unordered_map<COLTYPE, COLTYPE>> _Q_next;
};

template <ResizableDiagonal CSRMatrixType>
class ICCLevelNumericFixedPoint
{
public:
    using COLTYPE = typename CSRMatrixType::COLTYPE;
    using ROWTYPE = typename CSRMatrixType::ROWTYPE;
    using VALTYPE = typename CSRMatrixType::VALTYPE;

    ICCLevelNumericFixedPoint( const int num_threads ) : _num_threads( num_threads ) {}

    bool operator()( const COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj, VALTYPE const* av, CSRMatrixType& L );

private:
    int _num_threads;                // number of threads to use in parallel region
    int _sweeps{ 100 };              // number of sweeps to perform
    std::vector<VALTYPE> _av;        // av in L's sparsity pattern
    std::vector<COLTYPE> _ai;        // ai in COO format for L's sparsity pattern
    std::vector<VALTYPE> _L_av_init; // initial guess for L's av
    std::vector<VALTYPE> _L_av_next; // next iteration's L's av after a sweep
};

template <ResizableDiagonal CSRMatrixType>
bool ILULevel0Symbolic( const typename CSRMatrixType::COLTYPE size,
                        typename CSRMatrixType::ROWTYPE const* ai,
                        typename CSRMatrixType::COLTYPE const* aj,
                        CSRMatrixType& ilu );

template <ResizableDiagonal CSRMatrixType>
class ILULevelSymbolic
{
public:
    ILULevelSymbolic() = default;
    bool operator()( const typename CSRMatrixType::COLTYPE size,
                     typename CSRMatrixType::ROWTYPE const* ai,
                     typename CSRMatrixType::COLTYPE const* aj,
                     const int lvl,
                     CSRMatrixType& ilu );

private:
    // Local (col, level) pair used during symbolic pattern construction of a row
    struct ColLevel
    {
        typename CSRMatrixType::COLTYPE col;
        int level;
    };
    std::vector<int> _levels; // level for each element
    // marker array for O(1) membership / position lookup in current row (MAX sentinel if absent)
    std::vector<typename CSRMatrixType::ROWTYPE> _marker;
    // Reusable storage to avoid per-row allocations
    std::vector<ColLevel> _cl;
    std::deque<typename CSRMatrixType::COLTYPE> _q; // queue of pivot candidates < i
};

// GS-Urow ILU(k) symbolic factorization with parallel U-row construction
// New sequential and scalable parallel algorithms for incomplete LU factor preconditioning
// Hysom 2001
template <ResizableDiagonal CSRMatrixType, bool keepdiag = false>
class ILULevelSymbolicParallelU
{
public:
    using COLTYPE = typename CSRMatrixType::COLTYPE;
    using ROWTYPE = typename CSRMatrixType::ROWTYPE;
    ILULevelSymbolicParallelU( const int nthreads )
        : _nthreads( nthreads ), _visited( nthreads ), _Q( nthreads ), _Q_next( nthreads )
    {
    }

    bool operator()( const typename CSRMatrixType::COLTYPE size,
                     typename CSRMatrixType::ROWTYPE const* ai,
                     typename CSRMatrixType::COLTYPE const* aj,
                     const int lvl,
                     CSRMatrixType& U );

private:
    int _nthreads;
    std::vector<std::vector<COLTYPE>> _visited;
    std::vector<std::vector<COLTYPE>> _Q;
    std::vector<std::vector<COLTYPE>> _Q_next;
    std::vector<std::vector<COLTYPE>> _U;

    ROWTYPE BuildURow( const COLTYPE i,
                       ROWTYPE const* ai,
                       COLTYPE const* aj,
                       const int lvl,
                       const COLTYPE base,
                       std::vector<COLTYPE>& visited_thread,
                       std::vector<COLTYPE>& Q_thread,
                       std::vector<COLTYPE>& Q_next_thread );
};

// GS-Lrow ILU(k) symbolic factorization with parallel L-row construction
// Supports L and LU variants via TriangularMatrix (LU requires keepdiag = true).
template <ResizableDiagonal CSRMatrixType, enums::matrix_utils::TriangularMatrix Triangular = enums::matrix_utils::L, bool keepdiag = false>
class ILULevelSymbolicParallel
{
public:
    using COLTYPE = typename CSRMatrixType::COLTYPE;
    using ROWTYPE = typename CSRMatrixType::ROWTYPE;

    struct NodeInfo
    {
        COLTYPE index;
        COLTYPE peak;
    };

    ILULevelSymbolicParallel( const int nthreads )
        : _nthreads( nthreads ), _visited( nthreads ), _Q( nthreads ), _Q_next( nthreads )
    {
    }

    bool operator()( const typename CSRMatrixType::COLTYPE size,
                     typename CSRMatrixType::ROWTYPE const* ai,
                     typename CSRMatrixType::COLTYPE const* aj,
                     const int lvl,
                     CSRMatrixType& ILU );

private:
    static_assert( Triangular == enums::matrix_utils::L || Triangular == enums::matrix_utils::LU,
                   "ILULevelSymbolicParallel supports L and LU only" );
    static_assert( Triangular != enums::matrix_utils::LU || keepdiag,
                   "ILULevelSymbolicParallel with LU requires keepdiag = true" );

    int _nthreads;
    // Per-thread scratch buffers (indexed by thread id) to avoid synchronization.
    std::vector<std::vector<NodeInfo>> _visited;
    std::vector<std::vector<NodeInfo>> _Q;
    std::vector<std::vector<NodeInfo>> _Q_next;
    // Per-thread row outputs before final assembly.
    std::vector<std::vector<COLTYPE>> _L;
    std::vector<std::vector<COLTYPE>> _U;

    ROWTYPE BuildRow( const COLTYPE i,
                      ROWTYPE const* ai,
                      COLTYPE const* aj,
                      const int lvl,
                      const COLTYPE base,
                      std::vector<NodeInfo>& visited_thread,
                      std::vector<NodeInfo>& Q_thread,
                      std::vector<NodeInfo>& Q_next_thread );
};

// Gustavson-style ILU(k) symbolic factorization.
// V2 builds each row as repeated Op * A products: an inner K-way merge coalesces
// row expansion candidates, then an outer merge applies those candidates to the
// sorted visited list for the current row.
template <ResizableDiagonal CSRMatrixType, enums::matrix_utils::TriangularMatrix Triangular = enums::matrix_utils::L, bool keepdiag = false>
class ILULevelSymbolicParallelV2
{
public:
    using COLTYPE = typename CSRMatrixType::COLTYPE;
    using ROWTYPE = typename CSRMatrixType::ROWTYPE;

    struct NodeInfo
    {
        COLTYPE index;
        COLTYPE peak;
    };

    ILULevelSymbolicParallelV2( const int nthreads )
        : _nthreads( nthreads ),
          _visited( nthreads ),
          _visited_next( nthreads ),
          _op( nthreads ),
          _op_next( nthreads ),
          _candidates( nthreads ),
          _merge_cursors( nthreads )
    {
    }

    bool apply( const typename CSRMatrixType::COLTYPE size,
                typename CSRMatrixType::ROWTYPE const* ai,
                typename CSRMatrixType::COLTYPE const* aj,
                const int lvl,
                CSRMatrixType& ILU );

private:
    static_assert( Triangular == enums::matrix_utils::L || Triangular == enums::matrix_utils::LU,
                   "ILULevelSymbolicParallelV2 supports L and LU only" );
    static_assert( Triangular != enums::matrix_utils::LU || keepdiag,
                   "ILULevelSymbolicParallelV2 with LU requires keepdiag = true" );

    struct MergeCursor
    {
        // Candidate emitted by the current cursor position in A(row, :).
        COLTYPE index;
        // Candidate peak after semiring multiplication: max(source_peak, index).
        COLTYPE peak;
        // CSR range currently scanned by this cursor.
        ROWTYPE pos;
        ROWTYPE end;
        // Peak carried by the Op/frontier node that owns this cursor.
        COLTYPE source_peak;
    };

    int _nthreads;
    // Per-thread scratch buffers; no synchronization is needed while building rows.
    std::vector<std::vector<NodeInfo>> _visited;
    std::vector<std::vector<NodeInfo>> _visited_next;
    std::vector<std::vector<NodeInfo>> _op;
    std::vector<std::vector<NodeInfo>> _op_next;
    std::vector<std::vector<NodeInfo>> _candidates;
    std::vector<std::vector<MergeCursor>> _merge_cursors;
    // Per-row outputs assembled after prefix sum.
    std::vector<std::vector<COLTYPE>> _L;
    std::vector<std::vector<COLTYPE>> _U;

    ROWTYPE BuildRow( const COLTYPE i,
                      ROWTYPE const* ai,
                      COLTYPE const* aj,
                      const int lvl,
                      const COLTYPE base,
                      std::vector<NodeInfo>& visited,
                      std::vector<NodeInfo>& visited_next,
                      std::vector<NodeInfo>& op,
                      std::vector<NodeInfo>& op_next,
                      std::vector<NodeInfo>& candidates,
                      std::vector<MergeCursor>& merge_cursors );

    void BuildMergedCandidates( const COLTYPE i,
                                ROWTYPE const* ai,
                                COLTYPE const* aj,
                                const COLTYPE base,
                                std::vector<NodeInfo> const& op,
                                std::vector<NodeInfo>& candidates,
                                std::vector<MergeCursor>& merge_cursors ) const;

    void MergeCandidatesWithVisited( const COLTYPE i,
                                     std::vector<NodeInfo> const& candidates,
                                     std::vector<NodeInfo> const& visited,
                                     std::vector<NodeInfo>& visited_next,
                                     std::vector<NodeInfo>& op_next ) const;
};

template <ResizableDiagonal CSRMatrixType>
bool ILUNumeric( const typename CSRMatrixType::COLTYPE size,
                 typename CSRMatrixType::ROWTYPE const* ai,
                 typename CSRMatrixType::COLTYPE const* aj,
                 typename CSRMatrixType::VALTYPE const* av,
                 CSRMatrixType& ilu );

template <ResizableDiagonal CSRMatrixType>
bool ILUTNumeric( const typename CSRMatrixType::COLTYPE size,
                  typename CSRMatrixType::ROWTYPE const* ai,
                  typename CSRMatrixType::COLTYPE const* aj,
                  typename CSRMatrixType::VALTYPE const* av,
                  const typename CSRMatrixType::VALTYPE tau,
                  CSRMatrixType& ilu );

template <typename VT>
class IdentityPrec
{
public:
    using VALTYPE = VT;
    IdentityPrec( const std::size_t size ) : _size( size ) {}

    std::size_t size() const { return _size; }

    bool operator()( VALTYPE const* const b, VALTYPE* const x ) const
    {
        for ( std::size_t i = 0; i < _size; i++ )
        {
            x[i] = b[i];
        }
        return true;
    }
    std::size_t _size;
};

// Jacobi (diagonal) preconditioner: x = D^{-1} b
// Requires diagonal entries present in the matrix; zeros are treated as 1.0 to avoid division by zero.
template <ResizableDiagonal CSRMatrixType>
class JacobiPrec
{
public:
    using ROWTYPE = typename CSRMatrixType::ROWTYPE;
    using COLTYPE = typename CSRMatrixType::COLTYPE;
    using VALTYPE = typename CSRMatrixType::VALTYPE;

    JacobiPrec( const CSRMatrixType& A, int nthreads = omp_get_max_threads() )
        : _n( A.rows ), _invD( A.rows ), _nthreads( nthreads )
    {
        // Build inverse diagonal using utility function Diagonal
        // Ask Diagonal to compute the inverted diagonal directly (invert=true)
        const bool ok = matrix_utils::Diagonal(
            A.rows, A.AI(), A.AJ(), A.AV(), static_cast<ROWTYPE*>( nullptr ), _invD.data(), true );
    }
    COLTYPE size() const { return _n; }

    bool operator()( VALTYPE const* const b, VALTYPE* const x ) const
    {
        // Apply inverse diagonal
#pragma omp parallel for num_threads( _nthreads )
        for ( COLTYPE i = 0; i < _n; ++i )
        {
            x[i] = _invD[i] * b[i];
        }
        return true;
    }

private:
    COLTYPE _n;
    std::vector<VALTYPE> _invD;
    int _nthreads;
};
} // namespace matrix_utils
