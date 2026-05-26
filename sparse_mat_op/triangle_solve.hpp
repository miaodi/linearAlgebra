#pragma once

#include "BitVector.hpp"
#include "utils.h"
#include <chrono>
#include <execution>
#include <fstream>
#include <memory>
#include <numeric>
#include <algorithm>
#include <omp.h>
#include <span>
#include <tuple>
#include <type_traits>

#include "matrix_utils.hpp"
#include "graph_algs.hpp"
#include "permutation.hpp"
#include "spmv.hpp"

namespace matrix_utils
{

/// @brief Combined triangular solve function using TriangularMatrix enum with standard CSR format
/// @tparam TM TriangularMatrix::L for forward substitution, TriangularMatrix::U for backward substitution
/// @tparam ROWTYPE Row pointer type
/// @tparam COLTYPE Column index type
/// @tparam VALTYPE Value type
/// @param size Matrix dimension
/// @param ai Row pointers (ai[i] to ai[i+1]-1 are indices for row i, ai[0] is base)
/// @param aj Column indices
/// @param av Matrix values
/// @param diag Diagonal values (nullptr for unit diagonal)
/// @param b Right-hand side vector
/// @param x Solution vector
template <TriangularMatrix TM, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void TriangularSolve( const COLTYPE size,
                      ROWTYPE const* ai,
                      COLTYPE const* aj,
                      VALTYPE const* av,
                      VALTYPE const* diag,
                      VALTYPE const* const b,
                      VALTYPE* const x );

/// @brief Combined triangular solve function using TriangularMatrix enum with standard CSC format
/// @tparam TM TriangularMatrix::L for forward substitution, TriangularMatrix::U for backward substitution
/// @tparam ROWTYPE Column pointer type
/// @tparam COLTYPE Row index type
/// @tparam VALTYPE Value type
/// @param size Matrix dimension
/// @param col_ptr Column pointers (col_ptr[j] to col_ptr[j+1]-1 are indices for column j, col_ptr[0] is base)
/// @param row_idx Row indices
/// @param av Matrix values
/// @param diag Diagonal values (nullptr for unit diagonal)
/// @param b Right-hand side vector
/// @param x Solution vector
template <TriangularMatrix TM, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void TriangularSolveCSC( const COLTYPE size,
                         ROWTYPE const* col_ptr,
                         COLTYPE const* row_idx,
                         VALTYPE const* av,
                         VALTYPE const* diag,
                         VALTYPE const* const b,
                         VALTYPE* const x );

/// @brief Level-scheduled triangular substitution with OpenMP parallelization
/// @tparam TM TriangularMatrix::L for forward substitution, TriangularMatrix::U for backward substitution
/// @tparam ROWTYPE Row pointer type
/// @tparam COLTYPE Column index type
/// @tparam VALTYPE Value type
/// @param iperm Permutation array for level scheduling
/// @param prefix Level prefix array (prefix[l] to prefix[l+1]-1 are nodes in level l)
/// @param lvls Number of levels
/// @param rows Number of rows
/// @param ai Row pointers
/// @param aj Column indices
/// @param av Matrix values
/// @param diag Diagonal values (required for backward substitution, ignored for forward)
/// @param b Right-hand side vector
/// @param x Solution vector
template <TriangularMatrix TM, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
struct LevelScheduleTriangularSubstitution
{
    LevelScheduleTriangularSubstitution( const int num_threads = omp_get_max_threads() )
        : _nthreads{ num_threads }
    {
        if ( _nthreads < 1 )
            throw std::runtime_error( "Number of threads must be at least 1." );
    }

    void analysis( const COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj, VALTYPE const* av, VALTYPE const* diag = nullptr );

    void operator()( VALTYPE const* const b, VALTYPE* const x ) const;
    void set_num_threads( const int num_threads ) { _nthreads = num_threads; }

    int _nthreads;
    std::vector<COLTYPE> _iperm;
    std::vector<COLTYPE> _levelPrefix;
    COLTYPE _levels;
    graph::TopologicalSort2<ROWTYPE, COLTYPE, TM> _topSort;

    ROWTYPE const* _ai;
    COLTYPE const* _aj;
    VALTYPE const* _av;
    VALTYPE const* _diag{ nullptr };
    COLTYPE _size{ 0 };
};

template <TriangularMatrix TM, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
struct P2PTriangularSubstitution
{
    P2PTriangularSubstitution( const int num_threads = omp_get_max_threads(), const COLTYPE task_maximum_size = 128 )
        : _nthreads{ num_threads },
          _task_maximum_size{ task_maximum_size },
          _graphProjector{ num_threads },
          _transitiveReducer{ num_threads }
    {
        if ( _nthreads < 1 )
            throw std::runtime_error( "Number of threads must be at least 1." );
    }

    void analysis( const COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj, VALTYPE const* av, VALTYPE const* diag = nullptr );

    void operator()( VALTYPE const* const b, VALTYPE* const x ) const;

private:
    void computeLevelSchedule( const COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj );
    void buildTaskPartition( const COLTYPE size, ROWTYPE const* ai );
    void reportTaskPartitionSummary( ROWTYPE const* ai ) const;
    void verifyTaskMappings() const;
    COLTYPE buildTaskGraphs( const COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj );
    COLTYPE reduceTaskGraph( COLTYPE task_edges_before );
    void partitionTasksToThreads();
    void pruneThreadIntraEdgesFromReducedGraph();
    void createThreadLocalizedPermutation( const COLTYPE size );
    void reorderMatrixForCacheLocality( const COLTYPE size,
                                        ROWTYPE const* ai,
                                        COLTYPE const* aj,
                                        VALTYPE const* av,
                                        VALTYPE const* diag );
    void outputTaskGraphDebugInfo( COLTYPE task_edges_before, COLTYPE task_edges_after ) const;

private:
    int _nthreads;
    COLTYPE _task_maximum_size; // maximum node size for a task

    // Level scheduling data
    std::vector<COLTYPE> _iperm;
    std::vector<COLTYPE> _levelPrefix;
    COLTYPE _levels;

    // Task partitioning data
    std::vector<COLTYPE> _taskPrefix; // CSR row pointer for task-to-node mapping (task -> node indices)
    std::vector<COLTYPE> _taskToNode; // CSR column indices for task-to-node mapping
    std::vector<COLTYPE> _taskToReorderedNode; // CSR column indices for task-to-node mapping after reordering
    std::vector<COLTYPE> _taskToLevel;     // Map from task ID to level
    std::vector<COLTYPE> _levelTaskPrefix; // Prefix of tasks for each level, base = 0
    std::vector<COLTYPE> _nodeToTask;      // Map from node index to task ID, base = 0
    CSRStructVec<ROWTYPE, COLTYPE> _taskInGraph; // Task dependency graph (projected from original graph) (base = 0)
    CSRStructVec<ROWTYPE, COLTYPE> _taskOutGraph; // Transpose of task dependency graph (base = 0)
    CSRStructVec<ROWTYPE, COLTYPE> _taskOutGraphTransitiveReduced; // Transitive reduction of task out-graph (base = 0)
    CSRStructVec<ROWTYPE, COLTYPE> _taskOutGraphIntraReduced; // Task out-graph with intra-thread edges removed (base = 0)
    CSRStructVec<ROWTYPE, COLTYPE> _taskInGraphIntraReduced; // Transpose of intra-thread pruned task graph (base = 0)
    std::vector<COLTYPE> _taskPartition; // Partition assignment for each task (0 to _nthreads-1)

    // Thread-task mapping: _threadTasks[i] contains all task IDs assigned to thread i
    std::vector<std::vector<COLTYPE>> _threadTasks;

    // Cache-optimized matrix reordering for thread locality
    std::vector<COLTYPE> _threadLocalizedRowPerm;    // Row permutation: original_row -> new_row
    std::vector<COLTYPE> _threadLocalizedRowInvPerm; // Inverse permutation: new_row -> original_row
    CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE> _reorderedMatrix; // Matrix reordered for thread locality
    std::vector<VALTYPE> _reorderedDiag;                     // Diagonal values for reordered matrix
    mutable std::unique_ptr<std::atomic<bool>[]> _taskReady; // Task readiness flags
    mutable COLTYPE _taskReadyCapacity{ 0 };

    COLTYPE _totalTasks;

    graph::TopologicalSort2<ROWTYPE, COLTYPE, TM> _topSort;
    graph::ProjectGraphToTaskGraph<ROWTYPE, COLTYPE> _graphProjector; // Reusable graph projection instance
    graph::TransitiveReduction<ROWTYPE, COLTYPE> _transitiveReducer; // Reusable transitive reduction instance

private:
    std::vector<std::vector<COLTYPE>> _taskScratch; // temporary workspace (zero-based neighbors per task)
};

// Jacobi iteration based triangular substitution using a pluggable SPMV
// Works for both L and U by treating the input as a general sparse matrix
// and performing Jacobi iterations: x^{k+1} = D^{-1} (b - (A - D) x^{k})
// If diag == nullptr, assumes unit diagonal.
template <TriangularMatrix TM, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
struct JacobiTriangularSubstitution
{
    // Default SPMV type: ALBUSSPMV with SIMD kernel
    using SpmvType = ALBUSSPMV<ROWTYPE, COLTYPE, VALTYPE, RowDotKernel::Simd>;

    JacobiTriangularSubstitution( const int num_threads = omp_get_num_threads(),
                                  const int max_iters = 100,
                                  const VALTYPE tol = static_cast<VALTYPE>( 1e-10 ) )
        : _nthreads{ num_threads }, _max_iters{ max_iters }, _tol{ tol }
    {
        if ( _nthreads < 1 )
            throw std::runtime_error( "Number of threads must be at least 1." );
        if ( _max_iters < 1 )
            throw std::runtime_error( "Max iterations must be at least 1." );
        if ( _tol < static_cast<VALTYPE>( 0 ) )
            throw std::runtime_error( "Tolerance must be non-negative." );
    }

    void set_num_threads( const int num_threads ) { _nthreads = num_threads; }
    void set_max_iters( const int iters ) { _max_iters = iters; }
    void set_tol( const VALTYPE tol ) { _tol = tol; }

    bool analysis( const COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj, VALTYPE const* av, VALTYPE const* diag = nullptr );

    bool operator()( VALTYPE const* const b, VALTYPE* const x ) const;

private:
    int _nthreads;
    int _max_iters;
    VALTYPE _tol;

    VALTYPE const* _diag{ nullptr };

    mutable std::vector<VALTYPE> _dinv;

    CSRMatrixRawPtr<ROWTYPE, COLTYPE, VALTYPE> _mat;
    SPMV<CSRMatrixRawPtr<ROWTYPE, COLTYPE, VALTYPE>, SpmvType> _spmv;
};

enum class FBSubstitutionType
{
    Barrier,
    NoBarrier,
    NoBarrierSuperNode
};

template <FBSubstitutionType FBST = FBSubstitutionType::Barrier, TriangularMatrix TS = TriangularMatrix::L, typename ROWTYPE = int, typename COLTYPE = int, typename VALTYPE = double>
class OptimizedTriangularSolve
{
public:
    OptimizedTriangularSolve( const int num_threads = omp_get_num_threads() )
        : _nthreads{ num_threads }
    {
        if ( _nthreads < 1 )
            throw std::runtime_error( "Number of threads must be at least 1." );
    }

    void analysis( const COLTYPE rows,
                   const int base,
                   ROWTYPE const* ai,
                   COLTYPE const* aj,
                   VALTYPE const* av,
                   VALTYPE const* diag = nullptr );

    void operator()( const VALTYPE* const b, VALTYPE* const x ) const;

    void BarrierOp( const VALTYPE* const b, VALTYPE* const x ) const;

    void NoBarrierOp( const VALTYPE* const b, VALTYPE* const x ) const;

    void NoBarrierSuperNodeOp( const VALTYPE* const b, VALTYPE* const x ) const;

    void build_task_graph();

    int get_num_threads() const { return _nthreads; }

protected:
    int _nthreads;
    COLTYPE _size;
    std::vector<COLTYPE> _iperm;
    std::vector<COLTYPE> _levelPrefix;
    mutable std::vector<double> _vec;

    COLTYPE _levels;
    std::vector<std::vector<COLTYPE>> _threadlevels; // level prefix for each thread, zero based
    std::vector<COLTYPE> _threadiperm;
    std::vector<COLTYPE> _threadperm;
    CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE> _reorderedMat;
    VALTYPE const* _diag{ nullptr };

    // super node level scheduling data
    // always zero based
    COLTYPE _tasks;
    CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE> _taskAdjGraph;    // children
    CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE> _taskInvAdjGraph; // parents
    CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE> _taskInvAdjGraph2; // parents after transisive edge removal
    std::vector<COLTYPE> _threadTaskPrefix;                    // tasks on each thread
    std::vector<COLTYPE> _taskBoundaryPrefix; // num of rows in each task size task + 1
    std::vector<ROWTYPE> _threadPrefixSum;    //
    // std::vector<ROWTYPE> _threadPrefixSum2; //
    std::vector<COLTYPE> _reorderedRowIdToTaskId;
    std::vector<std::vector<COLTYPE>> _taskInvAdj; // thread local
    std::vector<COLTYPE> _transitiveEdgeRemoveAj;

    mutable utils::BitVector<COLTYPE> _bv;

    // debugging
    // std::vector<COLTYPE> taskSizes;
};
} // namespace matrix_utils
