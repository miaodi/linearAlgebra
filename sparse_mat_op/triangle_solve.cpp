#include "triangle_solve.hpp"
#include "config.h"
#include "BitVector.hpp"
#include "utils.h"
#include <atomic>
#include <chrono>
#include <execution>
#include <fstream>
#include <memory>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <omp.h>
#include <tuple>
#include <type_traits>
#include <thread>

#include "matrix_utils.hpp"
#include "permutation.hpp"
#include "io.hpp"

namespace matrix_utils
{
/// @brief Combined triangular solve function using TriangularMatrix enum with standard CSR format
template <TriangularMatrix TM, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void TriangularSolve( const COLTYPE size,
                      ROWTYPE const* ai,
                      COLTYPE const* aj,
                      VALTYPE const* av,
                      VALTYPE const* diag,
                      VALTYPE const* const b,
                      VALTYPE* const x )
{
    // Extract base from ai[0]
    const ROWTYPE base = ai[0];

    // Lambda to process a single row - eliminates code duplication
    auto process_row = [&]( COLTYPE row_idx )
    {
        VALTYPE val = VALTYPE( 0 );

        for ( ROWTYPE j = ai[row_idx] - base; j < ai[row_idx + 1] - base; j++ )
        {
            COLTYPE col_idx = aj[j] - base;

            // For forward substitution (L), only use strict lower triangular elements (col < row)
            // For backward substitution (U), only use strict upper triangular elements (col > row)
            if constexpr ( TM == TriangularMatrix::L )
            {
                if ( col_idx < row_idx ) // Only lower triangular part
                {
                    val += av[j] * x[col_idx];
                }
            }
            else // TM == TriangularMatrix::U
            {
                if ( col_idx > row_idx ) // Only upper triangular part
                {
                    val += av[j] * x[col_idx];
                }
            }
        }

        // Apply diagonal
        if ( diag )
        {
            x[row_idx] = ( b[row_idx] - val ) / diag[row_idx];
        }
        else
        {
            x[row_idx] = b[row_idx] - val; // Unit diagonal
        }
    };

    // Use different loop strategies for forward vs backward to handle unsigned types
    if constexpr ( TM == TriangularMatrix::L ) // Forward substitution
    {
        for ( COLTYPE i = 0; i < size; i++ )
        {
            process_row( i );
        }
    }
    else // Backward substitution (TM == TriangularMatrix::U)
    {
        for ( COLTYPE i = size; i > 0; i-- )
        {
            process_row( i - 1 );
        }
    }
}


/// @brief Combined triangular solve function using TriangularMatrix enum with standard CSC format
template <TriangularMatrix TM, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void TriangularSolveCSC( const COLTYPE size,
                         ROWTYPE const* col_ptr,
                         COLTYPE const* row_idx,
                         VALTYPE const* av,
                         VALTYPE const* diag,
                         VALTYPE const* const b,
                         VALTYPE* const x )
{
    const ROWTYPE base = col_ptr[0];
    std::copy( b, b + size, x );

    auto process_column = [&]( const COLTYPE col )
    {
        if ( diag )
        {
            x[col] /= diag[col];
        }

        const VALTYPE x_col = x[col];
        for ( ROWTYPE p = col_ptr[col] - base; p < col_ptr[col + 1] - base; ++p )
        {
            const COLTYPE row = row_idx[p] - base;
            if constexpr ( TM == TriangularMatrix::L )
            {
                if ( row > col )
                {
                    x[row] -= av[p] * x_col;
                }
            }
            else // TM == TriangularMatrix::U
            {
                if ( row < col )
                {
                    x[row] -= av[p] * x_col;
                }
            }
        }
    };

    if constexpr ( TM == TriangularMatrix::L )
    {
        for ( COLTYPE col = 0; col < size; ++col )
        {
            process_column( col );
        }
    }
    else // TM == TriangularMatrix::U
    {
        for ( COLTYPE col = size; col > 0; --col )
        {
            process_column( col - 1 );
        }
    }
}


template <TriangularMatrix TM, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void LevelScheduleTriangularSubstitution<TM, ROWTYPE, COLTYPE, VALTYPE>::analysis(
    const COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj, VALTYPE const* av, VALTYPE const* diag )
{
    if constexpr ( TM == TriangularMatrix::U )
    {
        assert( diag != nullptr &&
                "Diagonal must be provided for backward substitution." );
    }

    _ai = ai;
    _aj = aj;
    _av = av;
    _diag = diag;
    _size = size;

    const auto base = _ai[0];
    _iperm.resize(_size);
    _levelPrefix.resize(_size + 1);
    _levels = _topSort( _size, _ai, _aj, _iperm.data(), _levelPrefix.data() );
}


// operator() runtime execution for level-scheduled triangular substitution
// Applies previously computed permutation (_iperm) and level prefixes
// to perform forward or backward substitution with optional diagonal.
template <TriangularMatrix TM, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void LevelScheduleTriangularSubstitution<TM, ROWTYPE, COLTYPE, VALTYPE>::operator()(
    VALTYPE const* const b, VALTYPE* const x ) const
{
    const auto base = _ai[0];

#pragma omp parallel num_threads( _nthreads )
    {
        for ( COLTYPE lvl = 0; lvl < _levels; ++lvl )
        {
#pragma omp for
            for ( COLTYPE p = _levelPrefix[lvl]; p < _levelPrefix[lvl + 1]; ++p )
            {
                const COLTYPE row = _iperm[p] - base; // logical row index
                VALTYPE accum = VALTYPE( 0 );
                // Traverse adjacency of the original matrix for this row.
                for ( auto jj = _ai[row] - base; jj < _ai[row + 1] - base; ++jj )
                {
                    const COLTYPE col = _aj[jj] - base;
                    // For L: use strictly lower entries; For U: use strictly upper entries.
                    if constexpr ( TM == TriangularMatrix::L )
                    {
                        if ( col < row )
                            accum += _av[jj] * x[col];
                    }
                    else // U
                    {
                        if ( col > row )
                            accum += _av[jj] * x[col];
                    }
                }
                // Diagonal handling (same formula for L and U after accum defined by direction)
                x[row] = _diag ? ( b[row] - accum ) / _diag[row] : ( b[row] - accum );
            }
#pragma omp barrier
        }
    }
}


// JacobiTriangularSubstitution moved from header: analysis and operator()
template <TriangularMatrix TM, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
bool JacobiTriangularSubstitution<TM, ROWTYPE, COLTYPE, VALTYPE>::analysis(
    const COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj, VALTYPE const* av,
    VALTYPE const* diag)
{
    if (size < static_cast<COLTYPE>(0) || ai == nullptr || aj == nullptr || av == nullptr)
        return false;

    _diag = diag;

    if constexpr (TM == TriangularMatrix::U)
    {
        if (_diag == nullptr)
        {
            return false;
        }
        for (COLTYPE i = 0; i < size; ++i)
        {
            if (_diag[i] == static_cast<VALTYPE>(0))
                return false;
        }
    }

    _mat.rows = size;
    _mat.cols = size;
    _mat.ai = ai;
    _mat.aj = aj;
    _mat.av = av;
    _spmv.setMatrix(&_mat);
    _spmv.preprocess();

    if constexpr (TM == TriangularMatrix::U)
    {
        _dinv.resize(_mat.rows);
#pragma omp parallel for num_threads(_nthreads)
        for (COLTYPE i = 0; i < _mat.rows; ++i)
        {
            _dinv[i] = static_cast<VALTYPE>(1) / _diag[i];
        }
    }

    return true;
}

template <TriangularMatrix TM, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
bool JacobiTriangularSubstitution<TM, ROWTYPE, COLTYPE, VALTYPE>::operator()(VALTYPE const* const b, VALTYPE* const x) const
{
    if (_mat.rows == 0)
        return true;

#pragma omp parallel for num_threads(_nthreads)
    for (COLTYPE i = 0; i < _mat.rows; ++i)
        x[i] = static_cast<VALTYPE>(0);

    std::vector<VALTYPE> y(_mat.rows);
    std::vector<VALTYPE> xnew(_mat.rows);

    for (int k = 0; k < _max_iters; ++k)
    {
        _spmv(x, y.data(), static_cast<VALTYPE>(1), static_cast<VALTYPE>(0));

        if constexpr (TM == TriangularMatrix::U)
        {
#pragma omp parallel for num_threads(_nthreads)
            for (COLTYPE i = 0; i < _mat.rows; ++i)
            {
                xnew[i] = (b[i] - y[i]) * _dinv[i];
            }
        }
        else
        {
#pragma omp parallel for num_threads(_nthreads)
            for (COLTYPE i = 0; i < _mat.rows; ++i)
            {
                xnew[i] = b[i] - y[i];
            }
        }

        VALTYPE diff_max = static_cast<VALTYPE>(0);
#pragma omp parallel for reduction(max : diff_max) num_threads(_nthreads)
        for (COLTYPE i = 0; i < _mat.rows; ++i)
        {
            VALTYPE d = std::abs(xnew[i] - x[i]);
            if (d > diff_max)
                diff_max = d;
            x[i] = xnew[i];
        }
        if (diff_max <= _tol)
            break;
    }
    return true;
}

template <TriangularMatrix TM, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void P2PTriangularSubstitution<TM, ROWTYPE, COLTYPE, VALTYPE>::analysis(
    const COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj, VALTYPE const* av, VALTYPE const* diag )
{
    if constexpr ( TM == TriangularMatrix::U )
    {
        assert( diag != nullptr &&
                "Diagonal must be provided for backward substitution." );
    }

    computeLevelSchedule( size, ai, aj );

    _taskPrefix.clear();
    _taskToLevel.clear();
    _taskToNode.clear();
    _nodeToTask.resize( size );
    _levelTaskPrefix.resize( _levels + 1 );
    _taskPrefix.push_back( 0 );
    _threadTasks.resize( _nthreads );
    for(int i = 0;i<_nthreads;++i)
    {
      _threadTasks[i].clear();
    }

    _totalTasks = 0;

    buildTaskPartition( size, ai );
    reportTaskPartitionSummary( ai );
    verifyTaskMappings();

    const auto task_edges = buildTaskGraphs( size, ai, aj );
    reduceTaskGraph( task_edges );
    partitionTasksToThreads();
    pruneThreadIntraEdgesFromReducedGraph();
    
    if ( _taskOutGraphIntraReduced.ai.empty() )
    {
        throw std::runtime_error("Error: Task graph intra-reduced adjacency array is empty after pruning. "
                               "This indicates a failure in the pruning operation.");
    }
    
    const auto pruned_edges = static_cast<COLTYPE>( _taskOutGraphIntraReduced.ai[_totalTasks] );
    createThreadLocalizedPermutation( size );
    reorderMatrixForCacheLocality( size, ai, aj, av, diag );
    outputTaskGraphDebugInfo( task_edges, pruned_edges );
}

template <TriangularMatrix TM, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void P2PTriangularSubstitution<TM, ROWTYPE, COLTYPE, VALTYPE>::operator()(
    VALTYPE const* const b, VALTYPE* const x ) const
{
    const COLTYPE size = _reorderedMatrix.rows;
    if ( size == 0 )
        return;

    const ROWTYPE matrix_base = _reorderedMatrix.ai[0];

    const ROWTYPE dep_base = _taskInGraphIntraReduced.ai.empty() ? ROWTYPE( 0 ) : _taskInGraphIntraReduced.ai[0];

    auto solve_row = [&]( COLTYPE original_row_idx, COLTYPE reordered_row_idx )
    {
        const ROWTYPE row_begin = _reorderedMatrix.ai[reordered_row_idx] - matrix_base;
        const ROWTYPE row_end = _reorderedMatrix.ai[reordered_row_idx + 1] - matrix_base;

        VALTYPE diag = _reorderedDiag.empty() ? VALTYPE( 0 ) : _reorderedDiag[reordered_row_idx];
        VALTYPE accum = VALTYPE( 0 );

        for ( ROWTYPE jj = row_begin; jj < row_end; ++jj )
        {
            const COLTYPE col = _reorderedMatrix.aj[jj] - matrix_base;
            const VALTYPE val = _reorderedMatrix.av[jj];

            accum += val * x[col];
        }

        x[original_row_idx] = b[original_row_idx] - accum;
        x[original_row_idx] = ( diag == VALTYPE( 0 ) ) ? x[original_row_idx] : x[original_row_idx] / diag;
    };

    if ( _taskReadyCapacity < _totalTasks )
    {
        // Allocate 1.5x the needed capacity to reduce future reallocations
        _taskReadyCapacity = _totalTasks * 1.5;
        _taskReady = std::make_unique<std::atomic<bool>[]>( _taskReadyCapacity );
    }
    for ( COLTYPE i = 0; i < _totalTasks; ++i )
    {
        _taskReady[i].store( false, std::memory_order_relaxed );
    }

#pragma omp parallel num_threads( _nthreads )
    {
        const int tid = omp_get_thread_num();
        const auto& assigned_tasks = _threadTasks[tid];

        for ( COLTYPE task_id : assigned_tasks )
        {
            // #pragma omp critical
            // {
            //     std::cout << "tid: " << tid << ", task_id: " << task_id << std::endl;
            // }
            // if (tid == 1) {
            //     std::cout << "Thread 1 is pausing..." << std::endl;
            //     while (true) {
            //         std::this_thread::yield();
            //     }
            // }
            const ROWTYPE dep_begin = _taskInGraphIntraReduced.ai[task_id] - dep_base;
            const ROWTYPE dep_end = _taskInGraphIntraReduced.ai[task_id + 1] - dep_base;
            for ( ROWTYPE dep_idx = dep_begin; dep_idx < dep_end; ++dep_idx )
            {
                const COLTYPE dep_task = _taskInGraphIntraReduced.aj[dep_idx] - dep_base;
                while ( !_taskReady[static_cast<std::size_t>( dep_task )].load( std::memory_order_acquire ) )
                {
                    // std::this_thread::yield();
                    _mm_pause();
                }
            }

            const COLTYPE task_start = _taskPrefix[task_id];
            const COLTYPE task_end = _taskPrefix[task_id + 1];

            for ( COLTYPE offset = task_start; offset < task_end; ++offset )
            {
                const COLTYPE original_row_idx = _taskToNode[offset];
                const COLTYPE reordered_row_idx = _taskToReorderedNode[offset];
                solve_row( original_row_idx, reordered_row_idx );
            }

            _taskReady[static_cast<std::size_t>( task_id )].store( true, std::memory_order_release );
        }
    }

}

template <TriangularMatrix TM, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void P2PTriangularSubstitution<TM, ROWTYPE, COLTYPE, VALTYPE>::computeLevelSchedule(
    const COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj )
{
    _iperm.resize( size );
    _levelPrefix.resize( size + 1 );
    _levels = _topSort( size, ai, aj, _iperm.data(), _levelPrefix.data() );

    std::cout << "size: " << size << ", levels: " << _levels << std::endl;
}

template <TriangularMatrix TM, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void P2PTriangularSubstitution<TM, ROWTYPE, COLTYPE, VALTYPE>::buildTaskPartition(
    const COLTYPE size, ROWTYPE const* ai )
{
    // Partition each topological level into tasks that respect the maximum node cap,
    // aim for roughly k · _nthreads per level, and balance by nonzero workload.
    const auto base = ai[0];
    COLTYPE current_task = 0;
    _levelTaskPrefix[0] = 0;

    for ( COLTYPE level = 0; level < _levels; ++level )
    {
        const COLTYPE level_start = _levelPrefix[level] - base;
        const COLTYPE level_end = _levelPrefix[level + 1] - base;
        const COLTYPE nodes_in_level = level_end - level_start;

        if ( nodes_in_level == 0 )
        {
            _levelTaskPrefix[level + 1] = current_task;
            continue;
        }

        // Prefix-sum workloads so we can slice levels using cumulative load instead of raw counts.
        std::vector<COLTYPE> cumulative_workload( nodes_in_level + 1, 0 );
        COLTYPE total_workload = 0;

        for ( COLTYPE i = 0; i < nodes_in_level; ++i )
        {
            const COLTYPE node_idx = _iperm[level_start + i] - base;
            const COLTYPE node_workload = ai[node_idx + 1] - ai[node_idx];
            total_workload += node_workload;
            cumulative_workload[i + 1] = total_workload;
        }

        const COLTYPE min_tasks_by_capacity = static_cast<COLTYPE>( std::max<COLTYPE>(
            1, ( nodes_in_level + _task_maximum_size - 1 ) / _task_maximum_size ) );

        const COLTYPE capacity = _task_maximum_size * static_cast<COLTYPE>( _nthreads );
        COLTYPE k_multiplier = 1;
        if ( capacity > 0 )
        {
            k_multiplier =
                ( std::max<COLTYPE>( 1, ( nodes_in_level + capacity - 1 ) / capacity ) );
        }

        const COLTYPE desired_tasks = std::min<COLTYPE>(
            nodes_in_level, k_multiplier * static_cast<COLTYPE>( std::max( 1, _nthreads ) ) );

        COLTYPE num_tasks = std::max<COLTYPE>( min_tasks_by_capacity, desired_tasks );
        num_tasks = std::min<COLTYPE>( num_tasks, nodes_in_level );

        std::cout << "    Level " << level << " task calculation:" << std::endl;
        std::cout << "      nodes_in_level: " << nodes_in_level << std::endl;
        std::cout << "      total_workload: " << total_workload << std::endl;
        std::cout << "      _task_maximum_size: " << _task_maximum_size << std::endl;
        std::cout << "      min_tasks_by_capacity: " << min_tasks_by_capacity << std::endl;
        std::cout << "      k (ceil(nodes / (max_size * threads))): " << k_multiplier << std::endl;
        std::cout << "      desired_tasks (k*threads): " << desired_tasks << std::endl;
        std::cout << "      num_tasks: " << num_tasks << std::endl;

        COLTYPE current_node_start = 0;

        for ( COLTYPE task_idx = 0; task_idx < num_tasks; ++task_idx )
        {
            const COLTYPE tasks_left = num_tasks - task_idx;
            const COLTYPE nodes_left = nodes_in_level - current_node_start;
            // Clamp the task's node count so we leave enough room for what remains and never exceed the cap.

            // Keep enough nodes for the remaining tasks while honoring the maximum size cap.
            COLTYPE min_nodes_for_task = std::max<COLTYPE>(
                1, nodes_left - _task_maximum_size * static_cast<COLTYPE>( tasks_left - 1 ) );
            COLTYPE max_nodes_for_task = std::min<COLTYPE>(
                _task_maximum_size, nodes_left - static_cast<COLTYPE>( tasks_left - 1 ) );

            if ( min_nodes_for_task > max_nodes_for_task )
            {
                min_nodes_for_task = max_nodes_for_task = std::min<COLTYPE>( nodes_left, _task_maximum_size );
            }

            COLTYPE current_node_end = current_node_start + min_nodes_for_task;

            const COLTYPE absolute_min = current_node_start + min_nodes_for_task;
            const COLTYPE absolute_max = current_node_start + max_nodes_for_task;

            if ( total_workload > 0 && absolute_min < absolute_max )
            {
                // Target cumulative workload is proportional to the task index to equalize effort.
                const COLTYPE desired_cumulative = static_cast<COLTYPE>(
                    ( static_cast<long long>( total_workload ) * ( task_idx + 1 ) + num_tasks - 1 ) / num_tasks );

                // Search only inside the feasible window so we never form an oversized or empty task.
                auto range_begin = cumulative_workload.begin() + absolute_min;
                auto range_end = cumulative_workload.begin() + absolute_max + 1;
                auto it = std::lower_bound( range_begin, range_end, desired_cumulative );
                COLTYPE candidate = static_cast<COLTYPE>(
                    std::distance( cumulative_workload.begin(), it ) );
                candidate = std::clamp( candidate, absolute_min, absolute_max );
                current_node_end = candidate;
            }
            else if ( total_workload == 0 && absolute_min < absolute_max )
            {
                COLTYPE ideal_split = current_node_start + static_cast<COLTYPE>(
                    ( static_cast<long long>( nodes_in_level ) * ( task_idx + 1 ) + num_tasks - 1 ) / num_tasks );
                ideal_split = std::clamp( ideal_split, absolute_min, absolute_max );
                current_node_end = ideal_split;
            }
            else
            {
                current_node_end = absolute_min;
            }

            if ( current_node_end <= current_node_start )
            {
                // Fall back to the tightest admissible boundary rather than emitting a zero-length task.
                current_node_end = std::min<COLTYPE>( absolute_max, nodes_in_level );
            }

            // Finalize task metadata and map each node in the slice back to the global node index.
            _taskToLevel.push_back( level );
            _taskPrefix.push_back( _taskPrefix.back() + ( current_node_end - current_node_start ) );

            for ( COLTYPE idx = current_node_start; idx < current_node_end; ++idx )
            {
                const COLTYPE node_idx = _iperm[level_start + idx] - base;
                _taskToNode.push_back( node_idx );
                _nodeToTask[node_idx] = current_task;
            }
            current_task++;
            current_node_start = current_node_end;
        }

        _levelTaskPrefix[level + 1] = current_task;
    }

    _totalTasks = current_task;
}

template <TriangularMatrix TM, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void P2PTriangularSubstitution<TM, ROWTYPE, COLTYPE, VALTYPE>::reportTaskPartitionSummary(
    ROWTYPE const* ai ) const
{
    std::cout << "Created " << _totalTasks << " tasks across " << _levels << " levels:" << std::endl;
    std::cout << "Task-to-node mapping uses CSR format:" << std::endl;
    std::cout << "  _taskPrefix size: " << _taskPrefix.size() << " (should be " << _totalTasks + 1 << ")"
              << std::endl;
    std::cout << "  _taskToNode size: " << _taskToNode.size() << " (total node items)" << std::endl;
    std::cout << "  _nthreads: " << _nthreads << ", _task_maximum_size: " << _task_maximum_size << std::endl;

    const auto base = ai[0];
    COLTYPE total_balanced_levels = 0;
    COLTYPE total_perfectly_balanced_levels = 0;

    for ( COLTYPE level = 0; level < _levels; ++level )
    {
        COLTYPE nodes_in_level = _levelPrefix[level + 1] - _levelPrefix[level];
        COLTYPE tasks_in_level = _levelTaskPrefix[level + 1] - _levelTaskPrefix[level];
        if ( tasks_in_level > 0 )
        {
            bool is_multiple = ( tasks_in_level % _nthreads == 0 );
            bool is_perfect_multiple = ( tasks_in_level == _nthreads || tasks_in_level == 2 * _nthreads ||
                                         tasks_in_level == 3 * _nthreads || tasks_in_level == 4 * _nthreads );

            if ( is_multiple )
                total_balanced_levels++;
            if ( is_perfect_multiple )
                total_perfectly_balanced_levels++;

            std::cout << "  Level " << level << ": " << nodes_in_level << " works -> " << tasks_in_level
                      << " tasks";
            if ( is_multiple )
            {
                std::cout << " (BALANCED: " << tasks_in_level / _nthreads << "x" << _nthreads << ")";
            }
            else
            {
                std::cout << " (unbalanced)";
            }
            std::cout << std::endl;

            bool all_tasks_within_maximum = true;
            for ( COLTYPE task_idx = _levelTaskPrefix[level]; task_idx < _levelTaskPrefix[level + 1]; ++task_idx )
            {
                COLTYPE task_start = _taskPrefix[task_idx];
                COLTYPE task_end = _taskPrefix[task_idx + 1];
                COLTYPE task_nodes = task_end - task_start;

                COLTYPE task_workload = 0;
                for ( COLTYPE csr_idx = task_start; csr_idx < task_end; ++csr_idx )
                {
                    COLTYPE node_idx = _taskToNode[csr_idx];
                    COLTYPE node_workload = ai[node_idx + 1] - ai[node_idx];
                    task_workload += node_workload;
                }

                if ( task_workload > _task_maximum_size && tasks_in_level > 1 )
                {
                    all_tasks_within_maximum = false;
                }

                std::cout << "    Task " << task_idx << ": " << task_nodes << " nodes (workload: "
                          << task_workload << ")";
                if ( task_workload > _task_maximum_size && tasks_in_level > 1 )
                {
                    std::cout << " [WARNING: exceeds maximum workload]";
                }
                std::cout << std::endl;
            }

            if ( !all_tasks_within_maximum )
            {
                std::cout << "    WARNING: Some tasks exceed maximum workload!" << std::endl;
            }
        }
    }

    std::cout << "Threading balance summary:" << std::endl;
    std::cout << "  Levels with task count multiple of " << _nthreads << ": " << total_balanced_levels
              << "/" << _levels << " (" << ( 100.0 * total_balanced_levels / _levels ) << "%)" << std::endl;
    std::cout << "  Levels with perfect balance (1-4x" << _nthreads << "): "
              << total_perfectly_balanced_levels << "/" << _levels << " ("
              << ( 100.0 * total_perfectly_balanced_levels / _levels ) << "%)" << std::endl;
}

template <TriangularMatrix TM, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void P2PTriangularSubstitution<TM, ROWTYPE, COLTYPE, VALTYPE>::verifyTaskMappings() const
{
    std::cout << "Verifying CSR task-work mappings..." << std::endl;
    bool mapping_ok = true;

    if ( _taskPrefix.size() != _totalTasks + 1 )
    {
        std::cout << "ERROR: _taskPrefix size (" << _taskPrefix.size() << ") != _totalTasks + 1 ("
                  << _totalTasks + 1 << ")" << std::endl;
        mapping_ok = false;
    }

    if ( _taskToNode.size() != _taskPrefix.back() )
    {
        std::cout << "ERROR: _taskToNode size (" << _taskToNode.size() << ") != _taskPrefix.back() ("
                  << _taskPrefix.back() << ")" << std::endl;
        mapping_ok = false;
    }

    for ( COLTYPE task_idx = 0; task_idx < _totalTasks; ++task_idx )
    {
        COLTYPE task_start = _taskPrefix[task_idx];
        COLTYPE task_end = _taskPrefix[task_idx + 1];

        for ( COLTYPE csr_idx = task_start; csr_idx < task_end; ++csr_idx )
        {
            COLTYPE node_idx = _taskToNode[csr_idx];
            if ( _nodeToTask[node_idx] != task_idx )
            {
                std::cout << "ERROR: Inconsistent mapping for node " << node_idx
                          << ": _nodeToTask=" << _nodeToTask[node_idx]
                          << " but CSR indicates task " << task_idx << std::endl;
                mapping_ok = false;
                break;
            }
        }
        if ( !mapping_ok )
            break;
    }

    if ( mapping_ok )
    {
        std::cout << "CSR task-work mappings verified successfully!" << std::endl;
    }
}

template <TriangularMatrix TM, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
COLTYPE P2PTriangularSubstitution<TM, ROWTYPE, COLTYPE, VALTYPE>::buildTaskGraphs(
    const COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj )
{
    std::cout << "\nProjecting original graph to task graph..." << std::endl;

    _taskInGraph.ai.resize( _totalTasks + 1 );
    _taskInGraph.aj.resize( _totalTasks * _totalTasks );
    _taskInGraph.ai[0] = 0;

    COLTYPE task_edges = _graphProjector( size, ai, aj, _totalTasks, _taskPrefix.data(), _taskToNode.data(),
                                         _nodeToTask.data(), _taskInGraph.ai.data(), _taskInGraph.aj.data() );

    _taskInGraph.aj.resize( task_edges );
    _taskInGraph.rows = _totalTasks;
    _taskInGraph.cols = _totalTasks;

    std::cout << "Task graph projection completed:" << std::endl;
    std::cout << "  Task graph nodes: " << _totalTasks << std::endl;
    std::cout << "  Task graph edges: " << task_edges << std::endl;
    std::cout << "  Task graph density: " << ( 100.0 * task_edges ) / ( _totalTasks * _totalTasks ) << "%"
              << std::endl;

    std::cout << "  Creating transpose task graph (_taskOutGraph)..." << std::endl;
    _taskOutGraph.rows = _totalTasks;
    _taskOutGraph.cols = _totalTasks;
    _taskOutGraph.ai.resize( _totalTasks + 1 );
    _taskOutGraph.aj.resize( task_edges );

    matrix_utils::ParallelTranspose2( _taskInGraph.rows, _taskInGraph.cols,
                                      _taskInGraph.ai.data(), _taskInGraph.aj.data(), (VALTYPE const*)nullptr,
                                      _taskOutGraph.ai.data(), _taskOutGraph.aj.data(), (VALTYPE*)nullptr );

    std::cout << "  Task graph transpose completed." << std::endl;

    return task_edges;
}

template <TriangularMatrix TM, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
COLTYPE P2PTriangularSubstitution<TM, ROWTYPE, COLTYPE, VALTYPE>::reduceTaskGraph(
    COLTYPE task_edges_before )
{
    std::cout << "  Computing transitive reduction of task out-graph..." << std::endl;

    _taskOutGraphTransitiveReduced.rows = _totalTasks;
    _taskOutGraphTransitiveReduced.cols = _totalTasks;
    _taskOutGraphTransitiveReduced.ai.resize( _totalTasks + 1 );
    _taskOutGraphTransitiveReduced.aj.resize( task_edges_before );

    _transitiveReducer( _totalTasks,
                        _taskOutGraph.ai.data(),
                        _taskOutGraph.aj.data(),
                        _taskOutGraphTransitiveReduced.ai.data(),
                        _taskOutGraphTransitiveReduced.aj.data(),
                        false );

    COLTYPE task_edges_after =
        _taskOutGraphTransitiveReduced.ai[_totalTasks] - _taskOutGraphTransitiveReduced.ai[0];
    _taskOutGraphTransitiveReduced.aj.resize( task_edges_after );

    std::cout << "  Transitive reduction completed:" << std::endl;
    std::cout << "    Edges before reduction: " << task_edges_before << std::endl;
    std::cout << "    Edges after reduction:  " << task_edges_after << std::endl;
    std::cout << "    Edges removed:          " << ( task_edges_before - task_edges_after ) << std::endl;
    std::cout << "    Reduction ratio:        " << (double)task_edges_after / task_edges_before * 100.0
              << "%" << std::endl;

    return task_edges_after;
}

template <TriangularMatrix TM, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void P2PTriangularSubstitution<TM, ROWTYPE, COLTYPE, VALTYPE>::partitionTasksToThreads()
{
    std::cout << "  Partitioning tasks to threads..." << std::endl;

    std::cout << "    Using level-by-level partitioning with direct vector construction..." << std::endl;
    _taskPartition.resize( _totalTasks );

    for (COLTYPE level = 0; level < _levels; ++level) {
        COLTYPE level_start_task = _levelTaskPrefix[level];
        COLTYPE level_end_task = _levelTaskPrefix[level + 1];
        COLTYPE tasks_in_level = level_end_task - level_start_task;
        
        if (tasks_in_level == 0) {
            throw std::runtime_error("Error: Level " + std::to_string(level) + 
                                   " has zero tasks. This indicates a problem with task partitioning.");
        }
        
        // Distribute tasks in this level among threads, keeping adjacent tasks together
        COLTYPE tasks_per_thread = (tasks_in_level + _nthreads - 1) / _nthreads;  // ceil division
        
        // Process tasks level by level, thread by thread to maintain adjacency
        for (int thread = 0; thread < _nthreads; ++thread) {
            COLTYPE start_task_in_level = thread * tasks_per_thread;
            COLTYPE end_task_in_level = std::min((thread + 1) * tasks_per_thread, tasks_in_level);
            
            if (start_task_in_level >= tasks_in_level) break;
            
            // Assign consecutive tasks to this thread
            for (COLTYPE task_in_level = start_task_in_level; task_in_level < end_task_in_level; ++task_in_level) {
                COLTYPE task_id = level_start_task + task_in_level;
                _taskPartition[task_id] = thread;
                _threadTasks[thread].push_back(task_id);
            }
        }
    }
    
    std::cout << "    Level-by-level partitioning with direct vector construction completed." << std::endl;
    
    // Report partition statistics using the vector structure
    std::cout << "    Partition statistics:" << std::endl;
    for (int thread = 0; thread < _nthreads; ++thread) {
        COLTYPE task_count = _threadTasks[thread].size();
        std::cout << "      Thread " << thread << ": " << task_count << " tasks";
        
        // Show first few task IDs for each thread
        if (task_count > 0) {
            std::cout << " [";
            COLTYPE show_count = std::min(static_cast<COLTYPE>(5), task_count);
            
            for (COLTYPE i = 0; i < show_count; ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << _threadTasks[thread][i];
            }
            if (task_count > 5) {
                std::cout << ", ...";
            }
            std::cout << "]";
        }
        std::cout << std::endl;
    }
    
    // Calculate load balance ratio using vector structure
    COLTYPE min_tasks = _totalTasks;
    COLTYPE max_tasks = 0;
    for (int thread = 0; thread < _nthreads; ++thread) {
        COLTYPE task_count = _threadTasks[thread].size();
        min_tasks = std::min(min_tasks, task_count);
        max_tasks = std::max(max_tasks, task_count);
    }
    double balance_ratio = (max_tasks > 0) ? (double)min_tasks / max_tasks : 1.0;
    std::cout << "      Load balance ratio: " << balance_ratio << " (1.0 = perfect)" << std::endl;
    
    // Verify vector structure integrity
    std::cout << "    Vector structure verification:" << std::endl;
    std::cout << "      _threadTasks size: " << _threadTasks.size() << " (should be " << _nthreads << ")" << std::endl;
    COLTYPE total_assigned_tasks = 0;
    for (int thread = 0; thread < _nthreads; ++thread) {
        total_assigned_tasks += _threadTasks[thread].size();
    }
    std::cout << "      Total assigned tasks: " << total_assigned_tasks << " (should be " << _totalTasks << ")" << std::endl;
}

template <TriangularMatrix TM, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void P2PTriangularSubstitution<TM, ROWTYPE, COLTYPE, VALTYPE>::pruneThreadIntraEdgesFromReducedGraph()
{
    if ( _taskOutGraphTransitiveReduced.ai.empty() || _totalTasks == 0 )
    {
        throw std::runtime_error("Error: Cannot prune thread intra edges - reduced graph is empty or no tasks exist. "
                               "This indicates a problem with task graph construction.");
    }

    if ( static_cast<COLTYPE>( _taskPartition.size() ) != _totalTasks )
    {
        throw std::runtime_error("Error: Cannot prune thread intra edges - task partitioning incomplete. "
                               "Expected " + std::to_string(_totalTasks) + " tasks, but partition size is " + 
                               std::to_string(_taskPartition.size()) + ".");
    }

    const ROWTYPE* trans_ai = _taskOutGraphTransitiveReduced.ai.data();
    const COLTYPE* trans_aj = _taskOutGraphTransitiveReduced.aj.data();
    const ROWTYPE original_edges = trans_ai[_totalTasks];

    _taskScratch.resize( _totalTasks );

    ROWTYPE final_edges = 0;

#pragma omp parallel num_threads( _nthreads )
    {
#pragma omp for schedule( dynamic ) nowait
        for ( COLTYPE task = 0; task < _totalTasks; ++task )
        {
            const auto row_start = trans_ai[task];
            const auto row_end = trans_ai[task + 1];
            const auto task_partition = _taskPartition[task];

            auto& scratch = _taskScratch[task];
            scratch.clear();

            for ( ROWTYPE idx = row_start; idx < row_end; ++idx )
            {
                const COLTYPE neighbor = trans_aj[idx];
                if ( task_partition != _taskPartition[neighbor] )
                {
                    scratch.push_back( neighbor );
                }
            }
        }

#pragma omp single
        {
            _taskOutGraphIntraReduced.rows = _totalTasks;
            _taskOutGraphIntraReduced.cols = _totalTasks;
            auto& new_ai = _taskOutGraphIntraReduced.ai;
            new_ai.resize( _totalTasks + 1 );
            new_ai[0] = 0;

            for ( COLTYPE task = 0; task < _totalTasks; ++task )
            {
                new_ai[task + 1] =
                    new_ai[task] + static_cast<ROWTYPE>( _taskScratch[task].size() );
            }

            auto& new_aj = _taskOutGraphIntraReduced.aj;
            final_edges = new_ai[_totalTasks];
            new_aj.resize( final_edges );
        }

#pragma omp for schedule( dynamic )
        for ( COLTYPE task = 0; task < _totalTasks; ++task )
        {
            const auto write_begin = _taskOutGraphIntraReduced.ai[task];
            const auto& scratch = _taskScratch[task];
            auto* dest = _taskOutGraphIntraReduced.aj.data() + write_begin;
            for ( std::size_t i = 0; i < scratch.size(); ++i )
            {
                dest[i] = scratch[i];
            }
        }
    }

    // convert back to in-graph by transposing
    _taskInGraphIntraReduced.rows = _totalTasks;
    _taskInGraphIntraReduced.cols = _totalTasks;
    _taskInGraphIntraReduced.ai.resize( _totalTasks + 1 );
    _taskInGraphIntraReduced.aj.resize( final_edges );

    matrix_utils::ParallelTranspose2( _taskOutGraphIntraReduced.rows,
                                      _taskOutGraphIntraReduced.cols,
                                      _taskOutGraphIntraReduced.ai.data(),
                                      _taskOutGraphIntraReduced.aj.data(),
                                      static_cast<VALTYPE const*>( nullptr ),
                                      _taskInGraphIntraReduced.ai.data(),
                                      _taskInGraphIntraReduced.aj.data(),
                                      static_cast<VALTYPE*>( nullptr ) );

    std::cout << "    Removed intra-thread edges from reduced task graph (" << original_edges << " -> "
              << final_edges << ")." << std::endl;
}

template <TriangularMatrix TM, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void P2PTriangularSubstitution<TM, ROWTYPE, COLTYPE, VALTYPE>::outputTaskGraphDebugInfo(
    COLTYPE task_edges_before, COLTYPE task_edges_after ) const
{
    std::cout << "  Task graph edge count (raw -> transitive -> intra-thread): " << task_edges_before
              << " -> " << task_edges_after << " -> "
              << ( _taskOutGraphIntraReduced.ai[_totalTasks] - _taskOutGraphIntraReduced.ai[0] ) << std::endl;

    std::cout << "  Transitive-reduced task out-graph structure (first "
              << std::min( _totalTasks, static_cast<COLTYPE>( 5 ) ) << " tasks):" << std::endl;
    for ( COLTYPE task_idx = 0; task_idx < std::min( _totalTasks, static_cast<COLTYPE>( 5 ) ); ++task_idx )
    {
        COLTYPE edge_start = _taskOutGraphTransitiveReduced.ai[task_idx] - _taskOutGraphTransitiveReduced.ai[0];
        COLTYPE edge_end = _taskOutGraphTransitiveReduced.ai[task_idx + 1] - _taskOutGraphTransitiveReduced.ai[0];
        std::cout << "    Task " << task_idx << " has " << ( edge_end - edge_start ) << " outgoing edges: [";
        for ( COLTYPE edge_idx = edge_start; edge_idx < edge_end; ++edge_idx )
        {
            if ( edge_idx > edge_start )
                std::cout << ", ";
            std::cout << ( _taskOutGraphTransitiveReduced.aj[edge_idx] - _taskOutGraphTransitiveReduced.ai[0] );
        }
        std::cout << "]" << std::endl;
    }

    std::cout << "  Thread-pruned task out-graph structure (first "
              << std::min( _totalTasks, static_cast<COLTYPE>( 5 ) ) << " tasks):" << std::endl;
    for ( COLTYPE task_idx = 0; task_idx < std::min( _totalTasks, static_cast<COLTYPE>( 5 ) ); ++task_idx )
    {
        COLTYPE edge_start = _taskOutGraphIntraReduced.ai[task_idx] - _taskOutGraphIntraReduced.ai[0];
        COLTYPE edge_end = _taskOutGraphIntraReduced.ai[task_idx + 1] - _taskOutGraphIntraReduced.ai[0];
        std::cout << "    Task " << task_idx << " has " << ( edge_end - edge_start ) << " outgoing edges: [";
        for ( COLTYPE edge_idx = edge_start; edge_idx < edge_end; ++edge_idx )
        {
            if ( edge_idx > edge_start )
                std::cout << ", ";
            std::cout << ( _taskOutGraphIntraReduced.aj[edge_idx] - _taskOutGraphIntraReduced.ai[0] );
        }
        std::cout << "]" << std::endl;
    }

#ifdef USE_BOOST_LIB
    std::string dot_filename_reduced = "task_out_graph_transitive_reduced.dot";
    std::string dot_filename_intra = "task_out_graph_intra_reduced.dot";
    std::string dot_filename_partitioned = "task_level_partitioned.dot";
    std::string dot_filename_threaded = "task_thread_partitioned.dot";

    std::cout << "  Writing transitive-reduced task out-graph to DOT file: " << dot_filename_reduced << std::endl;
    utils::writeAdjacencyGraphDOT( _totalTasks,
                                   _taskOutGraphTransitiveReduced.ai.data(),
                                   _taskOutGraphTransitiveReduced.aj.data(),
                                   dot_filename_reduced,
                                   "P2P Task Out-Graph Transitive Reduced" );

    std::cout << "  Writing intra-thread-pruned task out-graph to DOT file: " << dot_filename_intra << std::endl;
    utils::writeAdjacencyGraphDOT( _totalTasks,
                                   _taskOutGraphIntraReduced.ai.data(),
                                   _taskOutGraphIntraReduced.aj.data(),
                                   dot_filename_intra,
                                   "P2P Task Out-Graph After Pruning" );

    std::cout << "  Writing level-partitioned task graph to DOT file: " << dot_filename_partitioned << std::endl;
    utils::writeAdjacencyGraphDOT( _totalTasks,
                                   _taskOutGraphIntraReduced.ai.data(),
                                   _taskOutGraphIntraReduced.aj.data(),
                                   _taskToLevel.data(), _levels, dot_filename_partitioned,
                                   "P2P Task Level Partitioning (Post-Pruning)" );

    std::cout << "  Writing thread-partitioned task graph to DOT file: " << dot_filename_threaded << std::endl;
    utils::writeAdjacencyGraphDOT( _totalTasks,
                                   _taskOutGraphIntraReduced.ai.data(),
                                   _taskOutGraphIntraReduced.aj.data(),
                                   _taskPartition.data(), _nthreads, dot_filename_threaded,
                                   "P2P Task Thread Partitioning (Post-Pruning)" );

    std::cout << "  Task graph DOT files created. Visualize with:" << std::endl;
    std::cout << "    dot -Tpng " << dot_filename_reduced << " -o task_transitive_reduced.png" << std::endl;
    std::cout << "    dot -Tpng " << dot_filename_intra << " -o task_intra_reduced.png" << std::endl;
    std::cout << "    dot -Tpng " << dot_filename_partitioned << " -o task_level_partitioned.png" << std::endl;
    std::cout << "    dot -Tpng " << dot_filename_threaded << " -o task_thread_partitioned.png" << std::endl;
#else
    std::cout << "  Task graph DOT generation skipped (USE_BOOST_LIB not defined)" << std::endl;
#endif

    std::cout << "Task dependencies (first " << std::min( _totalTasks, static_cast<COLTYPE>( 5 ) )
              << " tasks):" << std::endl;
    for ( COLTYPE task_idx = 0; task_idx < std::min( _totalTasks, static_cast<COLTYPE>( 5 ) ); ++task_idx )
    {
        COLTYPE dep_start = _taskInGraph.ai[task_idx] - _taskPrefix[0];
        COLTYPE dep_end = _taskInGraph.ai[task_idx + 1] - _taskPrefix[0];
        std::cout << "  Task " << task_idx << " depends on " << ( dep_end - dep_start ) << " tasks: [";
        for ( COLTYPE dep_idx = dep_start; dep_idx < dep_end; ++dep_idx )
        {
            if ( dep_idx > dep_start )
                std::cout << ", ";
            std::cout << ( _taskInGraph.aj[dep_idx] - _taskPrefix[0] );
        }
        std::cout << "]" << std::endl;
    }
}

template <TriangularMatrix TM, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void P2PTriangularSubstitution<TM, ROWTYPE, COLTYPE, VALTYPE>::createThreadLocalizedPermutation(
    const COLTYPE size )
{
    std::cout << "\n  Creating thread-localized row permutation for cache optimization..." << std::endl;

    _threadLocalizedRowPerm.resize( size );
    _threadLocalizedRowInvPerm.resize( size );
    _taskToReorderedNode.resize( size );

    COLTYPE current_permuted_row = 0;
    
    // Create permutation: Thread by thread, level by level within each thread
    for ( int thread = 0; thread < _nthreads; ++thread )
    {
        COLTYPE thread_start_row = current_permuted_row;
        
        // Process tasks for this thread in order (they're already sorted by level due to construction)
        for ( COLTYPE task_id : _threadTasks[thread] )
        {
            // Get all node items (matrix rows) for this task
            COLTYPE task_start = _taskPrefix[task_id];
            COLTYPE task_end = _taskPrefix[task_id + 1];

            for ( COLTYPE idx = task_start; idx < task_end; ++idx )
            {
                COLTYPE original_row = _taskToNode[idx];
                _threadLocalizedRowPerm[original_row] = current_permuted_row;
                _taskToReorderedNode[idx] = current_permuted_row;
                _threadLocalizedRowInvPerm[current_permuted_row] = original_row;
                current_permuted_row++;
            }
        }
        
        COLTYPE thread_end_row = current_permuted_row;
        
        std::cout << "    Thread " << thread << ": rows [" << thread_start_row 
                  << ", " << thread_end_row << ") (" << (thread_end_row - thread_start_row) << " rows)" << std::endl;
    }
    
    std::cout << "  Thread-localized permutation created. Total rows: " << current_permuted_row << std::endl;
}

template <TriangularMatrix TM, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void P2PTriangularSubstitution<TM, ROWTYPE, COLTYPE, VALTYPE>::reorderMatrixForCacheLocality(
    const COLTYPE size,
    ROWTYPE const* ai,
    COLTYPE const* aj,
    VALTYPE const* av,
    VALTYPE const* diag )
{
    std::cout << "  Reordering matrix using thread-localized permutation with permuteMat..." << std::endl;
    
    const auto base = ai[0];
    const auto nnz = ai[size] - ai[0];
    
    // Allocate reordered matrix
    _reorderedMatrix.ai.resize( size + 1 );
    _reorderedMatrix.aj.resize( nnz );
    _reorderedMatrix.av.resize( nnz );
    _reorderedMatrix.rows = size;
    _reorderedMatrix.cols = size;

    // Print out nnz of each row of original CSR
    std::cout << "Original CSR row nnz counts:" << std::endl;
    for (COLTYPE row = 0; row < size; ++row) {
      std::cout << "  Row " << row << ": " << (ai[row + 1] - ai[row]) << " nnz" << std::endl;
    }
    
    // Use permuteMat to perform row permutation only: pA = P * A
    // For triangular solve, we only need row permutation (no column permutation)
    // Use nullptr for column permutation (keeps original column indices)
    matrix_utils::permuteMat<ROWTYPE, COLTYPE, VALTYPE>(
        size, size,
        _threadLocalizedRowInvPerm.data(),  // Row permutation: new -> original
        nullptr,                            // No column permutation
        ai, aj, av,
        _reorderedMatrix.ai.data(), _reorderedMatrix.aj.data(), _reorderedMatrix.av.data()
    );
    std::cout << "  _threadLocalizedRowInvPerm: [";
    for (COLTYPE i = 0; i < size; ++i) {
      if (i > 0) std::cout << ", ";
      std::cout << _threadLocalizedRowInvPerm[i];
    }
    std::cout << "]" << std::endl;

    if ( diag )
    {
        _reorderedDiag.resize( size );
        matrix_utils::permVec<COLTYPE, VALTYPE>( size,
                                                 static_cast<COLTYPE>( base ),
                                                 diag,
                                                 _threadLocalizedRowInvPerm.data(),
                                                 _reorderedDiag.data() );
    }
    else
    {
        _reorderedDiag.clear();
    }

    std::cout << "  Matrix reordering completed using permuteMat. NNZ: " << (_reorderedMatrix.ai[size] - _reorderedMatrix.ai[0]) << std::endl;
    
    // Debug: Write SVG visualizations of original and reordered matrices with diagonal terms
    {
        // Helper lambda to add diagonal entries to CSR matrix structure (no values)
        auto addDiagonalToCSR = [](const COLTYPE rows, const ROWTYPE base, 
                                   ROWTYPE const* src_ai, COLTYPE const* src_aj,
                                   std::vector<ROWTYPE>& dst_ai, 
                                   std::vector<COLTYPE>& dst_aj) {
            dst_ai.resize(rows + 1);
            dst_ai[0] = base;
            
            // Count entries per row including diagonal
            for (COLTYPE i = 0; i < rows; ++i) {
                COLTYPE const* row_start = src_aj + src_ai[i] - base;
                COLTYPE const* row_end = src_aj + src_ai[i + 1] - base;
                auto diag_it = std::lower_bound(row_start, row_end, i + base);
                bool has_diag = (diag_it != row_end && *diag_it == i + base);
                ROWTYPE row_nnz = src_ai[i + 1] - src_ai[i];
                dst_ai[i + 1] = row_nnz + (has_diag ? 0 : 1); // Add diagonal if missing
            }
            
            // Prefix sum to get row pointers
            for (COLTYPE i = 0; i < rows; ++i) {
                dst_ai[i + 1] += dst_ai[i];
            }
            
            ROWTYPE total_nnz = dst_ai[rows] - base;
            dst_aj.resize(total_nnz);
            
            // Fill column indices with diagonal inserted
            for (COLTYPE i = 0; i < rows; ++i) {
                COLTYPE const* src_row_start = src_aj + src_ai[i] - base;
                COLTYPE const* src_row_end = src_aj + src_ai[i + 1] - base;
                COLTYPE* dst_row_start = dst_aj.data() + dst_ai[i] - base;
                
                auto diag_pos = std::lower_bound(src_row_start, src_row_end, i + base);
                bool has_diag = (diag_pos != src_row_end && *diag_pos == i + base);
                
                if (has_diag) {
                    // Diagonal already exists, just copy
                    std::copy(src_row_start, src_row_end, dst_row_start);
                } else {
                    // Insert diagonal entry
                    COLTYPE* insert_pos = std::copy(src_row_start, diag_pos, dst_row_start);
                    *insert_pos = i + base;
                    std::copy(diag_pos, src_row_end, insert_pos + 1);
                }
            }
        };
        
        // Create original matrix with diagonal
        std::vector<ROWTYPE> orig_with_diag_ai;
        std::vector<COLTYPE> orig_with_diag_aj;
        if (1) {
            addDiagonalToCSR(size, base, ai, aj, orig_with_diag_ai, orig_with_diag_aj);
            std::ofstream original_svg("debug_original_matrix_with_diag.svg");
            if (original_svg.is_open()) {
                matrix_utils::writeSVG(size, size, 
                                       orig_with_diag_ai.data(), 
                                       orig_with_diag_aj.data(), 
                                       original_svg);
                original_svg.close();
                std::cout << "  Debug: Original matrix (with diag) written to debug_original_matrix_with_diag.svg" << std::endl;
            }
            
            // Now permute the original matrix with diagonal using the same permutation
            const auto nnz_with_diag = orig_with_diag_ai[size] - base;
            
            std::vector<ROWTYPE> reord_with_diag_ai(size + 1);
            std::vector<COLTYPE> reord_with_diag_aj(nnz_with_diag);
            
            matrix_utils::permuteMat<ROWTYPE, COLTYPE, VALTYPE>(
                size, size,
                _threadLocalizedRowInvPerm.data(),  // Same row permutation as before
                nullptr,                             // No column permutation
                orig_with_diag_ai.data(), orig_with_diag_aj.data(), nullptr,
                reord_with_diag_ai.data(), reord_with_diag_aj.data(), nullptr
            );
            
            std::ofstream reordered_svg("debug_reordered_matrix_with_diag.svg");
            if (reordered_svg.is_open()) {
                matrix_utils::writeSVG(size, size, 
                                       reord_with_diag_ai.data(), 
                                       reord_with_diag_aj.data(), 
                                       reordered_svg);
                reordered_svg.close();
                std::cout << "  Debug: Reordered matrix (with diag) written to debug_reordered_matrix_with_diag.svg" << std::endl;
            }
        }
    }
    
    // Report cache locality improvement - compute thread row ranges on-demand
    std::cout << "  Cache locality analysis:" << std::endl;
    COLTYPE current_row = 0;
    
    for ( int thread = 0; thread < _nthreads; ++thread )
    {
        COLTYPE thread_start_row = current_row;
        
        // Compute number of rows for this thread based on task assignments
        COLTYPE thread_rows = 0;
        for ( COLTYPE task_id : _threadTasks[thread] )
        {
            COLTYPE task_start = _taskPrefix[task_id];
            COLTYPE task_end = _taskPrefix[task_id + 1];
            thread_rows += task_end - task_start; // Number of node items (rows) in this task
        }
        
        COLTYPE thread_end_row = thread_start_row + thread_rows;
        current_row = thread_end_row;
        
        // Count nonzeros for this thread's rows
        COLTYPE thread_nnz = 0;
        for ( COLTYPE row = thread_start_row; row < thread_end_row; ++row )
        {
            thread_nnz += _reorderedMatrix.ai[row + 1] - _reorderedMatrix.ai[row];
        }
        
        std::cout << "    Thread " << thread << ": " << thread_rows << " rows, " 
                  << thread_nnz << " nonzeros (" 
                  << (100.0 * thread_nnz / nnz) << "% of total)" << std::endl;
    }
}

template <FBSubstitutionType FBST, TriangularMatrix TS, typename ROWTYPE,
          typename COLTYPE, typename VALTYPE>
void OptimizedTriangularSolve<FBST, TS, ROWTYPE, COLTYPE, VALTYPE>::analysis(
    const COLTYPE rows, const int base, ROWTYPE const *ai, COLTYPE const *aj,
    VALTYPE const *av, VALTYPE const *diag) {
  _diag = diag;
  _size = rows;
  _vec.resize(_size);
  const auto nnz = ai[rows] - base;
  _reorderedMat.ai.resize(rows + 1);
  _reorderedMat.aj.resize(nnz);
  _reorderedMat.av.resize(nnz);
  _reorderedMat.ai[0] = base;
  _reorderedMat.rows = rows;
  graph::TopologicalSort2<int, int, TS> topSort;
  _iperm.resize(rows);
  _levelPrefix.resize(rows + 1);
  _levels = topSort( rows, ai, aj, _iperm.data(), _levelPrefix.data() );
  _threadlevels.resize(_nthreads);
  _threadiperm.resize(rows);

#pragma omp parallel num_threads(_nthreads)
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    // #pragma omp single
    //       std::cout << "nthreads: " << nthreads << std::endl;

    // prepare cost for load balance of each level

    _threadlevels[tid].resize(_levels + 1);
    _threadlevels[tid][0] = 0;

    for (COLTYPE l = 0; l < _levels; l++) {
      // TODO: a better load balancing is needed
      auto [start, end] = utils::LoadBalancedPartitionPos(
          _levelPrefix[l + 1] - _levelPrefix[l], tid, nthreads);
      const COLTYPE size = end - start;
      // #pragma omp critical
      //         std::cout << "tid: " << tid << " , size: " << size <<
      //         std::endl;
      _threadlevels[tid][l + 1] = _threadlevels[tid][l] + size;
    }

#pragma omp barrier
#pragma omp single
    {
      COLTYPE size = 0;
      for (int tid = 1; tid < nthreads; tid++) {
        size += _threadlevels[tid - 1][_levels];
        _threadlevels[tid][0] = size;
      }
    }

    for (COLTYPE l = 0; l < _levels; l++) {
      _threadlevels[tid][l + 1] += _threadlevels[tid][0];
    }
    // up to this point, _threadlevels becomes the prefix of size of each
    // super task

#pragma omp barrier
    COLTYPE cur = _threadlevels[tid][0];

    for (COLTYPE l = 0; l < _levels; l++) {
      auto [start, end] = utils::LoadBalancedPartitionPos(
          _levelPrefix[l + 1] - _levelPrefix[l], tid, nthreads);
      for (auto i = start; i != end; i++) {
        _threadiperm[cur++] = _iperm[i + _levelPrefix[l]];
      }
    }
  }

  utils::inversePermute(_threadperm, _threadiperm, base);

  // matrix_utils::permute(rows, base, ai, aj, av, _threadiperm.data(),
  //                       _threadperm.data(), _reorderedMat.ai.data(),
  //                       _reorderedMat.aj.data(), _reorderedMat.av.data());

  matrix_utils::permuteMat(rows, rows, _threadiperm.data(),
                           static_cast<COLTYPE const*>(nullptr),
                           ai, aj, av, _reorderedMat.ai.data(),
                           _reorderedMat.aj.data(), _reorderedMat.av.data());

  if constexpr (FBST == FBSubstitutionType::NoBarrierSuperNode) {
    build_task_graph();
    // for (auto i = 0; i < _taskInvAdjGraph.rows; i++) {
    //   std::cout << "taks " << i << ": ";
    //   for (auto j = _taskInvAdjGraph.ai[i]; j < _taskInvAdjGraph.ai[i + 1];
    //        j++) {
    //     std::cout << _taskInvAdjGraph.aj[j] << " ";
    //   }
    //   std::cout << std::endl;
    // }
  }

  if constexpr (FBST == FBSubstitutionType::NoBarrier)
    _bv.resize(_size);
  else if constexpr (FBST == FBSubstitutionType::NoBarrierSuperNode)
    _bv.resize(_tasks);
}

template <FBSubstitutionType FBST, TriangularMatrix TS, typename ROWTYPE,
          typename COLTYPE, typename VALTYPE>
void OptimizedTriangularSolve<FBST, TS, ROWTYPE, COLTYPE, VALTYPE>::operator()(
    const VALTYPE *const b, VALTYPE *const x) const {
  if constexpr (FBST == FBSubstitutionType::Barrier)
    BarrierOp(b, x);
  else if constexpr (FBST == FBSubstitutionType::NoBarrier)
    NoBarrierOp(b, x);
  else if constexpr (FBST == FBSubstitutionType::NoBarrierSuperNode)
    NoBarrierSuperNodeOp(b, x);
}

template <FBSubstitutionType FBST, TriangularMatrix TS, typename ROWTYPE,
          typename COLTYPE, typename VALTYPE>
void OptimizedTriangularSolve<FBST, TS, ROWTYPE, COLTYPE, VALTYPE>::BarrierOp(
    const VALTYPE *const b, VALTYPE *const x) const {
#pragma omp parallel num_threads(_nthreads)
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    for (COLTYPE l = 0; l < _levels; l++) {
      const COLTYPE start = _threadlevels[tid][l];
      const COLTYPE end = _threadlevels[tid][l + 1];
      for (COLTYPE i = start; i < end; i++) {
        const COLTYPE idx = _threadiperm[i] - _reorderedMat.Base();
        VALTYPE val = 0;
#pragma unroll
        for (auto j = _reorderedMat.ai[i] - _reorderedMat.Base();
             j < _reorderedMat.ai[i + 1] - _reorderedMat.Base(); j++) {
          const COLTYPE j_idx = _reorderedMat.aj[j] - _reorderedMat.Base();
          val += _reorderedMat.av[j] * x[j_idx];
        }
        x[idx] = _diag ? (b[idx] - val) / _diag[idx] : (b[idx] - val);
      }
#pragma omp barrier
    }
  }
  // std::copy(_vec.begin(), _vec.end(), x);
  // matrix_utils::permuteVec(_size, _reorderedMat.Base(), _vec.data(),
  //                          _threadperm.data(), x);
}

template <FBSubstitutionType FBST, TriangularMatrix TS, typename ROWTYPE,
          typename COLTYPE, typename VALTYPE>
void OptimizedTriangularSolve<FBST, TS, ROWTYPE, COLTYPE, VALTYPE>::NoBarrierOp(
    const VALTYPE *const b, VALTYPE *const x) const {
  _bv.clearAll();
#pragma omp parallel num_threads(_nthreads)
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    for (COLTYPE l = 0; l < _levels; l++) {
      const COLTYPE start = _threadlevels[tid][l];
      const COLTYPE end = _threadlevels[tid][l + 1];
      for (COLTYPE i = start; i < end; i++) {
        const COLTYPE idx = _threadiperm[i] - _reorderedMat.Base();
        VALTYPE val = 0;
        for (auto j = _reorderedMat.ai[i] - _reorderedMat.Base();
             j < _reorderedMat.ai[i + 1] - _reorderedMat.Base(); j++) {
          const COLTYPE j_idx = _reorderedMat.aj[j] - _reorderedMat.Base();
          while (!_bv.get(j_idx)) {
            // std::cout << "tid: " << tid << "yield\n";
            // std::this_thread::yield();
            _mm_pause();
          }
          val += _reorderedMat.av[j] * x[j_idx];
        }
        x[idx] = _diag ? (b[idx] - val) / _diag[idx] : (b[idx] - val);
        _bv.set(idx);
      }
    }
  }
}

template <FBSubstitutionType FBST, TriangularMatrix TS, typename ROWTYPE,
          typename COLTYPE, typename VALTYPE>
void OptimizedTriangularSolve<FBST, TS, ROWTYPE, COLTYPE, VALTYPE>::
    NoBarrierSuperNodeOp(const VALTYPE *const b, VALTYPE *const x) const {
  _bv.clearAll();
#pragma omp parallel num_threads(_nthreads)
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    for (COLTYPE task = _threadTaskPrefix[tid];
         task < _threadTaskPrefix[tid + 1]; task++) {

      for (COLTYPE i = _taskInvAdjGraph2.ai[task];
           i < _taskInvAdjGraph2.ai[task + 1]; i++) {
        const COLTYPE j_idx = _taskInvAdjGraph2.aj[i];
        while (!_bv.get(j_idx)) {
          // std::this_thread::yield();
          _mm_pause();
          // std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
      }

      for (COLTYPE i = _taskBoundaryPrefix[task];
           i < _taskBoundaryPrefix[task + 1]; i++) {
        const COLTYPE idx = _threadiperm[i] - _reorderedMat.Base();
        VALTYPE val = 0;
#pragma unroll
        for (auto j = _reorderedMat.ai[i] - _reorderedMat.Base();
             j < _reorderedMat.ai[i + 1] - _reorderedMat.Base(); j++) {
          const COLTYPE j_idx = _reorderedMat.aj[j] - _reorderedMat.Base();
          val += _reorderedMat.av[j] * x[j_idx];
        }
        x[idx] = _diag ? (b[idx] - val) / _diag[idx] : (b[idx] - val);
      }
      _bv.set(task);
    }
  }
}

template <FBSubstitutionType FBST, TriangularMatrix TS, typename ROWTYPE,
          typename COLTYPE, typename VALTYPE>
void OptimizedTriangularSolve<FBST, TS, ROWTYPE, COLTYPE,
                              VALTYPE>::build_task_graph() {
  _taskInvAdj.resize(_nthreads);
  _threadTaskPrefix.resize(_nthreads + 1);
  _threadPrefixSum.resize(_nthreads + 1);
  _threadPrefixSum[0] = 0;
  // _threadPrefixSum2.resize(_nthreads + 1);
  // std::fill(_threadPrefixSum2.begin(), _threadPrefixSum2.end(), 0);
  _reorderedRowIdToTaskId.resize(_size);
  // std::cout << "levels: " << _levels << std::endl;

#pragma omp parallel num_threads(_nthreads)
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();

    // count the number of tasks for each thread
    COLTYPE cnt = 0;
    for (COLTYPE l = 0; l < _levels; l++) {
      if (_threadlevels[tid][l + 1] > _threadlevels[tid][l])
        ++cnt;
    }
    _threadTaskPrefix[tid + 1] = cnt; // tasks in each thread

#pragma omp barrier
#pragma omp single
    {
      _threadTaskPrefix[0] = 0;
      std::inclusive_scan(_threadTaskPrefix.begin(), _threadTaskPrefix.end(),
                          _threadTaskPrefix.begin());
      _tasks = _threadTaskPrefix[_nthreads];

      // taskSizes.resize(_tasks);

      // std::cout << "tasks: " << _tasks << std::endl;
      _taskBoundaryPrefix.resize(_tasks + 1);

      _taskInvAdjGraph.rows = _tasks;
      _taskInvAdjGraph.cols = _tasks;
      _taskInvAdjGraph.ai.resize(_tasks + 1);
      _taskInvAdjGraph.ai[0] = 0; // zero based
      _taskInvAdjGraph.aj.resize(
          _reorderedMat.NNZ() +
          _tasks); // added _tasks for edges within super-tasks

      _taskAdjGraph.rows = _tasks;
      _taskAdjGraph.cols = _tasks;
      _taskAdjGraph.ai.resize(_tasks + 1);
      _taskAdjGraph.ai[0] = 0; // zero based
      // _taskAdjGraph.aj.resize(_reorderedMat.NNZ());
    }

    // build task boundary prefix (prefix of task sizes)
    COLTYPE taskOffset = _threadTaskPrefix[tid];
    for (COLTYPE l = 0; l < _levels; l++) {
      if (_threadlevels[tid][l + 1] > _threadlevels[tid][l]) {
        _taskBoundaryPrefix[++taskOffset] =
            _threadlevels[tid][l + 1] - _threadlevels[tid][l];
      }
    }

#pragma omp barrier
#pragma omp single
    {
      _taskBoundaryPrefix[0] = 0;
      std::inclusive_scan(_taskBoundaryPrefix.begin(),
                          _taskBoundaryPrefix.end(),
                          _taskBoundaryPrefix.begin());
    }

    // split tasks to each thread
    auto [start, end] = utils::LoadBalancedPartitionPos(_tasks, tid, nthreads);
    _threadPrefixSum[tid + 1] = 0;
    for (COLTYPE task = start; task < end; task++) {
      COLTYPE invAdjSizePerTask = 0;
      for (COLTYPE i = _taskBoundaryPrefix[task];
           i < _taskBoundaryPrefix[task + 1]; i++) {
        invAdjSizePerTask += _reorderedMat.ai[i + 1] - _reorderedMat.ai[i];
        _reorderedRowIdToTaskId[i] = task;
      }
      invAdjSizePerTask += 1; // added 1 for task -> task-1 dependency
                              // within each super-tasks
      _threadPrefixSum[tid + 1] += invAdjSizePerTask;
      _taskInvAdjGraph.ai[task + 1] = _threadPrefixSum[tid + 1];
      // #pragma omp critical
      //         {
      //           std::cout << "tid: " << tid << " task: " << task
      //                     << " ai:  " << _taskInvAdjGraph.ai[task + 1] <<
      //                     std::endl;
      //         }
    }

#pragma omp barrier
#pragma omp single
    {
      std::inclusive_scan(_threadPrefixSum.begin(), _threadPrefixSum.end(),
                          _threadPrefixSum.begin());
    }

    for (COLTYPE task = start; task < end; task++) {
      _taskInvAdjGraph.ai[task + 1] += _threadPrefixSum[tid];
    }

#pragma omp barrier
    _threadPrefixSum[tid + 1] = 0; // reset
    // #pragma omp barrier
    // #pragma omp single
    //       {
    //         for (auto i = 0; i <= _tasks; i++) {
    //           std::cout << _taskInvAdjGraph.ai[i] << std::endl;
    //         }
    //       }

    // rebalance the work load
    auto [start2, end2] = utils::LoadPrefixBalancedPartitionPos(
        _taskInvAdjGraph.ai.begin(), _taskInvAdjGraph.ai.begin() + _tasks, tid,
        nthreads);

    COLTYPE maxInvAdjSize = 0;
    for (auto task = start2; task < end2; task++) {
      maxInvAdjSize = std::max(maxInvAdjSize, _taskInvAdjGraph.ai[task + 1] -
                                                  _taskInvAdjGraph.ai[task]);
    }

    auto startThread =
        std::distance(_threadTaskPrefix.begin(),
                      upper_bound(_threadTaskPrefix.begin(),
                                  _threadTaskPrefix.end(),
                                  static_cast<COLTYPE>(start2))) -
        1;
    auto endThread = std::distance(_threadTaskPrefix.begin(),
                                   upper_bound(_threadTaskPrefix.begin(),
                                               _threadTaskPrefix.end(),
                                               static_cast<COLTYPE>(end2))) -
                     1;
    endThread =
        std::min(endThread, static_cast<decltype(endThread)>(_nthreads) - 1);

    // building task inverse adjacency graph
    _taskInvAdj[tid].resize(maxInvAdjSize);
    for (auto thread = startThread; thread <= endThread; thread++) {
      ROWTYPE threadCount = 0;
      const COLTYPE threadBegin = _threadTaskPrefix[thread];
      const COLTYPE threadEnd = _threadTaskPrefix[thread + 1];
      const COLTYPE startTask =
          std::max(static_cast<COLTYPE>(start2), threadBegin);
      const COLTYPE endTask = std::min(static_cast<COLTYPE>(end2), threadEnd);

      for (auto task = startTask; task < endTask; task++) {
        maxInvAdjSize = 0;
        if (task != threadBegin)
          _taskInvAdj[tid][maxInvAdjSize++] = task - 1;
        for (COLTYPE row = _taskBoundaryPrefix[task];
             row < _taskBoundaryPrefix[task + 1]; row++) {
          for (COLTYPE i = _reorderedMat.ai[row] - _reorderedMat.Base();
               i < _reorderedMat.ai[row + 1] - _reorderedMat.Base(); i++) {
            COLTYPE j = _reorderedMat.aj[i] - _reorderedMat.Base();
            auto col =
                _reorderedRowIdToTaskId[_threadperm[j] - _reorderedMat.Base()];
            if (col < threadBegin || col >= threadEnd) {
              _taskInvAdj[tid][maxInvAdjSize++] = col;
            }
          }
        }
        std::sort(_taskInvAdj[tid].begin(),
                  _taskInvAdj[tid].begin() + maxInvAdjSize);
        maxInvAdjSize = std::distance(
            _taskInvAdj[tid].begin(),
            std::unique(_taskInvAdj[tid].begin(),
                        _taskInvAdj[tid].begin() + maxInvAdjSize));

        _taskAdjGraph.ai[task + 1] = maxInvAdjSize;
        std::copy(_taskInvAdj[tid].begin(),
                  _taskInvAdj[tid].begin() + maxInvAdjSize,
                  _taskInvAdjGraph.aj.begin() + _taskInvAdjGraph.ai[task]);
        threadCount += maxInvAdjSize;
      }
      __atomic_add_fetch(&_threadPrefixSum[thread + 1], threadCount,
                         __ATOMIC_RELAXED);
      // #pragma omp critical
      //         std::cout << "tid: " << tid << " threadCount: " <<
      //         threadCount
      //                   << std::endl;
    }

#pragma omp barrier
#pragma omp single
    {
      std::inclusive_scan(_threadPrefixSum.begin(), _threadPrefixSum.end(),
                          _threadPrefixSum.begin());
      _taskAdjGraph.aj.resize(_threadPrefixSum[_nthreads]);
      _taskAdjGraph.ai[_tasks] = _threadPrefixSum[_nthreads];
    }

    _taskAdjGraph.ai[_threadTaskPrefix[tid]] = _threadPrefixSum[tid];
    for (auto task = _threadTaskPrefix[tid];
         task < _threadTaskPrefix[tid + 1] - 1; task++) {
      _taskAdjGraph.ai[task + 1] += _taskAdjGraph.ai[task];
    }

#pragma omp barrier
    for (auto task = start2; task < end2; task++) {
      std::copy_n(_taskInvAdjGraph.aj.begin() + _taskInvAdjGraph.ai[task],
                  _taskAdjGraph.ai[task + 1] - _taskAdjGraph.ai[task],
                  _taskAdjGraph.aj.begin() + _taskAdjGraph.ai[task]);
    }
  }

  std::swap(_taskAdjGraph, _taskInvAdjGraph);

  // std::ifstream f("test.bin");
  // if (!f.good()) {
  //   std::ofstream ofs("test.bin", std::ios::binary);
  //   std::stringstream ss;
  //   cereal::BinaryOutputArchive oarchive(ss);
  //   oarchive(_taskInvAdjGraph);
  //   ofs << ss.rdbuf();
  // } else {
  //   std::ifstream ofs("test.bin", std::ios::binary);
  //   std::stringstream ss;
  //   ss << ofs.rdbuf();
  //   ofs.close();
  //   CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE> temp;
  //   cereal::BinaryInputArchive iarchive(ss);
  //   iarchive(temp);
  //   for (auto i = 0; i < temp.aj.size(); i++) {
  //     if (temp.aj[i] != _taskInvAdjGraph.aj[i])
  //       std::cout << "fucked\n";
  //   }
  //   for (auto i = 0; i < temp.ai.size(); i++) {
  //     if (temp.ai[i] != _taskInvAdjGraph.ai[i])
  //       std::cout << "fucked\n";
  //   }
  // }

  _taskAdjGraph.aj.resize(_taskInvAdjGraph.NNZ());
  matrix_utils::ParallelTranspose2(
      _taskInvAdjGraph.rows, _taskInvAdjGraph.cols,
      _taskInvAdjGraph.ai.data(), _taskInvAdjGraph.aj.data(),
      (VALTYPE const *)nullptr, _taskAdjGraph.ai.data(),
      _taskAdjGraph.aj.data(), (VALTYPE *)nullptr);
  // matrix_utils::SerialTranspose(
  //     _taskInvAdjGraph.rows, _taskInvAdjGraph.cols, _taskInvAdjGraph.Base(),
  //     _taskInvAdjGraph.ai.data(), _taskInvAdjGraph.aj.data(),
  //     (VALTYPE const *)nullptr, _taskAdjGraph.ai.data(),
  //     _taskAdjGraph.aj.data(), (VALTYPE *)nullptr);

  _taskInvAdjGraph2.rows = _tasks;
  _taskInvAdjGraph2.cols = _tasks;
  _taskInvAdjGraph2.ai.resize(_tasks + 1);
  _taskInvAdjGraph2.ai[0] = 0; // zero based
  // _taskInvAdjGraph2.aj.resize(_taskInvAdjGraph.aj.size());
  _transitiveEdgeRemoveAj.resize(_taskInvAdjGraph.aj.size());

#ifdef DEBUG
  std::cout << "_taskAdjGraph is valid: "
            << matrix_utils::ValidCSR(
                   _taskAdjGraph.rows, _taskAdjGraph.cols, _taskAdjGraph.Base(),
                   _taskAdjGraph.ai.data(), _taskAdjGraph.aj.data())
            << std::endl;

  std::cout << "_taskInvAdjGraph is valid: "
            << matrix_utils::ValidCSR(
                   _taskInvAdjGraph.rows, _taskInvAdjGraph.cols,
                   _taskInvAdjGraph.Base(), _taskInvAdjGraph.ai.data(),
                   _taskInvAdjGraph.aj.data())
            << std::endl;
#endif

  // for (ROWTYPE i = 0; i < _taskInvAdjGraph.rows; i++) {
  //   for (COLTYPE j = _taskInvAdjGraph.ai[i]; j < _taskInvAdjGraph.ai[i +
  //   1];
  //        j++) {
  //     std::cout << _taskInvAdjGraph.aj[j] << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // std::cout << std::endl;

#pragma omp parallel num_threads(_nthreads)
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();

    // rebalance the work load
    auto [start3, end3] = utils::LoadPrefixBalancedPartitionPos(
        _taskAdjGraph.ai.begin(), _taskAdjGraph.ai.begin() + _tasks, tid,
        nthreads);

    auto startThread =
        std::distance(_threadTaskPrefix.begin(),
                      upper_bound(_threadTaskPrefix.begin(),
                                  _threadTaskPrefix.end(),
                                  static_cast<COLTYPE>(start3))) -
        1;
    auto endThread = std::distance(_threadTaskPrefix.begin(),
                                   upper_bound(_threadTaskPrefix.begin(),
                                               _threadTaskPrefix.end(),
                                               static_cast<COLTYPE>(end3))) -
                     1;
    endThread =
        std::min(endThread, static_cast<decltype(endThread)>(_nthreads) - 1);

    ROWTYPE threadCount = 0;
    COLTYPE maxInvAdjSize = 0;
    COLTYPE parent;
    _threadPrefixSum[tid + 1] = 0;
    for (auto thread = startThread; thread <= endThread; thread++) {
      threadCount = 0;
      const COLTYPE threadBegin = _threadTaskPrefix[thread];
      const COLTYPE threadEnd = _threadTaskPrefix[thread + 1];
      const COLTYPE startTask =
          std::max(static_cast<COLTYPE>(start3), threadBegin);
      const COLTYPE endTask = std::min(static_cast<COLTYPE>(end3), threadEnd);
      for (auto task = startTask; task < endTask; task++) {
        maxInvAdjSize = 0;

        for (ROWTYPE parentID = _taskInvAdjGraph.ai[task];
             parentID < _taskInvAdjGraph.ai[task + 1]; parentID++) {
          parent = _taskInvAdjGraph.aj[parentID];
          auto parentPtr =
              _taskInvAdjGraph.aj.data() + _taskInvAdjGraph.ai[task];
          auto parentEndPtr =
              _taskInvAdjGraph.aj.data() + _taskInvAdjGraph.ai[task + 1];
          auto childPtr = _taskAdjGraph.aj.data() + _taskAdjGraph.ai[parent];
          auto childEndPtr =
              _taskAdjGraph.aj.data() + _taskAdjGraph.ai[parent + 1];

          bool remove = false;
          if (parentPtr < parentEndPtr) {
            childPtr = std::lower_bound(childPtr, childEndPtr, *parentPtr);
          }
          if (childPtr < childEndPtr) {
            parentPtr = std::lower_bound(parentPtr, parentEndPtr, *childPtr);
          }

          while (parentPtr != parentEndPtr && childPtr != childEndPtr) {
            COLTYPE cmp = *parentPtr - *childPtr;
            if (0 == cmp) {
              remove = true;
              break;
            } else if (cmp < 0)
              ++parentPtr;
            else
              ++childPtr;
          }
#pragma omp critical
{

          if (!remove) {
            std::cout << "tid: " << tid << " task " << task
                      << " removing edge to parent: " << parent << " maxInvAdjSize: " << maxInvAdjSize << std::endl;
            std::cout << _taskInvAdj[tid].size() << std::endl;
            _taskInvAdj[tid][maxInvAdjSize++] = parent;
            std::cout<<"hello"<<std::endl;
          }
}
        }
        _taskInvAdjGraph2.ai[task + 1] = maxInvAdjSize;
        std::copy(_taskInvAdj[tid].begin(),
                  _taskInvAdj[tid].begin() + maxInvAdjSize,
                  _transitiveEdgeRemoveAj.begin() + _taskInvAdjGraph.ai[task]);
        threadCount += maxInvAdjSize;
        // #pragma omp critical
        //           {
        //             std::cout << "tid: " << tid << " task " << task
        //                       << " : start point: " <<
        //                       _taskInvAdjGraph.ai[task]
        //                       << " | ";
        //             for (int i = 0; i < maxInvAdjSize; i++) {
        //               std::cout << _taskInvAdj[tid][i] << " ";
        //             }
        //             std::cout << std::endl;
        //           }
      }
      __atomic_add_fetch(&_threadPrefixSum[thread + 1], threadCount,
                         __ATOMIC_RELAXED);
    }
#pragma omp barrier
#pragma omp single
    {
      std::inclusive_scan(_threadPrefixSum.begin(), _threadPrefixSum.end(),
                          _threadPrefixSum.begin());
      _taskInvAdjGraph2.aj.resize(_threadPrefixSum[_nthreads]);
      _taskInvAdjGraph2.ai[_tasks] = _threadPrefixSum[_nthreads];
    }

    _taskInvAdjGraph2.ai[_threadTaskPrefix[tid]] = _threadPrefixSum[tid];
    for (auto task = _threadTaskPrefix[tid];
         task < _threadTaskPrefix[tid + 1] - 1; task++) {
      _taskInvAdjGraph2.ai[task + 1] += _taskInvAdjGraph2.ai[task];
    }

#pragma omp barrier

    for (auto task = start3; task < end3; task++) {
      std::copy(_transitiveEdgeRemoveAj.begin() + _taskInvAdjGraph.ai[task],
                _transitiveEdgeRemoveAj.begin() + _taskInvAdjGraph.ai[task] +
                    _taskInvAdjGraph2.ai[task + 1] - _taskInvAdjGraph2.ai[task],
                _taskInvAdjGraph2.aj.begin() + _taskInvAdjGraph2.ai[task]);
    }
  }

  // // sanity check
  // {
  //   std::cout << "_taskInvAdjGraph2 is valid: "
  //             << matrix_utils::ValidCSR(
  //                    _taskInvAdjGraph2.rows, _taskInvAdjGraph2.cols,
  //                    _taskInvAdjGraph2.Base(), _taskInvAdjGraph2.ai.data(),
  //                    _taskInvAdjGraph2.aj.data())
  //             << std::endl;

  //   std::ifstream f("test.bin");
  //   if (!f.good()) {
  //     std::ofstream ofs("test.bin", std::ios::binary);
  //     std::stringstream ss;
  //     cereal::BinaryOutputArchive oarchive(ss);
  //     oarchive(_taskInvAdjGraph2);
  //     ofs << ss.rdbuf();
  //   } else {
  //     std::ifstream ofs("test.bin", std::ios::binary);
  //     std::stringstream ss;
  //     ss << ofs.rdbuf();
  //     ofs.close();
  //     CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE> temp;
  //     cereal::BinaryInputArchive iarchive(ss);
  //     iarchive(temp);
  //     for (auto i = 0; i < temp.aj.size(); i++) {
  //       if (temp.aj[i] != _taskInvAdjGraph2.aj[i])
  //         std::cout << "fucked\n";
  //     }
  //     for (auto i = 0; i < temp.ai.size(); i++) {
  //       if (temp.ai[i] != _taskInvAdjGraph2.ai[i])
  //         std::cout << "fucked\n";
  //     }
  //     std::cout << _taskInvAdjGraph.NNZ() << " " <<
  //     _taskInvAdjGraph2.NNZ()
  //               << std::endl;
  //     std::cout << "finished check\n";
  //   }
  // }
}

// template class TriangularSolve<int, int, double>;
template void TriangularSolve<TriangularMatrix::L, int, int, double>(
    const int, int const*, int const*, double const*, double const*, double const*, double* );

template void TriangularSolve<TriangularMatrix::U, int, int, double>(
    const int, int const*, int const*, double const*, double const*, double const*, double* );

template void TriangularSolveCSC<TriangularMatrix::L, int, int, double>(
    const int, int const*, int const*, double const*, double const*, double const*, double* );

template void TriangularSolveCSC<TriangularMatrix::U, int, int, double>(
    const int, int const*, int const*, double const*, double const*, double const*, double* );

template class LevelScheduleTriangularSubstitution<TriangularMatrix::L, int, int, double>;
template class LevelScheduleTriangularSubstitution<TriangularMatrix::U, int, int, double>;

template class JacobiTriangularSubstitution<TriangularMatrix::L, int, int, double>;
template class JacobiTriangularSubstitution<TriangularMatrix::U, int, int, double>;

template class P2PTriangularSubstitution<TriangularMatrix::L, int, int, double>;
template class P2PTriangularSubstitution<TriangularMatrix::U, int, int, double>;

template class OptimizedTriangularSolve<FBSubstitutionType::Barrier, TriangularMatrix::L, int, int, double>;

template class OptimizedTriangularSolve<FBSubstitutionType::Barrier, TriangularMatrix::U, int, int, double>;

template class OptimizedTriangularSolve<FBSubstitutionType::NoBarrier, TriangularMatrix::L, int, int, double>;

template class OptimizedTriangularSolve<FBSubstitutionType::NoBarrier, TriangularMatrix::U, int, int, double>;

template class OptimizedTriangularSolve<FBSubstitutionType::NoBarrierSuperNode, TriangularMatrix::L, int, int, double>;

template class OptimizedTriangularSolve<FBSubstitutionType::NoBarrierSuperNode, TriangularMatrix::U, int, int, double>;

} // namespace matrix_utils
