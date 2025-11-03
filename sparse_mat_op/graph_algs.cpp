#include "graph_algs.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <limits>
#include <omp.h>
#include <queue>
#include <iostream>
#include <thread>
#include <vector>
#include <iterator>
#include <immintrin.h>

#ifndef TRANSITIVE_REDUCTION_USE_PARALLEL
#define TRANSITIVE_REDUCTION_USE_PARALLEL 1
#endif

namespace matrix_utils
{

template <typename ROWTYPE, typename COLTYPE>
void ElimTree( const COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj, COLTYPE* parent )
{
    const int base = ai[0];
    const COLTYPE empty_tag = std::numeric_limits<COLTYPE>::max();
    // initialize parent
    std::fill_n( parent, rows, empty_tag );

    COLTYPE jroot = empty_tag;
    for ( COLTYPE i = 0; i < rows; i++ )
    {
        for ( ROWTYPE j = ai[i] - base, jroot = aj[j] - base;
              j < ai[i + 1] - base && jroot < i; j++ )
        {
            while ( parent[jroot] != empty_tag && parent[jroot] != i + base )
            {
                jroot = parent[jroot] - base;
            }
            if ( parent[jroot] == empty_tag )
                parent[jroot] = i + base;
        }
    }
}

template <typename ROWTYPE, typename COLTYPE>
bool IsDAG( const COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj )
{
    const int base = ai[0];
    std::vector<int> visited( rows, 0 ); // 0: unvisited, 1: visiting, 2: visited

    std::function<bool( COLTYPE )> dfs = [&]( COLTYPE u )
    {
        visited[u] = 1;
        for ( ROWTYPE j = ai[u] - base; j < ai[u + 1] - base; ++j )
        {
            COLTYPE v = aj[j] - base;
            if ( visited[v] == 1 )
                return false; // cycle detected
            if ( visited[v] == 0 && !dfs( v ) )
                return false;
        }
        visited[u] = 2;
        return true;
    };

    for ( COLTYPE i = 0; i < rows; ++i )
    {
        if ( visited[i] == 0 && !dfs( i ) )
            return false;
    }
    return true;
}

template <typename ROWTYPE, typename COLTYPE>
COLTYPE ProjectGraphToTaskGraph<ROWTYPE, COLTYPE>::operator()( const COLTYPE work_graph_rows,
                                                               ROWTYPE const* work_ai,
                                                               COLTYPE const* work_aj,
                                                               const COLTYPE num_tasks,
                                                               COLTYPE const* task_prefix,
                                                               COLTYPE const* task_to_work,
                                                               COLTYPE const* work_to_task,
                                                               ROWTYPE* task_ai,
                                                               COLTYPE* task_aj )
{
    const ROWTYPE work_base = work_ai[0];
    const COLTYPE task_base = task_prefix[0];

    // Initialize task graph row pointers
    task_ai[0] = task_base;

    // Resize and clear reusable memory for task dependencies
    _task_dependencies.resize( num_tasks );
    for ( auto& deps : _task_dependencies )
    {
        deps.clear();
    }

    // Clear thread-local storage (size already set in constructor)
    for ( auto& thread_vec : _thread_local_storage )
    {
        thread_vec.clear();
    }

    // First pass: compute dependency counts for each task in parallel
#pragma omp parallel num_threads( _nthreads )
    {
        // Get thread-specific storage
        int thread_id = omp_get_thread_num();
        std::vector<COLTYPE>& local_dependent_tasks = _thread_local_storage[thread_id];

#pragma omp for schedule( dynamic, 16 )
        for ( COLTYPE task_id = 0; task_id < num_tasks; ++task_id )
        {
            local_dependent_tasks.clear();

            // Get all work items assigned to this task
            COLTYPE task_work_start = task_prefix[task_id] - task_base;
            COLTYPE task_work_end = task_prefix[task_id + 1] - task_base;

            // For each work item in this task
            for ( COLTYPE work_offset = task_work_start;
                  work_offset < task_work_end; ++work_offset )
            {
                COLTYPE work_idx = task_to_work[work_offset] - work_base;

                // Look at all dependencies of this work item in the original graph
                for ( ROWTYPE adj_idx = work_ai[work_idx] - work_base;
                      adj_idx < work_ai[work_idx + 1] - work_base; ++adj_idx )
                {
                    COLTYPE dep_work_idx = work_aj[adj_idx] - work_base;

                    // Find which task this dependency work belongs to
                    COLTYPE dep_task_id = work_to_task[dep_work_idx] - task_base;

                    // Only add edge if dependency is from a different task
                    if ( dep_task_id != task_id )
                    {
                        local_dependent_tasks.push_back( dep_task_id );
                    }
                }
            }

            // Remove duplicates and sort
            if ( !local_dependent_tasks.empty() )
            {
                std::sort( local_dependent_tasks.begin(), local_dependent_tasks.end() );
                auto unique_end = std::unique( local_dependent_tasks.begin(),
                                               local_dependent_tasks.end() );
                local_dependent_tasks.erase( unique_end, local_dependent_tasks.end() );
            }

            // Store the dependencies for this task (reusing memory)
            _task_dependencies[task_id] =
                local_dependent_tasks; // Copy instead of move to preserve thread-local storage
        }
    }

    // Second pass: compute row pointers sequentially (dependencies needed)
    ROWTYPE total_task_edges = 0;
    for ( COLTYPE task_id = 0; task_id < num_tasks; ++task_id )
    {
        task_ai[task_id + 1] =
            task_ai[task_id] + static_cast<ROWTYPE>( _task_dependencies[task_id].size() );
        total_task_edges += static_cast<ROWTYPE>( _task_dependencies[task_id].size() );
    }

    // Third pass: fill adjacency array in parallel using computed offsets
#pragma omp parallel for schedule( dynamic, 16 ) num_threads( _nthreads )
    for ( COLTYPE task_id = 0; task_id < num_tasks; ++task_id )
    {
        ROWTYPE start_offset = task_ai[task_id] - task_base;
        const auto& deps = _task_dependencies[task_id];

        for ( size_t i = 0; i < deps.size(); ++i )
        {
            task_aj[start_offset + i] = deps[i] + task_base;
        }
    }

    return total_task_edges;
}

template <typename ROWTYPE, typename COLTYPE>
void TransitiveReduction<ROWTYPE, COLTYPE>::operator()( const COLTYPE rows,
                                                        ROWTYPE const* ai,
                                                        COLTYPE const* aj,
                                                        ROWTYPE* out_ai,
                                                        COLTYPE* out_aj,
                                                        bool has_self_loops )
{
    const int base = ai[0];

    // Initialize reachability storage - resize instead of assign to reuse memory
    if ( _reachable.size() != rows )
    {
        _reachable.resize( rows );
    }
    for ( auto& reachable_row : _reachable )
    {
        reachable_row.clear(); // Clear contents but keep allocated memory
    }

    // Lambda to check if node u can reach node v
    auto can_reach = [&]( COLTYPE u, COLTYPE v ) -> bool
    {
        const auto& reachable_from_u = _reachable[u];
        return std::binary_search( reachable_from_u.begin(), reachable_from_u.end(), v );
    };

    const int nthreads = std::max( 1, _nthreads );

    const auto reachability_start = std::chrono::steady_clock::now();
#if TRANSITIVE_REDUCTION_USE_PARALLEL
    std::vector<std::atomic<bool>> ready_flags( static_cast<std::size_t>( rows ) );
    for ( auto& flag : ready_flags )
    {
        flag.store( false, std::memory_order_relaxed );
    }

#pragma omp parallel num_threads( nthreads )
    {
        std::vector<COLTYPE> merge_buffer;

        auto merge_into = [&]( const COLTYPE* begin, const COLTYPE* end, std::vector<COLTYPE>& result )
        {
            if ( begin == end )
                return;
            merge_buffer.clear();
            merge_buffer.reserve( result.size() + static_cast<std::size_t>( end - begin ) );
            std::merge( result.begin(), result.end(), begin, end, std::back_inserter( merge_buffer ) );
            merge_buffer.erase( std::unique( merge_buffer.begin(), merge_buffer.end() ), merge_buffer.end() );
            result.swap( merge_buffer );
        };

#pragma omp for schedule( dynamic )
        for ( COLTYPE offset = 0; offset < rows; ++offset )
        {
            const COLTYPE node_j = rows - 1 - offset;

            auto& result = _reachable[node_j];
            result.clear();

            if ( !has_self_loops )
            {
                result.push_back( node_j );
            }

            const ROWTYPE row_begin = ai[node_j] - base;
            const ROWTYPE row_end = ai[node_j + 1] - base;

            for ( ROWTYPE k = row_end; k-- > row_begin; )
            {
                const COLTYPE neighbor = aj[k] - base;

                while ( neighbor > node_j && !ready_flags[neighbor].load( std::memory_order_acquire ) )
                {
                    _mm_pause();
                }

                const auto& reachable_from_neighbor = _reachable[neighbor];
                if ( !reachable_from_neighbor.empty() )
                {
                    merge_into( reachable_from_neighbor.data(),
                                reachable_from_neighbor.data() + reachable_from_neighbor.size(), result );
                }
            }

            ready_flags[node_j].store( true, std::memory_order_release );
        }
    }
#else
    std::vector<std::pair<COLTYPE*, COLTYPE*>> sequences;
    sequences.reserve( 16 );
    std::vector<COLTYPE> direct_neighbors;
    direct_neighbors.reserve( 64 );

    for ( COLTYPE j = rows; j > 0; --j )
    {
        const COLTYPE node_j = j - 1;

        sequences.clear();
        direct_neighbors.clear();

        if ( has_self_loops )
        {
            for ( ROWTYPE k = ai[node_j] - base; k < ai[node_j + 1] - base; ++k )
            {
                direct_neighbors.push_back( aj[k] - base );
            }
        }
        else
        {
            direct_neighbors.push_back( node_j );
            for ( ROWTYPE k = ai[node_j] - base; k < ai[node_j + 1] - base; ++k )
            {
                direct_neighbors.push_back( aj[k] - base );
            }
        }

        if ( !direct_neighbors.empty() )
        {
            sequences.emplace_back( direct_neighbors.data(),
                                    direct_neighbors.data() + direct_neighbors.size() );
        }

        for ( ROWTYPE k = ai[node_j] - base; k < ai[node_j + 1] - base; ++k )
        {
            const COLTYPE neighbor = aj[k] - base;
            const auto& reachable_from_neighbor = _reachable[neighbor];
            if ( !reachable_from_neighbor.empty() )
            {
                sequences.emplace_back( const_cast<COLTYPE*>( reachable_from_neighbor.data() ),
                                        const_cast<COLTYPE*>( reachable_from_neighbor.data() +
                                                              reachable_from_neighbor.size() ) );
            }
        }

        auto& result = _reachable[node_j];
        result.clear();

        if ( !sequences.empty() )
        {
            using HeapElement = std::pair<COLTYPE, std::pair<COLTYPE*, COLTYPE*>>;
            std::priority_queue<HeapElement, std::vector<HeapElement>, std::greater<HeapElement>> heap;

            for ( auto& seq : sequences )
            {
                if ( seq.first < seq.second )
                {
                    heap.emplace( *seq.first, std::make_pair( seq.first, seq.second ) );
                }
            }

            COLTYPE last_added = std::numeric_limits<COLTYPE>::max();
            while ( !heap.empty() )
            {
                auto [value, ptrs] = heap.top();
                heap.pop();

                if ( value != last_added )
                {
                    result.push_back( value );
                    last_added = value;
                }

                ptrs.first++;
                if ( ptrs.first < ptrs.second )
                {
                    heap.emplace( *ptrs.first, ptrs );
                }
            }
        }
    }
#endif

    const auto reachability_end = std::chrono::steady_clock::now();
    const auto reachability_ms = std::chrono::duration_cast<std::chrono::milliseconds>( reachability_end - reachability_start ).count();

    if ( _reduced_edges.size() != static_cast<std::size_t>( rows ) )
    {
        _reduced_edges.resize( rows );
    }

#pragma omp parallel for num_threads( nthreads ) schedule( dynamic )
    for ( COLTYPE u = 0; u < rows; ++u )
    {
        auto& row_edges = _reduced_edges[u];
        row_edges.clear();

        const ROWTYPE row_start = ai[u] - base;
        const ROWTYPE row_end = ai[u + 1] - base;
        if ( row_end > row_start )
        {
            row_edges.reserve( static_cast<std::size_t>( row_end - row_start ) );
        }

        for ( ROWTYPE j = row_start; j < row_end; ++j )
        {
            COLTYPE v = aj[j] - base;
            bool is_transitive = false;

            for ( ROWTYPE k = row_start; k < row_end; ++k )
            {
                COLTYPE w = aj[k] - base;
                if ( w != v && can_reach( w, v ) )
                {
                    is_transitive = true;
                    break;
                }
            }

            if ( !is_transitive )
            {
                row_edges.push_back( v + base );
            }
        }
    }

    const auto reduction_start = std::chrono::steady_clock::now();

    out_ai[0] = base;
    ROWTYPE out_nnz = 0;
    for ( COLTYPE u = 0; u < rows; ++u )
    {
        const auto& row_edges = _reduced_edges[u];
        const ROWTYPE edge_count = static_cast<ROWTYPE>( row_edges.size() );
        for ( COLTYPE edge : row_edges )
        {
            out_aj[out_nnz++] = edge;
        }
        out_ai[u + 1] = out_ai[u] + edge_count;
    }

    const auto reduction_end = std::chrono::steady_clock::now();
    const auto reduction_ms = std::chrono::duration_cast<std::chrono::milliseconds>( reduction_end - reachability_end ).count();

    std::cout << "Transitive reduction timing -- reachability: " << reachability_ms
              << " ms, reduction: " << reduction_ms << " ms" << std::endl;
}

// Template instantiations
#define INSTANTIATE_GRAPH_ALGS( ROWTYPE, COLTYPE )                                   \
    template void ElimTree<ROWTYPE, COLTYPE>(                                        \
        const COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj, COLTYPE* parent ); \
    template bool IsDAG<ROWTYPE, COLTYPE>(                                           \
        const COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj );                  \
    template struct ProjectGraphToTaskGraph<ROWTYPE, COLTYPE>;                       \
    template struct TransitiveReduction<ROWTYPE, COLTYPE>;

// INSTANTIATE_GRAPH_ALGS(int, int)
INSTANTIATE_GRAPH_ALGS( std::int32_t, std::int32_t )
INSTANTIATE_GRAPH_ALGS( std::int64_t, std::int64_t )

#undef INSTANTIATE_GRAPH_ALGS

} // namespace matrix_utils
