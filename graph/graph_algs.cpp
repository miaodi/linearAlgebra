#include "graph_algs.hpp"
#include "spadd.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <limits>
#include <omp.h>
#include <iostream>
#include <vector>
#include <set>
#include <iterator>
#include <immintrin.h>
#include <numeric>
#include <stdexcept>
#include "matrix_utils.hpp"

namespace graph
{
using enums::matrix_utils::TriangularMatrix;

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
        for ( ROWTYPE j = ai[i] - base, jroot = aj[j] - base; j < ai[i + 1] - base && jroot < i; j++ )
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

template <typename ROWTYPE, typename COLTYPE, bool KEEPDIAG>
COLTYPE ProjectGraphToTaskGraph<ROWTYPE, COLTYPE, KEEPDIAG>::operator()( const COLTYPE work_graph_rows,
                                                                         ROWTYPE const* work_ai,
                                                                         COLTYPE const* work_aj,
                                                                         const COLTYPE num_tasks,
                                                                         COLTYPE const* task_prefix,
                                                                         COLTYPE const* task_to_node,
                                                                         COLTYPE const* node_to_task,
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

    ROWTYPE total_task_edges = 0;

    // Parallel region handles both passes and the sequential prefix calculation
#pragma omp parallel num_threads( _nthreads )
    {
#pragma omp for schedule( dynamic, 16 )
        for ( COLTYPE task_id = 0; task_id < num_tasks; ++task_id )
        {
            auto& deps = _task_dependencies[task_id];
            deps.clear();

            // Get all work items assigned to this task
            COLTYPE task_work_start = task_prefix[task_id] - task_base;
            COLTYPE task_work_end = task_prefix[task_id + 1] - task_base;

            // For each work item in this task
            for ( COLTYPE work_offset = task_work_start; work_offset < task_work_end; ++work_offset )
            {
                COLTYPE work_idx = task_to_node[work_offset] - work_base;

                // Look at all dependencies of this work item in the original graph
                for ( ROWTYPE adj_idx = work_ai[work_idx] - work_base;
                      adj_idx < work_ai[work_idx + 1] - work_base; ++adj_idx )
                {
                    COLTYPE dep_work_idx = work_aj[adj_idx] - work_base;

                    // Find which task this dependency work belongs to
                    COLTYPE dep_task_id = node_to_task[dep_work_idx] - task_base;

                    // Only add edge if dependency is from a different task (unless KEEPDIAG is true)
                    if constexpr ( KEEPDIAG )
                    {
                        deps.insert( dep_task_id ); // set automatically handles duplicates and sorting
                    }
                    else
                    {
                        if ( dep_task_id != task_id )
                        {
                            deps.insert( dep_task_id ); // set automatically handles duplicates and sorting
                        }
                    }
                }
            }
        }

        // Compute row pointers sequentially once dependencies are ready
#pragma omp single
        {
            total_task_edges = 0;
            for ( COLTYPE task_id = 0; task_id < num_tasks; ++task_id )
            {
                task_ai[task_id + 1] =
                    task_ai[task_id] + static_cast<ROWTYPE>( _task_dependencies[task_id].size() );
                total_task_edges += static_cast<ROWTYPE>( _task_dependencies[task_id].size() );
            }
        }

#pragma omp for schedule( dynamic, 16 )
        for ( COLTYPE task_id = 0; task_id < num_tasks; ++task_id )
        {
            ROWTYPE start_offset = task_ai[task_id] - task_base;
            const auto& deps = _task_dependencies[task_id];

            // Copy from set to output array using iterator
            ROWTYPE idx = start_offset;
            for ( COLTYPE dep : deps )
            {
                task_aj[idx++] = dep + task_base;
            }
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

    std::vector<std::atomic<bool>> ready_flags( static_cast<std::size_t>( rows ) );
    for ( auto& flag : ready_flags )
    {
        flag.store( false, std::memory_order_relaxed );
    }

    std::chrono::steady_clock::time_point reachability_end;

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
            merge_buffer.erase( std::unique( merge_buffer.begin(), merge_buffer.end() ),
                                merge_buffer.end() );
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

#pragma omp single
        {
            reachability_end = std::chrono::steady_clock::now();
            if ( _reduced_edges.size() != static_cast<std::size_t>( rows ) )
            {
                _reduced_edges.resize( rows );
            }
        }

#pragma omp for schedule( dynamic )
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
    }

    const auto reachability_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>( reachability_end - reachability_start )
            .count();

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
    const auto reduction_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>( reduction_end - reachability_end ).count();

    std::cout << "Transitive reduction timing -- reachability: " << reachability_ms
              << " ms, reduction: " << reduction_ms << " ms" << std::endl;
}

template <typename ROWTYPE, typename COLTYPE>
COLTYPE FindStronglyConnectedComponents( const COLTYPE rows,
                                         ROWTYPE const* ai,
                                         COLTYPE const* aj,
                                         ROWTYPE* scc_prefix,
                                         COLTYPE* scc_to_node,
                                         COLTYPE* node_to_scc )
{
    // Tarjan's algorithm for finding strongly connected components
    // Returns the number of SCCs and populates output arrays
    const int base = ai[0];

    // Algorithm state
    std::vector<COLTYPE> index( rows, -1 );   // Discovery index (-1 = unvisited)
    std::vector<COLTYPE> lowlink( rows, -1 ); // Lowest reachable index
    std::vector<char> on_stack( rows, false );
    std::vector<COLTYPE> stack;
    stack.reserve( rows );

    COLTYPE current_index = 0;
    COLTYPE scc_count = 0;

    // Temporary storage for SCC nodes
    std::vector<std::vector<COLTYPE>> scc_nodes;

    // Iterative DFS to avoid stack overflow on large graphs
    std::vector<COLTYPE> dfs_stack;
    std::vector<ROWTYPE> edge_iter; // Iterator for edges of each node on stack
    dfs_stack.reserve( rows );
    edge_iter.reserve( rows );

    for ( COLTYPE start = 0; start < rows; ++start )
    {
        if ( index[start] != -1 )
            continue;

        // Start DFS from unvisited node
        dfs_stack.clear();
        edge_iter.clear();
        dfs_stack.push_back( start );
        edge_iter.push_back( ai[start] - base );

        while ( !dfs_stack.empty() )
        {
            COLTYPE u = dfs_stack.back();
            ROWTYPE& edge_pos = edge_iter.back();

            // First visit to this node
            if ( index[u] == -1 )
            {
                index[u] = current_index;
                lowlink[u] = current_index;
                ++current_index;
                stack.push_back( u );
                on_stack[u] = true;
            }

            // Process edges
            bool found_unvisited = false;
            const ROWTYPE edge_end = ai[u + 1] - base;

            while ( edge_pos < edge_end )
            {
                COLTYPE v = aj[edge_pos] - base;
                ++edge_pos;

                if ( index[v] == -1 )
                {
                    // Unvisited neighbor - recurse
                    dfs_stack.push_back( v );
                    edge_iter.push_back( ai[v] - base );
                    found_unvisited = true;
                    break;
                }
                else if ( on_stack[v] )
                {
                    // Back edge to node on stack
                    lowlink[u] = std::min( lowlink[u], index[v] );
                }
            }

            if ( found_unvisited )
                continue;

            // All edges processed - backtrack
            dfs_stack.pop_back();
            edge_iter.pop_back();

            // Update parent's lowlink if we're not the root
            if ( !dfs_stack.empty() )
            {
                COLTYPE parent = dfs_stack.back();
                lowlink[parent] = std::min( lowlink[parent], lowlink[u] );
            }

            // Check if u is a root of an SCC
            if ( lowlink[u] == index[u] )
            {
                // Pop all nodes in this SCC from stack
                std::vector<COLTYPE> current_scc;
                COLTYPE v;
                do
                {
                    v = stack.back();
                    stack.pop_back();
                    on_stack[v] = false;
                    node_to_scc[v] = scc_count + base;
                    current_scc.push_back( v );
                } while ( v != u );

                scc_nodes.push_back( std::move( current_scc ) );
                ++scc_count;
            }
        }
    }

    // Build scc_prefix and scc_to_node arrays
    scc_prefix[0] = base;
    ROWTYPE offset = 0;
    for ( COLTYPE scc = 0; scc < scc_count; ++scc )
    {
        const auto& nodes = scc_nodes[scc];
        for ( COLTYPE node : nodes )
        {
            scc_to_node[offset++] = node + base;
        }
        scc_prefix[scc + 1] = offset + base;
    }

    return scc_count;
}

template <typename ROWTYPE, typename COLTYPE>
COLTYPE MISPerm( COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj, COLTYPE* perm, COLTYPE* iperm )
{
    const ROWTYPE base = ai[0];
    std::vector<int> visited( size, 0 ); // 0: not seen, 1: seen, 2: visited

    std::vector<COLTYPE> degree( size, 0 );
    for ( COLTYPE i = 0; i < size; ++i )
    {
        degree[i] = ai[i + 1] - ai[i];
    }

    // Initialize perm with identity permutation
    std::iota( perm, perm + size, base );

    // Sort nodes by degree while keeping perm and degree vectors in sync
    std::vector<std::pair<COLTYPE, COLTYPE>> degree_perm( size );
    for ( COLTYPE i = 0; i < size; ++i )
    {
        degree_perm[i] = { degree[i], perm[i] };
    }

    std::sort( degree_perm.begin(), degree_perm.end(),
               []( const auto& a, const auto& b )
               {
                   if ( a.first != b.first )
                   {
                       return a.first > b.first; // Descending by degree
                   }
                   return a.second < b.second; // Ascending by original index for ties
               } );

    COLTYPE idx = 0;
    // First pass: find all "not seen" nodes (state 0) and add them to perm
    for ( COLTYPE i = 0; i < size; ++i )
    {
        COLTYPE node = degree_perm[i].second - base;
        if ( visited[node] == 0 ) // Not seen
        {
            perm[idx++] = degree_perm[i].second;
            visited[node] = 2; // Mark as visited

            // Mark neighbors as seen (state 1) - branchless
            for ( ROWTYPE j = ai[node] - base; j < ai[node + 1] - base; ++j )
            {
                COLTYPE neighbor = aj[j] - base;
                // If visited[neighbor] == 0, set it to 1; otherwise keep current value
                visited[neighbor] = visited[neighbor] | ( visited[neighbor] == 0 );
            }
        }
    }

    const COLTYPE is_size = idx;

    // Second pass: fill perm with remaining "seen but not visited" nodes (state 1)
    for ( COLTYPE i = 0; i < size; ++i )
    {
        COLTYPE node = degree_perm[i].second - base;
        if ( visited[node] == 1 ) // Seen but not visited
        {
            perm[idx++] = degree_perm[i].second;
        }
    }

    if ( iperm != nullptr )
    {
        // Compute inverse permutation
        for ( COLTYPE i = 0; i < size; ++i )
        {
            iperm[perm[i] - base] = i + base;
        }
    }
    return is_size;
}

template <typename ROWTYPE, typename COLTYPE>
void BuildPermutationFromSccLevels( COLTYPE num_sccs,
                                    ROWTYPE const* scc_prefix,
                                    COLTYPE const* scc_to_node,
                                    COLTYPE const* scc_perm,
                                    ROWTYPE const* scc_level_prefix,
                                    COLTYPE scc_levels,
                                    COLTYPE* node_perm,
                                    COLTYPE* node_iperm )
{
    const auto base = scc_prefix[0];
    COLTYPE pos = 0;

    for ( COLTYPE lvl = 0; lvl < scc_levels; ++lvl )
    {
        const auto level_begin = scc_level_prefix[lvl] - base;
        const auto level_end = scc_level_prefix[lvl + 1] - base;
        for ( auto idx = level_begin; idx < level_end; ++idx )
        {
            const COLTYPE scc_id = scc_perm[idx] - base;
            const auto node_begin = scc_prefix[scc_id] - base;
            const auto node_end = scc_prefix[scc_id + 1] - base;
            for ( auto n = node_begin; n < node_end; ++n )
            {
                node_perm[pos++] = scc_to_node[n];
            }
        }
    }

    if ( node_iperm != nullptr )
    {
        for ( COLTYPE i = 0; i < pos; ++i )
        {
            node_iperm[node_perm[i] - base] = i + base;
        }
    }
}

template <typename ROWTYPE, typename COLTYPE>
COLTYPE KahnSerial<ROWTYPE, COLTYPE>::operator()( const COLTYPE nodes,
                                                  ROWTYPE const* ai,
                                                  COLTYPE const* aj,
                                                  COLTYPE* perm,
                                                  COLTYPE* prefix )
{
    _degrees.resize( nodes );
    const auto base = ai[0];
    const auto nnz = ai[nodes] - base;

    _t_ai.resize( nodes + 1 );
    _t_aj.resize( nnz );
    COLTYPE processed = 0;
    COLTYPE level = 0;

    // reverse graph to get out edges
    matrix_utils::ParallelTranspose2( nodes, nodes, ai, aj, (double*)nullptr, _t_ai.data(),
                                      _t_aj.data(), (double*)nullptr );

    prefix[0] = base;

    auto degree_for = [&]( const COLTYPE i )
    {
        COLTYPE degree = 0;
        const auto row_start = ai[i] - base;
        const auto row_end = ai[i + 1] - base;
        for ( auto pos = row_start; pos < row_end; ++pos )
        {
            if ( aj[pos] - base != i )
            {
                ++degree;
            }
        }
        return degree;
    };

    for ( COLTYPE i = 0; i < nodes; ++i )
    {
        _degrees[i] = degree_for( i );
        if ( _degrees[i] == 0 )
        {
            perm[processed++] = i + base;
        }
    }

    prefix[1] = processed + base;

    auto process_row = [&]( const COLTYPE idx, const auto& handle_neighbor )
    {
        const auto row_start = _t_ai[idx] - base;
        const auto row_end = _t_ai[idx + 1] - base;
        for ( auto pos = row_start; pos < row_end; ++pos )
        {
            if ( _t_aj[pos] - base == idx )
            {
                continue;
            }
            handle_neighbor( _t_aj[pos] );
        }
    };

    // process levels
    while ( processed != nodes )
    {
        for ( COLTYPE i = prefix[level] - base; i < prefix[level + 1] - base; ++i )
        {
            const auto idx = perm[i] - base;
            process_row( idx,
                         [&]( const COLTYPE neighbor )
                         {
                             if ( --_degrees[neighbor - base] == 0 )
                             {
                                 perm[processed++] = neighbor;
                             }
                         } );
        }
        ++level;
        prefix[level + 1] = processed + base;
    }

    return level + 1;
}

template <typename ROWTYPE, typename COLTYPE>
COLTYPE KahnParallel<ROWTYPE, COLTYPE>::operator()( const COLTYPE nodes,
                                                    ROWTYPE const* ai,
                                                    COLTYPE const* aj,
                                                    COLTYPE* perm,
                                                    COLTYPE* prefix )
{
    if ( _degrees_size < nodes )
    {
        _degrees.reset( new std::atomic<COLTYPE>[nodes] );
        _degrees_size = nodes;
    }
    const auto base = ai[0];
    const auto nnz = ai[nodes] - base;

    _t_ai.resize( nodes + 1 );
    _t_aj.resize( nnz );
    COLTYPE processed = 0;
    COLTYPE level = 0;

    // reverse graph
    matrix_utils::ParallelTranspose2( nodes, nodes, ai, aj, (double*)nullptr, _t_ai.data(),
                                      _t_aj.data(), (double*)nullptr );

    prefix[0] = base;

    auto degree_for = [&]( const COLTYPE i )
    {
        COLTYPE degree = 0;
        const auto row_start = ai[i] - base;
        const auto row_end = ai[i + 1] - base;
        for ( auto pos = row_start; pos < row_end; ++pos )
        {
            if ( aj[pos] - base != i )
            {
                ++degree;
            }
        }
        return degree;
    };

    auto process_row = [&]( const COLTYPE idx, const auto& handle_neighbor )
    {
        const auto row_start = _t_ai[idx] - base;
        const auto row_end = _t_ai[idx + 1] - base;
        for ( auto pos = row_start; pos < row_end; ++pos )
        {
            if ( _t_aj[pos] - base == idx )
            {
                continue;
            }
            handle_neighbor( _t_aj[pos] );
        }
    };

#pragma omp parallel num_threads( _nthreads )
    {
        const int thread_id = omp_get_thread_num();
        _threads_nodes[thread_id].clear();
        _threads_prefix[thread_id + 1] = 0;

        auto chunk_begin = thread_id * nodes / _nthreads;
        auto chunk_end = ( thread_id + 1 ) * nodes / _nthreads;

        for ( COLTYPE i = chunk_begin; i < chunk_end; ++i )
        {
            _degrees[i] = degree_for( i );
            if ( _degrees[i] == 0 )
            {
                _threads_nodes[thread_id].push_back( i + base );
            }
        }
        _threads_prefix[thread_id + 1] = _threads_nodes[thread_id].size();
#pragma omp barrier
#pragma omp single
        {
            _threads_prefix[0] = base;
            for ( size_t i = 1; i < _threads_prefix.size(); ++i )
            {
                _threads_prefix[i] += _threads_prefix[i - 1];
            }
            prefix[1] = _threads_prefix[_nthreads];
            processed = _threads_prefix[_nthreads] - base;
        }
        auto thread_start = _threads_prefix[thread_id] - base;
        for ( const auto node : _threads_nodes[thread_id] )
        {
            perm[thread_start++] = node;
        }
#pragma omp barrier
        while ( processed != nodes )
        {
            _threads_prefix[thread_id + 1] = 0;
            _threads_nodes[thread_id].clear();
            auto start_range =
                prefix[level] - base + ( prefix[level + 1] - prefix[level] ) * thread_id / _nthreads;
            auto end_range = prefix[level] - base +
                             ( prefix[level + 1] - prefix[level] ) * ( thread_id + 1 ) / _nthreads;

            for ( COLTYPE idx_pos = start_range; idx_pos < end_range; ++idx_pos )
            {
                const auto idx = perm[idx_pos] - base;
                process_row( idx,
                             [&]( const COLTYPE neighbor )
                             {
                                 if ( _degrees[neighbor - base].fetch_sub( 1, std::memory_order_relaxed ) == 1 )
                                 {
                                     _threads_nodes[thread_id].push_back( neighbor );
                                 }
                             } );
            }
            _threads_prefix[thread_id + 1] = _threads_nodes[thread_id].size();
#pragma omp barrier
#pragma omp single
            {
                _threads_prefix[0] = processed + base;
                for ( size_t i = 1; i < _threads_prefix.size(); ++i )
                {
                    _threads_prefix[i] += _threads_prefix[i - 1];
                }
                ++level;
                prefix[level + 1] = _threads_prefix[_nthreads];
                processed = _threads_prefix[_nthreads] - base;
            }
            auto local_start = _threads_prefix[thread_id] - base;
            for ( const auto node : _threads_nodes[thread_id] )
            {
                perm[local_start++] = node;
            }
#pragma omp barrier
        }
    }

    return level + 1;
}

template <typename ROWTYPE, typename COLTYPE, TriangularMatrix TS>
COLTYPE TopologicalSort2<ROWTYPE, COLTYPE, TS>::operator()( const COLTYPE nodes,
                                                            ROWTYPE const* ai,
                                                            COLTYPE const* aj,
                                                            COLTYPE* perm,
                                                            COLTYPE* prefix )
{
    _depths.resize( nodes );
    std::fill( _depths.begin(), _depths.end(), 0 );

    const auto base = ai[0];

    COLTYPE level = 0;
    for ( COLTYPE offset = 0; offset < nodes; ++offset )
    {
        COLTYPE i = offset;
        if constexpr ( TS == TriangularMatrix::U )
        {
            i = nodes - 1 - offset;
        }
        auto row_begin = ai[i] - base;
        auto row_end = ai[i + 1] - base;
        for ( auto j = row_begin; j < row_end; ++j )
        {
            const COLTYPE dependency = aj[j] - base;
            if constexpr ( TS == TriangularMatrix::U )
            {
                if ( dependency <= i )
                {
                    continue;
                }
            }
            else
            {
                if ( dependency >= i )
                {
                    continue;
                }
            }
            _depths[i] = std::max( _depths[i], static_cast<COLTYPE>( _depths[dependency] + 1 ) );
        }
        level = std::max( level, _depths[i] + 1 );
    }
    std::fill( prefix, prefix + level + 1, 0 );
    prefix[0] = base;
    for ( COLTYPE i = 0; i < nodes; i++ )
    {
        prefix[_depths[i] + 1]++;
    }
    std::inclusive_scan( prefix, prefix + level + 1, prefix );

    for ( COLTYPE i = 0; i < nodes; i++ )
    {
        perm[prefix[_depths[i]]++ - base] = i + base;
    }
    for ( COLTYPE i = level; i > 0; i-- )
    {
        prefix[i] = prefix[i - 1];
    }
    prefix[0] = base;
    return level;
}

// Template instantiations
#define INSTANTIATE_GRAPH_ALGS( ROWTYPE, COLTYPE )                                                        \
    template void ElimTree<ROWTYPE, COLTYPE>( const COLTYPE rows, ROWTYPE const* ai,                      \
                                              COLTYPE const* aj, COLTYPE* parent );                       \
    template bool IsDAG<ROWTYPE, COLTYPE>( const COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj );    \
    template COLTYPE FindStronglyConnectedComponents<ROWTYPE, COLTYPE>(                                   \
        const COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj, ROWTYPE* scc_prefix,                    \
        COLTYPE* scc_to_node, COLTYPE* node_to_scc );                                                     \
    template struct ProjectGraphToTaskGraph<ROWTYPE, COLTYPE, false>;                                     \
    template struct ProjectGraphToTaskGraph<ROWTYPE, COLTYPE, true>;                                      \
    template struct TransitiveReduction<ROWTYPE, COLTYPE>;                                                \
    template COLTYPE MISPerm<ROWTYPE, COLTYPE>(                                                           \
        const COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj, COLTYPE* perm, COLTYPE* iperm );        \
    template void BuildPermutationFromSccLevels<ROWTYPE, COLTYPE>(                                        \
        COLTYPE num_sccs, ROWTYPE const* scc_prefix, COLTYPE const* scc_to_node, COLTYPE const* scc_perm, \
        ROWTYPE const* scc_level_prefix, COLTYPE scc_levels, COLTYPE* node_perm, COLTYPE* node_iperm );   \
    template struct KahnSerial<ROWTYPE, COLTYPE>;                                                         \
    template struct KahnParallel<ROWTYPE, COLTYPE>;                                                       \
    template struct TopologicalSort2<ROWTYPE, COLTYPE, TriangularMatrix::L>;                              \
    template struct TopologicalSort2<ROWTYPE, COLTYPE, TriangularMatrix::U>;                              \
    template struct TopologicalSort2<ROWTYPE, COLTYPE, TriangularMatrix::LU>;

// INSTANTIATE_GRAPH_ALGS(int, int)
INSTANTIATE_GRAPH_ALGS( std::int32_t, std::int32_t )
INSTANTIATE_GRAPH_ALGS( std::int64_t, std::int64_t )

#undef INSTANTIATE_GRAPH_ALGS

template <typename ROWTYPE, typename COLTYPE>
double jaccardSimilarity( const COLTYPE A_rows,
                          const COLTYPE A_cols,
                          const ROWTYPE* A_ai,
                          const COLTYPE* A_aj,
                          const COLTYPE B_rows,
                          const COLTYPE B_cols,
                          const ROWTYPE* B_ai,
                          const COLTYPE* B_aj,
                          int num_threads )
{
    // Check dimensions
    if ( A_rows != B_rows || A_cols != B_cols )
    {
        throw std::invalid_argument( "Matrix dimensions do not match for Jaccard similarity" );
    }

    const ROWTYPE A_base = A_ai[0];
    const ROWTYPE B_base = B_ai[0];
    const ROWTYPE A_nnz = A_ai[A_rows] - A_base;
    const ROWTYPE B_nnz = B_ai[B_rows] - B_base;

    // Handle empty matrices
    if ( A_nnz == 0 && B_nnz == 0 )
    {
        return 1.0; // Both empty, perfectly similar
    }
    if ( A_nnz == 0 || B_nnz == 0 )
    {
        return 0.0; // One empty, no similarity
    }

    // Create value vectors filled with 1
    std::vector<int8_t> A_av( A_nnz, 1 );
    std::vector<int8_t> B_av( B_nnz, 1 );

    // Use SpADD to compute union: C = 1*A + 1*B
    matrix_utils::SpADD<matrix_utils::CSRMatrixVec<ROWTYPE, COLTYPE, int8_t>> spadd( num_threads );
    matrix_utils::CSRMatrixVec<ROWTYPE, COLTYPE, int8_t> C;

    // Analysis phase
    spadd.analysis( A_rows, A_cols, A_ai, A_aj, B_rows, B_cols, B_ai, B_aj, C );

    // Numerical phase with alpha=1, beta=1
    spadd( A_rows, A_cols, A_ai, A_aj, A_av.data(), static_cast<int8_t>( 1 ), B_rows, B_cols, B_ai,
           B_aj, B_av.data(), static_cast<int8_t>( 1 ), C );

    const ROWTYPE C_base = C.AI()[0];
    const ROWTYPE C_nnz = C.AI()[C.rows] - C_base;

    // Count entries with value 2 (intersection)
    ROWTYPE intersection_count = 0;
    const int8_t* C_av = C.AV();

#pragma omp parallel for reduction( + : intersection_count ) num_threads( num_threads )
    for ( ROWTYPE i = 0; i < C_nnz; i++ )
    {
        if ( C_av[i] == 2 ) // Exact comparison for int8_t
        {
            intersection_count++;
        }
    }

    // Jaccard similarity = |A ∩ B| / |A ∪ B|
    return static_cast<double>( intersection_count ) / static_cast<double>( C_nnz );
}

// Explicit instantiation for common types
template double jaccardSimilarity<int, int>( const int,
                                             const int,
                                             const int*,
                                             const int*,
                                             const int,
                                             const int,
                                             const int*,
                                             const int*,
                                             int );
template double jaccardSimilarity<int64_t, int64_t>( const int64_t,
                                                     const int64_t,
                                                     const int64_t*,
                                                     const int64_t*,
                                                     const int64_t,
                                                     const int64_t,
                                                     const int64_t*,
                                                     const int64_t*,
                                                     int );

} // namespace graph
