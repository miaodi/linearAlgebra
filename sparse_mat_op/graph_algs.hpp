#pragma once
#include "matrix_utils.hpp"
#include <algorithm>
#include <omp.h>
#include <set>
#include <vector>
namespace matrix_utils
{

/// @brief Compute the elimination tree of a sparse matrix
/// @tparam ROWTYPE Row pointer type (typically int or int64_t)
/// @tparam COLTYPE Column index type (typically int or int64_t)
/// @param rows Number of rows in the matrix
/// @param ai Row pointers array (ai[0] contains the base indexing)
/// @param aj Column indices array
/// @param parent Output array for parent nodes in elimination tree
template <typename ROWTYPE, typename COLTYPE>
void ElimTree(const COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj, COLTYPE* parent);

/// @brief Check if a graph represented by CSR format is a Directed Acyclic Graph (DAG)
/// @tparam ROWTYPE Row pointer type (typically int or int64_t)
/// @tparam COLTYPE Column index type (typically int or int64_t)
/// @param rows Number of rows in the matrix
/// @param ai Row pointers array (ai[0] contains the base indexing)
/// @param aj Column indices array
/// @return true if the graph is a DAG, false if it contains cycles
template <typename ROWTYPE, typename COLTYPE>
bool IsDAG(const COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj);

/// @brief Project an adjacency graph to a task graph based on task-to-work and work-to-task mappings
/// @tparam ROWTYPE Row pointer type (typically int or int64_t)
/// @tparam COLTYPE Column index type (typically int or int64_t)
/// @tparam KEEPDIAG Whether to keep diagonal elements in the projection
template <typename ROWTYPE, typename COLTYPE, bool KEEPDIAG = false>
struct ProjectGraphToTaskGraph
{
    /// @brief Constructor
    /// @param nthreads Number of threads to use for parallel computation
    ProjectGraphToTaskGraph(const int nthreads = 1) : _nthreads(nthreads) {}

    /// @brief Project work graph to task graph
    /// @param work_graph_rows Number of rows in the original work graph
    /// @param work_ai Row pointers of the original work adjacency graph (work_ai[0] is base)
    /// @param work_aj Column indices of the original work adjacency graph
    /// @param num_tasks Number of tasks in the task graph
    /// @param task_prefix CSR row pointers for task-to-work mapping (task_prefix[0] is task graph base)
    /// @param task_to_node CSR column indices for task-to-work mapping
    /// @param node_to_task Map from work index to task ID
    /// @param task_ai Output row pointers for task adjacency graph (pre-allocated, size = num_tasks + 1)
    /// @param task_aj Output column indices for task adjacency graph (pre-allocated, sufficient size)
    /// @return Number of edges in the projected task graph
    COLTYPE operator()(COLTYPE work_graph_rows, ROWTYPE const* work_ai, COLTYPE const* work_aj,
                       COLTYPE num_tasks, COLTYPE const* task_prefix, COLTYPE const* task_to_node,
                       COLTYPE const* node_to_task, ROWTYPE* task_ai, COLTYPE* task_aj);

private:
    // Data members for reusable memory allocation
    int _nthreads; // Number of threads for parallel computation
    mutable std::vector<std::set<COLTYPE>> _task_dependencies; // dependencies for each task (set for auto-dedup)
};

/// @brief Compute transitive reduction of a topologically sorted adjacency graph
/// @tparam ROWTYPE Row pointer type (typically int or int64_t)
/// @tparam COLTYPE Column index type (typically int or int64_t)
template <typename ROWTYPE, typename COLTYPE>
struct TransitiveReduction
{
    TransitiveReduction(int num_threads = omp_get_max_threads())
        : _nthreads{std::max(1, num_threads)}
    {
    }

    void set_num_threads(int num_threads) { _nthreads = std::max(1, num_threads); }

    /// @brief Compute transitive reduction
    /// @param rows Number of rows in the graph
    /// @param ai Input row pointers array (ai[0] contains the base indexing)
    /// @param aj Input column indices array
    /// @param out_ai Output row pointers array (pre-allocated)
    /// @param out_aj Output column indices array (pre-allocated)
    /// @param has_self_loops If true, assumes every node has a self-loop in the input graph
    void operator()(COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj, ROWTYPE* out_ai,
                    COLTYPE* out_aj, bool has_self_loops = false);

private:
    int _nthreads{1};
    // Data members for reusable memory allocation
    mutable std::vector<std::vector<COLTYPE>> _reachable; // reachable[i] = list of nodes reachable from i
    mutable std::vector<std::vector<COLTYPE>> _reduced_edges; // reusable storage for reduced edges per row
};

/// @brief Compute a Maximal Independent Set (MIS) permutation of a graph
///
/// This function implements Algorithm 2.2 from:
/// Jones, M. T., & Plassmann, P. E. (1996). "Incomplete Cholesky factorizations
/// with limited memory." SIAM Journal on Scientific Computing, 21(1), 24-45.
/// https://doi.org/10.1137/0917054
///
/// The algorithm sorts nodes by degree (descending) and greedily selects nodes
/// into a maximal independent set. The resulting permutation places MIS nodes first,
/// followed by remaining nodes, which can improve ILU factorization quality.
///
/// @tparam ROWTYPE Row pointer type (typically int or int64_t)
/// @tparam COLTYPE Column index type (typically int or int64_t)
/// @param size Number of nodes in the graph
/// @param ai Row pointers array (ai[0] contains the base indexing)
/// @param aj Column indices array
/// @param perm Output permutation array (maps old index to new index)
/// @param iperm Output inverse permutation array (maps new index to old index)
template <typename ROWTYPE, typename COLTYPE>
COLTYPE MISPerm(COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj, COLTYPE* perm, COLTYPE* iperm);

/// @brief Find Strongly Connected Components (SCCs) using Tarjan's algorithm
/// @tparam ROWTYPE Row pointer type (typically int or int64_t)
/// @tparam COLTYPE Column index type (typically int or int64_t)
/// @param rows Number of rows in the graph
/// @param ai Row pointers array (ai[0] contains the base indexing)
/// @param aj Column indices array
/// @param scc_prefix Output CSR row pointers for SCC-to-node mapping (pre-allocated, size = num_sccs + 1)
/// @param scc_to_node Output CSR column indices for SCC-to-node mapping (pre-allocated, size = rows)
/// @param node_to_scc Output array mapping each node to its SCC ID (pre-allocated, size = rows)
/// @return Number of strongly connected components found
/// @note SCCs are numbered from 0 to (num_sccs - 1) in reverse topological order
///       (i.e., if there's an edge from SCC i to SCC j, then i > j)
///       scc_prefix[i] to scc_prefix[i+1]-1 gives the range of nodes in SCC i
///       scc_to_node contains the node IDs for each SCC
template <typename ROWTYPE, typename COLTYPE>
COLTYPE FindStronglyConnectedComponents(const COLTYPE rows,
                                        ROWTYPE const* ai,
                                        COLTYPE const* aj,
                                        ROWTYPE* scc_prefix,
                                        COLTYPE* scc_to_node,
                                        COLTYPE* node_to_scc);
} // namespace matrix_utils
