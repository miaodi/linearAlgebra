#pragma once
#include "matrix_enums.hpp"
#include <algorithm>
#include <omp.h>
#include <set>
#include <vector>
#include <memory>

namespace graph
{
using enums::matrix_utils::TriangularMatrix;

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

/// @brief Expand SCC level order into a node permutation.
///
/// The resulting permutation first groups SCCs by their topological level
/// (computed, e.g., by TopologicalSort2 on the SCC condensation graph) and
/// keeps all nodes that belong to the same SCC contiguous.
/// @tparam ROWTYPE Row pointer type
/// @tparam COLTYPE Column index type
/// @param num_sccs Number of SCCs
/// @param scc_prefix CSR prefix for SCC-to-node mapping (size = num_sccs + 1)
/// @param scc_to_node CSR column indices for SCC-to-node mapping (size = rows)
/// @param scc_perm Permutation of SCC ids produced by the topological sort
/// @param scc_level_prefix Level boundaries over scc_perm (size = scc_levels + 1)
/// @param scc_levels Number of levels returned by the topological sort
/// @param node_perm Output permutation over original nodes (size = rows)
/// @param node_iperm Optional inverse permutation over original nodes (size = rows),
///                   mapping original node id -> position in node_perm
template <typename ROWTYPE, typename COLTYPE>
void BuildPermutationFromSccLevels(COLTYPE num_sccs,
                                   ROWTYPE const* scc_prefix,
                                   COLTYPE const* scc_to_node,
                                   COLTYPE const* scc_perm,
                                   ROWTYPE const* scc_level_prefix,
                                   COLTYPE scc_levels,
                                   COLTYPE* node_perm,
                                   COLTYPE* node_iperm = nullptr);

/// @brief Serial implementation of Kahn's algorithm for topological sorting
///
/// Kahn's algorithm computes a topological ordering of a directed acyclic graph (DAG)
/// by repeatedly removing nodes with zero in-degree. This implementation also computes
/// level sets (nodes at the same topological distance from sources).
///
/// @tparam ROWTYPE Row pointer type (typically int or int64_t)
/// @tparam COLTYPE Column index type (typically int or int64_t)
template <typename ROWTYPE, typename COLTYPE>
struct KahnSerial
{
    /// @brief Compute topological sort from a successor graph and dependency counts
    /// @param nodes Number of nodes in the graph
    /// @param successor_ai Row pointers for outgoing edges (node -> dependent nodes)
    /// @param successor_aj Column indices for outgoing edges
    /// @param in_degree Number of incoming dependencies for each node (self-loops excluded)
    /// @param perm Output array for topological ordering (size = nodes)
    /// @param prefix Output array for level set boundaries (size = num_levels + 1)
    /// @note Self-loop entries in successor_aj are ignored.
    /// @return Number of levels in the topological ordering
    COLTYPE fromSuccessors( const COLTYPE nodes,
                            ROWTYPE const* successor_ai,
                            COLTYPE const* successor_aj,
                            COLTYPE const* in_degree,
                            COLTYPE* perm,
                            COLTYPE* prefix );

    /// @brief Compute topological sort using Kahn's algorithm
    /// @param nodes Number of nodes in the graph
    /// @param ai Row pointers array for incoming dependencies (ai[0] contains the base indexing)
    /// @param aj Column indices array for incoming dependencies
    /// @param perm Output array for topological ordering (size = nodes)
    /// @param prefix Output array for level set boundaries (size = num_levels + 1)
    /// @note Self-loop entries are ignored.
    /// @return Number of levels in the topological ordering
    COLTYPE operator()( const COLTYPE nodes,
                        ROWTYPE const* ai,
                        COLTYPE const* aj,
                        COLTYPE* perm,
                        COLTYPE* prefix );

    std::vector<COLTYPE> _degrees;    // In-degree of each node
    std::vector<COLTYPE> _initial_degrees; // Reusable storage for dependency counts
    std::vector<ROWTYPE> _t_ai;       // Transpose graph row pointers
    std::vector<COLTYPE> _t_aj;       // Transpose graph column indices
};

/// @brief Parallel implementation of Kahn's algorithm for topological sorting
///
/// This parallel version uses atomic operations on in-degrees and thread-local
/// queues to process nodes in parallel while maintaining topological constraints.
/// Level sets are computed by synchronizing between BFS-style iterations.
///
/// @tparam ROWTYPE Row pointer type (typically int or int64_t)
/// @tparam COLTYPE Column index type (typically int or int64_t)
template <typename ROWTYPE, typename COLTYPE>
struct KahnParallel
{
    /// @brief Constructor
    /// @param nthreads Number of threads to use for parallel computation
    KahnParallel( int nthreads )
        : _nthreads( std::max( 1, nthreads ) ),
          _threads_nodes( _nthreads ),
          _threads_prefix( _nthreads + 1 )
    {
    }

    /// @brief Compute topological sort from a successor graph and dependency counts
    /// @param nodes Number of nodes in the graph
    /// @param successor_ai Row pointers for outgoing edges (node -> dependent nodes)
    /// @param successor_aj Column indices for outgoing edges
    /// @param in_degree Number of incoming dependencies for each node (self-loops excluded)
    /// @param perm Output array for topological ordering (size = nodes)
    /// @param prefix Output array for level set boundaries (size = num_levels + 1)
    /// @note Self-loop entries in successor_aj are ignored.
    /// @return Number of levels in the topological ordering
    COLTYPE fromSuccessors( const COLTYPE nodes,
                            ROWTYPE const* successor_ai,
                            COLTYPE const* successor_aj,
                            COLTYPE const* in_degree,
                            COLTYPE* perm,
                            COLTYPE* prefix );

    /// @brief Compute topological sort using parallel Kahn's algorithm
    /// @param nodes Number of nodes in the graph
    /// @param ai Row pointers array for incoming dependencies (ai[0] contains the base indexing)
    /// @param aj Column indices array for incoming dependencies
    /// @param perm Output array for topological ordering (size = nodes)
    /// @param prefix Output array for level set boundaries (size = num_levels + 1)
    /// @note Self-loop entries are ignored.
    /// @return Number of levels in the topological ordering
    COLTYPE operator()( const COLTYPE nodes,
                        ROWTYPE const* ai,
                        COLTYPE const* aj,
                        COLTYPE* perm,
                        COLTYPE* prefix );

    int _nthreads;                                      // Number of threads
    std::unique_ptr<std::atomic<COLTYPE>[]> _degrees;  // Atomic in-degrees for parallel updates
    COLTYPE _degrees_size{ 0 };                        // Current size of _degrees array
    std::vector<COLTYPE> _initial_degrees;              // Reusable storage for dependency counts
    std::vector<ROWTYPE> _t_ai;                        // Transpose graph row pointers
    std::vector<COLTYPE> _t_aj;                        // Transpose graph column indices
    std::vector<std::vector<COLTYPE>> _threads_nodes;  // Per-thread queue of nodes to process
    std::vector<COLTYPE> _threads_prefix;              // Per-thread prefix sums for output positions
};

/// @brief Topological sort using dependency-depth level ordering
///
/// This algorithm assigns each node to one plus the maximum depth of its
/// dependencies, then groups nodes with the same depth into level sets. The
/// template parameter TS determines whether rows are traversed as lower or upper
/// triangular dependencies. TriangularMatrix::LU is treated as lower triangular
/// and uses only lower-side dependencies.
///
/// @tparam ROWTYPE Row pointer type (typically int or int64_t)
/// @tparam COLTYPE Column index type (typically int or int64_t)
/// @tparam TS Triangular matrix type (U for upper, L/LU for lower)
template <typename ROWTYPE, typename COLTYPE, TriangularMatrix TS>
struct TopologicalSort2
{
    /// @brief Compute depth-based topological level sets
    /// @param nodes Number of nodes in the graph
    /// @param ai Row pointers array (ai[0] contains the base indexing)
    /// @param aj Column indices array
    /// @param perm Output array for topological ordering (size = nodes)
    /// @param prefix Output array for level set boundaries (size = num_levels + 1)
    ///               After sorting, the base of prefix matches ai[0]
    /// @note Self-loop entries are ignored.
    /// @return Number of levels in the topological ordering
    COLTYPE operator()( const COLTYPE nodes,
                        ROWTYPE const* ai,
                        COLTYPE const* aj,
                        COLTYPE* perm,
                        COLTYPE* prefix );
                        
    std::vector<COLTYPE> _depths;  // Dependency depth of each node
};

/**
 * @brief Compute Jaccard similarity between two CSR structures
 * 
 * Jaccard similarity = |A ∩ B| / |A ∪ B|
 * Where A and B are the sets of non-zero positions in the matrices.
 * 
 * This function treats both matrices as binary (presence/absence of entries),
 * computes their union using SpADD with all values set to 1, and then
 * calculates the ratio of overlapping entries to total unique entries.
 * 
 * @tparam ROWTYPE Type for row pointers
 * @tparam COLTYPE Type for column indices
 * @param A_rows Number of rows in matrix A
 * @param A_cols Number of columns in matrix A (must equal B_cols)
 * @param A_ai Row pointers of matrix A
 * @param A_aj Column indices of matrix A
 * @param B_rows Number of rows in matrix B (must equal A_rows)
 * @param B_cols Number of columns in matrix B (must equal A_cols)
 * @param B_ai Row pointers of matrix B
 * @param B_aj Column indices of matrix B
 * @param num_threads Number of OpenMP threads to use
 * @return Jaccard similarity coefficient in range [0, 1]
 */
template <typename ROWTYPE, typename COLTYPE>
double jaccardSimilarity( const COLTYPE A_rows,
                          const COLTYPE A_cols,
                          const ROWTYPE* A_ai,
                          const COLTYPE* A_aj,
                          const COLTYPE B_rows,
                          const COLTYPE B_cols,
                          const ROWTYPE* B_ai,
                          const COLTYPE* B_aj,
                          int num_threads = omp_get_max_threads() );

} // namespace graph
