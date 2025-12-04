#pragma once
#include "config.h"
#include <algorithm>
#include <limits>
#include <omp.h>
#include <ranges>
#include <utility>
#include <vector>
#include "bfs.hpp"
#include "parallel_sort.hpp"
#include "UnionFind.h"
#include <iostream>

namespace reordering {

// Modern interface with raw CSR pointers
template <typename ROWTYPE, typename COLTYPE>
void NodeDegree(COLTYPE rows, const ROWTYPE* ai, COLTYPE* degrees, int numthreads = 1);

template <typename T>
void PairReduce(std::pair<T, T> &inout, const std::pair<T, T> &in) {
  if (in.second < inout.second) {
    inout = in;
  } else if (in.second == inout.second) {
    inout.first = std::min(in.first, inout.first);
  }
}

// returns node index and degree
template <typename Iter>
auto MinDegreeNode(const typename std::iterator_traits<Iter>::value_type* degrees, 
                   const typename std::iterator_traits<Iter>::value_type base, 
                   Iter begin, Iter end, int numthreads = 1)
{
    using T = typename std::iterator_traits<Iter>::value_type;
    // Initialize with sentinels for unsigned-safe operation
    constexpr T INVALID = std::numeric_limits<T>::max();
    constexpr T DEG_MAX = std::numeric_limits<T>::max();
    std::pair<T, T> res(INVALID, DEG_MAX);
    
    if (numthreads == 1) {
        // Serial path
        for (auto it = begin; it != end; ++it)
        {
            const T i = *it;
            if (degrees[i - base] < res.second)
            {
                res.first = i;
                res.second = degrees[i - base];
            }
        }
    } else {
        // Parallel path
#pragma omp declare reduction(                                                 \
        pairreduce : std::pair<T, T> : PairReduce<T>(                          \
                omp_out, omp_in)) initializer(omp_priv = omp_orig)

        const std::size_t n = std::distance(begin, end);
#pragma omp parallel for num_threads(numthreads) reduction(pairreduce : res)
        for (std::size_t idx = 0; idx < n; ++idx) {
            const T i = *(begin + idx);
            PairReduce(res, std::make_pair(i, degrees[i - base]));
        }
    }
    return res;
}

/// @brief Compute pseudo-diameter of a graph component using BFS
/// @details Implements heuristic from Duff (1989) "The use of profile reduction algorithms with a frontal code"
/// @see https://github.com/dralves/sp1-sp2-galois/blob/1597f1f510cc1aa75f5595f0d42f5701dfc34a91/lonestar/experimental/cuthill/serial/cuthill.cpp#L815
/// @tparam ROWTYPE Row pointer type
/// @tparam COLTYPE Column index type
/// @tparam Iter Iterator type for the component nodes
/// @param rows Number of rows in the graph
/// @param ai Row pointer array (CSR format)
/// @param aj Column index array (CSR format)
/// @param degrees Degree array (will be modified - nodes are marked off)
/// @param base Index base (0 or 1)
/// @param begin Iterator to start of component nodes
/// @param end Iterator to end of component nodes
/// @param source Output: source node of pseudo-diameter
/// @param target Output: target node of pseudo-diameter
/// @param numthreads Number of threads for parallel operations
/// @return Pseudo-diameter length
template <typename ROWTYPE, typename COLTYPE, typename Iter>
COLTYPE PseudoDiameter(COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj, COLTYPE* degrees,
                       COLTYPE base, Iter begin, Iter end, COLTYPE& source, COLTYPE& target,
                       int numthreads = 1)
{
    const COLTYPE INVALID = std::numeric_limits<COLTYPE>::max();
    source = MinDegreeNode(degrees, base, begin, end, numthreads).first;
    target = INVALID;
    std::vector<COLTYPE> chosen;
    COLTYPE diameter;
    COLTYPE forwardWidth;
    COLTYPE backwardWidth;
    
    while (target == INVALID)
    {
        chosen.resize(0);
        
        // BFS from current source
        COLTYPE height = 0;
        COLTYPE width = 0;
        std::vector<COLTYPE> levels;
        std::vector<COLTYPE> lastLevel;
        
        graph::BFSFunc<ROWTYPE, COLTYPE, true, true>(rows, ai, aj, source, 
            INVALID, height, width, levels, lastLevel, numthreads);
        
        diameter = height;
        forwardWidth = width;

        // First five strategy: select up to 5 min-degree nodes from last level
        while (chosen.size() < 5)
        {
            COLTYPE minDeg = INVALID;
            COLTYPE sel = INVALID;
            for (auto i : lastLevel)
            {
                if (degrees[i - base] < minDeg)
                {
                    minDeg = degrees[i - base];
                    sel = i;
                }
                else if (degrees[i - base] == minDeg)
                {
                    // Ensure deterministic tie-breaking
                    sel = std::min(sel, i);
                }
            }
            if (minDeg == INVALID)
                break;

            chosen.push_back(sel);
            degrees[sel - base] = INVALID; // mark-off selected node
            
            // Mark off neighbors of selected node
            for (ROWTYPE k = ai[sel - base] - base; k < ai[sel - base + 1] - base; k++)
            {
                degrees[aj[k] - base] = INVALID;
            }
        }

        if (chosen.size() == 0)
        {
            // No candidates found - end of search
            target = source;
            break;
        }

        backwardWidth = INVALID;
        for (auto i : chosen)
        {
            // BFS from candidate with shortcut
            if (!graph::BFSFunc<ROWTYPE, COLTYPE, false, true>(
                    rows, ai, aj, i, backwardWidth, height, width, levels, lastLevel, numthreads))
                continue; // short-circuited
            
            if (height > diameter)
            {
                // Found a farther node - restart from it
                source = i;
                target = INVALID;
                break;
            }
            else if (width < backwardWidth)
            {
                // Same diameter, narrower width - better peripheral node
                backwardWidth = width;
                target = i;
            }
        }
    }
    
    if (forwardWidth > backwardWidth)
        std::swap(source, target);
    
    return diameter;
}

/// @brief RCM ordering for a single component
/// @details Helper function that performs RCM on a subset of nodes
/// @tparam ROWTYPE Row pointer type
/// @tparam COLTYPE Column index type
/// @tparam Iter Iterator type for component nodes
/// @param rows Number of rows in the graph
/// @param ai Row pointer array (CSR format)
/// @param aj Column index array (CSR format)
/// @param base Index base (0 or 1)
/// @param comp_begin Iterator to start of component nodes
/// @param comp_end Iterator to end of component nodes
/// @param perm Output: permutation array where perm[new_pos] = old_node
///              (i.e., new position i contains old node perm[i])
/// @param perm_offset Starting offset for this component's permutation
/// @param numthreads Number of threads for parallel operations
template <typename ROWTYPE, typename COLTYPE, typename Iter>
void RCM_Component(COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj,
                   COLTYPE base, Iter comp_begin, Iter comp_end,
                   COLTYPE* perm, COLTYPE* iperm, COLTYPE perm_offset, int numthreads = 1)
{
    COLTYPE comp_size = std::distance(comp_begin, comp_end);
#define RCM_DEBUG 
#ifdef RCM_DEBUG
    std::cerr << "start RCM_Component: comp_size=" << comp_size << std::endl;
#endif
    
    // Compute node degrees
    std::vector<COLTYPE> degrees(rows);
    NodeDegree(rows, ai, degrees.data(), numthreads);
    std::vector<COLTYPE> degrees_copy = degrees;

    // Find pseudo-peripheral node for this component
    COLTYPE source, target;
    PseudoDiameter(rows, ai, aj, degrees_copy.data(), base, 
                   comp_begin, comp_end, source, target, numthreads);
#ifdef RCM_DEBUG
    std::cerr << "[RCM] PseudoDiameter chosen source=" << (source) 
              << " target=" << (target) << std::endl;
#endif
    
    // BFS from source to get level structure
    COLTYPE height, width;
    std::vector<COLTYPE> levels;
    std::vector<COLTYPE> lastLevel;
    graph::BFSFunc<ROWTYPE, COLTYPE, false, false>(rows, ai, aj, source,
        std::numeric_limits<COLTYPE>::max(), height, width, levels, lastLevel, numthreads);
#ifdef RCM_DEBUG
    std::cerr << "[RCM] BFS from source=" << (source) 
              << " height=" << (height) 
              << " width=" << (width) 
              << " levels.size=" << levels.size() << std::endl;
#endif
    
    // Create (level, degree, node) tuples for nodes in this component
    std::vector<std::tuple<COLTYPE, COLTYPE, COLTYPE>> level_degree_node(comp_size);

    // Parallel path
#pragma omp parallel for num_threads(numthreads)
    for (COLTYPE idx = 0; idx < comp_size; ++idx)
    {
        COLTYPE node = *(comp_begin + idx) - base;
        level_degree_node[idx] = std::make_tuple(levels[node], degrees[node], node);
    }

    // Sort by level, then by degree (ascending), then by node index
    utils::sort(level_degree_node.begin(), level_degree_node.end(), numthreads);
#ifdef RCM_DEBUG
    std::cerr << "[RCM] Sorted component nodes by (level, degree). comp_size=" 
              << comp_size << std::endl;
#endif
    
    
    // Generate RCM ordering for this component (reverse of sorted order)
    // perm[new_pos] = old_node (permutation) and iperm[old_node] = new_pos
#pragma omp parallel for num_threads(numthreads)
    for (COLTYPE i = 0; i < comp_size; ++i)
    {
        COLTYPE old_node = std::get<2>(level_degree_node[comp_size - 1 - i]);
        COLTYPE new_pos = perm_offset + i;
        perm[new_pos + base] = old_node + base;
        iperm[old_node + base] = new_pos + base;
    #ifdef RCM_DEBUG
        if (i < 5) {
            std::cerr << "[RCM] perm[" << (new_pos + base) << "] = " << (old_node + base)
                  << ", iperm[" << (old_node + base) << "] = " << (new_pos + base) << std::endl;
        }
    #endif
    }
}

/// @brief Reverse Cuthill-McKee (RCM) reordering algorithm
/// @details Orders nodes by reverse BFS levels from a pseudo-peripheral node,
/// with nodes at each level sorted by increasing degree. Produces both
/// permutation (perm) and inverse permutation (iperm) suitable for matrix
/// permutation routines.
/// @tparam ROWTYPE Row pointer type
/// @tparam COLTYPE Column index type
/// @param rows Number of rows in the graph
/// @param ai Row pointer array (CSR format)
/// @param aj Column index array (CSR format)
/// @param perm Output: permutation array where perm[new_pos] = old_node
/// @param iperm Output: inverse permutation where iperm[old_node] = new_pos
/// @param numthreads Number of threads for parallel operations
template <typename ROWTYPE, typename COLTYPE>
void RCM(COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj,
         COLTYPE* perm, COLTYPE* iperm, int numthreads = 1)
{
    const COLTYPE base = ai[0];
    
    // Create component with all nodes
    std::vector<COLTYPE> component(rows);
    for (COLTYPE i = 0; i < rows; ++i) {
        component[i] = i + base;
    }
    
    RCM_Component(rows, ai, aj, base, component.begin(), component.end(),
                  perm, iperm, 0, numthreads);
}

/// @brief Reverse Cuthill-McKee (RCM) reordering for multi-component graphs
/// @details Orders nodes by reverse BFS levels from pseudo-peripheral nodes,
/// processing each connected component separately. Nodes at each level are
/// sorted by increasing degree. Produces both permutation (perm) and inverse
/// permutation (iperm).
/// @tparam ROWTYPE Row pointer type
/// @tparam COLTYPE Column index type
/// @param rows Number of rows in the graph
/// @param ai Row pointer array (CSR format)
/// @param aj Column index array (CSR format)
/// @param perm Output: permutation array where perm[new_pos] = old_node
/// @param iperm Output: inverse permutation where iperm[old_node] = new_pos
/// @param numthreads Number of threads for parallel operations
template <typename ROWTYPE, typename COLTYPE>
void RCM_MultiComponent(COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj,
                        COLTYPE* perm, COLTYPE* iperm, int numthreads = 1)
{
    const COLTYPE base = ai[0];
    
    // Find connected components using union-find
    std::vector<COLTYPE> parents(rows);
    ParUnionFindRem(rows, ai, aj, parents.data(), numthreads);
    
    std::vector<COLTYPE> compRoots, sortedComp, compPrefSum;
    ComponentsStat(parents.data(), rows, base, compRoots, sortedComp, compPrefSum, numthreads);
    
    COLTYPE perm_offset = 0; // Offset from base (0, 1, 2, ...)
    
    // Process each connected component separately
    for (size_t comp = 0; comp < compRoots.size(); ++comp)
    {
        COLTYPE comp_start = compPrefSum[comp];
        COLTYPE comp_size = compPrefSum[comp + 1] - comp_start;
        
        // Skip small components (singletons, pairs, triples) - trivial ordering
        if (comp_size <= 3)
        {
            for (COLTYPE i = 0; i < comp_size; ++i)
            {
                COLTYPE node = sortedComp[comp_start + i];
                perm[perm_offset + i + base] = node;
            }
            perm_offset += comp_size;
            continue;
        }
        
        // Get component nodes as iterator range
        auto comp_begin = sortedComp.begin() + comp_start;
        auto comp_end = sortedComp.begin() + comp_start + comp_size;
        
        // Apply RCM to this component
        RCM_Component(rows, ai, aj, base, comp_begin, comp_end,
                      perm, iperm, perm_offset, numthreads);
        
        perm_offset += comp_size;
    }

    // iperm already built in RCM_Component
}

#ifdef USE_METIS_LIB
/// @brief Configuration options for METIS nested dissection
struct MetisNDOptions {
  /// @brief Number of different separators to try (1-10+, default: 1)
  /// Higher values may produce better quality orderings but take longer
  int nseps = 1;
  
  /// @brief Number of refinement iterations (default: 10)
  /// Higher values may produce better quality orderings but take longer
  int niter = 10;
  
  /// @brief Random seed for reproducibility (default: -1 for random)
  /// Set to a fixed value (e.g., 0, 42) for reproducible results
  int seed = -1;
  
  /// @brief Compress graph by removing self-loops and duplicate edges (default: true)
  bool compress = true;
  
  /// @brief Order connected components of the graph separately (default: false)
  /// Useful for disconnected graphs
  bool ccorder = false;
  
  /// @brief Coarsening type (default: 1 = METIS_CTYPE_SHEM - sorted heavy-edge matching)
  /// 0 = METIS_CTYPE_RM (random matching)
  /// 1 = METIS_CTYPE_SHEM (sorted heavy-edge matching, usually better)
  int ctype = 1;
  
  /// @brief Refinement type (default: 3 = METIS_RTYPE_SEP1SIDED)
  /// 0 = METIS_RTYPE_FM (Fiduccia-Mattheyses)
  /// 1 = METIS_RTYPE_GREEDY
  /// 2 = METIS_RTYPE_SEP2SIDED (2-sided separator refinement)
  /// 3 = METIS_RTYPE_SEP1SIDED (1-sided separator refinement, usually best for ND)
  int rtype = 3;
  
  /// @brief Debug level (default: 0 = no output)
  /// Higher values produce more diagnostic output
  int dbglvl = 0;
};

/// @brief METIS nested dissection reordering for general CSR matrices
/// @tparam ROWTYPE Type for row pointers (xadj array) - supports int32_t or int64_t
/// @tparam COLTYPE Type for column indices (adjncy array) - supports int32_t or int64_t
/// @param nrows Number of rows in the matrix
/// @param ncols Number of columns in the matrix (must equal nrows for square matrix)
/// @param xadj Row pointer array of size (nrows + 1), zero-based, with diagonals removed
/// @param adjncy Column index array, zero-based, with diagonals removed
/// @param perm Inverse permutation array: perm[i] = k means new row i comes from old row k
/// @param iperm Permutation array: iperm[i] = k means old row i goes to new row k
/// @param opts METIS options (optional, uses defaults if not provided)
/// @return 0 on success, non-zero on failure
template <typename ROWTYPE, typename COLTYPE>
int MetisND(const COLTYPE nrows, const COLTYPE ncols,
            const ROWTYPE* xadj, const COLTYPE* adjncy,
            COLTYPE* perm, COLTYPE* iperm,
            const MetisNDOptions& opts = MetisNDOptions());
#endif


// // TODO: implement parallel one
// void SerialCM(mkl_wrapper::mkl_sparse_mat const *const mat,
//               std::vector<MKL_INT> &iperm, std::vector<MKL_INT> &perm);

} // namespace reordering