#pragma once
#include <functional>
#include <limits>
#include <memory>
#include <vector>

namespace graph
{

/// @brief Helper to extract ROWTYPE and COLTYPE from BFS function pointer
template <typename Func>
struct BFSFuncTraits;

// Specialization for BFSFunc with nthreads parameter
template <typename R, typename C>
struct BFSFuncTraits<bool ( * )( C, R const*, C const*, C, C, C&, C&, std::vector<C>&, std::vector<C>&, int )>
{
    using ROWTYPE = R;
    using COLTYPE = C;
};

/// @brief Breadth-First Search (BFS) class for graph traversal
/// @details Provides both serial and parallel BFS implementations for graphs
/// represented in CSR (Compressed Sparse Row) format
/// @tparam BFSFuncType BFS function pointer (e.g., BFSFunc)
template <auto BFSFuncType>
class BFS
{
public:
    using ROWTYPE = typename BFSFuncTraits<decltype( BFSFuncType )>::ROWTYPE;
    using COLTYPE = typename BFSFuncTraits<decltype( BFSFuncType )>::COLTYPE;

    /// @brief Constructor
    BFS() = default;

    /// @brief Perform BFS traversal
    /// @param rows Number of rows in the graph
    /// @param ai Row pointers array (ai[0] contains the base indexing)
    /// @param aj Column indices array
    /// @param source Source node for BFS (in original indexing)
    /// @param nthreads Number of threads to use (default: 1 for serial)
    /// @return true if successful, false if shortcut width exceeded
    bool operator()( COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj, const COLTYPE source, int nthreads = 1 )
    {
        return BFSFuncType( rows, ai, aj, source, _shortCut, _height, _width, _levels, _lastLevel, nthreads );
    }

    /// @brief Get the BFS levels for each node
    const std::vector<COLTYPE>& getLevels() const { return _levels; }
    std::vector<COLTYPE>& getLevels() { return _levels; }

    /// @brief Get the number of levels in BFS tree
    COLTYPE getHeight() const { return _height; }

    /// @brief Get the maximum width across all levels
    COLTYPE getWidth() const { return _width; }

    /// @brief Set shortcut width threshold
    void setShortCut( const COLTYPE sc ) { _shortCut = sc; }

    /// @brief Get the nodes in the last level
    const std::vector<COLTYPE>& getLastLevel() const { return _lastLevel; }

private:
    std::vector<COLTYPE> _lastLevel;
    std::vector<COLTYPE> _levels;
    COLTYPE _height;
    COLTYPE _width;
    COLTYPE _shortCut{ std::numeric_limits<COLTYPE>::max() };
};

/// @brief BFS implementation with automatic serial/parallel selection
/// @tparam ROWTYPE Row pointer type (typically int or int64_t)
/// @tparam COLTYPE Column index type (typically int or int64_t)
/// @tparam LASTLEVEL If true, records nodes in the last level
/// @tparam TRACK Controls both width tracking and early-exit by width.
/// @param rows Number of rows in the graph
/// @param ai Row pointers array (ai[0] contains the base indexing)
/// @param aj Column indices array
/// @param source Source node for BFS (in original indexing)
/// @param shortCutWidth Width threshold for early termination when TRACK=true
/// @param height Output: number of levels in BFS tree
/// @param width Output: maximum width across all levels (0 when TRACK=false)
/// @param levels Output: BFS level for each node (INVALID if not visited)
/// @param lastLevel Output: nodes in the last level (if LASTLEVEL=true)
/// @param nthreads Number of threads to use (1 = serial, >1 = parallel)
/// @return true if successful, false if early-exit triggered (only when TRACK=true)
// When TRACK=false: no shortcut check and width is not updated/returned (left as 0).
// When nthreads=1: uses serial implementation
// When nthreads>1: uses parallel implementation with OpenMP
template <typename ROWTYPE, typename COLTYPE, bool LASTLEVEL = false, bool TRACK = true>
bool BFSFunc( COLTYPE rows,
              ROWTYPE const* ai,
              COLTYPE const* aj,
              COLTYPE source,
              COLTYPE shortCutWidth,
              COLTYPE& height,
              COLTYPE& width,
              std::vector<COLTYPE>& levels,
              std::vector<COLTYPE>& lastLevel,
              int nthreads = 1 );

} // namespace graph
