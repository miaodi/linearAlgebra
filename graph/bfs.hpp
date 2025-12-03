#pragma once
#include <functional>
#include <limits>
#include <memory>
#include <vector>

namespace graph {

/// @brief Breadth-First Search (BFS) class for graph traversal
/// @details Provides both serial and parallel BFS implementations for graphs
/// represented in CSR (Compressed Sparse Row) format
template <typename ROWTYPE, typename COLTYPE>
class BFS {
public:
  /// @brief Function signature for BFS implementations
  using FN = std::function<bool(COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj,
                                 COLTYPE source, COLTYPE shortCut, COLTYPE& height,
                                 COLTYPE& width, std::vector<COLTYPE>& levels,
                                 std::vector<COLTYPE>& lastLevel)>;

  /// @brief Constructor
  /// @param fn BFS function to use (serial or parallel)
  BFS(FN fn) : _fn{fn} {}

  /// @brief Perform BFS traversal
  /// @tparam LASTLEVEL If true, records the nodes in the last level
  /// @param rows Number of rows in the graph
  /// @param ai Row pointers array (ai[0] contains the base indexing)
  /// @param aj Column indices array
  /// @param source Source node for BFS (in original indexing)
  /// @return true if successful, false if shortcut width exceeded
  template <bool LASTLEVEL = false>
  bool operator()(COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj,
                  const COLTYPE source) {
    return _fn(rows, ai, aj, source, _shortCut, _height, _width, _levels, _lastLevel);
  }

  /// @brief Get the BFS levels for each node
  const std::vector<COLTYPE>& getLevels() const { return _levels; }
  std::vector<COLTYPE>& getLevels() { return _levels; }

  /// @brief Get the number of levels in BFS tree
  COLTYPE getHeight() const { return _height; }

  /// @brief Get the maximum width across all levels
  COLTYPE getWidth() const { return _width; }

  /// @brief Set shortcut width threshold
  void setShortCut(const COLTYPE sc) { _shortCut = sc; }

  /// @brief Get the nodes in the last level
  const std::vector<COLTYPE>& getLastLevel() const { return _lastLevel; }

private:
  FN _fn;
  std::vector<COLTYPE> _lastLevel;
  std::vector<COLTYPE> _levels;
  COLTYPE _height;
  COLTYPE _width;
  COLTYPE _shortCut{std::numeric_limits<COLTYPE>::max()};
};

/// @brief Serial BFS implementation
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
/// @return true if successful, false if early-exit triggered (only when TRACK=true)
// When TRACK=false: no shortcut check and width is not updated/returned (left as 0).
template <typename ROWTYPE, typename COLTYPE, bool LASTLEVEL = false, bool TRACK = true>
bool BFSFunc(COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj,
             COLTYPE source, COLTYPE shortCutWidth, COLTYPE& height,
             COLTYPE& width, std::vector<COLTYPE>& levels,
             std::vector<COLTYPE>& lastLevel);

/// @brief Parallel BFS implementation
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
/// @param width Output: maximum width across all levels
/// @param levels Output: BFS level for each node (INVALID if not visited)
/// @param lastLevel Output: nodes in the last level (if LASTLEVEL=true)
/// @param nthreads Number of threads to use (default = 1)
/// @return true if successful, false if early-exit triggered (only when TRACK=true)
template <typename ROWTYPE, typename COLTYPE, bool LASTLEVEL = false, bool TRACK = true>
bool PBFSFunc(COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj,
              COLTYPE source, COLTYPE shortCutWidth, COLTYPE& height,
              COLTYPE& width, std::vector<COLTYPE>& levels,
              std::vector<COLTYPE>& lastLevel, int nthreads = 1);

} // namespace graph
