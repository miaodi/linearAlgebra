#pragma once
#include "circularbuffer.hpp"
#include "sparse_mat_traits.hpp"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

namespace factorization {

/// @brief Compute the row and optional column counts of the Cholesky factor L.
///
/// This is the elimination-tree path-compression count algorithm described in
/// @cite liu1990role.
/// @tparam ROWTYPE row index type
/// @tparam COLTYPE column index type
/// @param nnodes number of nodes
/// @param base base index of the matrix (usually 0 or 1)
/// @param parent parent vector from the elimination tree
/// @param row_count output vector containing the nonzero count of each row in L
/// @param col_count optional output vector containing the nonzero count of each
/// column in L
/// @param mark mark vector, helper for path compression
template <typename ROWTYPE, typename COLTYPE>
void nnzCount(const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,
              const COLTYPE *parent, ROWTYPE *row_count, ROWTYPE *col_count,
              COLTYPE *mark);

/// @brief Build the skeleton matrix from the leaf vertices of row subtrees.
///
/// The row-subtree/skeleton-matrix idea comes from @cite liu1986compact. The
/// leaf test used here follows Corollary 4.11 in @cite scott2023algorithms,
/// using subtree sizes from Algorithm 4.5.
template <matrix_utils::ResizableCSR CSRMatrixType>
class SkeletonGraph {
public:
  using ROWTYPE = typename CSRMatrixType::ROWTYPE;
  using COLTYPE = typename CSRMatrixType::COLTYPE;

  SkeletonGraph(const int nthreads)
      : _nthreads(nthreads), _XLEAFs(nthreads), _LEAFs(nthreads),
        _XLEAF_prefix(nthreads + 1) {}

  void apply(const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,
             const COLTYPE *parent, CSRMatrixType &leaf);

private:
  int _nthreads;
  std::vector<std::vector<COLTYPE>> _XLEAFs;
  std::vector<std::vector<COLTYPE>> _LEAFs;
  std::vector<COLTYPE> _XLEAF_prefix;
  std::vector<COLTYPE> _subtree_size;
};

template <matrix_utils::ResizableCSR CSRMatrixType>
class SymbolicCholesky {
public:
  using ROWTYPE = typename CSRMatrixType::ROWTYPE;
  using COLTYPE = typename CSRMatrixType::COLTYPE;

  SymbolicCholesky(const int nthreads)
      : _nthreads(nthreads), _ais_prefix(nthreads + 1), _ais(nthreads),
        _ajs(nthreads) {}

  void apply(const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,
             const COLTYPE *parent, const COLTYPE *XLEAF,
             const COLTYPE *LEAF, CSRMatrixType &L);

private:
  int _nthreads;
  std::vector<ROWTYPE> _ais_prefix;
  std::vector<std::vector<ROWTYPE>> _ais;
  std::vector<std::vector<COLTYPE>> _ajs;
};

template <matrix_utils::ResizableCSR CSRMatrixType>
class SymbolicCholeskyCol {
public:
  using ROWTYPE = typename CSRMatrixType::ROWTYPE;
  using COLTYPE = typename CSRMatrixType::COLTYPE;

  SymbolicCholeskyCol(const int nthreads) : _nthreads(nthreads) {}

  /// @brief Column-oriented symbolic Cholesky using Theorem 4.8 of
  /// @cite scott2023algorithms.
  ///
  /// For each column j, the pattern of L(:,j) is the adjacency of the
  /// elimination-tree subtree T(j). Equivalently, Algorithm 4.4 computes it as
  /// the union of the original adjacency of j and the already-computed patterns
  /// of j's children in the elimination tree. This version schedules ready
  /// columns through one shared queue: leaves are ready initially, and a parent
  /// becomes ready after all of its children have been processed.
  bool apply(const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,
             const COLTYPE *parent, CSRMatrixType &L);

private:
  void task(const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,
            const COLTYPE *parent, const int tid, CSRMatrixType &L);

private:
  int _nthreads;
  std::vector<ROWTYPE> _diag;
  std::vector<std::vector<COLTYPE>> _aj;

  std::vector<COLTYPE> _degrees;
  std::vector<COLTYPE> _childOffsets;
  std::vector<COLTYPE> _children;
  std::vector<COLTYPE> _visited;
  std::mutex _mutex;
  std::condition_variable _cv;
  utils::CircularBuffer<COLTYPE> _queue;
  COLTYPE _finished{0};
};

template <matrix_utils::ResizableCSR CSRMatrixType>
class SymbolicCholeskyColV2 {
public:
  using ROWTYPE = typename CSRMatrixType::ROWTYPE;
  using COLTYPE = typename CSRMatrixType::COLTYPE;

  SymbolicCholeskyColV2(const int nthreads) : _nthreads(nthreads) {}

  /// @brief Work-stealing variant of the Theorem 4.8/Algorithm 4.4 symbolic
  /// Cholesky column pattern construction.
  ///
  /// The computed pattern is the same as SymbolicCholeskyCol: merge the
  /// structural column of A with child column patterns in the elimination tree.
  /// This version keeps per-thread ready queues to reduce contention; a worker
  /// pops locally first and steals from other queues when needed.
  bool apply(const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,
             const COLTYPE *parent, CSRMatrixType &L);

private:
  void task(const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,
            const COLTYPE *parent, const int tid, CSRMatrixType &L);
  void pushReady(const int tid, const COLTYPE task);
  bool popReady(const int tid, COLTYPE &task);

private:
  int _nthreads;
  std::vector<ROWTYPE> _diag;
  std::vector<std::vector<COLTYPE>> _aj;

  std::vector<COLTYPE> _degrees;
  std::vector<COLTYPE> _childOffsets;
  std::vector<COLTYPE> _children;
  std::vector<COLTYPE> _visited;
  std::vector<std::deque<COLTYPE>> _queues;
  std::unique_ptr<std::mutex[]> _queueMutexes;
  std::atomic<COLTYPE> _readyTasks;
  std::atomic<COLTYPE> _readyEpoch;
  std::atomic<COLTYPE> _finished;
};

template <matrix_utils::ResizableCSR CSRMatrixType>
class SymbolicCholeskyColV3 {
public:
  using ROWTYPE = typename CSRMatrixType::ROWTYPE;
  using COLTYPE = typename CSRMatrixType::COLTYPE;

  SymbolicCholeskyColV3(const int nthreads) : _nthreads(nthreads) {}

  /// @brief Topological-order variant of the Theorem 4.8/Algorithm 4.4
  /// symbolic Cholesky column pattern construction.
  ///
  /// The topological order gives a dependency-respecting traversal of the
  /// elimination tree. Threads claim columns from that order and spin only on
  /// children that have not finished yet, then merge the child patterns to form
  /// colL{j} as in Theorem 4.8.
  bool apply(const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,
             const COLTYPE *parent, CSRMatrixType &L);

private:
  void task(const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,
            const int tid, CSRMatrixType &L);

private:
  int _nthreads;
  std::vector<ROWTYPE> _diag;
  std::vector<std::vector<COLTYPE>> _aj;

  std::vector<COLTYPE> _degrees;
  std::vector<COLTYPE> _childOffsets;
  std::vector<COLTYPE> _children;
  std::vector<COLTYPE> _visited;
  std::vector<std::vector<COLTYPE>> _pendingChildren;
  std::vector<COLTYPE> _topoPerm;
  std::vector<COLTYPE> _topoPrefix;
  std::unique_ptr<std::atomic<int>[]> _ready;
  COLTYPE _readySize{0};
  std::atomic<COLTYPE> _nextTask;
};
} // namespace factorization
