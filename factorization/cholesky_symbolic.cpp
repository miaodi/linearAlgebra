#include "cholesky_symbolic.hpp"
#include "tree.hpp"
#include "matrix_utils.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <omp.h>
#include <thread>

namespace factorization {

template <typename ROWTYPE, typename COLTYPE>
void nnzCount(const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,
              const COLTYPE *parent, ROWTYPE *row_count, ROWTYPE *col_count,
              COLTYPE *mark) {
  const ROWTYPE base = ai[0];
  if (col_count != nullptr) {
    std::fill(col_count, col_count + nnodes, 1);
  }
  for (COLTYPE i = 0; i < nnodes; i++) {
    row_count[i] = 1;
    mark[i] = i;
    for (ROWTYPE j_idx = ai[i] - base; j_idx < ai[i + 1] - base; j_idx++) {
      COLTYPE jroot = aj[j_idx] - base;
      if (jroot >= i) // break if jroot is not in the lower triangle
        break;
      // Walk from the matrix entry toward row i in the elimination tree. A
      // node marked with i was already reached by an earlier entry in this row,
      // so the remaining path has already contributed to row_count[i].
      while (mark[jroot] != i) {
        row_count[i] += 1;
        if (col_count != nullptr) {
          col_count[jroot] += 1;
        }
        mark[jroot] = i;
        jroot = parent[jroot] - base;
      }
    }
  }
}

template <matrix_utils::ResizableCSR CSRMatrixType>
void SkeletonGraph<CSRMatrixType>::apply(const COLTYPE nnodes, const ROWTYPE *ai,
                                         const COLTYPE *aj,
                                         const COLTYPE *parent,
                                         CSRMatrixType &leaf) {
  const ROWTYPE base = ai[0];
  _subtree_size.resize(nnodes);
  graph::subtreeSize(nnodes, base, parent, _subtree_size.data());
#pragma omp parallel num_threads(_nthreads)
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    auto [start, end] =
        utils::LoadPrefixBalancedPartitionPos(ai, ai + nnodes, tid, nthreads);
    _XLEAFs[tid].clear();
    _XLEAFs[tid].push_back(0);
    _LEAFs[tid].clear();
    for (auto i = start; i < end; i++) {
      COLTYPE count = 0;
      auto j = ai[i] - base;
      if (j < ai[i + 1] - base && aj[j] - base < i) {
        _LEAFs[tid].push_back(aj[j++]);
        count++;
      }

      for (; j < ai[i + 1] - base && aj[j] - base < i; j++) {
        // Consecutive lower-triangular entries aj[j - 1], aj[j] are leaves of
        // different row subtrees when aj[j - 1] is outside the subtree rooted at
        // aj[j]. This is the Liu row-subtree criterion stated as Scott/Tuma
        // Corollary 4.11.
        if (aj[j - 1] + _subtree_size[aj[j] - base] - 1 < aj[j]) {
          _LEAFs[tid].push_back(aj[j]);
          count++;
        }
      }
      _XLEAFs[tid].push_back(_XLEAFs[tid].back() + count);
    }

#pragma omp barrier
#pragma omp single
    {
      _XLEAF_prefix[0] = 0;
      for (int i = 0; i < _nthreads; i++) {
        _XLEAF_prefix[i + 1] = _XLEAF_prefix[i] + _LEAFs[i].size();
      }
      leaf.ResizeAI(nnodes + 1);
      leaf.ResizeAJ(_XLEAF_prefix.back());
    }
    for (auto i = start; i <= end; i++) {
      leaf.AI()[i] = _XLEAFs[tid][i - start] + _XLEAF_prefix[tid] + base;
    }
    COLTYPE pos = _XLEAF_prefix[tid];
    for (auto i : _LEAFs[tid]) {
      leaf.AJ()[pos++] = i;
    }
  }
}

template <matrix_utils::ResizableCSR CSRMatrixType>
void SymbolicCholesky<CSRMatrixType>::apply(
    const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,
    const COLTYPE *parent, const COLTYPE *XLEAF, const COLTYPE *LEAF,
    CSRMatrixType &L) {
  const auto base = ai[0];
#pragma omp parallel num_threads(_nthreads)
  {
    const int tid = omp_get_thread_num();
    auto [start, end] =
        utils::LoadPrefixBalancedPartitionPos(ai, ai + nnodes, tid, _nthreads);
    _ais[tid].clear();
    _ais[tid].push_back(0);
    _ajs[tid].clear();
    for (auto i = start; i < end; i++) {
      ROWTYPE count = 0;
      for (auto j = XLEAF[i] - base; j < XLEAF[i + 1] - base; j++) {
        COLTYPE node = LEAF[j] - base;
        COLTYPE nextleaf;
        if (j + 1 == XLEAF[i + 1] - base) {
          nextleaf = i;
        } else {
          nextleaf = LEAF[j + 1] - base;
        }
        while (node < nextleaf) {
          _ajs[tid].push_back(node + base);
          count++;
          node = parent[node] - base;
        }
      }
      _ajs[tid].push_back(i + base);
      _ais[tid].push_back(_ais[tid].back() + count + 1);
    }

#pragma omp barrier
#pragma omp single
    {
      _ais_prefix[0] = 0;
      for (int i = 0; i < _nthreads; i++) {
        _ais_prefix[i + 1] = _ais_prefix[i] + _ajs[i].size();
      }
      L.ResizeAI(nnodes + 1);
      L.ResizeAJ(_ais_prefix.back());
      L.rows = nnodes;
      L.cols = nnodes;
    }
    for (auto i = start; i <= end; i++) {
      L.AI()[i] = _ais[tid][i - start] + _ais_prefix[tid] + base;
    }
    COLTYPE pos = _ais_prefix[tid];
    for (auto i : _ajs[tid]) {
      L.AJ()[pos++] = i;
    }
  }
  L.ResizeAV(_ais_prefix.back());
}

template <matrix_utils::ResizableCSR CSRMatrixType>
bool SymbolicCholeskyCol<CSRMatrixType>::apply(const COLTYPE nnodes,
                                               const ROWTYPE *ai,
                                               const COLTYPE *aj,
                                               const COLTYPE *parent,
                                               CSRMatrixType &L) {
  if (_nthreads <= 0) {
    return false;
  }
  const auto base = ai[0];

  _diag.resize(nnodes);
  const bool has_diagonal = matrix_utils::Diagonal(
      nnodes, ai, aj, (double *)nullptr, _diag.data(), (double *)nullptr);
  if (!has_diagonal) {
    return false;
  }

  // initizlize internal data
  _aj.resize(nnodes);
  const auto visited_size = static_cast<std::size_t>(_nthreads) *
                            static_cast<std::size_t>(nnodes);
  _visited.resize(visited_size);
  const auto unvisited = std::numeric_limits<COLTYPE>::max();
#pragma omp parallel for simd num_threads(_nthreads)
  for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(visited_size);
       i++) {
    _visited[static_cast<std::size_t>(i)] = unvisited;
  }
  _queue.clear();

  // initialize L
  L.ResizeAI(nnodes + 1);
  L.AI()[0] = base;
  L.rows = nnodes;
  L.cols = nnodes;

  // Children in the elimination tree are exactly the child{j} sets used by
  // Algorithm 4.4; _degrees counts unprocessed children for ready scheduling.
  _degrees.resize(nnodes);
  _firstChild.resize(nnodes);
  _nextSibling.resize(nnodes);
  graph::parentToChildSibling(nnodes, base, parent, _firstChild.data(),
                              _nextSibling.data(),
                              static_cast<COLTYPE *>(nullptr),
                              _degrees.data());

  for (COLTYPE i = 0; i < nnodes; i++) {
    if (_degrees[i] == 0) {
      _queue.push_back(i);
    }
  }

  _finished = 0;

  std::vector<std::thread> threads;
  for (int i = 0; i < _nthreads; i++) {
    threads.emplace_back(&SymbolicCholeskyCol<CSRMatrixType>::task, this,
                         nnodes, ai, aj, parent, i, std::ref(L));
  }
  for (auto &thread : threads) {
    thread.join();
  }

  std::inclusive_scan(L.AI(), L.AI() + nnodes + 1, L.AI());
  L.ResizeAJ(L.AI()[nnodes] - base);
  L.ResizeAV(L.AI()[nnodes] - base);

#pragma omp parallel num_threads(_nthreads)
  {
    const int tid = omp_get_thread_num();
    auto [start, end] = utils::LoadPrefixBalancedPartitionPos(
        L.AI(), L.AI() + nnodes, tid, _nthreads);
    for (auto i = start; i < end; i++) {
      assert(_aj[i].size() ==
             static_cast<std::size_t>(L.AI()[i + 1] - L.AI()[i]));
      std::memcpy(L.AJ() + L.AI()[i] - base, _aj[i].data(),
                  _aj[i].size() * sizeof(COLTYPE));
    }
  }
  return true;
}

template <matrix_utils::ResizableCSR CSRMatrixType>
void SymbolicCholeskyCol<CSRMatrixType>::task(const COLTYPE nnodes,
                                              const ROWTYPE *ai,
                                              const COLTYPE *aj,
                                              const COLTYPE *parent,
                                              const int tid, CSRMatrixType &L) {
  const auto base = ai[0];
  auto *visited = _visited.data() + static_cast<std::size_t>(tid) *
                                        static_cast<std::size_t>(nnodes);

  auto work = [&, this](const COLTYPE task) {
    _aj[task].clear();

    auto appendIfNew = [&](const COLTYPE node) {
      const auto node_id = node - base;
      if (visited[node_id] != task) {
        visited[node_id] = task;
        _aj[task].push_back(node);
      }
    };

    // Start with adj_G(A){task} restricted to the lower triangle, including the
    // diagonal. For a symmetric pattern this is the structural column of A.
    for (auto j = _diag[task] - base; j < ai[task + 1] - base; j++) {
      appendIfNew(aj[j]);
    }

    // Theorem 4.8 / Algorithm 4.4: colL{task} is the union of A's structural
    // column and all already-computed child column patterns, excluding task
    // itself from each child pattern by skipping the child's diagonal entry.
    auto child = _firstChild[task];
    while (child != std::numeric_limits<COLTYPE>::max()) {
      auto start = _aj[child].data() + 1; // skip diagonal
      auto end = _aj[child].data() + _aj[child].size();
      for (auto it = start; it < end; it++) {
        appendIfNew(*it);
      }
      child = _nextSibling[child];
    }
    std::sort(_aj[task].begin(), _aj[task].end());
    L.AI()[task + 1] = _aj[task].size();
  };

  // std::cout << "thread " << tid << " starting..." << std::endl;
  while (true) {
    COLTYPE task;
    {
      std::unique_lock<std::mutex> lock(_mutex);
      _cv.wait(lock, [this, nnodes] {
        return !_queue.empty() || _finished == nnodes;
      });
      if (_finished == nnodes) {
        break;
      }
      task = _queue.pop_front();
    }

    // process the task
    work(task);

    // A parent becomes ready exactly when all child column patterns needed by
    // the union above have been computed.
    auto parent_task = parent[task] - base;
    COLTYPE deg;
    if (parent_task != task) {
      std::atomic_ref<COLTYPE> degree(_degrees[parent_task]);
      deg = degree.fetch_sub(1);

      // notify one thread if new task is available
      if (deg == 1) {
        {
          std::unique_lock<std::mutex> lock(_mutex);
          _queue.push_back(parent_task);
        }
        _cv.notify_one();
      }
    }

    // if finished notify all threads
    auto finished = _finished.fetch_add(1);
    if (finished + 1 == nnodes) {
      _cv.notify_all();
      break;
    }
  }
}

template <matrix_utils::ResizableCSR CSRMatrixType>
bool SymbolicCholeskyColV2<CSRMatrixType>::apply(const COLTYPE nnodes,
                                                 const ROWTYPE *ai,
                                                 const COLTYPE *aj,
                                                 const COLTYPE *parent,
                                                 CSRMatrixType &L) {
  if (_nthreads <= 0) {
    return false;
  }
  const auto base = ai[0];

  _diag.resize(nnodes);
  const bool has_diagonal = matrix_utils::Diagonal(
      nnodes, ai, aj, (double *)nullptr, _diag.data(), (double *)nullptr);
  if (!has_diagonal) {
    return false;
  }

  _aj.resize(nnodes);
  const auto visited_size = static_cast<std::size_t>(_nthreads) *
                            static_cast<std::size_t>(nnodes);
  _visited.resize(visited_size);
  const auto unvisited = std::numeric_limits<COLTYPE>::max();
#pragma omp parallel for simd num_threads(_nthreads)
  for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(visited_size);
       i++) {
    _visited[static_cast<std::size_t>(i)] = unvisited;
  }

  _queues.resize(_nthreads);
  for (auto &queue : _queues) {
    queue.clear();
  }
  _queueMutexes = std::make_unique<std::mutex[]>(_nthreads);

  L.ResizeAI(nnodes + 1);
  L.AI()[0] = base;
  L.rows = nnodes;
  L.cols = nnodes;

  // Children in the elimination tree are exactly the child{j} sets used by
  // Algorithm 4.4; _degrees counts unprocessed children for ready scheduling.
  _degrees.resize(nnodes);
  _firstChild.resize(nnodes);
  _nextSibling.resize(nnodes);
  graph::parentToChildSibling(nnodes, base, parent, _firstChild.data(),
                              _nextSibling.data(),
                              static_cast<COLTYPE *>(nullptr),
                              _degrees.data());

  COLTYPE ready_tasks = 0;
  for (COLTYPE i = 0; i < nnodes; i++) {
    if (_degrees[i] == 0) {
      _queues[ready_tasks % _nthreads].push_back(i);
      ready_tasks++;
    }
  }
  _readyTasks = ready_tasks;
  _finished = 0;

  std::vector<std::thread> threads;
  for (int i = 0; i < _nthreads; i++) {
    threads.emplace_back(&SymbolicCholeskyColV2<CSRMatrixType>::task, this,
                         nnodes, ai, aj, parent, i, std::ref(L));
  }
  for (auto &thread : threads) {
    thread.join();
  }

  std::inclusive_scan(L.AI(), L.AI() + nnodes + 1, L.AI());
  L.ResizeAJ(L.AI()[nnodes] - base);
  L.ResizeAV(L.AI()[nnodes] - base);

#pragma omp parallel num_threads(_nthreads)
  {
    const int tid = omp_get_thread_num();
    auto [start, end] = utils::LoadPrefixBalancedPartitionPos(
        L.AI(), L.AI() + nnodes, tid, _nthreads);
    for (auto i = start; i < end; i++) {
      assert(_aj[i].size() ==
             static_cast<std::size_t>(L.AI()[i + 1] - L.AI()[i]));
      std::memcpy(L.AJ() + L.AI()[i] - base, _aj[i].data(),
                  _aj[i].size() * sizeof(COLTYPE));
    }
  }
  return true;
}

template <matrix_utils::ResizableCSR CSRMatrixType>
void SymbolicCholeskyColV2<CSRMatrixType>::pushReady(const int tid,
                                                     const COLTYPE task) {
  _readyTasks.fetch_add(1, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(_queueMutexes[tid]);
    _queues[tid].push_back(task);
  }
  _readyCv.notify_one();
}

template <matrix_utils::ResizableCSR CSRMatrixType>
bool SymbolicCholeskyColV2<CSRMatrixType>::popReady(const int tid,
                                                    COLTYPE &task) {
  auto popFrom = [&](const int queue_id, const bool local) {
    std::lock_guard<std::mutex> lock(_queueMutexes[queue_id]);
    if (_queues[queue_id].empty()) {
      return false;
    }
    if (local) {
      task = _queues[queue_id].back();
      _queues[queue_id].pop_back();
    } else {
      task = _queues[queue_id].front();
      _queues[queue_id].pop_front();
    }
    return true;
  };

  if (popFrom(tid, true)) {
    _readyTasks.fetch_sub(1, std::memory_order_acq_rel);
    return true;
  }

  for (int i = 1; i < _nthreads; i++) {
    const int victim = (tid + i) % _nthreads;
    if (popFrom(victim, false)) {
      _readyTasks.fetch_sub(1, std::memory_order_acq_rel);
      return true;
    }
  }
  return false;
}

template <matrix_utils::ResizableCSR CSRMatrixType>
void SymbolicCholeskyColV2<CSRMatrixType>::task(
    const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,
    const COLTYPE *parent, const int tid, CSRMatrixType &L) {
  const auto base = ai[0];
  auto *visited = _visited.data() + static_cast<std::size_t>(tid) *
                                        static_cast<std::size_t>(nnodes);

  auto work = [&, this](const COLTYPE task) {
    _aj[task].clear();

    auto appendIfNew = [&](const COLTYPE node) {
      const auto node_id = node - base;
      if (visited[node_id] != task) {
        visited[node_id] = task;
        _aj[task].push_back(node);
      }
    };

    // Start with adj_G(A){task} restricted to the lower triangle, including the
    // diagonal. For a symmetric pattern this is the structural column of A.
    for (auto j = _diag[task] - base; j < ai[task + 1] - base; j++) {
      appendIfNew(aj[j]);
    }

    // Theorem 4.8 / Algorithm 4.4: merge the already-computed patterns of the
    // children of task in the elimination tree. Skipping the first child entry
    // removes the child's diagonal, so only the off-diagonal column pattern is
    // propagated upward.
    auto child = _firstChild[task];
    while (child != std::numeric_limits<COLTYPE>::max()) {
      auto start = _aj[child].data() + 1; // skip diagonal
      auto end = _aj[child].data() + _aj[child].size();
      for (auto it = start; it < end; it++) {
        appendIfNew(*it);
      }
      child = _nextSibling[child];
    }
    std::sort(_aj[task].begin(), _aj[task].end());
    L.AI()[task + 1] = _aj[task].size();
  };

  while (true) {
    COLTYPE task;
    if (!popReady(tid, task)) {
      std::unique_lock<std::mutex> lock(_readyMutex);
      _readyCv.wait(lock, [this, nnodes] {
        return _readyTasks.load(std::memory_order_acquire) > 0 ||
               _finished.load(std::memory_order_acquire) == nnodes;
      });
      if (_finished.load(std::memory_order_acquire) == nnodes) {
        break;
      }
      continue;
    }

    work(task);

    auto parent_task = parent[task] - base;
    if (parent_task != task) {
      std::atomic_ref<COLTYPE> degree(_degrees[parent_task]);
      if (degree.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        pushReady(tid, parent_task);
      }
    }

    auto finished = _finished.fetch_add(1, std::memory_order_acq_rel);
    if (finished + 1 == nnodes) {
      _readyCv.notify_all();
      break;
    }
  }
}

template <matrix_utils::ResizableCSR CSRMatrixType>
bool SymbolicCholeskyColV3<CSRMatrixType>::apply(const COLTYPE nnodes,
                                                 const ROWTYPE *ai,
                                                 const COLTYPE *aj,
                                                 const COLTYPE *parent,
                                                 CSRMatrixType &L) {
  if (_nthreads <= 0) {
    return false;
  }
  const auto base = ai[0];

  _diag.resize(nnodes);
  const bool has_diagonal = matrix_utils::Diagonal(
      nnodes, ai, aj, (double *)nullptr, _diag.data(), (double *)nullptr);
  if (!has_diagonal) {
    return false;
  }

  _aj.resize(nnodes);
  const auto thread_workspace_size = static_cast<std::size_t>(_nthreads) *
                                     static_cast<std::size_t>(nnodes);
  _visited.resize(thread_workspace_size);
  _pendingChildren.resize(_nthreads);

  L.ResizeAI(nnodes + 1);
  L.AI()[0] = base;
  L.rows = nnodes;
  L.cols = nnodes;

  // Children in the elimination tree are exactly the child{j} sets used by
  // Algorithm 4.4. V3 also reuses _degrees for child counts while merging.
  _degrees.resize(nnodes);
  _firstChild.resize(nnodes);
  _nextSibling.resize(nnodes);
  graph::parentToChildSibling(nnodes, base, parent, _firstChild.data(),
                              _nextSibling.data(),
                              static_cast<COLTYPE *>(nullptr),
                              _degrees.data());

  // Any topological order of the elimination tree is valid: a column only needs
  // its children processed before their patterns can be merged.
  _topoPerm.resize(nnodes);
  _topoPrefix.resize(nnodes + 1);
  const auto levels = graph::parentTopologicalOrder(
      nnodes, base, parent, _degrees.data(), _visited.data(), _topoPerm.data(),
      _topoPrefix.data());
  if (_topoPrefix[levels] - base != nnodes) {
    return false;
  }

  const auto unvisited = std::numeric_limits<COLTYPE>::max();
#pragma omp parallel for num_threads(_nthreads)
  for (std::ptrdiff_t i = 0;
       i < static_cast<std::ptrdiff_t>(thread_workspace_size); i++) {
    const auto idx = static_cast<std::size_t>(i);
    _visited[idx] = unvisited;
  }

  if (_readySize < nnodes) {
    _ready.reset(new std::atomic<int>[nnodes]);
    _readySize = nnodes;
  }
#pragma omp parallel for num_threads(_nthreads)
  for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(nnodes); i++) {
    _ready[static_cast<std::size_t>(i)].store(0, std::memory_order_relaxed);
  }
  _nextTask.store(0, std::memory_order_relaxed);

  std::vector<std::thread> threads;
  for (int i = 0; i < _nthreads; i++) {
    threads.emplace_back(&SymbolicCholeskyColV3<CSRMatrixType>::task, this,
                         nnodes, ai, aj, i, std::ref(L));
  }
  for (auto &thread : threads) {
    thread.join();
  }

  std::inclusive_scan(L.AI(), L.AI() + nnodes + 1, L.AI());
  L.ResizeAJ(L.AI()[nnodes] - base);
  L.ResizeAV(L.AI()[nnodes] - base);

#pragma omp parallel num_threads(_nthreads)
  {
    const int tid = omp_get_thread_num();
    auto [start, end] = utils::LoadPrefixBalancedPartitionPos(
        L.AI(), L.AI() + nnodes, tid, _nthreads);
    for (auto i = start; i < end; i++) {
      assert(_aj[i].size() ==
             static_cast<std::size_t>(L.AI()[i + 1] - L.AI()[i]));
      std::memcpy(L.AJ() + L.AI()[i] - base, _aj[i].data(),
                  _aj[i].size() * sizeof(COLTYPE));
    }
  }
  return true;
}

template <matrix_utils::ResizableCSR CSRMatrixType>
void SymbolicCholeskyColV3<CSRMatrixType>::task(
    const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj, const int tid,
    CSRMatrixType &L) {
  const auto base = ai[0];
  auto *visited = _visited.data() + static_cast<std::size_t>(tid) *
                                        static_cast<std::size_t>(nnodes);
  auto &pending_children = _pendingChildren[tid];

  auto work = [&, this](const COLTYPE task) {
    _aj[task].clear();

    auto appendIfNew = [&](const COLTYPE node) {
      const auto node_id = node - base;
      if (visited[node_id] != task) {
        visited[node_id] = task;
        _aj[task].push_back(node);
      }
    };

    // Start with adj_G(A){task} restricted to the lower triangle, including the
    // diagonal. For a symmetric pattern this is the structural column of A.
    for (auto j = _diag[task] - base; j < ai[task + 1] - base; j++) {
      appendIfNew(aj[j]);
    }

    auto mergeChild = [&](const COLTYPE child) {
      // Skip the child diagonal so the child contributes its off-diagonal
      // column pattern, matching the child-pattern union in (4.6).
      auto start = _aj[child].data() + 1; // skip diagonal
      auto end = _aj[child].data() + _aj[child].size();
      for (auto it = start; it < end; it++) {
        appendIfNew(*it);
      }
    };

    // Claiming tasks in topological order is cheap, but threads can overtake
    // one another; wait only for the children that are not ready yet.
    const auto child_count = static_cast<std::size_t>(_degrees[task]);
    if (pending_children.capacity() < child_count) {
      pending_children.reserve(child_count);
    }
    pending_children.clear();
    auto child = _firstChild[task];
    while (child != std::numeric_limits<COLTYPE>::max()) {
      if (_ready[child].load(std::memory_order_acquire) != 0) {
        mergeChild(child);
      } else {
        pending_children.push_back(child);
      }
      child = _nextSibling[child];
    }

    COLTYPE spin_count = 0;
    while (!pending_children.empty()) {
      bool merged_any = false;
      std::size_t write = 0;
      const auto pending_count = pending_children.size();
      for (std::size_t read = 0; read < pending_count; read++) {
        const auto child = pending_children[read];
        if (_ready[child].load(std::memory_order_acquire) != 0) {
          mergeChild(child);
          merged_any = true;
        } else {
          pending_children[write++] = child;
        }
      }
      pending_children.resize(write);

      if (merged_any) {
        spin_count = 0;
      } else if (++spin_count % 64 == 0) {
        std::this_thread::yield();
      }
    }

    std::sort(_aj[task].begin(), _aj[task].end());
    L.AI()[task + 1] = _aj[task].size();
    _ready[task].store(1, std::memory_order_release);
  };

  while (true) {
    const auto task_pos = _nextTask.fetch_add(1, std::memory_order_relaxed);
    if (task_pos >= nnodes) {
      break;
    }
    work(_topoPerm[task_pos] - base);
  }
}

// instantiate for common types
#define INSTANTIATE_CHOLESKY(ROWTYPE, COLTYPE)                                 \
  template void nnzCount<ROWTYPE, COLTYPE>(                                    \
      const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,              \
      const COLTYPE *parent, ROWTYPE *row_count, ROWTYPE *col_count,           \
      COLTYPE *mark);

INSTANTIATE_CHOLESKY(std::int32_t, std::int32_t)
INSTANTIATE_CHOLESKY(std::int64_t, std::int64_t)

template class SkeletonGraph<
    ::matrix_utils::CSRMatrix<std::int32_t, std::int32_t, double>>;
template class SkeletonGraph<
    ::matrix_utils::CSRMatrix<std::int64_t, std::int64_t, double>>;
template class SymbolicCholesky<
    ::matrix_utils::CSRMatrix<std::int32_t, std::int32_t, double>>;
template class SymbolicCholesky<
    ::matrix_utils::CSRMatrix<std::int64_t, std::int64_t, double>>;
template class SymbolicCholeskyCol<
    ::matrix_utils::CSRMatrix<std::int32_t, std::int32_t, double>>;
template class SymbolicCholeskyCol<
    ::matrix_utils::CSRMatrix<std::int64_t, std::int64_t, double>>;
template class SymbolicCholeskyColV2<
    ::matrix_utils::CSRMatrix<std::int32_t, std::int32_t, double>>;
template class SymbolicCholeskyColV2<
    ::matrix_utils::CSRMatrix<std::int64_t, std::int64_t, double>>;
template class SymbolicCholeskyColV3<
    ::matrix_utils::CSRMatrix<std::int32_t, std::int32_t, double>>;
template class SymbolicCholeskyColV3<
    ::matrix_utils::CSRMatrix<std::int64_t, std::int64_t, double>>;
} // namespace factorization
