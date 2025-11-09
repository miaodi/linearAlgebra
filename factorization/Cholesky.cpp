#include "Cholesky.hpp"
#include "matrix_utils.hpp"
#include "permutation.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <omp.h>
#include <span>
#include <thread>

namespace factorization {

template <typename ROWTYPE, typename COLTYPE>
void EliminationTree(const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,
                     COLTYPE *parent, COLTYPE *ancestor) {
  const ROWTYPE base = ai[0];
  COLTYPE jroot;
  for (COLTYPE i = 0; i < nnodes; i++) {
    parent[i] = i + base;
    ancestor[i] = i + base;
    for (ROWTYPE j_idx = ai[i] - base; j_idx < ai[i + 1] - base; j_idx++) {
      jroot = aj[j_idx] - base;
      if (jroot >= i) // break if jroot is not in the lower triangle
        break;

      while (ancestor[jroot] - base != jroot && ancestor[jroot] - base != i) {
        COLTYPE l = ancestor[jroot] - base;
        ancestor[jroot] = i + base;
        jroot = l;
      }
      if (jroot == ancestor[jroot] - base) {
        parent[jroot] = i + base;
        ancestor[jroot] = i + base;
      }
    }
  }
}

template <typename COLTYPE>
void PostOrder<COLTYPE>::BuildChildren(const COLTYPE nnodes, const COLTYPE base,
                                       const COLTYPE *parent) {
  _childrenPrefix.resize(nnodes + 1);
  _roots.clear();
  std::fill(_childrenPrefix.begin(), _childrenPrefix.end(), 0);
  for (COLTYPE i = 0; i < nnodes; i++) {
    if (parent[i] != i + base) {
      _childrenPrefix[parent[i] - base + 1]++;
    } else {
      _roots.push_back(i);
    }
  }
  std::inclusive_scan(_childrenPrefix.begin(), _childrenPrefix.end(),
                      _childrenPrefix.begin());

  _children.resize(_childrenPrefix.back());
  for (COLTYPE i = 0; i < nnodes; i++) {
    if (parent[i] != i + base) {
      _children[_childrenPrefix[parent[i] - base]++] = i;
    }
  }
  std::rotate(_childrenPrefix.rbegin(), _childrenPrefix.rbegin() + 1,
              _childrenPrefix.rend());
  _childrenPrefix[0] = 0;

  // for(COLTYPE i = 0; i < nnodes; i++){
  //   std::sort(_children.begin() + _childrenPrefix[i],
  //             _children.begin() + _childrenPrefix[i + 1]);
  // }
}

template <typename COLTYPE>
void PostOrder<COLTYPE>::DFS(const COLTYPE root, const COLTYPE base,
                             COLTYPE *&post) {
  for (COLTYPE i = _childrenPrefix[root]; i < _childrenPrefix[root + 1]; i++) {
    DFS(_children[i], base, post);
  }
  *post = root + base;
  post++;
}

template <typename COLTYPE>
void PostOrder<COLTYPE>::operator()(const COLTYPE nnodes, const COLTYPE base,
                                    const COLTYPE *parent,
                                    COLTYPE *permed_parent, COLTYPE *perm,
                                    COLTYPE *iperm) {
  auto perm_cp = perm;
  BuildChildren(nnodes, base, parent);
  for (auto root : _roots) {
    DFS(root, base, perm_cp);
  }
  assert(matrix_utils::isPermutationSerial(nnodes, base, perm));
  matrix_utils::invPerm(nnodes, base, perm, iperm);
  assert(matrix_utils::isPermutationSerial(nnodes, base, iperm));

#pragma omp parallel for
  for (COLTYPE i = 0; i < nnodes; i++) {
    permed_parent[i] = iperm[parent[perm[i] - base] - base];
  }
}

template <typename COLTYPE>
void PostOrderNoRecur<COLTYPE>::operator()(const COLTYPE nnodes,
                                           const COLTYPE base,
                                           const COLTYPE *parent,
                                           COLTYPE *permed_parent,
                                           COLTYPE *perm, COLTYPE *iperm) {
  _roots.clear();
  _firstChild.resize(nnodes);
  std::fill(_firstChild.begin(), _firstChild.end(),
            std::numeric_limits<COLTYPE>::max());
  _nextSibling.resize(nnodes);
  std::fill(_nextSibling.begin(), _nextSibling.end(),
            std::numeric_limits<COLTYPE>::max());
  for (COLTYPE i = 0; i < nnodes; i++) {
    auto parent_i = parent[i] - base;
    if (parent_i != i) {
      auto parent_first_child = _firstChild[parent_i];
      _firstChild[parent_i] = i;
      _nextSibling[i] = parent_first_child;
    } else {
      _roots.push_back(i);
    }
  }
  auto perm_cp = perm;
  while (!_roots.empty()) {
    auto root = _roots.back();
    if (_firstChild[root] == std::numeric_limits<COLTYPE>::max()) {
      _roots.pop_back();
      *perm_cp = root + base;
      perm_cp++;
    } else {
      _roots.push_back(_firstChild[root]);
      _firstChild[root] = _nextSibling[_firstChild[root]];
    }
  }
  assert(matrix_utils::isPermutationSerial(nnodes, base, perm));
  matrix_utils::invPerm(nnodes, base, perm, iperm);
  assert(matrix_utils::isPermutationSerial(nnodes, base, iperm));

#pragma omp parallel for
  for (COLTYPE i = 0; i < nnodes; i++) {
    permed_parent[i] = iperm[parent[perm[i] - base] - base];
  }
}

template <typename COLTYPE>
void SubtreeSize(const COLTYPE nnodes, const COLTYPE base,
                 const COLTYPE *parent, COLTYPE *subtree_size) {
  std::fill(subtree_size, subtree_size + nnodes, 1);
  for (COLTYPE i = 0; i < nnodes; i++) {
    auto k = parent[i] - base;
    if (k != i)
      subtree_size[k] += subtree_size[i];
  }
}

template <typename ROWTYPE, typename COLTYPE>
void NNZCount(const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,
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

template <matrix_utils::ResizableCSRMatrixType CSRMatrixType>
void SkeletonGraph<CSRMatrixType>::operator()(const COLTYPE nnodes,
                                              const ROWTYPE *ai,
                                              const COLTYPE *aj,
                                              const COLTYPE *parent,
                                              CSRMatrixType &leaf) {
  const ROWTYPE base = ai[0];
  _subtree_size.resize(nnodes);
  SubtreeSize(nnodes, base, parent, _subtree_size.data());
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

template <matrix_utils::ResizableCSRMatrixType CSRMatrixType>
void SymbolicCholesky<CSRMatrixType>::operator()(
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

template <matrix_utils::ResizableCSRMatrixType CSRMatrixType>
void SymbolicCholeskyCol<CSRMatrixType>::operator()(const COLTYPE nnodes,
                                                    const ROWTYPE *ai,
                                                    const COLTYPE *aj,
                                                    const COLTYPE *parent,
                                                    CSRMatrixType &L) {
  const auto base = ai[0];

  // initizlize internal data
  _aj.resize(nnodes);
  _queue.clear();

  // initialize L
  L.ResizeAI(nnodes + 1);
  L.AI()[0] = base;
  L.rows = nnodes;
  L.cols = nnodes;

  _diag.resize(nnodes);
  matrix_utils::Diagonal(nnodes, ai, aj, (double *)nullptr, _diag.data(),
                         (double *)nullptr);
  _firstChild.resize(nnodes);
  std::fill(_firstChild.begin(), _firstChild.end(),
            std::numeric_limits<COLTYPE>::max());
  _nextSibling.resize(nnodes);
  std::fill(_nextSibling.begin(), _nextSibling.end(),
            std::numeric_limits<COLTYPE>::max());
  _degrees.resize(nnodes);
  std::fill(_degrees.begin(), _degrees.end(), 0);
  for (COLTYPE i = 0; i < nnodes; i++) {
    auto parent_i = parent[i] - base;
    if (parent_i != i) {
      _degrees[parent_i]++;
      auto parent_first_child = _firstChild[parent_i];
      _firstChild[parent_i] = i;
      _nextSibling[i] = parent_first_child;
    }
  }

  for (int i = 0; i < nnodes; i++) {
    if (_degrees[i] == 0) {
      _queue.autoResizePush(i);
    }
  }

  _finished = 0;

  std::vector<std::thread> threads;
  for (int i = 0; i < _nthreads; i++) {
    threads.emplace_back(&SymbolicCholeskyCol<CSRMatrixType>::Task, this,
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
    for (auto i = start - base; i < end - base; i++) {
      for (auto j = L.AI()[i] - base; j < L.AI()[i + 1] - base; j++) {
        L.AJ()[j] = _aj[i][j - (L.AI()[i] - base)];
      }
    }
  }
}

template <class T, class Comp = std::less<T>>
void kway_merge_spans(const std::vector<std::span<const T>> &runs,
                      std::vector<T> &out, Comp comp = {}) {
  struct Node {
    std::size_t run; // which span
    std::size_t idx; // index within that span
  };

  // priority_queue is a max-heap; define comparator so the smallest element
  // is at the top by returning true if 'a' should come AFTER 'b'.
  struct NodeCmp {
    const std::vector<std::span<const T>> *runs;
    Comp comp;

    bool operator()(const Node &a, const Node &b) const {
      const T &ax = (*runs)[a.run][a.idx];
      const T &bx = (*runs)[b.run][b.idx];

      if (comp(ax, bx))
        return false; // a < b  => a before b
      if (comp(bx, ax))
        return true; // b < a  => a after b
      // tie-break by run index to be stable across runs
      return a.run > b.run;
    }
  };

  // Pre-size output capacity (optional but improves perf).
  std::size_t total = 0;
  for (auto s : runs)
    total += s.size();
  out.clear();
  out.reserve(total);

  std::priority_queue<Node, std::vector<Node>, NodeCmp> pq(
      NodeCmp{&runs, comp});

  // Seed heap with the first element of each non-empty span.
  for (std::size_t r = 0; r < runs.size(); ++r) {
    if (!runs[r].empty())
      pq.push(Node{r, 0});
  }

  while (!pq.empty()) {
    Node n = pq.top();
    pq.pop();
    if (out.empty() || out.back() != runs[n.run][n.idx]) {
      out.push_back(runs[n.run][n.idx]); // copy (spans are const)
    }

    if (++n.idx < runs[n.run].size()) {
      pq.push(n);
    }
  }
}

template <matrix_utils::ResizableCSRMatrixType CSRMatrixType>
void SymbolicCholeskyCol<CSRMatrixType>::Task(const COLTYPE nnodes,
                                              const ROWTYPE *ai,
                                              const COLTYPE *aj,
                                              const COLTYPE *parent,
                                              const int tid, CSRMatrixType &L) {
  const auto base = ai[0];

  auto work = [&, this](const COLTYPE task) {
    _aj[task].clear();
    std::vector<std::span<const COLTYPE>> Ljs;
    Ljs.emplace_back(
        std::span<const COLTYPE>(aj + _diag[task], aj + ai[task + 1] - base));

    auto child = _firstChild[task];
    while (child != std::numeric_limits<COLTYPE>::max()) {
      auto end = _aj[child].data() + _aj[child].size();
      auto start = _aj[child].data() + 2; // using the definition of parent
      Ljs.emplace_back(std::span<const COLTYPE>(start, end));
      child = _nextSibling[child];
    }
    kway_merge_spans(Ljs, _aj[task]);
    L.AI()[task + 1] = _aj[task].size();
  };

  std::cout << "thread " << tid << " starting..." << std::endl;
  while (true) {
    COLTYPE task;
    {
      std::unique_lock<std::mutex> lock(_mutex);
      _cv.wait(lock, [this, nnodes] {
        return !_queue.isEmpty() || _finished == nnodes;
      });
      if (_finished == nnodes) {
        break;
      }
      task = _queue.shift();
    }

    // process the task
    work(task);

    // update the queue
    auto parent_task = parent[task] - base;
    COLTYPE deg;
    if (parent_task != task) {
      std::atomic_ref<COLTYPE> degree(_degrees[parent_task]);
      deg = degree.fetch_sub(1);

      // notify one thread if new task is available
      if (deg == 1) {
        {
          std::unique_lock<std::mutex> lock(_mutex);
          _queue.autoResizePush(parent_task);
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

// instantiate for common types
#define INSTANTIATE_CHOLESKY(ROWTYPE, COLTYPE)                                 \
  template void EliminationTree<ROWTYPE, COLTYPE>(                             \
      const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,              \
      COLTYPE *parent, COLTYPE *ancestor);                                     \
  template void SubtreeSize<COLTYPE>(const COLTYPE nnodes, const COLTYPE base, \
                                     COLTYPE const *parent,                    \
                                     COLTYPE *subtree_size);                   \
  template void NNZCount<ROWTYPE, COLTYPE>(                                    \
      const COLTYPE nnodes, const ROWTYPE *ai, const COLTYPE *aj,              \
      const COLTYPE *parent, ROWTYPE *row_count, ROWTYPE *col_count,           \
      COLTYPE *mark);                                                          \
  template class PostOrder<COLTYPE>;                                           \
  template class PostOrderNoRecur<COLTYPE>;

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
} // namespace factorization