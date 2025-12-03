#include "UnionFind.h"
#include "mkl_sparse_mat.h"
#include "utils.h"
#include <iostream>
#include <numeric>
#include <omp.h>
#include <unordered_map>
#include <vector>

namespace reordering {

template <typename T> T Find(T *parents, T x) {
  while (x != parents[x]) {
    parents[x] = parents[parents[x]];
    x = parents[x];
  }
  return x;
};

template MKL_INT Find(MKL_INT *parents, MKL_INT x);

template <typename T> T UniteByRank(T *rank, T *parent, const T i, const T j) {
  T pi = Find(parent, i);
  T pj = Find(parent, j);
  if (pi == pj)
    return pi;
  if (rank[pi] < rank[pj]) {
    parent[pi] = pj;
    return pj;
  } else if (rank[pi] > rank[pj]) {
    parent[pj] = pi;
    return pi;
  } else {
    parent[pi] = pj;
    rank[pj]++;
    return pj;
  }
}

template MKL_INT UniteByRank(MKL_INT *rank, MKL_INT *parent, const MKL_INT i,
                             const MKL_INT j);

template <typename T> T Unite(T *parents, const T i, const T j) {
  T pi = Find(parents, i);
  T pj = Find(parents, j);
  if (pi == pj)
    return pi;
  parents[pj] = pi;
  return pi;
}
template int Unite(int *parent, const int i, const int j);

template <typename T, bool Rank>
UnionFind<T, Rank>::UnionFind(T size) : _parents(size) {
  if constexpr (Rank) {
    _ranks.resize(size);
  }
  for (T i = 0; i < size; ++i) {
    _parents[i] = i;
    if constexpr (Rank)
      _ranks[i] = 0;
  }
}

template <typename T, bool Rank> void UnionFind<T, Rank>::reset(const T size) {
  _parents.resize(size);
  if constexpr (Rank) {
    _ranks.resize(size);
  }
  for (T i = 0; i < size; ++i) {
    _parents[i] = i;
    if constexpr (Rank)
      _ranks[i] = 0;
  }
}

template class UnionFind<int, false>;
template class UnionFind<int, true>;

template <typename ROWTYPE, typename COLTYPE>
void UnionFindRank(COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj, COLTYPE* parents) {
  const COLTYPE base = ai[0];
  std::vector<COLTYPE> ranks(rows);
  std::iota(parents, parents + rows, 0);
  std::fill(ranks.begin(), ranks.end(), 0);
  for (COLTYPE i = 0; i < rows; i++) {
    for (ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; j++) {
      UniteByRank(ranks.data(), parents, i, aj[j] - base);
    }
  }
}

template <typename ROWTYPE, typename COLTYPE>
void UnionFindRem(COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj, COLTYPE* parents) {
  const COLTYPE base = ai[0];
  std::iota(parents, parents + rows, 0);

  auto unite = [parents](COLTYPE x, COLTYPE y) {
    while (parents[x] != parents[y]) {
      if (parents[x] < parents[y]) {
        if (x == parents[x]) {
          parents[x] = parents[y];
          break;
        }
        COLTYPE tmp = parents[x];
        parents[x] = parents[y];
        x = tmp;
      } else {
        if (y == parents[y]) {
          parents[y] = parents[x];
          break;
        }
        COLTYPE tmp = parents[y];
        parents[y] = parents[x];
        y = tmp;
      }
    }
  };
  for (COLTYPE i = 0; i < rows; i++) {
    for (ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; j++) {
      unite(i, aj[j] - base);
    }
  }
}

// Multi-core Spanning Forest Algorithms using the Disjoint-set Data Structure
template <typename ROWTYPE, typename COLTYPE>
void ParUnionFindRem(COLTYPE rows, ROWTYPE const *const ai,
                                COLTYPE const *const aj, COLTYPE* parents) {
  const COLTYPE base = ai[0];
  std::iota(parents, parents + rows, 0);

  auto unite = [parents](COLTYPE x, COLTYPE y) {
    while (true) {
      COLTYPE px = parents[x];
      COLTYPE py = parents[y];
      if (px == py)
        break;
      if (py < px) {
        std::swap(x, y);
        std::swap(px, py);
      }
      if (x == px && __sync_bool_compare_and_swap(&parents[x], px, py))
        break;
      if (__sync_bool_compare_and_swap(&parents[x], px, py))
        x = px;
    }
  };

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    auto [start, end] = utils::LoadPrefixBalancedPartition(
        ai, ai + rows, tid, nthreads);

    for (auto it = start; it != end; it++) {
      for (ROWTYPE j = *it - base; j < *(it + 1) - base; j++) {
        unite(it - ai, aj[j] - base);
      }
    }
  }
  // #pragma omp parallel for
  //   for (COLTYPE i = 0; i < rows; i++) {
  //     for (ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; j++) {

  //       unite(i + base, aj[j]);
  //     }
  //   }
}

DisjointSets::DisjointSets(uint32_t rows)
    : mData(rows) {
  for (uint32_t i = 0; i < mData.size(); ++i)
    mData[i] = (uint32_t)i;
}

uint32_t DisjointSets::find(uint32_t id) const {
  while (id != parent(id)) {
    uint64_t value = mData[id];
    uint32_t new_parent = parent((uint32_t)value);
    uint64_t new_value = (value & 0xFFFFFFFF00000000ULL) | new_parent;
    /* Try to update parent (may fail, that's ok) */
    if (value != new_value)
      mData[id].compare_exchange_weak(value, new_value);
    id = new_parent;
  }
  return id;
}

bool DisjointSets::same(uint32_t id1, uint32_t id2) const {
  for (;;) {
    id1 = find(id1);
    id2 = find(id2);
    if (id1 == id2)
      return true;
    if (parent(id1) == id1)
      return false;
  }
}

uint32_t DisjointSets::unite(uint32_t id1, uint32_t id2) {
  for (;;) {
    id1 = find(id1);
    id2 = find(id2);

    if (id1 == id2)
      return id1;

    uint32_t r1 = rank(id1), r2 = rank(id2);

    if (r1 > r2 || (r1 == r2 && id1 < id2)) {
      std::swap(r1, r2);
      std::swap(id1, id2);
    }

    uint64_t oldEntry = ((uint64_t)r1 << 32) | id1;
    uint64_t newEntry = ((uint64_t)r1 << 32) | id2;

    if (!mData[id1].compare_exchange_strong(oldEntry, newEntry))
      continue;

    if (r1 == r2) {
      oldEntry = ((uint64_t)r2 << 32) | id2;
      newEntry = ((uint64_t)(r2 + 1) << 32) | id2;
      /* Try to update the rank (may fail, that's ok) */
      mData[id2].compare_exchange_weak(oldEntry, newEntry);
    }

    break;
  }
  return id2;
}

template <typename ROWTYPE, typename COLTYPE>
void DisjointSets::execute(COLTYPE rows, ROWTYPE const *const ai,
                           COLTYPE const *const aj) {

  const COLTYPE base = ai[0];
#pragma omp parallel for
  for (COLTYPE i = 0; i < rows; i++) {
    for (ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; j++) {
      unite(i, aj[j] - base);
    }
  }
}

template <typename T>
int CountComponents(T* parents, T size) {
  int sum = 0;
#pragma omp parallel for reduction(+ : sum)
  for (T i = 0; i < size; i++) {
    if (Find(parents, i) == i)
      sum++;
  }
  return sum;
}

// Parallel algorithm to compute component statistics
// Groups all nodes by their connected component and computes per-component metadata
template <typename T>
void ComponentsStat(const T* parents, T size, const T base, std::vector<T>& compRoots,
                    std::vector<T>& sortedComp, std::vector<T>& compPrefSum, int numThreads)
{
    sortedComp.resize(size);

    // Ensure numThreads is valid (>= 1)
    if (numThreads <= 0)
    {
        numThreads = omp_get_max_threads();
    }

    // Per-thread storage for component roots discovered by each thread
    std::vector<std::vector<T>> rootsPerThread(numThreads);

    // Map from root node ID to its index in compRoots array
    std::unordered_map<T, T> rootToCompIndex;

    // Per-thread, per-component prefix sums for parallel output assignment
    // Layout: [thread][component] in row-major order
    std::vector<T> threadCompOffsets;

#pragma omp parallel num_threads(numThreads)
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();

        // Partition work across threads using load-balanced partitioning
        auto [workStart, workEnd] = utils::LoadBalancedPartition(parents, parents + size, tid, nthreads);

        // Phase 1: Each thread finds root nodes in its partition
        for (auto it = workStart; it != workEnd; it++)
        {
            const T nodeIndex = it - parents;
            if (Find(const_cast<T*>(parents), nodeIndex) == nodeIndex)
            {
                rootsPerThread[tid].push_back(nodeIndex);
            }
        }

#pragma omp barrier
#pragma omp single
        {
            // Phase 2: Gather all roots from all threads and build index mapping
            int totalRoots = 0;
            for (int i = 0; i < nthreads; i++)
            {
                totalRoots += rootsPerThread[i].size();
            }
            compRoots.reserve(totalRoots);
            compRoots.resize(0);

            for (int i = 0; i < nthreads; i++)
            {
                for (auto root : rootsPerThread[i])
                {
                    rootToCompIndex[root] = compRoots.size();
                    compRoots.push_back(root);
                }
            }

            // Allocate storage for per-thread, per-component counters
            // threadCompOffsets[t * numComponents + c] = count for thread t, component c
            threadCompOffsets = std::vector<T>(compRoots.size() * (nthreads + 1), 0);
            compPrefSum = std::vector<T>(compRoots.size() + 1, 0);
        }

        // Phase 3: Each thread counts nodes per component in its partition
        for (auto it = workStart; it != workEnd; it++)
        {
            const T nodeIndex = it - parents;
            const T rootNode = Find(const_cast<T*>(parents), nodeIndex);
            const T compIndex = rootToCompIndex[rootNode];
            threadCompOffsets[(tid + 1) * compRoots.size() + compIndex]++;
        }

#pragma omp barrier
#pragma omp single
        {
            // Phase 4: Compute prefix sums to determine output positions
            // First, compute per-component prefix sums across threads
            for (size_t compIndex = 0; compIndex < compRoots.size(); compIndex++)
            {
                threadCompOffsets[compIndex] = compPrefSum[compIndex];
                for (int threadId = 0; threadId < nthreads; threadId++)
                {
                    threadCompOffsets[(threadId + 1) * compRoots.size() + compIndex] +=
                        threadCompOffsets[threadId * compRoots.size() + compIndex];
                }
                compPrefSum[compIndex + 1] = threadCompOffsets[nthreads * compRoots.size() + compIndex];
            }
        }
        
        // Phase 5: Write nodes to output in component-sorted order
        // Each thread writes to its pre-computed position using prefix sums
        for (auto it = workStart; it != workEnd; it++)
        {
            const T nodeIndex = it - parents;
            const T rootNode = Find(const_cast<T*>(parents), nodeIndex);
            const T compIndex = rootToCompIndex[rootNode];

            // Get and increment the output position for this thread and component
            T& outputPos = threadCompOffsets[tid * compRoots.size() + compIndex];
            sortedComp[outputPos++] = nodeIndex + base;
        }
    }
}

// Explicit template instantiations for common types
template void UnionFindRank<int, int>(int rows, int const* ai, int const* aj, int* parents);
template void UnionFindRank<int64_t, int64_t>(int64_t rows, int64_t const* ai, int64_t const* aj, int64_t* parents);

template void UnionFindRem<int, int>(int rows, int const* ai, int const* aj, int* parents);
template void UnionFindRem<int64_t, int64_t>(int64_t rows, int64_t const* ai, int64_t const* aj, int64_t* parents);

template void ParUnionFindRem<int, int>(int rows, int const* ai, int const* aj, int* parents);
template void ParUnionFindRem<int64_t, int64_t>(int64_t rows, int64_t const* ai, int64_t const* aj, int64_t* parents);

template void DisjointSets::execute<int, int>(int rows, int const* ai, int const* aj);
template void DisjointSets::execute<int64_t, int64_t>(int64_t rows, int64_t const* ai, int64_t const* aj);

template int CountComponents<int>(int* parents, int size);
template int CountComponents<int64_t>(int64_t* parents, int64_t size);

template void ComponentsStat<int>(const int* parents, int size, const int base, std::vector<int> &compRoots, std::vector<int> &sortedComp, std::vector<int> &compPrefSum, int numThreads);
template void ComponentsStat<int64_t>(const int64_t* parents, int64_t size, const int64_t base, std::vector<int64_t> &compRoots, std::vector<int64_t> &sortedComp, std::vector<int64_t> &compPrefSum, int numThreads);

} // namespace reordering