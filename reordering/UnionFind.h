#pragma once

#include <atomic>
#include <cstdint>
#include <utility>
#include <vector>
#include <type_traits>
#include <variant>

namespace mkl_wrapper {
class mkl_sparse_mat;
}

namespace reordering {

template <typename T> T Find(T *parents, const T x);

template <typename T> T UniteByRank(T *rank, T *parent, const T i, const T j);

// Always use i as the root
template <typename T> T Unite(T *parents, const T i, const T j);

template <typename T, bool Rank> class UnionFind {
public:
  UnionFind() = default;
  UnionFind(T size);

  T Find(const T x) { return reordering::Find(_parents.data(), x); }

  T Unite(const T i, const T j) {
    if constexpr (Rank) {
      return reordering::UniteByRank(_ranks.data(), _parents.data(), i, j);
    } else {
      return reordering::Unite(_parents.data(), i, j);
    }
  }

  void reset(const T size);

private:
  std::vector<T> _parents;

  std::conditional_t<Rank, std::vector<T>, std::monostate> _ranks;
};

// NOTE: no matter the base of ai, the output parents vector is always 0
// based

template <typename ROWTYPE, typename COLTYPE>
void UnionFindRank(COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj, COLTYPE* parents);

template <typename ROWTYPE, typename COLTYPE>
void UnionFindRem(COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj, COLTYPE* parents);

// Multi-core Spanning Forest Algorithms using the Disjoint-set Data Structure
template <typename ROWTYPE, typename COLTYPE>
void ParUnionFindRem(COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj, COLTYPE* parents);

// Wait-free parallel algorithms for the union-find problem
// https://github.com/wjakob/dset
class DisjointSets {
public:
  DisjointSets(uint32_t rows);

  uint32_t find(uint32_t id) const;

  bool same(uint32_t id1, uint32_t id2) const;
  uint32_t unite(uint32_t id1, uint32_t id2);

  uint32_t size() const { return (uint32_t)mData.size(); }

  uint32_t rank(uint32_t id) const {
    return ((uint32_t)(mData[id] >> 32)) & 0x7FFFFFFFu;
  }

  template <typename ROWTYPE, typename COLTYPE>
  void execute(COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj);

  uint32_t parent(uint32_t id) const { return (uint32_t)mData[id]; }
  mutable std::vector<std::atomic<uint64_t>> mData;
};

template <typename T>
int CountComponents(T* parents, T size);

// Compute statistics about connected components in a union-find structure
// Input:
//   parents: union-find parent array (size elements)
//   size: number of elements in the union-find structure
//   base: base index (0 or 1) for output indexing
//   numThreads: number of threads to use (if <= 0, uses current OMP setting)
// Output:
//   compRoots: vector of root nodes for each component
//   sortedComp: nodes sorted by component (grouped by root)
//   compPrefSum: prefix sum of component sizes (compPrefSum[i+1] - compPrefSum[i] = size of component i)
template <typename T>
void ComponentsStat(const T* parents, T size, const T base, std::vector<T>& compRoots,
                    std::vector<T>& sortedComp, std::vector<T>& compPrefSum, int numThreads = 1);

} // namespace reordering