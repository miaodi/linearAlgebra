#pragma once

#include <atomic>
#include <cstdint>
#include <mkl_types.h>
#include <utility>
#include <vector>

namespace mkl_wrapper {
class mkl_sparse_mat;
}

namespace reordering {

template <typename T> T Find(std::vector<T> &parents, T x) {
  while (x != parents[x]) {
    parents[x] = parents[parents[x]];
    x = parents[x];
  }
  return x;
};

// template <typename T> T Find(std::vector<std::atomic<T>> &parents, T x) {
//   while (x != parents[x]) {
//     T px = parents[x];
//     if (px != x)
//       parents[x].compare_exchange_weak(px, parents[px]);
//     x = parents[x];
//   }
//   return x;
// };

std::vector<MKL_INT>
UnionFindRank(mkl_wrapper::mkl_sparse_mat const *const mat);

std::vector<MKL_INT> UnionFindRem(mkl_wrapper::mkl_sparse_mat const *const mat);

// Multi-core Spanning Forest Algorithms using the Disjoint-set Data Structure
std::vector<MKL_INT>
ParUnionFindRem(mkl_wrapper::mkl_sparse_mat const *const mat);

// Wait-free parallel algorithms for the union-find problem
// https://github.com/wjakob/dset
class DisjointSets {
public:
  DisjointSets(mkl_wrapper::mkl_sparse_mat const *const mat);

  uint32_t find(uint32_t id) const;

  bool same(uint32_t id1, uint32_t id2) const;
  uint32_t unite(uint32_t id1, uint32_t id2);

  uint32_t size() const { return (uint32_t)mData.size(); }

  uint32_t rank(uint32_t id) const {
    return ((uint32_t)(mData[id] >> 32)) & 0x7FFFFFFFu;
  }

  void execute();

  uint32_t parent(uint32_t id) const { return (uint32_t)mData[id]; }
  mutable std::vector<std::atomic<uint64_t>> mData;
  mkl_wrapper::mkl_sparse_mat const *const mMat;
};

int CountComponents(std::vector<MKL_INT> &parents, const MKL_INT base = 0);

} // namespace reordering