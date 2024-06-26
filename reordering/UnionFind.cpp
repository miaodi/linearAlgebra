#include "UnionFind.h"
#include "mkl_sparse_mat.h"
#include "utils.h"
#include <iostream>
#include <numeric>
#include <omp.h>
#include <unordered_map>
#include <vector>

namespace reordering {

std::vector<MKL_INT>
UnionFindRank(mkl_wrapper::mkl_sparse_mat const *const mat) {
  const MKL_INT base = mat->mkl_base();
  std::vector<MKL_INT> parents(mat->rows());
  std::vector<MKL_INT> ranks(mat->rows());
  std::iota(parents.begin(), parents.end(), 0);
  std::fill(ranks.begin(), ranks.end(), 0);
  auto unite = [&parents, &ranks](MKL_INT x, MKL_INT y) {
    MKL_INT px = Find(parents, x);
    MKL_INT py = Find(parents, y);
    if (px == py)
      return;
    if (ranks[px] < ranks[py]) {
      parents[px] = py;
    } else if (ranks[px] > ranks[py]) {
      parents[py] = px;
    } else {
      parents[px] = py;
      ranks[py]++;
    }
  };
  auto ai = mat->get_ai();
  auto aj = mat->get_aj();
  for (MKL_INT i = 0; i < mat->rows(); i++) {
    for (MKL_INT j = ai[i] - base; j < ai[i + 1] - base; j++) {

      unite(i, aj[j] - base);
    }
  }
  return parents;
}

std::vector<MKL_INT>
UnionFindRem(mkl_wrapper::mkl_sparse_mat const *const mat) {
  const MKL_INT base = mat->mkl_base();
  std::vector<MKL_INT> parents(mat->rows());
  std::iota(parents.begin(), parents.end(), 0);

  auto unite = [&parents](MKL_INT x, MKL_INT y) {
    while (parents[x] != parents[y]) {
      if (parents[x] < parents[y]) {
        if (x == parents[x]) {
          parents[x] = parents[y];
          break;
        }
        MKL_INT tmp = parents[x];
        parents[x] = parents[y];
        x = tmp;
      } else {
        if (y == parents[y]) {
          parents[y] = parents[x];
          break;
        }
        MKL_INT tmp = parents[y];
        parents[y] = parents[x];
        y = tmp;
      }
    }
  };
  auto ai = mat->get_ai();
  auto aj = mat->get_aj();
  for (MKL_INT i = 0; i < mat->rows(); i++) {
    for (MKL_INT j = ai[i] - base; j < ai[i + 1] - base; j++) {
      unite(i, aj[j] - base);
    }
  }
  return parents;
}

// Multi-core Spanning Forest Algorithms using the Disjoint-set Data Structure
std::vector<MKL_INT>
ParUnionFindRem(mkl_wrapper::mkl_sparse_mat const *const mat) {
  const MKL_INT base = mat->mkl_base();
  const MKL_INT rows = mat->rows();
  std::vector<MKL_INT> parents(mat->rows());
  std::iota(parents.begin(), parents.end(), 0);

  auto unite = [&parents](MKL_INT x, MKL_INT y) {
    while (true) {
      MKL_INT px = parents[x];
      MKL_INT py = parents[y];
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
  auto ai = mat->get_ai();
  auto aj = mat->get_aj();

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    auto [start, end] = utils::LoadPrefixBalancedPartition(
        ai.get(), ai.get() + rows, tid, nthreads);

    for (auto it = start; it != end; it++) {
      for (MKL_INT j = *it - base; j < *(it + 1) - base; j++) {
        unite(it - ai.get(), aj[j] - base);
      }
    }
  }
  // #pragma omp parallel for
  //   for (MKL_INT i = 0; i < mat->rows(); i++) {
  //     for (MKL_INT j = ai[i] - base; j < ai[i + 1] - base; j++) {

  //       unite(i + base, aj[j]);
  //     }
  //   }
  return parents;
}

DisjointSets::DisjointSets(mkl_wrapper::mkl_sparse_mat const *const mat)
    : mData(mat->rows()), mMat(mat) {
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

void DisjointSets::execute() {

  const MKL_INT base = mMat->mkl_base();
  auto ai = mMat->get_ai();
  auto aj = mMat->get_aj();
#pragma omp parallel for
  for (MKL_INT i = 0; i < mMat->rows(); i++) {
    for (MKL_INT j = ai[i] - base; j < ai[i + 1] - base; j++) {
      unite(i, aj[j] - base);
    }
  }
}

int CountComponents(std::vector<MKL_INT> &parents) {
  int sum = 0;
#pragma omp parallel for reduction(+ : sum)
  for (MKL_INT i = 0; i < parents.size(); i++) {
    if (Find(parents, i) == i)
      sum++;
  }
  return sum;
}

void ComponentsStat(std::vector<MKL_INT> &parents, const MKL_INT base,
                    std::vector<MKL_INT> &compRoots,
                    std::vector<MKL_INT> &sortedComp,
                    std::vector<MKL_INT> &compPrefSum) {
  sortedComp.resize(parents.size());
  std::vector<std::vector<MKL_INT>> rootsOfThread(omp_get_max_threads());
  std::unordered_map<MKL_INT, MKL_INT> rootToInd; // inverse map of compRoots
  std::vector<MKL_INT> compSizePrefixSum;
#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    auto [start, end] = utils::LoadBalancedPartition(
        parents.begin(), parents.end(), tid, nthreads);

    // find roots
    for (auto it = start; it != end; it++) {
      const MKL_INT index = it - parents.begin();
      if (Find(parents, index) == index)
        rootsOfThread[tid].push_back(index);
    }

#pragma omp barrier
#pragma omp master
    {
      // write all roots and inverse map
      int roots = 0;
      for (int i = 0; i < nthreads; i++) {
        roots += rootsOfThread[i].size();
      }
      compRoots.reserve(roots);
      compRoots.resize(0);
      for (int i = 0; i < nthreads; i++) {
        for (auto root : rootsOfThread[i]) {
          rootToInd[root] = compRoots.size();
          compRoots.push_back(root);
        }
      }

      // prefix sum of each component on each thread
      compSizePrefixSum =
          std::vector<MKL_INT>(compRoots.size() * (nthreads + 1), 0);
      compPrefSum = std::vector<MKL_INT>(compRoots.size() + 1, 0);
    }

    // size of each component on each thread
    // TODO: should optimize inner outer loop
#pragma omp barrier
    for (auto it = start; it != end; it++) {
      const MKL_INT index = it - parents.begin();
      compSizePrefixSum[(tid + 1) * compRoots.size() +
                        rootToInd[Find(parents, index)]]++;
    }

    // prefix sum
#pragma omp barrier
#pragma omp master
    {
      for (size_t i = 0; i < compRoots.size(); i++) {
        compSizePrefixSum[i] = compPrefSum[i];
        for (int j = 0; j < nthreads; j++) {
          compSizePrefixSum[(j + 1) * compRoots.size() + i] +=
              compSizePrefixSum[j * compRoots.size() + i];
        }
        compPrefSum[i + 1] = compSizePrefixSum[nthreads * compRoots.size() + i];
      }
    }

#pragma omp barrier
    for (auto it = start; it != end; it++) {
      const MKL_INT index = it - parents.begin();
      sortedComp[compSizePrefixSum[tid * compRoots.size() +
                                   rootToInd[Find(parents, index)]]++] =
          index + base;
    }
  }
}
} // namespace reordering