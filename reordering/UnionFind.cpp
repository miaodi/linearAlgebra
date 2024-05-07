#include "UnionFind.h"
#include "mkl_sparse_mat.h"
#include <iostream>
#include <numeric>
#include <omp.h>
#include <vector>

namespace reordering {

std::vector<MKL_INT>
UnionFindRank(mkl_wrapper::mkl_sparse_mat const *const mat) {
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
    for (MKL_INT j = ai[i]; j < ai[i + 1]; j++) {
      unite(i, aj[j]);
    }
  }
  return parents;
}

std::vector<MKL_INT>
UnionFindRem(mkl_wrapper::mkl_sparse_mat const *const mat) {
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
    for (MKL_INT j = ai[i]; j < ai[i + 1]; j++) {
      unite(i, aj[j]);
    }
  }
  return parents;
}

// Multi-core Spanning Forest Algorithms using the Disjoint-set Data Structure
std::vector<MKL_INT>
ParUnionFindRem(mkl_wrapper::mkl_sparse_mat const *const mat) {
  std::vector<MKL_INT> parents(mat->rows());
  std::iota(parents.begin(), parents.end(), 0);

  auto unite = [&parents](MKL_INT x, MKL_INT y) {
    while (true) {
      MKL_INT px = parents[x];
      MKL_INT py = parents[y];
      if (px < py) {
        if (x == px &&
            __sync_bool_compare_and_swap(&parents[x], px, parents[y]))
          break;
        if (__sync_bool_compare_and_swap(&parents[x], px, parents[y]))
          x = px;
      } else if (px > py) {
        if (y == py &&
            __sync_bool_compare_and_swap(&parents[y], py, parents[x]))
          break;
        if (__sync_bool_compare_and_swap(&parents[y], py, parents[x]))
          y = py;
      } else
        break;
    }
  };
  auto ai = mat->get_ai();
  auto aj = mat->get_aj();
#pragma omp parallel for
  for (MKL_INT i = 0; i < mat->rows(); i++) {
    for (MKL_INT j = ai[i]; j < ai[i + 1]; j++) {
      unite(i, aj[j]);
    }
  }
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

  auto ai = mMat->get_ai();
  auto aj = mMat->get_aj();
#pragma omp parallel for
  for (MKL_INT i = 0; i < mMat->rows(); i++) {
    for (MKL_INT j = ai[i]; j < ai[i + 1]; j++) {
      unite(i, aj[j]);
    }
  }
}

} // namespace reordering