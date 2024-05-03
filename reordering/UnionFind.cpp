#include "UnionFind.h"
#include "mkl_sparse_mat.h"
#include <numeric>
#include <omp.h>
#include <vector>

namespace reordering {

MKL_INT Find(std::vector<MKL_INT> &parents, MKL_INT x) {
  while (x != parents[x]) {
    parents[x] = parents[parents[x]];
    x = parents[x];
  }
  return x;
};

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
    while (parents[x] != parents[y]) {
      if (parents[x] < parents[y]) {
        if (x == parents[x]) {
          if (__sync_bool_compare_and_swap(&parents[x], x, parents[y]))
            break;
        }
        MKL_INT tmp = parents[x];
        parents[x] = parents[y];
        x = tmp;
      } else {
        if (y == parents[y]) {
          if (__sync_bool_compare_and_swap(&parents[y], y, parents[x]))
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
#pragma omp parallel for
  for (MKL_INT i = 0; i < mat->rows(); i++) {
    for (MKL_INT j = ai[i]; j < ai[i + 1]; j++) {
      unite(i, aj[j]);
    }
  }
  return parents;
}
} // namespace reordering