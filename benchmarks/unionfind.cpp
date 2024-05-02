#include "BFS.h"
#include "Reordering.h"
#include "mkl_sparse_mat.h"
#include "utils.h"
#include <benchmark/benchmark.h>
#include <functional>
#include <map>
#include <memory>
#include <omp.h>

void UnionFind(mkl_wrapper::mkl_sparse_mat const *const mat) {
  std::vector<MKL_INT> parents(mat->rows());
  std::vector<MKL_INT> ranks(mat->rows());
  parents.resize(mat->rows());
  ranks.resize(mat->rows());
  std::iota(parents.begin(), parents.end(), 0);
  std::fill(ranks.begin(), ranks.end(), 0);
  auto find = [&parents](MKL_INT x) {
    while (x != parents[x]) {
      parents[x] = parents[parents[x]];
      x = parents[x];
    }
    return x;
  };
  auto unite = [&parents, &ranks, find](MKL_INT x, MKL_INT y) {
    MKL_INT px = find(x);
    MKL_INT py = find(y);
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
}

void UnionFindUnOpt(mkl_wrapper::mkl_sparse_mat const *const mat) {
  std::vector<MKL_INT> parents(mat->rows());
  parents.resize(mat->rows());
  std::iota(parents.begin(), parents.end(), 0);
  std::function<MKL_INT &(MKL_INT)> find = [&parents,
                                            &find](MKL_INT x) -> MKL_INT & {
    if (x == parents[x])
      return parents[x];
    else
      return find(parents[x]);
  };
  auto unite = [&parents, find](MKL_INT x, MKL_INT y) {
    MKL_INT &px = find(x);
    MKL_INT &py = find(y);
    if (px == py)
      return;
    if (px < py) {
      px = py;
    } else {
      py = px;
    }
  };
  auto ai = mat->get_ai();
  auto aj = mat->get_aj();
  for (MKL_INT i = 0; i < mat->rows(); i++) {
    for (MKL_INT j = ai[i]; j < ai[i + 1]; j++) {
      unite(i, aj[j]);
    }
  }
}

static std::unique_ptr<std::map<MKL_INT, mkl_wrapper::mkl_sparse_mat>>
    graph_map_ptr;

class MyFixture : public benchmark::Fixture {

public:
  // add members as needed

  MyFixture() {
    if (graph_map_ptr == nullptr) {
      graph_map_ptr.reset(new std::map<MKL_INT, mkl_wrapper::mkl_sparse_mat>);
      graph_map_ptr->emplace(1000, mkl_wrapper::random_sparse(1000, 100));
      graph_map_ptr->emplace(10000, mkl_wrapper::random_sparse(10000, 100));
      graph_map_ptr->emplace(100000, mkl_wrapper::random_sparse(100000, 100));
      graph_map_ptr->emplace(1000000, mkl_wrapper::random_sparse(1000000, 100));
    }
  }
};

BENCHMARK_DEFINE_F(MyFixture, UnionGraph)(benchmark::State &state) {
  for (auto _ : state) {
    UnionFind(&graph_map_ptr->at(state.range(0)));
  }
}

BENCHMARK_REGISTER_F(MyFixture, UnionGraph)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000);

BENCHMARK_DEFINE_F(MyFixture, UnionGraphUnOpt)(benchmark::State &state) {
  for (auto _ : state) {
    UnionFindUnOpt(&graph_map_ptr->at(state.range(0)));
  }
}

BENCHMARK_REGISTER_F(MyFixture, UnionGraphUnOpt)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000);

BENCHMARK_MAIN();