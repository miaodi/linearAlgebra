#include "UnionFind.h"
#include "BFS.h"
#include "mkl_sparse_mat.h"
#include "utils.h"
#include <benchmark/benchmark.h>
#include <functional>
#include <map>
#include <memory>
#include <omp.h>

static std::unique_ptr<std::map<MKL_INT, mkl_wrapper::mkl_sparse_mat>>
    graph_map_ptr;

class MyFixture : public benchmark::Fixture {

public:
  // add members as needed

  MyFixture() {
    omp_set_num_threads(8);
    if (graph_map_ptr == nullptr) {
      graph_map_ptr.reset(new std::map<MKL_INT, mkl_wrapper::mkl_sparse_mat>);
      graph_map_ptr->emplace(1000, mkl_wrapper::random_sparse(1000, 10));
      graph_map_ptr->emplace(10000, mkl_wrapper::random_sparse(10000, 10));
      graph_map_ptr->emplace(100000, mkl_wrapper::random_sparse(100000, 10));
      // graph_map_ptr->emplace(1000000, mkl_wrapper::random_sparse(1000000, 10));
    }
  }
};

BENCHMARK_DEFINE_F(MyFixture, UnionFindRank)(benchmark::State &state) {
  for (auto _ : state) {
    reordering::UnionFindRank(&graph_map_ptr->at(state.range(0)));
  }
}

BENCHMARK_REGISTER_F(MyFixture, UnionFindRank)
    ->RangeMultiplier(10)
    ->Range(1000, 100000);

BENCHMARK_DEFINE_F(MyFixture, UnionFindRem)(benchmark::State &state) {
  for (auto _ : state) {
    reordering::UnionFindRem(&graph_map_ptr->at(state.range(0)));
  }
}

BENCHMARK_REGISTER_F(MyFixture, UnionFindRem)
    ->RangeMultiplier(10)
    ->Range(1000, 100000);

BENCHMARK_DEFINE_F(MyFixture, ParUnionFindRem)(benchmark::State &state) {
  for (auto _ : state) {
    reordering::ParUnionFindRem(&graph_map_ptr->at(state.range(0)));
  }
}

BENCHMARK_REGISTER_F(MyFixture, ParUnionFindRem)
    ->RangeMultiplier(10)
    ->Range(1000, 100000);

BENCHMARK_DEFINE_F(MyFixture, ParUnionFindRank)(benchmark::State &state) {
  for (auto _ : state) {
    reordering::DisjointSets ds(&graph_map_ptr->at(state.range(0)));
    ds.execute();
  }
}

BENCHMARK_REGISTER_F(MyFixture, ParUnionFindRank)
    ->RangeMultiplier(10)
    ->Range(1000, 100000);

BENCHMARK_MAIN();