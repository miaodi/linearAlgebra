#include "BFS.h"
#include "mkl_sparse_mat.h"
#include <benchmark/benchmark.h>
#include <memory>

static void BM_BFS(benchmark::State &state) {
  std::shared_ptr<MKL_INT[]> aiA(
      new MKL_INT[10]{0, 3, 5, 8, 12, 16, 20, 24, 27, 28});
  std::shared_ptr<MKL_INT[]> ajA(new MKL_INT[28]{1, 2, 3, 0, 2, 0, 1, 3, 0, 2,
                                                 4, 5, 3, 5, 6, 7, 3, 4, 6, 7,
                                                 4, 5, 7, 8, 4, 5, 6, 6});
  std::shared_ptr<double[]> avA(new double[28]);

  mkl_wrapper::mkl_sparse_mat A(9, 9, aiA, ajA, avA);
  MKL_INT level;
  for (auto _ : state) {
    auto levels = reordering::BFS(&A, 0, level);
  }
}
// Register the function as a benchmark
BENCHMARK(BM_BFS);

static void BM_PBFS(benchmark::State &state) {
  std::shared_ptr<MKL_INT[]> aiA(
      new MKL_INT[10]{0, 3, 5, 8, 12, 16, 20, 24, 27, 28});
  std::shared_ptr<MKL_INT[]> ajA(new MKL_INT[28]{1, 2, 3, 0, 2, 0, 1, 3, 0, 2,
                                                 4, 5, 3, 5, 6, 7, 3, 4, 6, 7,
                                                 4, 5, 7, 8, 4, 5, 6, 6});
  std::shared_ptr<double[]> avA(new double[28]);

  mkl_wrapper::mkl_sparse_mat A(9, 9, aiA, ajA, avA);
  MKL_INT level;
  for (auto _ : state) {
    auto levels = reordering::PBFS(&A, 0, level);
  }
}
// Register the function as a benchmark
BENCHMARK(BM_PBFS);

BENCHMARK_MAIN();