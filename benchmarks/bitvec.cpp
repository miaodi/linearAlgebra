#include "BitVector.h"
#include <atomic>
#include <benchmark/benchmark.h>
#include <memory>
#include <vector>

static void BM_BitVec(benchmark::State &state) {
  for (auto _ : state) {
    utils::BitVector bv(state.range(0));
    for (size_t i = 0; i < state.range(0); i++) {
      if (rand() % 2)
        bv.set(i);
    }
    for (size_t i = 0; i < state.range(0); i++) {
      benchmark::DoNotOptimize(bv.get(i));
    }
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)) / 8);
}
// Register the function as a benchmark
BENCHMARK(BM_BitVec)->Range(8, 8 << 12);

static void BM_BoolVec(benchmark::State &state) {
  for (auto _ : state) {
    std::vector<bool> vec(state.range(0), false);
    for (size_t i = 0; i < state.range(0); i++) {
      if (rand() % 2)
        vec[i] = true;
    }
    for (size_t i = 0; i < state.range(0); i++) {
      benchmark::DoNotOptimize(vec[i]);
    }
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)) / 8);
}
// Register the function as a benchmark
BENCHMARK(BM_BoolVec)->Range(8, 8 << 12);

static void BM_AtomicBoolVec(benchmark::State &state) {
  for (auto _ : state) {
    std::vector<std::atomic<bool>> vec(state.range(0));
    for (size_t i = 0; i < state.range(0); i++) {
      if (rand() % 2)
        vec[i] = true;
    }
    for (size_t i = 0; i < state.range(0); i++) {
      benchmark::DoNotOptimize(vec[i]);
    }
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)) / 8);
}
// Register the function as a benchmark
BENCHMARK(BM_AtomicBoolVec)->Range(8, 8 << 12);

static void BM_IntVec(benchmark::State &state) {
  for (auto _ : state) {
    std::vector<int> vec(state.range(0), 0);
    for (size_t i = 0; i < state.range(0); i++) {
      if (rand() % 2)
        vec[i] = 1;
    }
    for (size_t i = 0; i < state.range(0); i++) {
      benchmark::DoNotOptimize(vec[i]);
    }
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)) / 8);
}
// Register the function as a benchmark
BENCHMARK(BM_IntVec)->Range(8, 8 << 12);

BENCHMARK_MAIN();