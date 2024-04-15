#include "circularbuffer.hpp"
#include <benchmark/benchmark.h>
#include <iostream>
#include <memory>
#include <queue>
#include <vector>

static void BM_CircularBuffer_Push(benchmark::State &state) {
  for (auto _ : state) {
    utils::CircularBuffer<int> cb(state.range(0));
    for (int j = 0; j < state.range(1); ++j) {
      benchmark::DoNotOptimize(cb.push(j));
    }
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(1)) * 4);
}

BENCHMARK(BM_CircularBuffer_Push)
    ->Args({16, 1 << 12})
    ->Args({16, 1 << 14})
    ->Args({16, 1 << 16})
    ->Args({16, 1 << 18});

static void BM_Queue_Fix_Size_Push(benchmark::State &state) {
  for (auto _ : state) {
    std::queue<int> q;
    for (int j = 0; j < state.range(1); ++j) {
      q.push(j);
      if (q.size() > state.range(0))
        q.pop();
    }
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(1)) * 4);
}

BENCHMARK(BM_Queue_Fix_Size_Push)
    ->Args({16, 1 << 12})
    ->Args({16, 1 << 14})
    ->Args({16, 1 << 16})
    ->Args({16, 1 << 18});

static void BM_Vector_Push(benchmark::State &state) {
  for (auto _ : state) {
    std::vector<int> vec;
    for (int j = 0; j < state.range(1); ++j) {
      vec.push_back(j);
    }
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(1)) * 4);
}

BENCHMARK(BM_Vector_Push)
    ->Args({16, 1 << 12})
    ->Args({16, 1 << 14})
    ->Args({16, 1 << 16})
    ->Args({16, 1 << 18});

static void BM_Vector_Fix_Size_Push(benchmark::State &state) {
  for (auto _ : state) {
    std::vector<int> vec;
    for (int j = 0; j < state.range(1); ++j) {
      vec.push_back(j);
      if (vec.size() > state.range(0))
        vec.erase(vec.begin());
    }
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(1)) * 4);
}

BENCHMARK(BM_Vector_Fix_Size_Push)
    ->Args({16, 1 << 12})
    ->Args({16, 1 << 14})
    ->Args({16, 1 << 16})
    ->Args({16, 1 << 18});

BENCHMARK_MAIN();