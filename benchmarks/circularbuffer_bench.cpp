#include "circularbuffer.hpp"
#include <benchmark/benchmark.h>
#include <cstdint>
#include <iostream>
#include <memory>
#include <queue>
#include <vector>

static void BM_CircularBuffer_Push(benchmark::State &state) {
  for (auto _ : state) {
    utils::CircularBuffer<int> cb(state.range(0));
    for (int j = 0; j < state.range(1); j++) {
      cb.push_back(j);
      benchmark::DoNotOptimize(cb.size());
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

static void BM_CircularBuffer_Push_Preallocated(benchmark::State &state) {
  for (auto _ : state) {
    utils::CircularBuffer<int> cb(state.range(1)); // Pre-allocate full size
    for (int j = 0; j < state.range(1); j++) {
      cb.push_back(j);
      benchmark::DoNotOptimize(cb.size());
    }
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(1)) * 4);
}

BENCHMARK(BM_CircularBuffer_Push_Preallocated)
    ->Args({16, 1 << 12})
    ->Args({16, 1 << 14})
    ->Args({16, 1 << 16})
    ->Args({16, 1 << 18});

static void BM_Vector_Push_Reserved(benchmark::State &state) {
  for (auto _ : state) {
    std::vector<int> vec;
    vec.reserve(state.range(1)); // Pre-allocate like CircularBuffer
    for (int j = 0; j < state.range(1); ++j) {
      vec.push_back(j);
    }
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(1)) * 4);
}

BENCHMARK(BM_Vector_Push_Reserved)
    ->Args({16, 1 << 12})
    ->Args({16, 1 << 14})
    ->Args({16, 1 << 16})
    ->Args({16, 1 << 18});

static void BM_CircularBuffer_Push_Overwrite(benchmark::State &state) {
  for (auto _ : state) {
    utils::CircularBuffer<int> cb(state.range(0));
    for (int j = 0; j < state.range(1); ++j) {
      cb.push_back_overwrite(j);
      benchmark::DoNotOptimize(cb.size());
    }
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(1)) * 4);
}

BENCHMARK(BM_CircularBuffer_Push_Overwrite)
    ->Args({16, 1 << 12})
    ->Args({16, 1 << 14})
    ->Args({16, 1 << 16})
    ->Args({16, 1 << 18});

// Cache-level probe using the push_back_overwrite interface. Args: {capacity}.
// Each run performs the same total number of pushes so work is comparable
// across capacities; data is reused because the buffer is smaller than the
// total push count.
static void BM_CircularBuffer_CacheLevels(benchmark::State &state) {
  const int capacity = static_cast<int>(state.range(0));
  static constexpr int64_t kTotalPushes = 1 << 24; // 16M pushes per iteration

  for (auto _ : state) {
    utils::CircularBuffer<int> cb(capacity);

    std::uint64_t acc = 0;
    for (int64_t push = 0; push < kTotalPushes; ++push) {
      cb.push_back_overwrite(static_cast<int>(push));
      acc += static_cast<std::uint64_t>(cb.last()); // read the slot we just wrote
    }
    benchmark::DoNotOptimize(acc);
    benchmark::ClobberMemory();
  }

  const int64_t touches = static_cast<int64_t>(state.iterations()) * kTotalPushes;
  state.SetBytesProcessed(touches * sizeof(int) * 2); // write + read per push
}

// Capacities target typical cache sizes: L1 (~32KB), L2 (~256KB), L3 (~2MB),
// and memory (multi-MB). Adjust to match your CPU if needed.
BENCHMARK(BM_CircularBuffer_CacheLevels)
    ->Arg(64)        // ~256 B working set (single cache line)
    ->Arg(4 << 10)   // ~16 KB working set (smaller than L1)
    ->Arg(8 << 10)   // ~32 KB working set (L1)
    ->Arg(64 << 10)  // ~256 KB working set (L2)
    ->Arg(512 << 10) // ~2 MB working set (L3)
    ->Arg(2 << 20)   // ~8 MB working set (beyond most L3 slices)
    ->Arg(8 << 20);  // ~32 MB working set (memory)

static void BM_Queue_Fix_Size_Push(benchmark::State &state) {
  for (auto _ : state) {
    std::queue<int> q;
    for (int j = 0; j < state.range(1); ++j) {
      q.push(j);
      if (q.size() > static_cast<size_t>(state.range(0)))
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
      if (vec.size() > static_cast<size_t>(state.range(0)))
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
