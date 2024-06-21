#include <algorithm>
#include <benchmark/benchmark.h>
#include <iterator>
#include <memory>
#include <numeric>
#include <omp.h>
#include <random>
#include <stdlib.h>
#include <vector>

template <bool DO_PREFETCH>
int binarySearch(int *array, int number_of_elements, int key) {
  int low = 0, high = number_of_elements - 1, mid;

  while (low <= high) {
    mid = (low + high) / 2;

    if constexpr (DO_PREFETCH) {
      // low path
      __builtin_prefetch(&array[(mid + 1 + high) / 2], 0, 1);
      // high path
      __builtin_prefetch(&array[(low + mid - 1) / 2], 0, 1);
    }

    if (array[mid] < key)
      low = mid + 1;
    else if (array[mid] == key)
      return mid;
    else if (array[mid] > key)
      high = mid - 1;
  }

  return -1;
}

static const int SIZE = 1024 * 1024 * 512;
static const int NUM_LOOKUPS = 1024 * 1024 * 8;

std::vector<int> &GetArray() {
  static std::unique_ptr<std::vector<int>> ptr{nullptr};
  if (!ptr) {
    ptr.reset(new std::vector<int>(SIZE));
    std::iota(ptr->begin(), ptr->end(), 0);
  }
  return *ptr;
}

std::vector<int> &GetLookup() {
  static std::unique_ptr<std::vector<int>> ptr{nullptr};
  if (!ptr) {

    // First create an instance of an engine.
    std::random_device rnd_device;
    // Specify the engine and distribution.
    std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
    std::uniform_int_distribution<int> dist{0, SIZE - 1};

    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

    ptr.reset(new std::vector<int>(NUM_LOOKUPS));
    std::generate(std::begin(*ptr), std::end(*ptr), gen);
  }
  return *ptr;
}

static void BM_Prefetch(benchmark::State &state) {
  auto &vec = GetArray();
  auto &lookups = GetLookup();

  for (auto _ : state) {
    for (auto i : lookups) {
      benchmark::DoNotOptimize(
          binarySearch<true>(vec.data(), SIZE, lookups[i]));
    }
  }
}

BENCHMARK(BM_Prefetch);

static void BM_NonPrefetch(benchmark::State &state) {
  auto &vec = GetArray();
  auto &lookups = GetLookup();

  for (auto _ : state) {
    for (auto i : lookups) {
      benchmark::DoNotOptimize(
          binarySearch<false>(vec.data(), SIZE, lookups[i]));
    }
  }
}

BENCHMARK(BM_NonPrefetch);

BENCHMARK_MAIN();