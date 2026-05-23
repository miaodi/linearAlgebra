#include "BFS.h"
#include "io.hpp"
#include "Reordering.h"
#include "utils.h"
#include "bfs.hpp"
#include <benchmark/benchmark.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <omp.h>
#include <stdexcept>
#include <vector>

constexpr int MAX_THREADS = 8;

struct BFSCsrData {
  std::vector<int> ai;
  std::vector<int> aj;
  std::vector<double> av;
};

static std::unique_ptr<BFSCsrData> ptr{nullptr};
static std::once_flag bfs_load_flag;

static void load_bfs_matrix_once() {
  if (ptr) return; // already loaded
  std::ifstream f("data/nv2.mtx");
  if (!f) {
    throw std::runtime_error("Failed to open data/nv2.mtx for BFS benchmark");
  }
  ptr = std::make_unique<BFSCsrData>();
  matrix_utils::readMatrixMarket(f, ptr->ai, ptr->aj, ptr->av);
}
class MyFixture : public benchmark::Fixture {

public:
  // add members as needed

  MyFixture() { std::call_once(bfs_load_flag, load_bfs_matrix_once); }
};

BENCHMARK_F(MyFixture, BM_BFS)(benchmark::State &state) {
  reordering::BFS<int, int, double> bfs(reordering::BFS_Fn<false>);
  for (auto _ : state) {
    bfs(static_cast<int>(ptr->ai.size() - 1), ptr->ai.data(), ptr->aj.data(),
        ptr->av.data(), 0);
  }
}

BENCHMARK_DEFINE_F(MyFixture, BM_PBFS)(benchmark::State &state) {
  omp_set_num_threads(state.range(0));
  reordering::BFS<int, int, double> bfs(reordering::PBFS_Fn<false, true>);
  for (auto _ : state) {
    bfs(static_cast<int>(ptr->ai.size() - 1), ptr->ai.data(), ptr->aj.data(),
        ptr->av.data(), 0);
  }
}
BENCHMARK_REGISTER_F(MyFixture, BM_PBFS)->RangeMultiplier(2)->Range(1, MAX_THREADS);

BENCHMARK_DEFINE_F(MyFixture, BM_PBFS_NOLEVELS)(benchmark::State &state) {
  omp_set_num_threads(state.range(0));
  reordering::BFS<int, int, double> bfs(reordering::PBFS_Fn<false, false>);
  for (auto _ : state) {
    bfs(static_cast<int>(ptr->ai.size() - 1), ptr->ai.data(), ptr->aj.data(),
        ptr->av.data(), 0);
  }
}

BENCHMARK_REGISTER_F(MyFixture, BM_PBFS_NOLEVELS)
    ->RangeMultiplier(2)
    ->Range(1, MAX_THREADS);

// Direct benchmark using graph::BFSFunc (serial)
BENCHMARK_F(MyFixture, BM_BFS_DIRECT)(benchmark::State &state) {
  using ROW = int;
  using COL = int;
  for (auto _ : state) {
    COL rows = static_cast<COL>(ptr->ai.size() - 1);
    ROW const* ai = ptr->ai.data();
    COL const* aj = ptr->aj.data();
    COL source = 0;
    COL shortCutWidth = std::numeric_limits<COL>::max();
    COL height = 0;
    COL width = 0;
    std::vector<COL> levels;
    std::vector<COL> lastLevel;
    graph::BFSFunc<ROW, COL, false, false>(rows, ai, aj, source, shortCutWidth, height, width, levels, lastLevel, 1);
    benchmark::DoNotOptimize(height);
    benchmark::DoNotOptimize(width);
  }
}

// Direct benchmark using graph::PBFSFunc with TRACK=true
BENCHMARK_DEFINE_F(MyFixture, BM_PBFS_DIRECT_TRACK_ON)(benchmark::State &state) {
  using ROW = int;
  using COL = int;
  omp_set_num_threads(state.range(0));
  for (auto _ : state) {
    COL rows = static_cast<COL>(ptr->ai.size() - 1);
    ROW const* ai = ptr->ai.data();
    COL const* aj = ptr->aj.data();
    COL source = 0;
    COL shortCutWidth = std::numeric_limits<COL>::max();
    COL height = 0;
    COL width = 0;
    std::vector<COL> levels;
    std::vector<COL> lastLevel;
    graph::BFSFunc<ROW, COL, false, true>(rows, ai, aj, source, shortCutWidth, height, width, levels, lastLevel, state.range(0));
    benchmark::DoNotOptimize(height);
    benchmark::DoNotOptimize(width);
  }
}
BENCHMARK_REGISTER_F(MyFixture, BM_PBFS_DIRECT_TRACK_ON)->RangeMultiplier(2)->Range(1, MAX_THREADS);

// Direct benchmark using graph::PBFSFunc with TRACK=false
BENCHMARK_DEFINE_F(MyFixture, BM_PBFS_DIRECT_TRACK_OFF)(benchmark::State &state) {
  using ROW = int;
  using COL = int;
  omp_set_num_threads(state.range(0));
  for (auto _ : state) {
    COL rows = static_cast<COL>(ptr->ai.size() - 1);
    ROW const* ai = ptr->ai.data();
    COL const* aj = ptr->aj.data();
    COL source = 0;
    COL shortCutWidth = std::numeric_limits<COL>::max();
    COL height = 0;
    COL width = 0;
    std::vector<COL> levels;
    std::vector<COL> lastLevel;
    graph::BFSFunc<ROW, COL, false, false>(rows, ai, aj, source, shortCutWidth, height, width, levels, lastLevel, state.range(0));
    benchmark::DoNotOptimize(height);
    benchmark::DoNotOptimize(width);
  }
}
BENCHMARK_REGISTER_F(MyFixture, BM_PBFS_DIRECT_TRACK_OFF)->RangeMultiplier(2)->Range(1, MAX_THREADS);

// BENCHMARK_F(MyFixture, BM_NodeDegree)(benchmark::State &state) {
//   for (auto _ : state) {
//     std::vector<int> degrees;
//     reordering::NodeDegree(ptr.get(), degrees);
//   }
// }

// BENCHMARK_REGISTER_F(MyFixture, BM_NodeDegree)
//     ->RangeMultiplier(2)
//     ->Range(1, MAX_THREADS);

// BENCHMARK_DEFINE_F(MyFixture, BM_PNodeDegree)(benchmark::State &state) {
//   omp_set_num_threads(state.range(0));
//   for (auto _ : state) {
//     std::vector<int> degrees;
//     reordering::PNodeDegree(ptr.get(), degrees);
//   }
// }

// BENCHMARK_REGISTER_F(MyFixture, BM_PNodeDegree)
//     ->RangeMultiplier(2)
//     ->Range(1, MAX_THREADS);

int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);
  std::call_once(bfs_load_flag, load_bfs_matrix_once);
  if (ptr) {
    std::cout << "rows: " << ptr->ai.size() - 1 << ", nnz: " << ptr->aj.size()
              << std::endl;
  }
  ::benchmark::RunSpecifiedBenchmarks();
  return 0;
}
