#include "BFS.h"
#include "Reordering.h"
#include "mkl_sparse_mat.h"
#include "utils.h"
#include <benchmark/benchmark.h>
#include <memory>
#include <mutex>
#include <omp.h>

static std::unique_ptr<mkl_wrapper::mkl_sparse_mat> ptr{nullptr};
static std::once_flag bfs_load_flag;

static void load_bfs_matrix_once() {
  if (ptr) return; // already loaded
  std::ifstream f("data/nv2.mtx");
  if (!f) {
    throw std::runtime_error("Failed to open data/nv2.mtx for BFS benchmark");
  }
  std::vector<MKL_INT> csr_rows, csr_cols;
  std::vector<double> csr_vals;
  utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
  ptr.reset(new mkl_wrapper::mkl_sparse_mat(csr_rows.size() - 1,
                                            csr_rows.size() - 1, csr_rows,
                                            csr_cols, csr_vals));
}
class MyFixture : public benchmark::Fixture {

public:
  // add members as needed

  MyFixture() { std::call_once(bfs_load_flag, load_bfs_matrix_once); }
};

BENCHMARK_F(MyFixture, BM_BFS)(benchmark::State &state) {
  MKL_INT level;
  reordering::BFS bfs(reordering::BFS_Fn<false>);
  for (auto _ : state) {
    bfs(ptr.get(), 0);
  }
}

BENCHMARK_DEFINE_F(MyFixture, BM_PBFS)(benchmark::State &state) {
  omp_set_num_threads(state.range(0));
  reordering::BFS bfs(reordering::PBFS_Fn<false, true>);
  for (auto _ : state) {
    bfs(ptr.get(), 0);
  }
}
BENCHMARK_REGISTER_F(MyFixture, BM_PBFS)->RangeMultiplier(2)->Range(1, 1 << 5);

BENCHMARK_DEFINE_F(MyFixture, BM_PBFS_NOLEVELS)(benchmark::State &state) {
  omp_set_num_threads(state.range(0));
  reordering::BFS bfs(reordering::PBFS_Fn<false, false>);
  for (auto _ : state) {
    bfs(ptr.get(), 0);
  }
}

BENCHMARK_REGISTER_F(MyFixture, BM_PBFS_NOLEVELS)
    ->RangeMultiplier(2)
    ->Range(1, 1 << 5);

BENCHMARK_F(MyFixture, BM_NodeDegree)(benchmark::State &state) {
  for (auto _ : state) {
    std::vector<MKL_INT> degrees;
    reordering::NodeDegree(ptr.get(), degrees);
  }
}

BENCHMARK_DEFINE_F(MyFixture, BM_PNodeDegree)(benchmark::State &state) {
  omp_set_num_threads(state.range(0));
  for (auto _ : state) {
    std::vector<MKL_INT> degrees;
    reordering::PNodeDegree(ptr.get(), degrees);
  }
}

BENCHMARK_REGISTER_F(MyFixture, BM_PNodeDegree)
    ->RangeMultiplier(2)
    ->Range(1, 1 << 5);

int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);
  std::call_once(bfs_load_flag, load_bfs_matrix_once);
  if (ptr) {
    std::cout << "rows: " << ptr->rows() << ", nnz: " << ptr->nnz() << std::endl;
  }
  ::benchmark::RunSpecifiedBenchmarks();
  return 0;
}