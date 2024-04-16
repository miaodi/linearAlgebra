#include "BFS.h"
#include "mkl_sparse_mat.h"
#include "utils.h"
#include <benchmark/benchmark.h>
#include <memory>
#include <omp.h>

static std::unique_ptr<mkl_wrapper::mkl_sparse_mat> ptr{nullptr};
class MyFixture : public benchmark::Fixture {

public:
  // add members as needed

  MyFixture() {
    std::ifstream f("data/nv2.mtx");
    f.clear();
    f.seekg(0, std::ios::beg);
    std::vector<MKL_INT> csr_rows, csr_cols;
    std::vector<double> csr_vals;
    utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);

    ptr.reset(new mkl_wrapper::mkl_sparse_mat(csr_rows.size() - 1,
                                              csr_rows.size() - 1, csr_rows,
                                              csr_cols, csr_vals));
    std::cout << "rows: " << ptr->rows() << ", nnz: " << ptr->nnz()
              << std::endl;
  }
};

BENCHMARK_F(MyFixture, BM_PBFS_2)(benchmark::State &state) {
  MKL_INT level;
  omp_set_num_threads(2);
  for (auto _ : state) {
    auto levels = reordering::PBFS(ptr.get(), 0, level);
  }
}

BENCHMARK_F(MyFixture, BM_PBFS_4)(benchmark::State &state) {
  MKL_INT level;
  omp_set_num_threads(4);
  for (auto _ : state) {
    auto levels = reordering::PBFS(ptr.get(), 0, level);
  }
}

BENCHMARK_F(MyFixture, BM_PBFS_8)(benchmark::State &state) {
  MKL_INT level;
  omp_set_num_threads(8);
  for (auto _ : state) {
    auto levels = reordering::PBFS(ptr.get(), 0, level);
  }
}

BENCHMARK_F(MyFixture, BM_PBFS_16)(benchmark::State &state) {
  MKL_INT level;
  omp_set_num_threads(16);
  for (auto _ : state) {
    auto levels = reordering::PBFS(ptr.get(), 0, level);
  }
}

BENCHMARK_F(MyFixture, BM_BFS)(benchmark::State &state) {
  MKL_INT level;
  for (auto _ : state) {
    auto levels = reordering::BFS(ptr.get(), 0, level);
  }
}

BENCHMARK_MAIN();