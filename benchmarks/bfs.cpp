#include "BFS.h"
#include "mkl_sparse_mat.h"
#include "utils.h"
#include <benchmark/benchmark.h>
#include <memory>

// static void BM_BFS(benchmark::State &state) {
//   std::shared_ptr<MKL_INT[]> aiA(
//       new MKL_INT[10]{0, 3, 5, 8, 12, 16, 20, 24, 27, 28});
//   std::shared_ptr<MKL_INT[]> ajA(new MKL_INT[28]{1, 2, 3, 0, 2, 0, 1, 3, 0,
//   2,
//                                                  4, 5, 3, 5, 6, 7, 3, 4, 6,
//                                                  7, 4, 5, 7, 8, 4, 5, 6, 6});
//   std::shared_ptr<double[]> avA(new double[28]);

//   mkl_wrapper::mkl_sparse_mat A(9, 9, aiA, ajA, avA);
//   MKL_INT level;
//   for (auto _ : state) {
//     auto levels = reordering::BFS(&A, 0, level);
//   }
// }
// // Register the function as a benchmark
// BENCHMARK(BM_BFS);

// static void BM_PBFS(benchmark::State &state) {
//   std::shared_ptr<MKL_INT[]> aiA(
//       new MKL_INT[10]{0, 3, 5, 8, 12, 16, 20, 24, 27, 28});
//   std::shared_ptr<MKL_INT[]> ajA(new MKL_INT[28]{1, 2, 3, 0, 2, 0, 1, 3, 0,
//   2,
//                                                  4, 5, 3, 5, 6, 7, 3, 4, 6,
//                                                  7, 4, 5, 7, 8, 4, 5, 6, 6});
//   std::shared_ptr<double[]> avA(new double[28]);

//   mkl_wrapper::mkl_sparse_mat A(9, 9, aiA, ajA, avA);
//   MKL_INT level;
//   for (auto _ : state) {
//     auto levels = reordering::PBFS(&A, 0, level);
//   }
// }
// // Register the function as a benchmark
// BENCHMARK(BM_PBFS);

static std::unique_ptr<mkl_wrapper::mkl_sparse_mat> ptr;
class MyFixture : public benchmark::Fixture {

public:
  // add members as needed

  MyFixture() {
    std::ifstream f("../../data/linear_system/parabolic_fem.mtx");
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

BENCHMARK_F(MyFixture, BM_PBFS)(benchmark::State &state) {
  MKL_INT level;
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