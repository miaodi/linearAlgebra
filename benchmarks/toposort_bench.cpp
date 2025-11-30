#include "matrix_utils.hpp"
#include <benchmark/benchmark.h>
#include <cxxopts.hpp>
#include <random>

using CSRMatrixType = matrix_utils::CSRMatrix<int, int, double>;

// Create a matrix with better parallelization potential
void createRandomSparseL(CSRMatrixType &L, int rows, int base, int max_deps_per_row) {
  L.ResizeAI(rows + 1);
  L.rows = rows;
  L.cols = rows;
  auto ai = L.AI();
  ai[0] = base;
  
  std::vector<std::vector<int>> temp_rows(rows);
  std::random_device rd;
  std::mt19937 gen(42); // Fixed seed for reproducibility
  
  // For each row, randomly select dependencies from earlier rows
  for (int i = 0; i < rows; i++) {
    if (i == 0) continue;
    
    int num_deps = std::min(max_deps_per_row, i);
    if (num_deps > 0) {
      // Randomly select which earlier rows to depend on
      std::uniform_int_distribution<int> dep_dist(0, i-1);
      std::set<int> dependencies;
      
      // Add some random dependencies
      for (int d = 0; d < num_deps/2; d++) {
        dependencies.insert(dep_dist(gen));
      }
      
      for (int dep : dependencies) {
        temp_rows[i].push_back(dep + base);
      }
    }
    
    ai[i + 1] = ai[i] + temp_rows[i].size();
  }
  
  int total_nnz = ai[rows] - base;
  L.ResizeAJ(total_nnz);
  L.ResizeAV(total_nnz);
  auto aj = L.AJ();
  auto av = L.AV();
  
  int pos = 0;
  for (int i = 0; i < rows; i++) {
    for (int dep : temp_rows[i]) {
      aj[pos] = dep;
      av[pos] = 1.0;
      pos++;
    }
  }
}
int num_threads = 1;
auto KahnSerialBench = [](benchmark::State &state, const CSRMatrixType &mat) {
  std::vector<int> perm(mat.rows);
  std::vector<int> prefix(mat.rows + 1);
  graph::KahnSerial<int, int> kahn;
  for (auto _ : state) {
    kahn.operator()(
        mat.rows, mat.AI(), mat.AJ(), perm.data(), prefix.data(), false);
  }
};

auto KahnParallelBench = [](benchmark::State &state, const CSRMatrixType &mat) {
  std::vector<int> perm(mat.rows);
  std::vector<int> prefix(mat.rows + 1);
  graph::KahnParallel<int, int> kahn(num_threads);
  for (auto _ : state) {
    kahn.operator()(
        mat.rows, mat.AI(), mat.AJ(), perm.data(), prefix.data(), false);
  }
};

auto TopologicalSort2Bench = [](benchmark::State &state, const CSRMatrixType &mat) {
  std::vector<int> perm(mat.rows);
  std::vector<int> prefix(mat.rows + 1);
  graph::TopologicalSort2<int, int, matrix_utils::TriangularMatrix::L> topologicalSort;
  for (auto _ : state) {
    topologicalSort.operator()(
        mat.rows, mat.AI(), mat.AJ(), perm.data(), prefix.data(), false);
  }
};

int main(int argc, char **argv) {
  cxxopts::Options options("TopologicalSort benchmark",
                           "Benchmark different types of TopologicalSort");
  options.allow_unrecognised_options().add_options()(
      "n,nt", "Number of threads", cxxopts::value<int>()->default_value("1"))(
      "s,size", "Matrix size", cxxopts::value<int>()->default_value("100"))(
      "r,rnnz", "row nnz",
      cxxopts::value<int>()->default_value("10"))("h,help", "Print usage");

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    benchmark::Initialize(&argc, argv);
    benchmark::Shutdown();
    exit(0);
  }
  num_threads = result["n"].as<int>();
  int size = result["s"].as<int>();
  int rnnz = result["r"].as<int>();
  CSRMatrixType mat;
  matrix_utils::RandomL<CSRMatrixType>(size, 0, rnnz, mat);

  // Create a sparser matrix with better parallel potential
  CSRMatrixType sparse_mat;
  createRandomSparseL(sparse_mat, size, 0, std::min(rnnz/4, 10));

  benchmark::RegisterBenchmark("KahnSerial_Dense", KahnSerialBench, mat);
  benchmark::RegisterBenchmark("KahnParallel_Dense", KahnParallelBench, mat);
  benchmark::RegisterBenchmark("TopologicalSort2_Dense", TopologicalSort2Bench, mat);
  
  // Add a sparse matrix with better parallelization potential  
  benchmark::RegisterBenchmark("KahnSerial_Sparse", KahnSerialBench, sparse_mat);
  benchmark::RegisterBenchmark("KahnParallel_Sparse", KahnParallelBench, sparse_mat);
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
