#include "matrix_utils.hpp"
#include <benchmark/benchmark.h>
#include <cxxopts.hpp>

using CSRMatrixType = matrix_utils::CSRMatrix<int, int, double>;
int num_threads = 1;
auto KahnSerial = [](benchmark::State &state, const CSRMatrixType &mat) {
  std::vector<int> perm(mat.rows);
  std::vector<int> prefix(mat.rows + 1);
  matrix_utils::KahnSerial<int, int> kahn;
  for (auto _ : state) {
    kahn(matrix_utils::TriangularMatrix::L, mat.rows, mat.AI(), mat.AJ(),
         perm.data(), prefix.data());
  }
};

auto KahnParallel = [](benchmark::State &state, const CSRMatrixType &mat) {
  std::vector<int> perm(mat.rows);
  std::vector<int> prefix(mat.rows + 1);
  matrix_utils::KahnParallel<int, int> kahn(num_threads);
  for (auto _ : state) {
    kahn(matrix_utils::TriangularMatrix::L, mat.rows, mat.AI(), mat.AJ(),
         perm.data(), prefix.data());
  }
};

auto TopologicalSort2 = [](benchmark::State &state, const CSRMatrixType &mat) {
  std::vector<int> perm(mat.rows);
  std::vector<int> prefix(mat.rows + 1);
  matrix_utils::TopologicalSort2<int, int> topologicalSort;
  for (auto _ : state) {
    topologicalSort(matrix_utils::TriangularMatrix::L, mat.rows, mat.AI(),
                    mat.AJ(), perm.data(), prefix.data());
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

  benchmark::RegisterBenchmark("KahnSerial", KahnSerial, mat);
  benchmark::RegisterBenchmark("KahnParallel", KahnParallel, mat);
  benchmark::RegisterBenchmark("TopologicalSort2", TopologicalSort2, mat);
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}