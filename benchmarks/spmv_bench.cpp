#include "incomplete_lu.h"
#include "matrix_utils.hpp"
#include "mkl_sparse_mat.h"
#include "spmv.hpp"
#include <algorithm>
#include <benchmark/benchmark.h>
#include <cxxopts.hpp>
#include <memory>
#include <numeric>
#include <omp.h>
#include <string_view>
#include <vector>

using CSRTYPE = typename matrix_utils::CSRMatrixVec<int, int, double>;

auto Serial = [](benchmark::State &state, const CSRTYPE &mat, const int threads,
                 const int it) {
  std::vector<double> x(mat.rows, 0.0);
  std::vector<double> b(mat.rows, 1.0);

  matrix_utils::SPMV<CSRTYPE, matrix_utils::SerialSPMV> spmv;
  spmv.setMatrix(&mat);
  spmv.preprocess();
  for (auto _ : state) {
    for (int i = 0; i < it; i++) {
      spmv(b.data(), x.data());
    }
  }
  state.SetBytesProcessed(2 * 8 * int64_t(state.iterations()) * int64_t(it) *
                          int64_t(mat.NNZ()));
};

auto RowBalancedPar = [](benchmark::State &state, const CSRTYPE &mat,
                         const int threads, const int it) {
  omp_set_num_threads(threads);
  std::vector<double> x(mat.rows, 0.0);
  std::vector<double> b(mat.rows, 1.0);

  matrix_utils::SPMV<CSRTYPE, matrix_utils::ParallelSPMV> spmv;
  spmv.setMatrix(&mat);
  spmv.preprocess();
  for (auto _ : state) {
    for (int i = 0; i < it; i++) {
      spmv(b.data(), x.data());
    }
  }
  state.SetBytesProcessed(2 * 8 * int64_t(state.iterations()) * int64_t(it) *
                          int64_t(mat.NNZ()));
};

auto SegSum = [](benchmark::State &state, const CSRTYPE &mat, const int threads,
                 const int it) {
  std::vector<double> x(mat.rows, 0.0);
  std::vector<double> b(mat.rows, 1.0);

  matrix_utils::SPMV<CSRTYPE,
                     matrix_utils::SegSumSPMV<MKL_INT, MKL_INT, double>>
      spmv;
  spmv.setMatrix(&mat);
  spmv._spmv.setNumThreads(threads);
  spmv.preprocess();
  for (auto _ : state) {
    for (int i = 0; i < it; i++) {
      spmv(b.data(), x.data());
    }
  }
  state.SetBytesProcessed(2 * 8 * int64_t(state.iterations()) * int64_t(it) *
                          int64_t(mat.NNZ()));
};

auto ALBUSSum = [](benchmark::State &state, const CSRTYPE &mat,
                   const int threads, const int it) {
  std::vector<double> x(mat.rows, 0.0);
  std::vector<double> b(mat.rows, 1.0);

  matrix_utils::SPMV<CSRTYPE, matrix_utils::ALBUSSPMV<MKL_INT, MKL_INT, double>>
      spmv;
  spmv.setMatrix(&mat);
  spmv._spmv.setNumThreads(threads);
  spmv.preprocess();
  for (auto _ : state) {
    for (int i = 0; i < it; i++) {
      spmv(b.data(), x.data());
    }
  }
  state.SetBytesProcessed(2 * 8 * int64_t(state.iterations()) * int64_t(it) *
                          int64_t(mat.NNZ()));
};

int main(int argc, char **argv) {

  CSRTYPE mat;
  int num_threads = 1;
  int iterations = 1;
  cxxopts::Options options("SPMV benchmark",
                           "Benchmark different types of SPMV");
  options.allow_unrecognised_options().add_options()(
      "n,nt", "Number of threads", cxxopts::value<int>()->default_value("1"))(
      "i,it", "Number of iterations",
      cxxopts::value<int>()->default_value("100"))(
      "m,matrix", "Matrix location",
      cxxopts::value<std::string>()->default_value("data/thermal2.mtx"))(
      "h,help", "Print usage");

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    benchmark::Initialize(&argc, argv);
    benchmark::Shutdown();
    exit(0);
  }
  num_threads = result["n"].as<int>();
  iterations = result["i"].as<int>();
  std::string file = result["m"].as<std::string>();
  std::ifstream f(file);
  f.clear();
  f.seekg(0, std::ios::beg);
  utils::read_matrix_market_csr(f, mat.ai, mat.aj, mat.av);
  mat.rows = mat.ai.size() - 1;
  std::cout << "matrix size: " << mat.rows << "\n";

  benchmark::RegisterBenchmark("Serial", Serial, mat, num_threads, iterations);
  benchmark::RegisterBenchmark("RowBalancedPar", RowBalancedPar, mat,
                               num_threads, iterations);
  benchmark::RegisterBenchmark("SegSum", SegSum, mat, num_threads, iterations);
  benchmark::RegisterBenchmark("ALBUSSum", ALBUSSum, mat, num_threads,
                               iterations);
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}