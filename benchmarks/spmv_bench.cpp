#include "incomplete_lu.h"
#include "matrix_utils.hpp"
#include "spmv.hpp"
#include <algorithm>
#include <benchmark/benchmark.h>
#include <cxxopts.hpp>
#include <memory>
#include <numeric>
#include <omp.h>
#include <string_view>
#include <vector>

using CSRTYPE_DOUBLE = typename matrix_utils::CSRMatrixVec<int, int, double>;
using CSRTYPE_FLOAT = typename matrix_utils::CSRMatrixVec<int, int, float>;

template <typename VALTYPE>
auto Serial = [](benchmark::State &state, const auto &mat, const int threads,
                 const int it) {
  std::vector<VALTYPE> x(mat.rows, 0.0);
  std::vector<VALTYPE> b(mat.rows, 1.0);

  matrix_utils::SPMV<std::remove_cvref_t<decltype(mat)>,
                     matrix_utils::SerialSPMV>
      spmv;
  spmv.setMatrix(&mat);
  spmv.preprocess();
  for (auto _ : state) {
    for (int i = 0; i < it; i++) {
      spmv(b.data(), x.data());
    }
  }
  state.SetBytesProcessed(2 * sizeof(VALTYPE) * int64_t(state.iterations()) *
                          int64_t(it) * int64_t(mat.NNZ()));
};

template <typename VALTYPE>
auto Parallel = [](benchmark::State &state, const auto &mat,
                   const int threads, const int it) {
  std::vector<VALTYPE> x(mat.rows, 0.0);
  std::vector<VALTYPE> b(mat.rows, 1.0);

  matrix_utils::SPMV<
      std::remove_cvref_t<decltype(mat)>,
      matrix_utils::ParallelSPMV<int, int, VALTYPE>>
      spmv;
  spmv.setMatrix(&mat);
  spmv._spmv.setNumThreads(threads);
  spmv.preprocess();
  for (auto _ : state) {
    for (int i = 0; i < it; i++) {
      spmv(b.data(), x.data());
    }
  }
  state.SetBytesProcessed(2 * sizeof(VALTYPE) * int64_t(state.iterations()) *
                          int64_t(it) * int64_t(mat.NNZ()));
};

template <typename VALTYPE>
auto RowBalanced = [](benchmark::State &state, const auto &mat,
                      const int threads, const int it) {
  std::vector<VALTYPE> x(mat.rows, 0.0);
  std::vector<VALTYPE> b(mat.rows, 1.0);

  matrix_utils::SPMV<
      std::remove_cvref_t<decltype(mat)>,
      matrix_utils::RowBalancedParallelSPMV<int, int, VALTYPE>>
      spmv;
  spmv.setMatrix(&mat);
  spmv._spmv.setNumThreads(threads);
  spmv.preprocess();
  for (auto _ : state) {
    for (int i = 0; i < it; i++) {
      spmv(b.data(), x.data());
    }
  }
  state.SetBytesProcessed(2 * sizeof(VALTYPE) * int64_t(state.iterations()) *
                          int64_t(it) * int64_t(mat.NNZ()));
};

template <typename VALTYPE>
auto RowBalancedSimd = [](benchmark::State &state, const auto &mat,
                          const int threads, const int it) {
  std::vector<VALTYPE> x(mat.rows, 0.0);
  std::vector<VALTYPE> b(mat.rows, 1.0);

  matrix_utils::SPMV<
      std::remove_cvref_t<decltype(mat)>,
      matrix_utils::RowBalancedParallelSPMV<int, int, VALTYPE,
                                            matrix_utils::RowDotKernel::Simd>>
      spmv;
  spmv.setMatrix(&mat);
  spmv._spmv.setNumThreads(threads);
  spmv.preprocess();
  for (auto _ : state) {
    for (int i = 0; i < it; i++) {
      spmv(b.data(), x.data());
    }
  }
  state.SetBytesProcessed(2 * sizeof(VALTYPE) * int64_t(state.iterations()) *
                          int64_t(it) * int64_t(mat.NNZ()));
};

template <typename VALTYPE>
auto ALBUSSum = [](benchmark::State &state, const auto &mat,
                   const int threads, const int it) {
  std::vector<VALTYPE> x(mat.rows, 0.0);
  std::vector<VALTYPE> b(mat.rows, 1.0);

  matrix_utils::SPMV<std::remove_cvref_t<decltype(mat)>,
                     matrix_utils::ALBUSSPMV<int32_t, int32_t, VALTYPE>>
      spmv;
  spmv.setMatrix(&mat);
  spmv._spmv.setNumThreads(threads);
  spmv.preprocess();
  for (auto _ : state) {
    for (int i = 0; i < it; i++) {
      spmv(b.data(), x.data());
    }
  }
  state.SetBytesProcessed(2 * sizeof(VALTYPE) * int64_t(state.iterations()) *
                          int64_t(it) * int64_t(mat.NNZ()));
};

template <typename VALTYPE>
auto ALBUSSimd = [](benchmark::State &state, const auto &mat,
                    const int threads, const int it) {
  std::vector<VALTYPE> x(mat.rows, 0.0);
  std::vector<VALTYPE> b(mat.rows, 1.0);

  matrix_utils::SPMV<
      std::remove_cvref_t<decltype(mat)>,
      matrix_utils::ALBUSSPMV<int32_t, int32_t, VALTYPE,
                              matrix_utils::RowDotKernel::Simd>>
      spmv;
  spmv.setMatrix(&mat);
  spmv._spmv.setNumThreads(threads);
  spmv.preprocess();
  for (auto _ : state) {
    for (int i = 0; i < it; i++) {
      spmv(b.data(), x.data());
    }
  }
  state.SetBytesProcessed(2 * sizeof(VALTYPE) * int64_t(state.iterations()) *
                          int64_t(it) * int64_t(mat.NNZ()));
};

int main(int argc, char **argv) {

  CSRTYPE_DOUBLE mat_double;
  CSRTYPE_FLOAT mat_float;
  int num_threads = 1;
  int iterations = 1;
  cxxopts::Options options("SPMV benchmark",
                           "Benchmark different types of SPMV");
  options.allow_unrecognised_options().add_options()(
      "n,nt", "Number of threads", cxxopts::value<int>()->default_value("1"))(
      "i,it", "Number of iterations",
      cxxopts::value<int>()->default_value("100"))(
      "f,file", "Matrix location",
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
  std::string file = result["f"].as<std::string>();
  
  // Read matrix as double
  {
    std::ifstream f(file);
    f.clear();
    f.seekg(0, std::ios::beg);
    utils::read_matrix_market_csr(f, mat_double.ai, mat_double.aj, mat_double.av);
    mat_double.rows = mat_double.ai.size() - 1;
  }
  
  // Convert to float
  mat_float.rows = mat_double.rows;
  mat_float.ai = mat_double.ai;
  mat_float.aj = mat_double.aj;
  mat_float.av.resize(mat_double.av.size());
  std::transform(mat_double.av.begin(), mat_double.av.end(), mat_float.av.begin(),
                 [](double d) { return static_cast<float>(d); });
  
  std::cout << "matrix size: " << mat_double.rows << "\n";

  // Double precision benchmarks
  benchmark::RegisterBenchmark("Serial_double", Serial<double>, mat_double, num_threads, iterations);
  benchmark::RegisterBenchmark("Parallel_double", Parallel<double>, mat_double,
                               num_threads, iterations);
  benchmark::RegisterBenchmark("RowBalanced_double", RowBalanced<double>, mat_double,
                               num_threads, iterations);
  benchmark::RegisterBenchmark("RowBalancedSimd_double", RowBalancedSimd<double>, mat_double,
                               num_threads, iterations);
  benchmark::RegisterBenchmark("ALBUSSum_double", ALBUSSum<double>, mat_double, num_threads,
                               iterations);
  benchmark::RegisterBenchmark("ALBUSSimd_double", ALBUSSimd<double>, mat_double, num_threads,
                               iterations);
  
  // Float precision benchmarks
  benchmark::RegisterBenchmark("Serial_float", Serial<float>, mat_float, num_threads, iterations);
  benchmark::RegisterBenchmark("Parallel_float", Parallel<float>, mat_float,
                               num_threads, iterations);
  benchmark::RegisterBenchmark("RowBalanced_float", RowBalanced<float>, mat_float,
                               num_threads, iterations);
  benchmark::RegisterBenchmark("RowBalancedSimd_float", RowBalancedSimd<float>, mat_float,
                               num_threads, iterations);
  benchmark::RegisterBenchmark("ALBUSSum_float", ALBUSSum<float>, mat_float, num_threads,
                               iterations);
  benchmark::RegisterBenchmark("ALBUSSimd_float", ALBUSSimd<float>, mat_float, num_threads,
                               iterations);
  
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
