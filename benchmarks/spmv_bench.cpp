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

#ifdef USE_CUDA
#include "cuda_spmv.h"
#include <cuda_runtime.h>
#endif

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

template <typename VALTYPE>
auto CAMLBSum = [](benchmark::State &state, const auto &mat,
                   const int threads, const int it) {
  std::vector<VALTYPE> x(mat.rows, 0.0);
  std::vector<VALTYPE> b(mat.rows, 1.0);

  matrix_utils::SPMV<
      std::remove_cvref_t<decltype(mat)>,
      matrix_utils::ALBUSSPMV<int32_t, int32_t, VALTYPE,
                              matrix_utils::RowDotKernel::Scalar,
                              matrix_utils::WorkloadMode::CAMLB>>
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
auto CAMLBSimd = [](benchmark::State &state, const auto &mat,
                    const int threads, const int it) {
  std::vector<VALTYPE> x(mat.rows, 0.0);
  std::vector<VALTYPE> b(mat.rows, 1.0);

  matrix_utils::SPMV<
      std::remove_cvref_t<decltype(mat)>,
      matrix_utils::ALBUSSPMV<int32_t, int32_t, VALTYPE,
                              matrix_utils::RowDotKernel::Simd,
                              matrix_utils::WorkloadMode::CAMLB>>
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

#ifdef USE_CUDA
template <typename VALTYPE>
auto CuSparseSPMV_Bench = [](benchmark::State &state, const auto &mat,
                         const int threads, const int it) {
  // Allocate host memory
  std::vector<VALTYPE> x(mat.rows, 0.0);
  std::vector<VALTYPE> b(mat.rows, 1.0);
  
  // Allocate device memory for vectors
  VALTYPE* d_x;
  VALTYPE* d_y;
  cudaMalloc(&d_x, mat.rows * sizeof(VALTYPE));
  cudaMalloc(&d_y, mat.rows * sizeof(VALTYPE));
  
  // Copy input vector to device
  cudaMemcpy(d_x, b.data(), mat.rows * sizeof(VALTYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, x.data(), mat.rows * sizeof(VALTYPE), cudaMemcpyHostToDevice);
  
  // Create cuSPARSE handle and preprocess CUDA SpMV
  cusparseHandle_t handle;
  cusparseCreate(&handle);
  matrix_utils::CuSparseSPMV<int, int, VALTYPE> cuda_spmv(handle);
  cuda_spmv.preprocess(mat.rows, mat.ai.data(), mat.aj.data(), mat.av.data());
  
  // Warm-up run
  cuda_spmv(d_x, d_y);
  cudaDeviceSynchronize();
  
  // Benchmark
  for (auto _ : state) {
    for (int i = 0; i < it; i++) {
      cuda_spmv(d_x, d_y);
    }
    cudaDeviceSynchronize();
  }
  
  state.SetBytesProcessed(2 * sizeof(VALTYPE) * int64_t(state.iterations()) *
                          int64_t(it) * int64_t(mat.NNZ()));
  
  // Cleanup
  cusparseDestroy(handle);
  cudaFree(d_x);
  cudaFree(d_y);
};
#endif

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
  
  // Print matrix statistics
  std::size_t nnz = mat_double.av.size();
  std::cout << "Matrix information:\n";
  std::cout << "  Size: " << mat_double.rows << " x " << mat_double.rows << "\n";
  std::cout << "  NNZ: " << nnz << "\n";
  std::cout << "  Avg NNZ/row: " << static_cast<double>(nnz) / mat_double.rows << "\n";
  std::cout << "  Sparsity: " << (100.0 * nnz) / (static_cast<double>(mat_double.rows) * mat_double.rows) << "%\n";
  std::cout << "  Threads: " << num_threads << "\n";
  std::cout << "  Iterations: " << iterations << "\n";

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
  benchmark::RegisterBenchmark("CAMLBSum_double", CAMLBSum<double>, mat_double, num_threads,
                               iterations);
  benchmark::RegisterBenchmark("CAMLBSimd_double", CAMLBSimd<double>, mat_double, num_threads,
                               iterations);
  
#ifdef USE_CUDA
  benchmark::RegisterBenchmark("CuSparseSPMV_double", CuSparseSPMV_Bench<double>, mat_double, num_threads,
                               iterations);
#endif
  
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
  benchmark::RegisterBenchmark("CAMLBSum_float", CAMLBSum<float>, mat_float, num_threads,
                               iterations);
  benchmark::RegisterBenchmark("CAMLBSimd_float", CAMLBSimd<float>, mat_float, num_threads,
                               iterations);
  
#ifdef USE_CUDA
  benchmark::RegisterBenchmark("CuSparseSPMV_float", CuSparseSPMV_Bench<float>, mat_float, num_threads,
                               iterations);
#endif
  
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
