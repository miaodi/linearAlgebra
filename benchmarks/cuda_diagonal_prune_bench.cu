#include "cuda_csr_utils.cuh"
#include "matrix_utils.hpp"
#include "utils.h"
#include <benchmark/benchmark.h>
#include <cxxopts.hpp>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <omp.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace matrix_utils;
namespace cuda_utils = matrix_utils::sparse_cuda;

// Global variables to hold the matrix data
static std::string g_matrix_file = "data/nv2.mtx";
static std::vector<int> g_ai, g_aj;
static std::vector<double> g_av;
static int g_rows = 0;
static int g_base = 0;
static int g_original_nnz = 0;

// Helper function to check CUDA errors
static void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + message + " - " + 
                                cudaGetErrorString(error));
    }
}

// CPU version benchmark
static void BM_DiagonalScaledPrune_CPU(benchmark::State& state) {
    const double threshold = state.range(0) / 1000.0; // Convert from integer millis to double
    
    // Make a copy for each iteration
    for (auto _ : state) {
        std::vector<int> ai_cpu = g_ai;
        std::vector<int> aj_cpu = g_aj;
        std::vector<double> av_cpu = g_av;
        
        state.PauseTiming();
        // Ensure copy is complete before timing
        benchmark::DoNotOptimize(ai_cpu.data());
        benchmark::DoNotOptimize(aj_cpu.data());
        benchmark::DoNotOptimize(av_cpu.data());
        state.ResumeTiming();
        
        int removed = DiagonalScaledPrune(g_rows, ai_cpu.data(), aj_cpu.data(), av_cpu.data(), threshold);
        
        benchmark::DoNotOptimize(removed);
        benchmark::ClobberMemory();
    }
    
    state.counters["Rows"] = g_rows;
    state.counters["OrigNNZ"] = g_original_nnz;
    state.counters["Threshold"] = threshold;
}

// GPU version benchmark
static void BM_DiagonalScaledPrune_GPU(benchmark::State& state) {
    const double threshold = state.range(0) / 1000.0; // Convert from integer millis to double
    
    // Allocate device memory once
    thrust::device_vector<int> d_ai(g_ai);
    thrust::device_vector<int> d_aj(g_aj);
    thrust::device_vector<double> d_av(g_av);
    thrust::device_vector<int> d_mask(g_original_nnz);
    thrust::device_vector<int> d_ai_out(g_rows + 1);
    thrust::device_vector<int> d_aj_out(g_original_nnz);
    thrust::device_vector<double> d_av_out(g_original_nnz);
    
    // Warm-up
    cuda_utils::CSRGenDiagScaledPruneMask(
        g_rows,
        thrust::raw_pointer_cast(d_ai.data()),
        thrust::raw_pointer_cast(d_aj.data()),
        thrust::raw_pointer_cast(d_av.data()),
        threshold,
        thrust::raw_pointer_cast(d_mask.data()));
    cudaDeviceSynchronize();
    
    for (auto _ : state) {
        // Reset input data
        state.PauseTiming();
        thrust::copy(g_ai.begin(), g_ai.end(), d_ai.begin());
        thrust::copy(g_aj.begin(), g_aj.end(), d_aj.begin());
        thrust::copy(g_av.begin(), g_av.end(), d_av.begin());
        cudaDeviceSynchronize();
        state.ResumeTiming();
        
        // Step 1: Generate mask
        cuda_utils::CSRGenDiagScaledPruneMask(
            g_rows,
            thrust::raw_pointer_cast(d_ai.data()),
            thrust::raw_pointer_cast(d_aj.data()),
            thrust::raw_pointer_cast(d_av.data()),
            threshold,
            thrust::raw_pointer_cast(d_mask.data()));
        
        // Step 2: Apply mask
        int removed = cuda_utils::CSRSelectByMaskDevice(
            g_rows,
            thrust::raw_pointer_cast(d_ai.data()),
            thrust::raw_pointer_cast(d_aj.data()),
            thrust::raw_pointer_cast(d_av.data()),
            thrust::raw_pointer_cast(d_mask.data()),
            thrust::raw_pointer_cast(d_ai_out.data()),
            thrust::raw_pointer_cast(d_aj_out.data()),
            thrust::raw_pointer_cast(d_av_out.data()));
        
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(removed);
    }
    
    state.counters["Rows"] = g_rows;
    state.counters["OrigNNZ"] = g_original_nnz;
    state.counters["Threshold"] = threshold;
}

// GPU mask generation only benchmark
static void BM_DiagonalScaledPrune_GPU_MaskOnly(benchmark::State& state) {
    const double threshold = state.range(0) / 1000.0;
    
    thrust::device_vector<int> d_ai(g_ai);
    thrust::device_vector<int> d_aj(g_aj);
    thrust::device_vector<double> d_av(g_av);
    thrust::device_vector<int> d_mask(g_original_nnz);
    
    // Warm-up
    cuda_utils::CSRGenDiagScaledPruneMask(
        g_rows,
        thrust::raw_pointer_cast(d_ai.data()),
        thrust::raw_pointer_cast(d_aj.data()),
        thrust::raw_pointer_cast(d_av.data()),
        threshold,
        thrust::raw_pointer_cast(d_mask.data()));
    cudaDeviceSynchronize();
    
    for (auto _ : state) {
        cuda_utils::CSRGenDiagScaledPruneMask(
            g_rows,
            thrust::raw_pointer_cast(d_ai.data()),
            thrust::raw_pointer_cast(d_aj.data()),
            thrust::raw_pointer_cast(d_av.data()),
            threshold,
            thrust::raw_pointer_cast(d_mask.data()));
        
        cudaDeviceSynchronize();
    }
    
    state.counters["Rows"] = g_rows;
    state.counters["OrigNNZ"] = g_original_nnz;
    state.counters["Threshold"] = threshold;
}

// GPU mask application only benchmark
static void BM_DiagonalScaledPrune_GPU_SelectOnly(benchmark::State& state) {
    const double threshold = state.range(0) / 1000.0;
    
    thrust::device_vector<int> d_ai(g_ai);
    thrust::device_vector<int> d_aj(g_aj);
    thrust::device_vector<double> d_av(g_av);
    thrust::device_vector<int> d_mask(g_original_nnz);
    thrust::device_vector<int> d_ai_out(g_rows + 1);
    thrust::device_vector<int> d_aj_out(g_original_nnz);
    thrust::device_vector<double> d_av_out(g_original_nnz);
    
    // Generate mask once
    cuda_utils::CSRGenDiagScaledPruneMask(
        g_rows,
        thrust::raw_pointer_cast(d_ai.data()),
        thrust::raw_pointer_cast(d_aj.data()),
        thrust::raw_pointer_cast(d_av.data()),
        threshold,
        thrust::raw_pointer_cast(d_mask.data()));
    cudaDeviceSynchronize();
    
    for (auto _ : state) {
        int removed = cuda_utils::CSRSelectByMaskDevice(
            g_rows,
            thrust::raw_pointer_cast(d_ai.data()),
            thrust::raw_pointer_cast(d_aj.data()),
            thrust::raw_pointer_cast(d_av.data()),
            thrust::raw_pointer_cast(d_mask.data()),
            thrust::raw_pointer_cast(d_ai_out.data()),
            thrust::raw_pointer_cast(d_aj_out.data()),
            thrust::raw_pointer_cast(d_av_out.data()));
        
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(removed);
    }
    
    state.counters["Rows"] = g_rows;
    state.counters["OrigNNZ"] = g_original_nnz;
    state.counters["Threshold"] = threshold;
}

// Register benchmarks for different threshold values (in millis: 1, 5, 10, 50, 100 = 0.001, 0.005, 0.01, 0.05, 0.1)
BENCHMARK(BM_DiagonalScaledPrune_CPU)->Arg(1)->Arg(5)->Arg(10)->Arg(50)->Arg(100)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_DiagonalScaledPrune_GPU)->Arg(1)->Arg(5)->Arg(10)->Arg(50)->Arg(100)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_DiagonalScaledPrune_GPU_MaskOnly)->Arg(1)->Arg(5)->Arg(10)->Arg(50)->Arg(100)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_DiagonalScaledPrune_GPU_SelectOnly)->Arg(1)->Arg(5)->Arg(10)->Arg(50)->Arg(100)->Unit(benchmark::kMicrosecond);

int main(int argc, char** argv) {
    // Parse command line arguments
    cxxopts::Options options("cuda_diagonal_prune_bench", "Benchmark diagonal scaled prune CPU vs GPU");
    options.add_options()
        ("f,file", "Matrix file path (MTX format)", cxxopts::value<std::string>()->default_value("data/nv2.mtx"))
        ("t,threads", "Number of OpenMP threads for CPU", cxxopts::value<int>()->default_value("8"))
        ("h,help", "Print usage");
    
    auto result = options.parse(argc, argv);
    
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }
    
    g_matrix_file = result["file"].as<std::string>();
    int num_threads = result["threads"].as<int>();
    
    // Set OpenMP thread count
    omp_set_num_threads(num_threads);
    std::cout << "Using " << num_threads << " OpenMP threads for CPU benchmarks" << std::endl;
    
    // Load matrix
    std::cout << "Loading matrix from: " << g_matrix_file << std::endl;
    std::ifstream f(g_matrix_file);
    if (!f.is_open()) {
        std::cerr << "Error: Could not open matrix file: " << g_matrix_file << std::endl;
        return 1;
    }
    
    utils::read_matrix_market_csr(f, g_ai, g_aj, g_av);
    f.close();
    
    g_rows = g_ai.size() - 1;
    g_base = 0;
    g_original_nnz = g_ai[g_rows] - g_base;
    
    std::cout << "Matrix loaded successfully:" << std::endl;
    std::cout << "  Rows: " << g_rows << std::endl;
    std::cout << "  NNZ: " << g_original_nnz << std::endl;
    std::cout << "  Avg NNZ/row: " << (double)g_original_nnz / g_rows << std::endl;
    std::cout << std::endl;
    
    // Initialize Google Benchmark with remaining arguments
    // Filter out our custom arguments
    std::vector<char*> bench_argv;
    bench_argv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.find("--file") == std::string::npos && arg.find("-f") == std::string::npos &&
            arg.find("--threads") == std::string::npos && arg.find("-t") == std::string::npos) {
            bench_argv.push_back(argv[i]);
        }
    }
    int bench_argc = bench_argv.size();
    
    ::benchmark::Initialize(&bench_argc, bench_argv.data());
    if (::benchmark::ReportUnrecognizedArguments(bench_argc, bench_argv.data())) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    
    return 0;
}
