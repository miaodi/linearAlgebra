#include "cuda_spmm.cuh"
#include "cuda_csr_utils.cuh"
#include "io.hpp"
#include "matrix_utils.hpp"
#include "utils.h"
#include <benchmark/benchmark.h>
#include <cxxopts.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <memory>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>

using namespace matrix_utils;
using namespace matrix_utils::sparse_cuda;

using HostCSRMatrix = CSRMatrixVec<int, int, double>;

// Global matrix cache
struct MatrixCache {
    static MatrixCache& instance() {
        static MatrixCache cache;
        return cache;
    }
    
    HostCSRMatrix mat;
    std::string filename;
    bool loaded = false;
    
    void load(const std::string& file) {
        if (loaded && filename == file) {
            return;  // Already loaded
        }
        
        std::ifstream f(file);
        if (!f.is_open()) {
            throw std::runtime_error("Failed to open matrix file: " + file);
        }
        
        matrix_utils::readMatrixMarket(f, mat.ai, mat.aj, mat.av);
        f.close();
        
        mat.rows = static_cast<int>(mat.ai.size()) - 1;
        mat.cols = mat.rows;
        
        filename = file;
        loaded = true;
        
        int base = mat.ai.empty() ? 0 : mat.ai[0];
        int nnz = mat.ai.empty() ? 0 : mat.ai[mat.rows] - base;
        
        std::cout << "Matrix loaded: " << mat.rows << " x " << mat.cols 
                  << ", NNZ: " << nnz << ", Base: " << base << std::endl;
    }
    
    void release() {
        mat.ai.clear();
        mat.aj.clear();
        mat.av.clear();
        loaded = false;
        filename.clear();
    }
    
private:
    MatrixCache() = default;
};

// Benchmark for SpMMStruct
static void BM_SpMMStruct(benchmark::State& state, const std::string& matrix_file,
                          OuterProductBuildMethod method) {
    MatrixCache& cache = MatrixCache::instance();
    cache.load(matrix_file);
    
    const HostCSRMatrix& mat = cache.mat;
    const int n = mat.rows;
    const int base = mat.ai.empty() ? 0 : mat.ai[0];
    const int nnz = mat.ai.empty() ? 0 : mat.ai[n] - base;
    
    if (mat.rows != mat.cols) {
        state.SkipWithError("SpMMStruct requires square matrices");
        return;
    }
    
    // Allocate device memory for input matrices
    // Note: For this benchmark, we use the same matrix structure for both A and B
    // In a real scenario:
    //   AT (CSC): column pointers and row indices
    //   A (CSR): row pointers and column indices
    // The function computes C = AT *  A sparsity pattern from outer products
    
    thrust::device_vector<int> d_ai_A(mat.ai);
    thrust::device_vector<int> d_aj_A(mat.aj);
    
    // Allocate output array for packed COO
    DeviceArray<uint64_t> packed_coo;
    
    // Warm-up call
    bool success = SpMMStruct(n, 
                              thrust::raw_pointer_cast(d_ai_A.data()),
                              thrust::raw_pointer_cast(d_aj_A.data()),
                              thrust::raw_pointer_cast(d_ai_A.data()),
                              thrust::raw_pointer_cast(d_aj_A.data()),
                              base, packed_coo, method);
    
    if (!success) {
        state.SkipWithError("SpMMStruct warm-up failed");
        return;
    }
    
    checkCudaError(cudaDeviceSynchronize(), "Failed to synchronize after warm-up");
    
    // Get the output size for metrics
    const size_t output_size = packed_coo.size();
    std::cout << "Output size: " << output_size << " entries" << std::endl;
    
    // Benchmark loop
    for (auto _ : state) {
        success = SpMMStruct(n,
                            thrust::raw_pointer_cast(d_ai_A.data()),
                            thrust::raw_pointer_cast(d_aj_A.data()),
                            thrust::raw_pointer_cast(d_ai_A.data()),
                            thrust::raw_pointer_cast(d_aj_A.data()),
                            base, packed_coo, method);
        
        if (!success) {
            state.SkipWithError("SpMMStruct call failed");
            return;
        }
        
        checkCudaError(cudaDeviceSynchronize(), "Failed to synchronize after SpMMStruct");
    }
    
    // Set metrics
    state.SetItemsProcessed(state.iterations() * nnz);
    state.SetBytesProcessed(state.iterations() * 
                           (mat.ai.size() * sizeof(int) +    // A pointers
                            mat.aj.size() * sizeof(int) +    // A indices
                            output_size * sizeof(uint64_t))); // Output COO
}

// Command line options
static void printUsage() {
    std::cout << "spmm_struct_bench - Benchmark for SpMMStruct\n"
              << "\nCustom Options:\n"
              << "  -m, --matrix FILE    Matrix Market file path (default: data/thermal2.mtx)\n"
              << "  -b, --builder MODE   outer-product builder: global|shared (default: global)\n"
              << "\nGoogle Benchmark Options:\n"
              << "  --help               Print Google Benchmark help\n"
              << "  --benchmark_help     Print Google Benchmark help\n"
              << "  --benchmark_list_tests\n"
              << "  --benchmark_filter=<regex>\n"
              << "  --benchmark_min_time=<seconds>\n"
              << "  See --help for more benchmark options\n";
}

int main(int argc, char** argv) {
    cxxopts::Options options("spmm_struct_bench", "Benchmark for SpMMStruct");
    options.allow_unrecognised_options()
        .add_options()
        ("m,matrix", "Matrix Market file path",
         cxxopts::value<std::string>()->default_value("data/thermal2.mtx"))
        ("b,builder", "Outer-product builder: global|shared",
         cxxopts::value<std::string>()->default_value("global"));
    
    try {
        auto result = options.parse(argc, argv);
        
        std::string matrix_file = result["m"].as<std::string>();
        std::string builder = result["b"].as<std::string>();
        OuterProductBuildMethod method = OuterProductBuildMethod::GlobalMemory;
        if (builder == "shared")
            method = OuterProductBuildMethod::SharedMemoryWarp;
        else if (builder != "global")
            throw std::runtime_error("Invalid --builder value: " + builder + " (expected global|shared)");
        
        // Register benchmark with the matrix file
        benchmark::RegisterBenchmark("BM_SpMMStruct", BM_SpMMStruct, matrix_file, method)
            ->Unit(benchmark::kMillisecond);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        printUsage();
        return 1;
    }
    
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    
    return 0;
}
