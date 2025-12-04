#include "matrix_utils.hpp"
#include "io.hpp"
#include "spmv.hpp"
#include "sp_ops.hpp"
#include "Reordering.h"
#include "permutation.hpp"
#include <benchmark/benchmark.h>
#include <cxxopts.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <mutex>
#include <iostream>

// Benchmark configuration
struct ReorderingConfig {
    std::string name;
    enum class Type { Natural, RCM_Parallel, RCM_Traditional, MetisND } type;
    
    ReorderingConfig(std::string n, Type t) : name(std::move(n)), type(t) {}
};

static const std::vector<ReorderingConfig> reordering_configs = {
    ReorderingConfig{"Natural", ReorderingConfig::Type::Natural},
    ReorderingConfig{"RCM-Parallel", ReorderingConfig::Type::RCM_Parallel},
    ReorderingConfig{"RCM-Traditional", ReorderingConfig::Type::RCM_Traditional},
#ifdef USE_METIS_LIB
    ReorderingConfig{"MetisND", ReorderingConfig::Type::MetisND},
#endif
};

// Global matrix cache to avoid reloading
struct MatrixCache {
    std::string filename;
    matrix_utils::CSRMatrix<int, int, double> original;
    std::vector<matrix_utils::CSRMatrix<int, int, double>> reordered_matrices;
    std::vector<std::string> reorder_names;
    bool loaded = false;
    
    void load(const std::string& file) {
        if (loaded && filename == file) return;
        
        filename = file;
        std::ifstream f(file);
        if (!f.is_open()) {
            throw std::runtime_error("Failed to open matrix file: " + file);
        }
        
        matrix_utils::readMatrixMarket(f, original);
        f.close();
        
        // Compute adjacency graph (A + A^T without diagonal)
        std::vector<int> xadj(original.rows + 1);
        matrix_utils::APlusATPrefix<int, int, false>(
            original.rows, original.AI(), original.AJ(), xadj.data());
        
        int actual_edges = xadj[original.rows] - xadj[0];
        std::vector<int> adjncy(actual_edges);
        matrix_utils::APlusATFill<int, int, false>(
            original.rows, original.AI(), original.AJ(), xadj.data(), adjncy.data());
        
        // Apply each reordering
        reordered_matrices.clear();
        reorder_names.clear();
        
        for (const auto& config : reordering_configs) {
            std::vector<int> perm(original.rows);
            std::vector<int> iperm(original.rows);
            
            switch (config.type) {
                case ReorderingConfig::Type::Natural:
                    // Identity permutation
                    for (int i = 0; i < original.rows; ++i) {
                        perm[i] = i;
                        iperm[i] = i;
                    }
                    break;
                    
                case ReorderingConfig::Type::RCM_Parallel:
                    reordering::RCM<reordering::RCMKernel::ParallelSort>(
                        original.rows, xadj.data(), adjncy.data(),
                        perm.data(), iperm.data());
                    break;
                    
                case ReorderingConfig::Type::RCM_Traditional:
                    reordering::RCM<reordering::RCMKernel::Traditional>(
                        original.rows, xadj.data(), adjncy.data(),
                        perm.data(), iperm.data());
                    break;
                    
#ifdef USE_METIS_LIB
                case ReorderingConfig::Type::MetisND:
                    reordering::MetisND<int, int>(
                        original.rows, original.cols,
                        xadj.data(), adjncy.data(),
                        iperm.data(), perm.data());
                    break;
#endif
            }
            
            // Create permuted matrix
            matrix_utils::CSRMatrix<int, int, double> permuted;
            permuted.rows = original.rows;
            permuted.cols = original.cols;
            permuted.ResizeAI(original.rows + 1);
            permuted.ResizeAJ(original.NNZ());
            permuted.ResizeAV(original.NNZ());
            
            matrix_utils::permuteMat(original.rows, original.cols,
                                    perm.data(), iperm.data(),
                                    original.AI(), original.AJ(), original.AV(),
                                    permuted.AI(), permuted.AJ(), permuted.AV());
            
            reordered_matrices.push_back(std::move(permuted));
            reorder_names.push_back(config.name);
        }
        
        // Write SVG files for visualization
        std::cout << "\nGenerating SVG visualizations...\n";
        for (size_t i = 0; i < reordered_matrices.size(); ++i) {
            std::string svg_filename = "matrix_" + reorder_names[i] + ".svg";
            std::ofstream svg_file(svg_filename);
            if (svg_file.is_open()) {
                matrix_utils::writeSVG(reordered_matrices[i].rows, 
                                      reordered_matrices[i].cols,
                                      reordered_matrices[i].AI(), 
                                      reordered_matrices[i].AJ(),
                                      svg_file);
                svg_file.close();
                std::cout << "  Saved: " << svg_filename << "\n";
            } else {
                std::cerr << "  Warning: Could not create " << svg_filename << "\n";
            }
        }
        
        loaded = true;
    }
};

static MatrixCache g_matrix_cache;

// Benchmark for ParallelSPMV
template<typename SPMVType>
static void BM_SPMV_Reordering(benchmark::State& state, 
                               const std::string& matrix_file,
                               const std::string& spmv_name) {
    // Load and prepare matrices
    try {
        g_matrix_cache.load(matrix_file);
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
        return;
    }
    
    const int reorder_idx = state.range(0);
    const int nthreads = state.range(1);
    
    if (reorder_idx >= g_matrix_cache.reordered_matrices.size()) {
        state.SkipWithError("Invalid reordering index");
        return;
    }
    
    const auto& matrix = g_matrix_cache.reordered_matrices[reorder_idx];
    const auto& reorder_name = g_matrix_cache.reorder_names[reorder_idx];
    
    // Setup SPMV
    SPMVType spmv_impl(nthreads);
    matrix_utils::SPMV<matrix_utils::CSRMatrix<int, int, double>, SPMVType> spmv;
    spmv._matrix = &matrix;
    spmv._spmv = spmv_impl;
    spmv.preprocess();
    
    // Allocate vectors
    std::vector<double> b(matrix.rows, 1.0);
    std::vector<double> x(matrix.rows, 0.0);
    
    // Warm-up
    spmv(b.data(), x.data(), 1.0, 0.0);
    
    // Benchmark
    for (auto _ : state) {
        spmv(b.data(), x.data(), 1.0, 0.0);
        benchmark::DoNotOptimize(x.data());
        benchmark::ClobberMemory();
    }
    
    // Compute metrics
    const double nnz = static_cast<double>(matrix.NNZ());
    const double flops = 2.0 * nnz; // One multiply and one add per non-zero
    
    state.counters["Rows"] = matrix.rows;
    state.counters["NNZ"] = nnz;
    state.counters["NNZ/Row"] = nnz / matrix.rows;
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, 
        benchmark::Counter::kIs1000);
    state.counters["GB/s"] = benchmark::Counter(
        nnz * (sizeof(int) + sizeof(double)) + matrix.rows * sizeof(double) * 2,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1024);
    
    state.SetLabel(spmv_name + "/" + reorder_name + "/threads=" + std::to_string(nthreads));
}

int main(int argc, char** argv) {
    // Parse command line options
    int num_threads = 1;
    std::string matrix_file;
    
    cxxopts::Options options("SPMV Reordering Benchmark",
                             "Compare SPMV performance across different reordering techniques");
    options.allow_unrecognised_options().add_options()
        ("n,nt", "Number of threads", cxxopts::value<int>()->default_value("1"))
        ("f,file", "Matrix file location", 
         cxxopts::value<std::string>()->default_value("data/thermal2.mtx"))
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        benchmark::Initialize(&argc, argv);
        benchmark::Shutdown();
        return 0;
    }
    
    num_threads = result["n"].as<int>();
    matrix_file = result["f"].as<std::string>();
    
    std::cout << "Matrix file: " << matrix_file << "\n";
    std::cout << "Threads: " << num_threads << "\n";
    
    // Load and prepare matrices with all reorderings
    try {
        g_matrix_cache.load(matrix_file);
    } catch (const std::exception& e) {
        std::cerr << "Error loading matrix: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "Matrix rows: " << g_matrix_cache.original.rows << "\n";
    std::cout << "Matrix NNZ: " << g_matrix_cache.original.NNZ() << "\n";
    std::cout << "Reorderings prepared: " << g_matrix_cache.reorder_names.size() << "\n";
    for (size_t i = 0; i < g_matrix_cache.reorder_names.size(); ++i) {
        std::cout << "  [" << i << "] " << g_matrix_cache.reorder_names[i] << "\n";
    }
    
    // Register benchmarks for ParallelSPMV with specified thread count only
    for (size_t reorder_idx = 0; reorder_idx < reordering_configs.size(); ++reorder_idx) {
        auto bm_parallel = [=](benchmark::State& state) {
            BM_SPMV_Reordering<matrix_utils::ParallelSPMV<int, int, double>>(
                state, matrix_file, "ParallelSPMV");
        };
        std::string name = "ParallelSPMV/" + 
                         reordering_configs[reorder_idx].name;
        benchmark::RegisterBenchmark(name.c_str(), bm_parallel)
            ->Args({static_cast<int64_t>(reorder_idx), static_cast<int64_t>(num_threads)})
            ->Unit(benchmark::kMicrosecond);
    }
    
    // Register benchmarks for ALBUSSPMV with specified thread count only
    for (size_t reorder_idx = 0; reorder_idx < reordering_configs.size(); ++reorder_idx) {
        auto bm_albus = [=](benchmark::State& state) {
            BM_SPMV_Reordering<matrix_utils::ALBUSSPMV<int, int, double, 
                               matrix_utils::RowDotKernel::Simd>>(
                state, matrix_file, "ALBUSSPMV");
        };
        std::string name = "ALBUSSPMV/" + 
                         reordering_configs[reorder_idx].name;
        benchmark::RegisterBenchmark(name.c_str(), bm_albus)
            ->Args({static_cast<int64_t>(reorder_idx), static_cast<int64_t>(num_threads)})
            ->Unit(benchmark::kMicrosecond);
    }
    
    // Initialize and run Google Benchmark with original argc/argv
    // This allows benchmark flags like --benchmark_min_time to work
    benchmark::Initialize(&argc, argv);
    
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    
    return 0;
}
