#include "Reordering.h"
#include "io.hpp"
#include "matrix_utils.hpp"
#include "permutation.hpp"
#include "sp_ops.hpp"
#include "spmv.hpp"
#include <benchmark/benchmark.h>
#include <cxxopts.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <omp.h>

#ifdef USE_CUDA
#include "cuda_spmv.h"
#include <cuda_runtime.h>
#endif

#ifdef USE_MKL
#include <mkl_types.h>
#include <mkl.h>
#endif

// Benchmark configuration
struct ReorderingConfig
{
    std::string name;
    enum class Type
    {
        Natural,
        RCM_Parallel,
        RCM_Traditional,
        MetisND
    } type;

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
struct MatrixCache
{
    std::string filename;
    matrix_utils::CSRMatrix<int, int, double> original;
    std::vector<matrix_utils::CSRMatrix<int, int, double>> reordered_matrices;
    std::vector<std::string> reorder_names;
    bool loaded = false;

    void load(const std::string& file)
    {
        const int num_threads = 10;
        if (loaded && filename == file)
            return;

        filename = file;
        std::ifstream f(file);
        if (!f.is_open())
        {
            throw std::runtime_error("Failed to open matrix file: " + file);
        }

        matrix_utils::readMatrixMarket(f, original);
        f.close();

        // Compute adjacency graph (A + A^T without diagonal)
        std::vector<int> xadj(original.rows + 1);
        matrix_utils::APlusATPrefix<int, int, false>(original.rows, original.AI(), original.AJ(), xadj.data());

        int actual_edges = xadj[original.rows] - xadj[0];
        std::vector<int> adjncy(actual_edges);
        matrix_utils::APlusATFill<int, int, false>(original.rows, original.AI(), original.AJ(),
                                                   xadj.data(), adjncy.data());

        // Apply each reordering
        reordered_matrices.clear();
        reorder_names.clear();

        for (const auto& config : reordering_configs)
        {
            std::vector<int> perm(original.rows);
            std::vector<int> iperm(original.rows);

            switch (config.type)
            {
            case ReorderingConfig::Type::Natural:
                // Identity permutation
                for (int i = 0; i < original.rows; ++i)
                {
                    perm[i] = i;
                    iperm[i] = i;
                }
                break;

            case ReorderingConfig::Type::RCM_Parallel:
                reordering::RCM_MultiComponent<reordering::RCMKernel::ParallelSort>(
                    original.rows, xadj.data(), adjncy.data(), perm.data(), iperm.data(), num_threads);
                break;

            case ReorderingConfig::Type::RCM_Traditional:
                reordering::RCM_MultiComponent<reordering::RCMKernel::Traditional>(
                    original.rows, xadj.data(), adjncy.data(), perm.data(), iperm.data(), num_threads);
                break;

#ifdef USE_METIS_LIB
            case ReorderingConfig::Type::MetisND:
                reordering::MetisND<int, int>(original.rows, original.cols, xadj.data(),
                                              adjncy.data(), iperm.data(), perm.data());
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

            matrix_utils::permuteMat(original.rows, original.cols, perm.data(), iperm.data(),
                                     original.AI(), original.AJ(), original.AV(), permuted.AI(),
                                     permuted.AJ(), permuted.AV());

            reordered_matrices.push_back(std::move(permuted));
            reorder_names.push_back(config.name);
        }

        // Write SVG files for visualization
        std::cout << "\nGenerating SVG visualizations...\n";
        for (size_t i = 0; i < reordered_matrices.size(); ++i)
        {
            std::string svg_filename = "matrix_" + reorder_names[i] + ".svg";
            std::ofstream svg_file(svg_filename);
            if (svg_file.is_open())
            {
                matrix_utils::writeSVG(reordered_matrices[i].rows, reordered_matrices[i].cols,
                                       reordered_matrices[i].AI(), reordered_matrices[i].AJ(), svg_file, 500);
                svg_file.close();
                std::cout << "  Saved: " << svg_filename << "\n";
            }
            else
            {
                std::cerr << "  Warning: Could not create " << svg_filename << "\n";
            }
        }

        loaded = true;
    }
};

static MatrixCache g_matrix_cache;

// Benchmark for ParallelSPMV
template <typename SPMVType>
static void BM_SPMV_Reordering(benchmark::State& state, const std::string& matrix_file, const std::string& spmv_name)
{
    // Load and prepare matrices
    try
    {
        g_matrix_cache.load(matrix_file);
    }
    catch (const std::exception& e)
    {
        state.SkipWithError(e.what());
        return;
    }

    const int reorder_idx = state.range(0);
    const int nthreads = state.range(1);

    if (reorder_idx >= g_matrix_cache.reordered_matrices.size())
    {
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
    for (auto _ : state)
    {
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
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["GB/s"] =
        benchmark::Counter(nnz * (sizeof(int) + sizeof(double)) + matrix.rows * sizeof(double) * 2,
                           benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1024);

    state.SetLabel(spmv_name + "/" + reorder_name + "/threads=" + std::to_string(nthreads));
}

#ifdef USE_CUDA
// Benchmark for CudaSPMV
static void BM_CudaSPMV_Reordering(benchmark::State& state, const std::string& matrix_file)
{
    // Load and prepare matrices
    try
    {
        g_matrix_cache.load(matrix_file);
    }
    catch (const std::exception& e)
    {
        state.SkipWithError(e.what());
        return;
    }

    const int reorder_idx = state.range(0);

    if (reorder_idx >= g_matrix_cache.reordered_matrices.size())
    {
        state.SkipWithError("Invalid reordering index");
        return;
    }

    const auto& matrix = g_matrix_cache.reordered_matrices[reorder_idx];
    const auto& reorder_name = g_matrix_cache.reorder_names[reorder_idx];

    // Setup CUDA SPMV - use move semantics to avoid copying
    matrix_utils::SPMV<matrix_utils::CSRMatrix<int, int, double>, 
                       matrix_utils::CudaSPMV<int, int, double>> spmv;
    spmv._matrix = &matrix;
    spmv._spmv = matrix_utils::CudaSPMV<int, int, double>(); // Move temporary
    spmv.preprocess();

    // Allocate host vectors
    std::vector<double> b(matrix.rows, 1.0);
    std::vector<double> x(matrix.rows, 0.0);

    // Allocate device vectors
    double* d_b = nullptr;
    double* d_x = nullptr;
    cudaMalloc(&d_b, matrix.rows * sizeof(double));
    cudaMalloc(&d_x, matrix.rows * sizeof(double));
    
    // Copy input vector to device
    cudaMemcpy(d_b, b.data(), matrix.rows * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_x, 0, matrix.rows * sizeof(double));

    // Warm-up
    spmv._spmv(d_b, d_x, 1.0, 0.0);
    cudaDeviceSynchronize();

    // Benchmark
    for (auto _ : state)
    {
        spmv._spmv(d_b, d_x, 1.0, 0.0);
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(d_x);
    }

    // Copy result back (optional, for verification)
    cudaMemcpy(x.data(), d_x, matrix.rows * sizeof(double), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_b);
    cudaFree(d_x);

    // Compute metrics
    const double nnz = static_cast<double>(matrix.NNZ());
    const double flops = 2.0 * nnz; // One multiply and one add per non-zero

    state.counters["Rows"] = matrix.rows;
    state.counters["NNZ"] = nnz;
    state.counters["NNZ/Row"] = nnz / matrix.rows;
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["GB/s"] =
        benchmark::Counter(nnz * (sizeof(int) + sizeof(double)) + matrix.rows * sizeof(double) * 2,
                           benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1024);

    state.SetLabel("CudaSPMV/" + reorder_name);
}
#endif

#ifdef USE_MKL
// Benchmark for MKLSPMV
static void BM_MKLSPMV_Reordering(benchmark::State& state, const std::string& matrix_file)
{
    // Load and prepare matrices
    try
    {
        g_matrix_cache.load(matrix_file);
    }
    catch (const std::exception& e)
    {
        state.SkipWithError(e.what());
        return;
    }

    const int reorder_idx = state.range(0);
    const int nthreads = state.range(1);

    if (reorder_idx >= g_matrix_cache.reordered_matrices.size())
    {
        state.SkipWithError("Invalid reordering index");
        return;
    }

    const auto& matrix = g_matrix_cache.reordered_matrices[reorder_idx];
    const auto& reorder_name = g_matrix_cache.reorder_names[reorder_idx];

    // Setup MKL SPMV - use MKL_INT for compatibility with MKL functions
    matrix_utils::SPMV<matrix_utils::CSRMatrix<int, int, double>, 
                       matrix_utils::MKLSPMV<MKL_INT, MKL_INT, double>> spmv;
    spmv._matrix = &matrix;
    spmv._spmv = matrix_utils::MKLSPMV<MKL_INT, MKL_INT, double>();
    spmv.preprocess();

    // Allocate vectors
    std::vector<double> b(matrix.rows, 1.0);
    std::vector<double> x(matrix.rows, 0.0);
    mkl_set_num_threads(nthreads);
    // Warm-up
    spmv._spmv(b.data(), x.data(), 1.0, 0.0);

    // Benchmark
    for (auto _ : state)
    {
        spmv._spmv(b.data(), x.data(), 1.0, 0.0);
        benchmark::DoNotOptimize(x.data());
        benchmark::ClobberMemory();
    }
    mkl_set_num_threads(1); // Reset to default

    // Compute metrics
    const double nnz = static_cast<double>(matrix.NNZ());
    const double flops = 2.0 * nnz; // One multiply and one add per non-zero

    state.counters["Rows"] = matrix.rows;
    state.counters["NNZ"] = nnz;
    state.counters["NNZ/Row"] = nnz / matrix.rows;
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["GB/s"] =
        benchmark::Counter(nnz * (sizeof(int) + sizeof(double)) + matrix.rows * sizeof(double) * 2,
                           benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1024);

    state.SetLabel("MKLSPMV/" + reorder_name + "/threads=" + std::to_string(nthreads));
}
#endif

int main(int argc, char** argv)
{
    // Parse command line options
    int num_threads = 1;
    std::vector<int> thread_list;
    std::string matrix_file;
    bool use_simd = true;

    cxxopts::Options options("SPMV Reordering Benchmark",
                             "Compare SPMV performance across different reordering techniques");
    options.allow_unrecognised_options()
        .add_options()("n,nt", "Number of threads",
                        cxxopts::value<int>()->default_value("1"))
        ("nt_list", "Comma-separated list of threads (e.g. 1,2,4,8)",
         cxxopts::value<std::string>()->default_value(""))
        ("f,file", "Matrix file location",
         cxxopts::value<std::string>()->default_value("data/thermal2.mtx"))
        ("simd", "Enable SIMD kernels (default: true)",
         cxxopts::value<bool>()->default_value("true"))
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        benchmark::Initialize(&argc, argv);
        benchmark::Shutdown();
        return 0;
    }

    num_threads = result["n"].as<int>();
    matrix_file = result["f"].as<std::string>();
    use_simd = result["simd"].as<bool>();

    // Parse optional thread list
    {
        const std::string tl = result["nt_list"].as<std::string>();
        if (!tl.empty())
        {
            size_t start = 0;
            while (start < tl.size())
            {
                size_t comma = tl.find(',', start);
                std::string token = tl.substr(start, comma == std::string::npos ? std::string::npos : (comma - start));
                if (!token.empty())
                {
                    try
                    {
                        int t = std::stoi(token);
                        if (t > 0)
                            thread_list.push_back(t);
                    }
                    catch (...)
                    {
                        // Ignore invalid entries
                    }
                }
                if (comma == std::string::npos)
                    break;
                start = comma + 1;
            }
        }
        // Fallback to single nt if no list provided
        if (thread_list.empty())
            thread_list.push_back(num_threads);
    }

    std::cout << "Matrix file: " << matrix_file << "\n";
    std::cout << "SIMD: " << (use_simd ? "enabled" : "disabled") << "\n";
    std::cout << "Threads: ";
    for (size_t i = 0; i < thread_list.size(); ++i)
    {
        std::cout << thread_list[i] << (i + 1 < thread_list.size() ? "," : "");
    }
    std::cout << "\n";

    // Load and prepare matrices with all reorderings
    try
    {
        g_matrix_cache.load(matrix_file);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error loading matrix: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Matrix rows: " << g_matrix_cache.original.rows << "\n";
    std::cout << "Matrix NNZ: " << g_matrix_cache.original.NNZ() << "\n";
    std::cout << "Reorderings prepared: " << g_matrix_cache.reorder_names.size() << "\n";
    for (size_t i = 0; i < g_matrix_cache.reorder_names.size(); ++i)
    {
        std::cout << "  [" << i << "] " << g_matrix_cache.reorder_names[i] << "\n";
    }

    // Register benchmarks for ParallelSPMV for each requested thread count
    for (size_t reorder_idx = 0; reorder_idx < reordering_configs.size(); ++reorder_idx)
    {
        for (int t : thread_list)
        {
            auto bm_parallel = [=](benchmark::State& state)
            {
                BM_SPMV_Reordering<matrix_utils::ParallelSPMV<int, int, double>>(state, matrix_file,
                                                                                 "ParallelSPMV");
            };
            std::string name = "ParallelSPMV/" + reordering_configs[reorder_idx].name + "/nt=" + std::to_string(t);
            benchmark::RegisterBenchmark(name.c_str(), bm_parallel)
                ->Args({static_cast<int64_t>(reorder_idx), static_cast<int64_t>(t)})
                ->Unit(benchmark::kMicrosecond);
        }
    }

    // Register benchmarks for ALBUSSPMV for each requested thread count
    for (size_t reorder_idx = 0; reorder_idx < reordering_configs.size(); ++reorder_idx)
    {
        for (int t : thread_list)
        {
            if (use_simd)
            {
                auto bm_albus = [=](benchmark::State& state)
                {
                    BM_SPMV_Reordering<matrix_utils::ALBUSSPMV<int, int, double, matrix_utils::RowDotKernel::Simd>>(
                        state, matrix_file, "ALBUSSPMV-Simd");
                };
                std::string name = "ALBUSSPMV-Simd/" + reordering_configs[reorder_idx].name + "/nt=" + std::to_string(t);
                benchmark::RegisterBenchmark(name.c_str(), bm_albus)
                    ->Args({static_cast<int64_t>(reorder_idx), static_cast<int64_t>(t)})
                    ->Unit(benchmark::kMicrosecond);
            }
            else
            {
                auto bm_albus = [=](benchmark::State& state)
                {
                    BM_SPMV_Reordering<matrix_utils::ALBUSSPMV<int, int, double, matrix_utils::RowDotKernel::Scalar>>(
                        state, matrix_file, "ALBUSSPMV-Scalar");
                };
                std::string name = "ALBUSSPMV-Scalar/" + reordering_configs[reorder_idx].name + "/nt=" + std::to_string(t);
                benchmark::RegisterBenchmark(name.c_str(), bm_albus)
                    ->Args({static_cast<int64_t>(reorder_idx), static_cast<int64_t>(t)})
                    ->Unit(benchmark::kMicrosecond);
            }
        }
    }

    // Register benchmarks for CAMLB-SPMV for each requested thread count
    for (size_t reorder_idx = 0; reorder_idx < reordering_configs.size(); ++reorder_idx)
    {
        for (int t : thread_list)
        {
            if (use_simd)
            {
                auto bm_camlb = [=](benchmark::State& state)
                {
                    BM_SPMV_Reordering<matrix_utils::ALBUSSPMV<int, int, double, matrix_utils::RowDotKernel::Simd,
                                                               matrix_utils::WorkloadMode::CAMLB>>(
                        state, matrix_file, "CAMLBSPMV-Simd");
                };
                std::string name = "CAMLBSPMV-Simd/" + reordering_configs[reorder_idx].name + "/nt=" + std::to_string(t);
                benchmark::RegisterBenchmark(name.c_str(), bm_camlb)
                    ->Args({static_cast<int64_t>(reorder_idx), static_cast<int64_t>(t)})
                    ->Unit(benchmark::kMicrosecond);
            }
            else
            {
                auto bm_camlb = [=](benchmark::State& state)
                {
                    BM_SPMV_Reordering<matrix_utils::ALBUSSPMV<int, int, double, matrix_utils::RowDotKernel::Scalar,
                                                               matrix_utils::WorkloadMode::CAMLB>>(
                        state, matrix_file, "CAMLBSPMV-Scalar");
                };
                std::string name = "CAMLBSPMV-Scalar/" + reordering_configs[reorder_idx].name + "/nt=" + std::to_string(t);
                benchmark::RegisterBenchmark(name.c_str(), bm_camlb)
                    ->Args({static_cast<int64_t>(reorder_idx), static_cast<int64_t>(t)})
                    ->Unit(benchmark::kMicrosecond);
            }
        }
    }

#ifdef USE_CUDA
    // Register benchmarks for CudaSPMV for each reordering
    std::cout << "\nCUDA enabled - registering CudaSPMV benchmarks\n";
    for (size_t reorder_idx = 0; reorder_idx < reordering_configs.size(); ++reorder_idx)
    {
        auto bm_cuda = [=](benchmark::State& state)
        {
            BM_CudaSPMV_Reordering(state, matrix_file);
        };
        std::string name = "CudaSPMV/" + reordering_configs[reorder_idx].name;
        benchmark::RegisterBenchmark(name.c_str(), bm_cuda)
            ->Args({static_cast<int64_t>(reorder_idx)})
            ->Unit(benchmark::kMicrosecond);
    }
#endif

#ifdef USE_MKL
    // Register benchmarks for MKLSPMV for each reordering
    std::cout << "\nMKL enabled - registering MKLSPMV benchmarks\n";
    for (size_t reorder_idx = 0; reorder_idx < reordering_configs.size(); ++reorder_idx)
    {
        for (int t : thread_list)
        {
            auto bm_mkl = [=](benchmark::State& state)
            {
                BM_MKLSPMV_Reordering(state, matrix_file);
            };
            std::string name = "MKLSPMV/" + reordering_configs[reorder_idx].name + "/nt=" + std::to_string(t);
            benchmark::RegisterBenchmark(name.c_str(), bm_mkl)
                ->Args({static_cast<int64_t>(reorder_idx), static_cast<int64_t>(t)})
                ->Unit(benchmark::kMicrosecond);
        }
    }
#endif

    // Initialize and run Google Benchmark with original argc/argv
    // This allows benchmark flags like --benchmark_min_time to work
    benchmark::Initialize(&argc, argv);

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    return 0;
}
