#include "cuda_ruiz_scale.cuh"
#include "cuda_tiled_sparse_mat.cuh"
#include "io.hpp"
#include "matrix_utils.hpp"

#include <benchmark/benchmark.h>
#include <cxxopts.hpp>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuda_utils = matrix_utils::sparse_cuda;

namespace
{
inline void checkCudaError(cudaError_t error, const char* message)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(std::string("CUDA error: ") + message + " - " + cudaGetErrorString(error));
    }
}

using HostCSRMatrix = matrix_utils::CSRMatrixVec<int, int, double>;

struct MatrixCache
{
    static MatrixCache& instance()
    {
        static MatrixCache cache;
        return cache;
    }

    HostCSRMatrix mat;
    std::string filename;
    bool loaded = false;

    void load(const std::string& file)
    {
        if (loaded && filename == file)
        {
            return;
        }

        std::ifstream f(file);
        if (!f.is_open())
        {
            throw std::runtime_error("Failed to open matrix file: " + file);
        }

        matrix_utils::readMatrixMarket(f, mat);
        f.close();

        filename = file;
        loaded = true;

        const int base = mat.ai.empty() ? 0 : mat.ai[0];
        const int nnz = mat.ai.empty() ? 0 : mat.ai[mat.rows] - base;

        std::cout << "Matrix loaded: " << mat.rows << " x " << mat.cols << ", NNZ: " << nnz
                  << ", Base: " << base << std::endl;
    }

private:
    MatrixCache() = default;
};

static void BM_RuizScaleCudaCSR(benchmark::State& state, const std::string& matrix_file, int max_iters)
{
    MatrixCache& cache = MatrixCache::instance();
    cache.load(matrix_file);
    const HostCSRMatrix& mat = cache.mat;

    const int rows = mat.rows;
    const int cols = mat.cols;
    const int nnz = static_cast<int>(mat.NNZ());

    thrust::device_vector<int> d_ai(mat.ai);
    thrust::device_vector<int> d_aj(mat.aj);
    thrust::device_vector<double> d_av0(mat.av);
    thrust::device_vector<double> d_av = d_av0;
    thrust::device_vector<double> d_dr(static_cast<size_t>(rows), 1.0);
    thrust::device_vector<double> d_dc(static_cast<size_t>(cols), 1.0);

    (void)cuda_utils::RuizScaleCuda<int, int, double, cuda_utils::CudaRuizScalingNormType::MaxNorm>(
        rows, cols, thrust::raw_pointer_cast(d_ai.data()), thrust::raw_pointer_cast(d_aj.data()),
        thrust::raw_pointer_cast(d_av.data()), thrust::raw_pointer_cast(d_dr.data()),
        thrust::raw_pointer_cast(d_dc.data()), 1);
    checkCudaError(cudaDeviceSynchronize(), "warm-up sync (csr)");

    for (auto _ : state)
    {
        state.PauseTiming();
        thrust::copy(d_av0.begin(), d_av0.end(), d_av.begin());
        state.ResumeTiming();

        (void)cuda_utils::RuizScaleCuda<int, int, double, cuda_utils::CudaRuizScalingNormType::MaxNorm>(
            rows, cols, thrust::raw_pointer_cast(d_ai.data()),
            thrust::raw_pointer_cast(d_aj.data()), thrust::raw_pointer_cast(d_av.data()),
            thrust::raw_pointer_cast(d_dr.data()), thrust::raw_pointer_cast(d_dc.data()), max_iters);
        checkCudaError(cudaDeviceSynchronize(), "benchmark sync (csr)");
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(nnz) * max_iters);
}

static void BM_RuizScaleCudaTiled(benchmark::State& state, const std::string& matrix_file, int max_iters)
{
    const int tile_k = static_cast<int>(state.range(0));

    constexpr int warps_per_block = 4;
    const size_t tile_size = size_t{1} << tile_k;
    const size_t shared_bytes = static_cast<size_t>(warps_per_block) * 2 * tile_size * sizeof(double);

    int max_shared_per_block = 0;
    checkCudaError(cudaDeviceGetAttribute(&max_shared_per_block, cudaDevAttrMaxSharedMemoryPerBlock, 0),
                   "query max shared memory per block");
    if (shared_bytes > static_cast<size_t>(max_shared_per_block))
    {
        state.SkipWithError("tile_k exceeds available per-block shared memory on this GPU");
        return;
    }

    MatrixCache& cache = MatrixCache::instance();
    cache.load(matrix_file);
    const HostCSRMatrix& mat = cache.mat;

    const int rows = mat.rows;
    const int cols = mat.cols;
    const int nnz = static_cast<int>(mat.NNZ());

    thrust::device_vector<int> d_ai(mat.ai);
    thrust::device_vector<int> d_aj(mat.aj);
    thrust::device_vector<double> d_av_for_tile(mat.av);
    thrust::device_vector<double> d_dr(static_cast<size_t>(rows), 1.0);
    thrust::device_vector<double> d_dc(static_cast<size_t>(cols), 1.0);

    cuda_utils::DeviceTileCOOMatrix<int, int, double> tile_mat;
    cuda_utils::CSRToTileCOO<int, int, double>(
        rows, cols, thrust::raw_pointer_cast(d_ai.data()), thrust::raw_pointer_cast(d_aj.data()),
        thrust::raw_pointer_cast(d_av_for_tile.data()), tile_k, tile_mat, nullptr);
    checkCudaError(cudaDeviceSynchronize(), "tile preprocess sync");

    thrust::device_ptr<double> tile_values_begin = thrust::device_pointer_cast(tile_mat.values.data());
    thrust::device_ptr<double> tile_values_end = tile_values_begin + tile_mat.values.size();
    thrust::device_vector<double> d_tile_values0(tile_values_begin, tile_values_end);

    (void)cuda_utils::RuizScaleCuda<int, int, double, cuda_utils::CudaRuizScalingNormType::MaxNorm>(
        tile_mat, thrust::raw_pointer_cast(d_dr.data()), thrust::raw_pointer_cast(d_dc.data()), 1);
    checkCudaError(cudaDeviceSynchronize(), "warm-up sync (tile)");

    for (auto _ : state)
    {
        state.PauseTiming();
        thrust::copy(d_tile_values0.begin(), d_tile_values0.end(), tile_values_begin);
        state.ResumeTiming();

        (void)cuda_utils::RuizScaleCuda<int, int, double, cuda_utils::CudaRuizScalingNormType::MaxNorm>(
            tile_mat, thrust::raw_pointer_cast(d_dr.data()), thrust::raw_pointer_cast(d_dc.data()), max_iters);
        checkCudaError(cudaDeviceSynchronize(), "benchmark sync (tile)");
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(nnz) * max_iters);
    state.counters["tile_k"] = tile_k;
    state.counters["n_tiles"] = static_cast<double>(tile_mat.n_tiles);
}

void printUsage()
{
    std::cout << "cuda_ruiz_scale_bench - Benchmark for RuizScaleCuda (CSR vs TileCOO)\n"
              << "\nCustom Options:\n"
              << "  -m, --matrix FILE    Matrix Market file path (default: data/ex27.mtx)\n"
              << "  -i, --iters N        Ruiz iterations per call (default: 5)\n"
              << "\nGoogle Benchmark Options:\n"
              << "  --help               Print Google Benchmark help\n"
              << "  --benchmark_help     Print Google Benchmark help\n"
              << "  --benchmark_filter=<regex>\n";
}

} // namespace

int main(int argc, char** argv)
{
    cxxopts::Options options("cuda_ruiz_scale_bench", "Benchmark for RuizScaleCuda");
    options.allow_unrecognised_options().add_options()(
        "m,matrix", "Matrix Market file path", cxxopts::value<std::string>()->default_value("data/thermal2.mtx"))(
        "i,iters", "Ruiz iterations per benchmark call", cxxopts::value<int>()->default_value("5"));

    std::string matrix_file;
    int max_iters = 5;

    try
    {
        auto result = options.parse(argc, argv);
        matrix_file = result["m"].as<std::string>();
        max_iters = result["i"].as<int>();
        if (max_iters <= 0)
        {
            throw std::runtime_error("--iters must be > 0");
        }

        benchmark::RegisterBenchmark("BM_RuizScaleCudaCSR", BM_RuizScaleCudaCSR, matrix_file, max_iters)
            ->Unit(benchmark::kMillisecond);

        benchmark::RegisterBenchmark("BM_RuizScaleCudaTiled/k=2", BM_RuizScaleCudaTiled, matrix_file, max_iters)
            ->Arg(2)
            ->Unit(benchmark::kMillisecond);
        benchmark::RegisterBenchmark("BM_RuizScaleCudaTiled/k=4", BM_RuizScaleCudaTiled, matrix_file, max_iters)
            ->Arg(4)
            ->Unit(benchmark::kMillisecond);
        benchmark::RegisterBenchmark("BM_RuizScaleCudaTiled/k=6", BM_RuizScaleCudaTiled, matrix_file, max_iters)
            ->Arg(6)
            ->Unit(benchmark::kMillisecond);
        benchmark::RegisterBenchmark("BM_RuizScaleCudaTiled/k=8", BM_RuizScaleCudaTiled, matrix_file, max_iters)
            ->Arg(8)
            ->Unit(benchmark::kMillisecond);
        benchmark::RegisterBenchmark("BM_RuizScaleCudaTiled/k=10", BM_RuizScaleCudaTiled, matrix_file, max_iters)
            ->Arg(10)
            ->Unit(benchmark::kMillisecond);
        // benchmark::RegisterBenchmark("BM_RuizScaleCudaTiled/k=12", BM_RuizScaleCudaTiled, matrix_file, max_iters)
        //     ->Arg(12)
        //     ->Unit(benchmark::kMillisecond);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        printUsage();
        return 1;
    }

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
