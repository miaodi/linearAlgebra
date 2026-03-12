#include <gtest/gtest.h>

#include "cuda_kernels.cuh"
#include "cuda_ruiz_scale.cuh"
#include "cuda_tiled_sparse_mat.cuh"
#include "io.hpp"
#include "matrix_utils.hpp"

#include <fstream>
#include <cstdlib>
#include <string>
#include <vector>

namespace cuda_utils = matrix_utils::sparse_cuda;

namespace {

std::string get_matrix_path()
{
    const char* env_path = std::getenv("MTX_FILE");
    if (env_path != nullptr && env_path[0] != '\0')
    {
        return std::string(env_path);
    }
    return "data/ex27.mtx";
}

} // namespace

TEST(CudaRuizScaleTiled, MultiIterationMatchesCSRWithValueUnpermute)
{
    const std::string matrix_path = get_matrix_path();
    std::cout << "Testing Ruiz scaling on matrix: " << matrix_path << std::endl;
    std::ifstream f(matrix_path);
    ASSERT_TRUE(f.is_open()) << "Failed to open " << matrix_path;

    using HostCSR = matrix_utils::CSRMatrixVec<int, int, double>;
    HostCSR h_csr;
    matrix_utils::readMatrixMarket(f, h_csr);

    const int rows = h_csr.rows;
    const int cols = h_csr.cols;
    const int nnz = static_cast<int>(h_csr.NNZ());

    cuda_utils::DeviceArray<int> d_ai;
    cuda_utils::DeviceArray<int> d_aj;
    cuda_utils::DeviceArray<double> d_av_csr;
    cuda_utils::DeviceArray<double> d_av_tile;
    d_ai.copyFromHost(h_csr.AI(), static_cast<size_t>(rows + 1));
    d_aj.copyFromHost(h_csr.AJ(), static_cast<size_t>(nnz));
    d_av_csr.copyFromHost(h_csr.AV(), static_cast<size_t>(nnz));
    d_av_tile.copyFromHost(h_csr.AV(), static_cast<size_t>(nnz));

    cuda_utils::DeviceArray<double> d_dr_csr;
    d_dr_csr.resize(static_cast<size_t>(rows));
    cuda_utils::fillArray(d_dr_csr.data(), static_cast<size_t>(rows), 1.0);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess) << cudaGetErrorString(cudaGetLastError());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << cudaGetErrorString(cudaGetLastError());

    cuda_utils::DeviceArray<double> d_dc_csr;
    d_dc_csr.resize(static_cast<size_t>(cols));
    cuda_utils::fillArray(d_dc_csr.data(), static_cast<size_t>(cols), 1.0);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess) << cudaGetErrorString(cudaGetLastError());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << cudaGetErrorString(cudaGetLastError());

    cuda_utils::DeviceArray<double> d_dr_tile;
    d_dr_tile.resize(static_cast<size_t>(rows));
    cuda_utils::fillArray(d_dr_tile.data(), static_cast<size_t>(rows), 1.0);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess) << cudaGetErrorString(cudaGetLastError());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << cudaGetErrorString(cudaGetLastError());

    cuda_utils::DeviceArray<double> d_dc_tile;
    d_dc_tile.resize(static_cast<size_t>(cols));
    cuda_utils::fillArray(d_dc_tile.data(), static_cast<size_t>(cols), 1.0);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess) << cudaGetErrorString(cudaGetLastError());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << cudaGetErrorString(cudaGetLastError());

    cuda_utils::DeviceTileCOOMatrix<int, int, double> tile_mat;
    constexpr int tile_k = 4;
    cuda_utils::CSRToTileCOO<int, int, double, true>(
        rows, cols, d_ai.data(), d_aj.data(), d_av_tile.data(), tile_k, tile_mat, nullptr);

    constexpr int max_iters = 10;

    ASSERT_TRUE((cuda_utils::RuizScaleCuda<int, int, double, cuda_utils::CudaRuizScalingNormType::MaxNorm>(
        tile_mat, d_dr_tile.data(), d_dc_tile.data(), max_iters)));

    ASSERT_TRUE((cuda_utils::RuizScaleCuda<int, int, double, cuda_utils::CudaRuizScalingNormType::MaxNorm>(
        rows, cols, d_ai.data(), d_aj.data(), d_av_csr.data(), d_dr_csr.data(), d_dc_csr.data(), max_iters)));

    std::vector<double> h_dr_csr(static_cast<size_t>(rows));
    std::vector<double> h_dc_csr(static_cast<size_t>(cols));
    std::vector<double> h_dr_tile(static_cast<size_t>(rows));
    std::vector<double> h_dc_tile(static_cast<size_t>(cols));
    std::vector<double> h_av_csr(static_cast<size_t>(nnz));
    std::vector<double> h_av_tile_sorted(static_cast<size_t>(nnz));
    std::vector<int> h_perm(static_cast<size_t>(nnz));
    std::vector<double> h_av_tile_original(static_cast<size_t>(nnz), 0.0);

    d_dr_csr.copyToHost(h_dr_csr.data());
    d_dc_csr.copyToHost(h_dc_csr.data());
    d_dr_tile.copyToHost(h_dr_tile.data());
    d_dc_tile.copyToHost(h_dc_tile.data());
    d_av_csr.copyToHost(h_av_csr.data());
    tile_mat.values.copyToHost(h_av_tile_sorted.data());
    tile_mat.permutation.copyToHost(h_perm.data());

    for (int sorted_idx = 0; sorted_idx < nnz; ++sorted_idx)
    {
        const int original_idx = h_perm[static_cast<size_t>(sorted_idx)];
        ASSERT_GE(original_idx, 0);
        ASSERT_LT(original_idx, nnz);
        h_av_tile_original[static_cast<size_t>(original_idx)] =
            h_av_tile_sorted[static_cast<size_t>(sorted_idx)];
    }

    constexpr double tol = 1e-12;
    for (int i = 0; i < rows; ++i)
    {
        EXPECT_NEAR(h_dr_csr[static_cast<size_t>(i)], h_dr_tile[static_cast<size_t>(i)],
                    tol * std::abs(h_dr_csr[static_cast<size_t>(i)]))
            << "row scale mismatch at row " << i;
    }
    for (int j = 0; j < cols; ++j)
    {
        EXPECT_NEAR(h_dc_csr[static_cast<size_t>(j)], h_dc_tile[static_cast<size_t>(j)],
                    tol * std::abs(h_dc_csr[static_cast<size_t>(j)]))
            << "col scale mismatch at col " << j;
    }
    for (int k = 0; k < nnz; ++k)
    {
        EXPECT_NEAR(h_av_csr[static_cast<size_t>(k)], h_av_tile_original[static_cast<size_t>(k)],
                    tol * std::abs(h_av_csr[static_cast<size_t>(k)]))
            << "value mismatch at original nnz index " << k;
    }
}
