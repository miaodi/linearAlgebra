#include <gtest/gtest.h>

#include "cuda_ruiz_scale.cuh"
#include "cuda_tiled_sparse_mat.cuh"
#include "io.hpp"
#include "matrix_utils.hpp"

#include <thrust/device_vector.h>

#include <fstream>
#include <vector>

namespace cuda_utils = matrix_utils::sparse_cuda;

TEST(CudaRuizScaleTiled, MultiIterationMatchesCSRWithValueUnpermute)
{
    std::ifstream f("data/ex27.mtx");
    ASSERT_TRUE(f.is_open()) << "Failed to open data/ex27.mtx";

    using HostCSR = matrix_utils::CSRMatrixVec<int, int, double>;
    HostCSR h_csr;
    matrix_utils::readMatrixMarket(f, h_csr);

    const int rows = h_csr.rows;
    const int cols = h_csr.cols;
    const int nnz = static_cast<int>(h_csr.NNZ());

    thrust::device_vector<int> d_ai(h_csr.AI(), h_csr.AI() + static_cast<size_t>(rows + 1));
    thrust::device_vector<int> d_aj(h_csr.AJ(), h_csr.AJ() + static_cast<size_t>(nnz));
    thrust::device_vector<double> d_av_csr(h_csr.AV(), h_csr.AV() + static_cast<size_t>(nnz));
    thrust::device_vector<double> d_av_tile = d_av_csr;

    thrust::device_vector<double> d_dr_csr(static_cast<size_t>(rows), 1.0);
    thrust::device_vector<double> d_dc_csr(static_cast<size_t>(cols), 1.0);
    thrust::device_vector<double> d_dr_tile(static_cast<size_t>(rows), 1.0);
    thrust::device_vector<double> d_dc_tile(static_cast<size_t>(cols), 1.0);

    cuda_utils::DeviceTileCOOMatrix<int, int, double> tile_mat;
    constexpr int tile_k = 4;
    cuda_utils::CSRToTileCOO<int, int, double>(
        rows, cols, thrust::raw_pointer_cast(d_ai.data()), thrust::raw_pointer_cast(d_aj.data()),
        thrust::raw_pointer_cast(d_av_tile.data()), tile_k, tile_mat, nullptr);

    constexpr int max_iters = 5;
    ASSERT_TRUE((cuda_utils::RuizScaleCuda<int, int, double, cuda_utils::CudaRuizScalingNormType::MaxNorm>(
        rows, cols, thrust::raw_pointer_cast(d_ai.data()), thrust::raw_pointer_cast(d_aj.data()),
        thrust::raw_pointer_cast(d_av_csr.data()), thrust::raw_pointer_cast(d_dr_csr.data()),
        thrust::raw_pointer_cast(d_dc_csr.data()), max_iters)));

    ASSERT_TRUE((cuda_utils::RuizScaleCuda<int, int, double, cuda_utils::CudaRuizScalingNormType::MaxNorm>(
        tile_mat, thrust::raw_pointer_cast(d_dr_tile.data()),
        thrust::raw_pointer_cast(d_dc_tile.data()), max_iters)));

    std::vector<double> h_dr_csr(static_cast<size_t>(rows));
    std::vector<double> h_dc_csr(static_cast<size_t>(cols));
    std::vector<double> h_dr_tile(static_cast<size_t>(rows));
    std::vector<double> h_dc_tile(static_cast<size_t>(cols));
    std::vector<double> h_av_csr(static_cast<size_t>(nnz));
    std::vector<double> h_av_tile_sorted(static_cast<size_t>(nnz));
    std::vector<int> h_perm(static_cast<size_t>(nnz));
    std::vector<double> h_av_tile_original(static_cast<size_t>(nnz), 0.0);

    thrust::copy(d_dr_csr.begin(), d_dr_csr.end(), h_dr_csr.begin());
    thrust::copy(d_dc_csr.begin(), d_dc_csr.end(), h_dc_csr.begin());
    thrust::copy(d_dr_tile.begin(), d_dr_tile.end(), h_dr_tile.begin());
    thrust::copy(d_dc_tile.begin(), d_dc_tile.end(), h_dc_tile.begin());
    thrust::copy(d_av_csr.begin(), d_av_csr.end(), h_av_csr.begin());
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
