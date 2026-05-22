#include <gtest/gtest.h>

#include "cuda_spmm.cuh"
#include "io.hpp"
#include "matrix_utils.hpp"
#include "spgemm.hpp"
#include "utils.h"

#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <vector>

using namespace matrix_utils::sparse_cuda;
using namespace matrix_utils;

class CudaSpMM : public ::testing::TestWithParam<const char*>
{
};

TEST_P(CudaSpMM, SpMMStruct_AA)
{
    const std::string matrix_path = GetParam();
    std::ifstream f(matrix_path);
    if (!f.is_open())
    {
        GTEST_SKIP() << "Could not open " << matrix_path;
    }

    std::vector<int> ai;
    std::vector<int> aj;
    std::vector<double> av;
    matrix_utils::readMatrixMarket(f, ai, aj, av);
    f.close();

    const int n = static_cast<int>(ai.size()) - 1;
    ASSERT_GT(n, 0);
    ASSERT_EQ(ai[0], 0) << "This test expects base-0 CSR input.";

    // Build CSC(A) on host because SpMMStruct expects A in CSC and B in CSR.
    std::vector<int> ai_csc(n + 1);
    std::vector<int> aj_csc(aj.size());
    SerialTranspose<int, int, double>(
        n, n, ai.data(), aj.data(), av.data(), ai_csc.data(), aj_csc.data(), nullptr);

    DeviceArray<int> d_ai_csc;
    DeviceArray<int> d_aj_csc;
    DeviceArray<int> d_ai;
    DeviceArray<int> d_aj;
    d_ai_csc.copyFromHost(ai_csc.data(), ai_csc.size());
    d_aj_csc.copyFromHost(aj_csc.data(), aj_csc.size());
    d_ai.copyFromHost(ai.data(), ai.size());
    d_aj.copyFromHost(aj.data(), aj.size());

    DeviceArray<uint64_t> d_packed_coo;
    ASSERT_TRUE((SpMMStruct<int, int>(
        n, d_ai_csc.data(), d_aj_csc.data(), d_ai.data(), d_aj.data(), 0, d_packed_coo)));

    DeviceCSRMatrix<int, int> d_csr;
    ASSERT_TRUE((PackedCOOtoCSR<int, int>(
        d_packed_coo.data(), static_cast<int>(d_packed_coo.size()), n, 0, d_csr)));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<int> gpu_ai(n + 1);
    d_csr.ai.copyToHost(gpu_ai.data());

    const int gpu_nnz = gpu_ai[n];
    ASSERT_GE(gpu_nnz, 0);
    std::vector<int> gpu_aj(gpu_nnz);
    d_csr.aj.copyToHost(gpu_aj.data());

    CSRMatrixVec<int, int, double> cpu_c;
    SpGEMM<CSRMatrixVec<int, int, double>> spgemm(1);
    spgemm.analysis(n, n, ai.data(), aj.data(), n, n, ai.data(), aj.data(), cpu_c);

    ASSERT_EQ(cpu_c.ai.size(), gpu_ai.size());
    for (int i = 0; i <= n; ++i)
    {
        EXPECT_EQ(cpu_c.ai[i], gpu_ai[i]) << "Row pointer mismatch at row " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(
    MatrixSet,
    CudaSpMM,
    ::testing::Values(
        "data/ex5.mtx",
        "data/nos5.mtx",
        "data/ex27.mtx",
        "data/rdist1.mtx",
        "data/s3rmt3m3.mtx",
        "data/bcsstk17.mtx",
        "data/jgl009.mtx"));
