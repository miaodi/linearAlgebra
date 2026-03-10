#include <gtest/gtest.h>
#include "cuda_csr_utils.cuh"
#include "cuda_tiled_sparse_mat.cuh"
#include "matrix_utils.hpp"
#include "utils.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>

using namespace matrix_utils;
namespace cuda_utils = matrix_utils::sparse_cuda;

// Helper function to create a test matrix with known diagonal values
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void CreateTestMatrixForDiagonalPrune(
    COLTYPE rows,
    int base,
    std::vector<ROWTYPE>& ai,
    std::vector<COLTYPE>& aj,
    std::vector<VALTYPE>& av)
{
    ai.resize(rows + 1);
    ai[0] = base;
    
    std::vector<std::vector<std::pair<COLTYPE, VALTYPE>>> entries(rows);
    
    // Create a structured matrix with varying diagonal values
    // This ensures some entries will be pruned and some will be kept
    for (COLTYPE i = 0; i < rows; ++i)
    {
        VALTYPE diag_val = 10.0 / (i + 1); // Decreasing diagonal values
        entries[i].push_back({i, diag_val}); // Diagonal
        
        // Add off-diagonal entries with varying magnitudes
        if (i > 0)
            entries[i].push_back({i - 1, diag_val * 0.3}); // Left neighbor
        if (i < rows - 1)
            entries[i].push_back({i + 1, diag_val * 0.2}); // Right neighbor
        if (i > 1)
            entries[i].push_back({i - 2, diag_val * 0.05}); // Should be pruned with threshold 0.01
        if (i < rows - 2)
            entries[i].push_back({i + 2, diag_val * 0.04}); // Should be pruned with threshold 0.01
    }
    
    // Build CSR structure
    ROWTYPE nnz_count = 0;
    for (COLTYPE i = 0; i < rows; ++i)
    {
        // Sort entries by column index
        std::sort(entries[i].begin(), entries[i].end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        
        for (const auto& [col, val] : entries[i])
        {
            aj.push_back(col + base);
            av.push_back(val);
            nnz_count++;
        }
        ai[i + 1] = nnz_count + base;
    }
}

TEST(CudaCSRUtils, CSRFindDiagonalDeviceBase0)
{
    const int rows = 10;
    const int base = 0;
    
    std::vector<int> ai_host, aj_host;
    std::vector<double> av_host;
    CreateTestMatrixForDiagonalPrune<int, int, double>(rows, base, ai_host, aj_host, av_host);
    
    // Copy to device
    thrust::device_vector<int> d_ai(ai_host);
    thrust::device_vector<int> d_aj(aj_host);
    thrust::device_vector<double> d_av(av_host);
    thrust::device_vector<int> d_diag_pos(rows);
    thrust::device_vector<double> d_diag_val(rows);
    
    // Call CSRFindDiagonalDevice
    cuda_utils::CSRFindDiagonalDevice(
        rows,
        thrust::raw_pointer_cast(d_ai.data()),
        thrust::raw_pointer_cast(d_aj.data()),
        thrust::raw_pointer_cast(d_av.data()),
        thrust::raw_pointer_cast(d_diag_pos.data()),
        thrust::raw_pointer_cast(d_diag_val.data()));
    
    // Copy results back
    std::vector<int> diag_pos(rows);
    std::vector<double> diag_val(rows);
    thrust::copy(d_diag_pos.begin(), d_diag_pos.end(), diag_pos.begin());
    thrust::copy(d_diag_val.begin(), d_diag_val.end(), diag_val.begin());
    
    // Verify diagonal positions and values
    for (int i = 0; i < rows; ++i)
    {
        bool found = false;
        for (int j = ai_host[i] - base; j < ai_host[i + 1] - base; ++j)
        {
            if (aj_host[j] - base == i)
            {
                found = true;
                EXPECT_EQ(diag_pos[i], j + base) << "Diagonal position mismatch at row " << i;
                EXPECT_DOUBLE_EQ(diag_val[i], av_host[j]) << "Diagonal value mismatch at row " << i;
                break;
            }
        }
        EXPECT_TRUE(found) << "Diagonal not found at row " << i;
    }
}

TEST(CudaCSRUtils, CSRFindDiagonalDeviceBase1)
{
    const int rows = 10;
    const int base = 1;
    
    std::vector<int> ai_host, aj_host;
    std::vector<double> av_host;
    CreateTestMatrixForDiagonalPrune<int, int, double>(rows, base, ai_host, aj_host, av_host);
    
    // Copy to device
    thrust::device_vector<int> d_ai(ai_host);
    thrust::device_vector<int> d_aj(aj_host);
    thrust::device_vector<double> d_av(av_host);
    thrust::device_vector<int> d_diag_pos(rows);
    thrust::device_vector<double> d_diag_val(rows);
    
    // Call CSRFindDiagonalDevice
    cuda_utils::CSRFindDiagonalDevice(
        rows,
        thrust::raw_pointer_cast(d_ai.data()),
        thrust::raw_pointer_cast(d_aj.data()),
        thrust::raw_pointer_cast(d_av.data()),
        thrust::raw_pointer_cast(d_diag_pos.data()),
        thrust::raw_pointer_cast(d_diag_val.data()));
    
    // Copy results back
    std::vector<int> diag_pos(rows);
    std::vector<double> diag_val(rows);
    thrust::copy(d_diag_pos.begin(), d_diag_pos.end(), diag_pos.begin());
    thrust::copy(d_diag_val.begin(), d_diag_val.end(), diag_val.begin());
    
    // Verify diagonal positions and values
    for (int i = 0; i < rows; ++i)
    {
        bool found = false;
        for (int j = ai_host[i] - base; j < ai_host[i + 1] - base; ++j)
        {
            if (aj_host[j] == i + base)
            {
                found = true;
                EXPECT_EQ(diag_pos[i], j + base);
                EXPECT_DOUBLE_EQ(diag_val[i], av_host[j]);
                break;
            }
        }
        EXPECT_TRUE(found);
    }
}

TEST(CudaCSRUtils, DiagonalScaledPruneGPUvsCPUBase0)
{
    const int rows = 20;
    const int base = 0;
    const double threshold = 0.01;
    
    std::vector<int> ai_orig, aj_orig;
    std::vector<double> av_orig;
    CreateTestMatrixForDiagonalPrune<int, int, double>(rows, base, ai_orig, aj_orig, av_orig);
    
    const int original_nnz = ai_orig[rows] - base;
    std::cout << "Original NNZ: " << original_nnz << ", Base: " << base << std::endl;
    
    // CPU version
    std::vector<int> ai_cpu = ai_orig;
    std::vector<int> aj_cpu = aj_orig;
    std::vector<double> av_cpu = av_orig;
    
    int removed_cpu = DiagonalScaledPrune(rows, ai_cpu.data(), aj_cpu.data(), av_cpu.data(), threshold);
    int cpu_nnz = ai_cpu[rows] - base;
    
    std::cout << "CPU removed: " << removed_cpu << ", CPU final NNZ: " << cpu_nnz << std::endl;
    
    // GPU version - Step 1: Generate mask
    thrust::device_vector<int> d_ai(ai_orig);
    thrust::device_vector<int> d_aj(aj_orig);
    thrust::device_vector<double> d_av(av_orig);
    thrust::device_vector<int> d_mask(original_nnz);
    
    cuda_utils::CSRGenDiagScaledPruneMask(
        rows,
        thrust::raw_pointer_cast(d_ai.data()),
        thrust::raw_pointer_cast(d_aj.data()),
        thrust::raw_pointer_cast(d_av.data()),
        threshold,
        thrust::raw_pointer_cast(d_mask.data()));
    
    // GPU version - Step 2: Apply mask
    thrust::device_vector<int> d_ai_out(rows + 1);
    thrust::device_vector<int> d_aj_out(original_nnz);
    thrust::device_vector<double> d_av_out(original_nnz);
    
    int removed_gpu = cuda_utils::CSRSelectByMaskDevice(
        rows,
        thrust::raw_pointer_cast(d_ai.data()),
        thrust::raw_pointer_cast(d_aj.data()),
        thrust::raw_pointer_cast(d_av.data()),
        thrust::raw_pointer_cast(d_mask.data()),
        thrust::raw_pointer_cast(d_ai_out.data()),
        thrust::raw_pointer_cast(d_aj_out.data()),
        thrust::raw_pointer_cast(d_av_out.data()));
    
    std::vector<int> ai_gpu(rows + 1);
    thrust::copy(d_ai_out.begin(), d_ai_out.end(), ai_gpu.begin());
    int gpu_nnz = ai_gpu[rows] - base;
    
    std::cout << "GPU removed: " << removed_gpu << ", GPU final NNZ: " << gpu_nnz << std::endl;
    
    // Compare results
    EXPECT_EQ(removed_cpu, removed_gpu) << "CPU and GPU should remove same number of entries";
    EXPECT_EQ(cpu_nnz, gpu_nnz) << "CPU and GPU should have same final NNZ";
    
    // Compare row pointers
    for (int i = 0; i <= rows; ++i)
    {
        EXPECT_EQ(ai_cpu[i], ai_gpu[i]) << "Row pointer mismatch at row " << i;
    }
    
    // Compare column indices
    std::vector<int> aj_gpu(gpu_nnz);
    thrust::copy(d_aj_out.begin(), d_aj_out.begin() + gpu_nnz, aj_gpu.begin());
    for (int i = 0; i < gpu_nnz; ++i)
    {
        EXPECT_EQ(aj_cpu[i], aj_gpu[i]) << "Column index mismatch at position " << i;
    }
    
    // Compare values
    std::vector<double> av_gpu(gpu_nnz);
    thrust::copy(d_av_out.begin(), d_av_out.begin() + gpu_nnz, av_gpu.begin());
    for (int i = 0; i < gpu_nnz; ++i)
    {
        EXPECT_DOUBLE_EQ(av_cpu[i], av_gpu[i]) << "Value mismatch at position " << i;
    }
    
    // Verify all diagonal entries are preserved
    for (int i = 0; i < rows; ++i)
    {
        bool found_cpu = false, found_gpu = false;
        
        for (int j = ai_cpu[i] - base; j < ai_cpu[i + 1] - base; ++j)
        {
            if (aj_cpu[j] - base == i) found_cpu = true;
        }
        
        for (int j = ai_gpu[i] - base; j < ai_gpu[i + 1] - base; ++j)
        {
            if (aj_gpu[j] - base == i) found_gpu = true;
        }
        
        EXPECT_TRUE(found_cpu) << "CPU: Diagonal missing at row " << i;
        EXPECT_TRUE(found_gpu) << "GPU: Diagonal missing at row " << i;
    }
}

TEST(CudaCSRUtils, DiagonalScaledPruneGPUvsCPUBase1)
{
    const int rows = 20;
    const int base = 1;
    const double threshold = 0.01;
    
    std::vector<int> ai_orig, aj_orig;
    std::vector<double> av_orig;
    CreateTestMatrixForDiagonalPrune<int, int, double>(rows, base, ai_orig, aj_orig, av_orig);
    
    const int original_nnz = ai_orig[rows] - base;
    std::cout << "Original NNZ: " << original_nnz << ", Base: " << base << std::endl;
    
    // CPU version
    std::vector<int> ai_cpu = ai_orig;
    std::vector<int> aj_cpu = aj_orig;
    std::vector<double> av_cpu = av_orig;
    
    int removed_cpu = DiagonalScaledPrune(rows, ai_cpu.data(), aj_cpu.data(), av_cpu.data(), threshold);
    int cpu_nnz = ai_cpu[rows] - base;
    
    std::cout << "CPU removed: " << removed_cpu << ", CPU final NNZ: " << cpu_nnz << std::endl;
    
    // GPU version
    thrust::device_vector<int> d_ai(ai_orig);
    thrust::device_vector<int> d_aj(aj_orig);
    thrust::device_vector<double> d_av(av_orig);
    thrust::device_vector<int> d_mask(original_nnz);
    
    cuda_utils::CSRGenDiagScaledPruneMask(
        rows,
        thrust::raw_pointer_cast(d_ai.data()),
        thrust::raw_pointer_cast(d_aj.data()),
        thrust::raw_pointer_cast(d_av.data()),
        threshold,
        thrust::raw_pointer_cast(d_mask.data()));
    
    thrust::device_vector<int> d_ai_out(rows + 1);
    thrust::device_vector<int> d_aj_out(original_nnz);
    thrust::device_vector<double> d_av_out(original_nnz);
    
    int removed_gpu = cuda_utils::CSRSelectByMaskDevice(
        rows,
        thrust::raw_pointer_cast(d_ai.data()),
        thrust::raw_pointer_cast(d_aj.data()),
        thrust::raw_pointer_cast(d_av.data()),
        thrust::raw_pointer_cast(d_mask.data()),
        thrust::raw_pointer_cast(d_ai_out.data()),
        thrust::raw_pointer_cast(d_aj_out.data()),
        thrust::raw_pointer_cast(d_av_out.data()));
    
    std::vector<int> ai_gpu(rows + 1);
    thrust::copy(d_ai_out.begin(), d_ai_out.end(), ai_gpu.begin());
    int gpu_nnz = ai_gpu[rows] - base;
    
    std::cout << "GPU removed: " << removed_gpu << ", GPU final NNZ: " << gpu_nnz << std::endl;
    
    // Compare results
    EXPECT_EQ(removed_cpu, removed_gpu) << "CPU and GPU should remove same number of entries";
    EXPECT_EQ(cpu_nnz, gpu_nnz) << "CPU and GPU should have same final NNZ";
    EXPECT_EQ(ai_cpu[0], base) << "CPU base should be preserved";
    EXPECT_EQ(ai_gpu[0], base) << "GPU base should be preserved";
    
    // Compare row pointers
    for (int i = 0; i <= rows; ++i)
    {
        EXPECT_EQ(ai_cpu[i], ai_gpu[i]) << "Row pointer mismatch at row " << i;
    }
    
    // Compare column indices
    std::vector<int> aj_gpu(gpu_nnz);
    thrust::copy(d_aj_out.begin(), d_aj_out.begin() + gpu_nnz, aj_gpu.begin());
    for (int i = 0; i < gpu_nnz; ++i)
    {
        EXPECT_EQ(aj_cpu[i], aj_gpu[i]) << "Column index mismatch at position " << i;
    }
    
    // Compare values
    std::vector<double> av_gpu(gpu_nnz);
    thrust::copy(d_av_out.begin(), d_av_out.begin() + gpu_nnz, av_gpu.begin());
    for (int i = 0; i < gpu_nnz; ++i)
    {
        EXPECT_DOUBLE_EQ(av_cpu[i], av_gpu[i]) << "Value mismatch at position " << i;
    }
}

TEST(CudaCSRUtils, DiagonalScaledPruneMultipleThresholds)
{
    const int rows = 15;
    const int base = 0;
    
    std::vector<double> thresholds = {0.001, 0.005, 0.01, 0.05, 0.1};
    
    for (double threshold : thresholds)
    {
        std::vector<int> ai_orig, aj_orig;
        std::vector<double> av_orig;
        CreateTestMatrixForDiagonalPrune<int, int, double>(rows, base, ai_orig, aj_orig, av_orig);
        
        // CPU version
        std::vector<int> ai_cpu = ai_orig;
        std::vector<int> aj_cpu = aj_orig;
        std::vector<double> av_cpu = av_orig;
        int removed_cpu = DiagonalScaledPrune(rows, ai_cpu.data(), aj_cpu.data(), av_cpu.data(), threshold);
        
        // GPU version
        thrust::device_vector<int> d_ai(ai_orig);
        thrust::device_vector<int> d_aj(aj_orig);
        thrust::device_vector<double> d_av(av_orig);
        int original_nnz = ai_orig[rows] - base;
        thrust::device_vector<int> d_mask(original_nnz);
        
        cuda_utils::CSRGenDiagScaledPruneMask(
            rows,
            thrust::raw_pointer_cast(d_ai.data()),
            thrust::raw_pointer_cast(d_aj.data()),
            thrust::raw_pointer_cast(d_av.data()),
            threshold,
            thrust::raw_pointer_cast(d_mask.data()));
        
        thrust::device_vector<int> d_ai_out(rows + 1);
        thrust::device_vector<int> d_aj_out(original_nnz);
        thrust::device_vector<double> d_av_out(original_nnz);
        
        int removed_gpu = cuda_utils::CSRSelectByMaskDevice(
            rows,
            thrust::raw_pointer_cast(d_ai.data()),
            thrust::raw_pointer_cast(d_aj.data()),
            thrust::raw_pointer_cast(d_av.data()),
            thrust::raw_pointer_cast(d_mask.data()),
            thrust::raw_pointer_cast(d_ai_out.data()),
            thrust::raw_pointer_cast(d_aj_out.data()),
            thrust::raw_pointer_cast(d_av_out.data()));
        
        EXPECT_EQ(removed_cpu, removed_gpu) 
            << "Mismatch at threshold " << threshold;
        
        std::vector<int> ai_gpu(rows + 1);
        thrust::copy(d_ai_out.begin(), d_ai_out.end(), ai_gpu.begin());
        
        for (int i = 0; i <= rows; ++i)
        {
            EXPECT_EQ(ai_cpu[i], ai_gpu[i]) 
                << "Row pointer mismatch at threshold " << threshold << ", row " << i;
        }
    }
}

TEST(CudaCSRUtils, DiagonalScaledPruneRealMatrixS3rmt3m3)
{
    const double threshold = 0.01;
    
    // Load s3rmt3m3 matrix from file
    std::vector<int> ai_orig, aj_orig;
    std::vector<double> av_orig;
    
    std::ifstream f("data/s3rmt3m3.mtx");
    ASSERT_TRUE(f.is_open()) << "Could not open s3rmt3m3.mtx";
    utils::read_matrix_market_csr(f, ai_orig, aj_orig, av_orig);
    f.close();
    
    int rows = ai_orig.size() - 1;
    int cols = rows;
    int base = 0;
    
    ASSERT_EQ(rows, 5357) << "s3rmt3m3 should have 5357 rows";
    
    const int original_nnz = ai_orig[rows] - base;
    std::cout << "\n=== Real Matrix Test: s3rmt3m3 ===" << std::endl;
    std::cout << "Matrix size: " << rows << " x " << cols << std::endl;
    std::cout << "Original NNZ: " << original_nnz << ", Base: " << base << std::endl;
    
    // CPU version
    std::vector<int> ai_cpu = ai_orig;
    std::vector<int> aj_cpu = aj_orig;
    std::vector<double> av_cpu = av_orig;
    
    int removed_cpu = DiagonalScaledPrune(rows, ai_cpu.data(), aj_cpu.data(), av_cpu.data(), threshold);
    int cpu_nnz = ai_cpu[rows] - base;
    
    std::cout << "CPU removed: " << removed_cpu << ", CPU final NNZ: " << cpu_nnz 
              << " (" << (100.0 * removed_cpu / original_nnz) << "% pruned)" << std::endl;
    
    // GPU version
    thrust::device_vector<int> d_ai(ai_orig);
    thrust::device_vector<int> d_aj(aj_orig);
    thrust::device_vector<double> d_av(av_orig);
    thrust::device_vector<int> d_mask(original_nnz);
    
    cuda_utils::CSRGenDiagScaledPruneMask(
        rows,
        thrust::raw_pointer_cast(d_ai.data()),
        thrust::raw_pointer_cast(d_aj.data()),
        thrust::raw_pointer_cast(d_av.data()),
        threshold,
        thrust::raw_pointer_cast(d_mask.data()));
    
    thrust::device_vector<int> d_ai_out(rows + 1);
    thrust::device_vector<int> d_aj_out(original_nnz);
    thrust::device_vector<double> d_av_out(original_nnz);
    
    int removed_gpu = cuda_utils::CSRSelectByMaskDevice(
        rows,
        thrust::raw_pointer_cast(d_ai.data()),
        thrust::raw_pointer_cast(d_aj.data()),
        thrust::raw_pointer_cast(d_av.data()),
        thrust::raw_pointer_cast(d_mask.data()),
        thrust::raw_pointer_cast(d_ai_out.data()),
        thrust::raw_pointer_cast(d_aj_out.data()),
        thrust::raw_pointer_cast(d_av_out.data()));
    
    std::vector<int> ai_gpu(rows + 1);
    thrust::copy(d_ai_out.begin(), d_ai_out.end(), ai_gpu.begin());
    int gpu_nnz = ai_gpu[rows] - base;
    
    std::cout << "GPU removed: " << removed_gpu << ", GPU final NNZ: " << gpu_nnz 
              << " (" << (100.0 * removed_gpu / original_nnz) << "% pruned)" << std::endl;
    
    // Compare results
    EXPECT_EQ(removed_cpu, removed_gpu) << "CPU and GPU should remove same number of entries";
    EXPECT_EQ(cpu_nnz, gpu_nnz) << "CPU and GPU should have same final NNZ";
    
    // Compare row pointers
    for (int i = 0; i <= rows; ++i)
    {
        EXPECT_EQ(ai_cpu[i], ai_gpu[i]) << "Row pointer mismatch at row " << i;
    }
    
    // Compare column indices
    std::vector<int> aj_gpu(gpu_nnz);
    thrust::copy(d_aj_out.begin(), d_aj_out.begin() + gpu_nnz, aj_gpu.begin());
    for (int i = 0; i < gpu_nnz; ++i)
    {
        EXPECT_EQ(aj_cpu[i], aj_gpu[i]) << "Column index mismatch at position " << i;
    }
    
    // Compare values
    std::vector<double> av_gpu(gpu_nnz);
    thrust::copy(d_av_out.begin(), d_av_out.begin() + gpu_nnz, av_gpu.begin());
    for (int i = 0; i < gpu_nnz; ++i)
    {
        EXPECT_DOUBLE_EQ(av_cpu[i], av_gpu[i]) << "Value mismatch at position " << i;
    }
    
    // Verify all diagonal entries are preserved
    int missing_diagonals = 0;
    for (int i = 0; i < rows; ++i)
    {
        bool found_cpu = false, found_gpu = false;
        
        for (int j = ai_cpu[i] - base; j < ai_cpu[i + 1] - base; ++j)
        {
            if (aj_cpu[j] - base == i) found_cpu = true;
        }
        
        for (int j = ai_gpu[i] - base; j < ai_gpu[i + 1] - base; ++j)
        {
            if (aj_gpu[j] - base == i) found_gpu = true;
        }
        
        if (!found_cpu || !found_gpu) missing_diagonals++;
    }
    
    EXPECT_EQ(missing_diagonals, 0) << "Missing " << missing_diagonals << " diagonal entries";
}

TEST(CudaCSRUtils, CSRFindDiagonalRealMatrixS3rmt3m3)
{
    // Load s3rmt3m3 matrix from file
    std::vector<int> ai_host, aj_host;
    std::vector<double> av_host;
    
    std::ifstream f("data/s3rmt3m3.mtx");
    ASSERT_TRUE(f.is_open()) << "Could not open s3rmt3m3.mtx";
    utils::read_matrix_market_csr(f, ai_host, aj_host, av_host);
    f.close();
    
    int rows = ai_host.size() - 1;
    int cols = rows;
    int base = 0;
    
    ASSERT_EQ(rows, 5357) << "s3rmt3m3 should have 5357 rows";
    
    std::cout << "\n=== Diagonal Find Test: s3rmt3m3 ===" << std::endl;
    std::cout << "Matrix size: " << rows << " x " << cols << ", Base: " << base << std::endl;
    
    // Copy to device
    thrust::device_vector<int> d_ai(ai_host);
    thrust::device_vector<int> d_aj(aj_host);
    thrust::device_vector<double> d_av(av_host);
    thrust::device_vector<int> d_diag_pos(rows);
    thrust::device_vector<double> d_diag_val(rows);
    
    // Call CSRFindDiagonalDevice
    cuda_utils::CSRFindDiagonalDevice(
        rows,
        thrust::raw_pointer_cast(d_ai.data()),
        thrust::raw_pointer_cast(d_aj.data()),
        thrust::raw_pointer_cast(d_av.data()),
        thrust::raw_pointer_cast(d_diag_pos.data()),
        thrust::raw_pointer_cast(d_diag_val.data()));
    
    // Copy results back
    std::vector<int> diag_pos(rows);
    std::vector<double> diag_val(rows);
    thrust::copy(d_diag_pos.begin(), d_diag_pos.end(), diag_pos.begin());
    thrust::copy(d_diag_val.begin(), d_diag_val.end(), diag_val.begin());
    
    // Verify diagonal positions and values
    int found_count = 0;
    int missing_count = 0;
    
    for (int i = 0; i < rows; ++i)
    {
        bool found = false;
        for (int j = ai_host[i] - base; j < ai_host[i + 1] - base; ++j)
        {
            if (aj_host[j] - base == i)
            {
                found = true;
                found_count++;
                EXPECT_EQ(diag_pos[i], j + base) << "Diagonal position mismatch at row " << i;
                EXPECT_DOUBLE_EQ(diag_val[i], av_host[j]) << "Diagonal value mismatch at row " << i;
                break;
            }
        }
        
        if (!found)
        {
            missing_count++;
            EXPECT_EQ(diag_pos[i], -1) << "Should be -1 for missing diagonal at row " << i;
            EXPECT_DOUBLE_EQ(diag_val[i], 0.0) << "Should be 0.0 for missing diagonal at row " << i;
        }
    }
    
    std::cout << "Diagonals found: " << found_count << "/" << rows << std::endl;
    if (missing_count > 0)
    {
        std::cout << "Missing diagonals: " << missing_count << std::endl;
    }
}

TEST(CudaCSRUtils, CSRPtrToCOORowDeviceBase0)
{
    const int rows = 8;
    const int base = 0;

    std::vector<int> ai_host, aj_host;
    std::vector<double> av_host;
    CreateTestMatrixForDiagonalPrune<int, int, double>(rows, base, ai_host, aj_host, av_host);

    const int nnz = ai_host.back() - base;
    thrust::device_vector<int> d_ai(ai_host);
    thrust::device_vector<int> d_coo_rows(nnz, -1);

    cuda_utils::CSRPtrToCOORowDevice(
        rows,
        thrust::raw_pointer_cast(d_ai.data()),
        thrust::raw_pointer_cast(d_coo_rows.data()));

    std::vector<int> coo_rows(nnz);
    thrust::copy(d_coo_rows.begin(), d_coo_rows.end(), coo_rows.begin());

    std::vector<int> expected(nnz, -1);
    for (int i = 0; i < rows; ++i)
    {
        for (int k = ai_host[i] - base; k < ai_host[i + 1] - base; ++k)
        {
            expected[k] = i + base;
        }
    }

    EXPECT_EQ(coo_rows, expected);
}

TEST(CudaCSRUtils, CSRPtrToCOORowDeviceBase1)
{
    const int rows = 8;
    const int base = 1;

    std::vector<int> ai_host, aj_host;
    std::vector<double> av_host;
    CreateTestMatrixForDiagonalPrune<int, int, double>(rows, base, ai_host, aj_host, av_host);

    const int nnz = ai_host.back() - base;
    thrust::device_vector<int> d_ai(ai_host);
    thrust::device_vector<int> d_coo_rows(nnz, -1);

    cuda_utils::CSRPtrToCOORowDevice(
        rows,
        thrust::raw_pointer_cast(d_ai.data()),
        thrust::raw_pointer_cast(d_coo_rows.data()));

    std::vector<int> coo_rows(nnz);
    thrust::copy(d_coo_rows.begin(), d_coo_rows.end(), coo_rows.begin());

    std::vector<int> expected(nnz, -1);
    for (int i = 0; i < rows; ++i)
    {
        for (int k = ai_host[i] - base; k < ai_host[i + 1] - base; ++k)
        {
            expected[k] = i + base;
        }
    }

    EXPECT_EQ(coo_rows, expected);
}

TEST(CudaCSRUtils, CSRToTileCOOBase1)
{
    const int rows = 4;
    const int cols = 4;
    const int base = 1;
    const int k = 1;

    std::vector<int> ai_host{1, 3, 4, 6, 7};
    std::vector<int> aj_host{1, 4, 3, 1, 4, 2};
    std::vector<double> av_host{10.0, 11.0, 12.0, 13.0, 14.0, 15.0};

    thrust::device_vector<int> d_ai(ai_host);
    thrust::device_vector<int> d_aj(aj_host);
    thrust::device_vector<double> d_av(av_host);

    cuda_utils::DeviceTileCOOMatrix<int, int, double> tiled;
    cuda_utils::CSRToTileCOO(
        rows,
        cols,
        thrust::raw_pointer_cast(d_ai.data()),
        thrust::raw_pointer_cast(d_aj.data()),
        thrust::raw_pointer_cast(d_av.data()),
        k,
        tiled);

    ASSERT_EQ(tiled.n_rows, rows);
    ASSERT_EQ(tiled.n_cols, cols);
    ASSERT_EQ(tiled.base, base);
    ASSERT_EQ(tiled.tile_k, k);
    ASSERT_EQ(tiled.tile_col_bits, 1);
    ASSERT_EQ(tiled.n_tile_rows, 2);
    ASSERT_EQ(tiled.n_tile_cols, 2);
    ASSERT_EQ(tiled.n_tiles, 4);

    std::vector<int> perm_host(av_host.size());
    std::vector<int> tile_nnz_prefix_host(5);
    std::vector<int> tile_rows_host(4);
    std::vector<int> tile_cols_host(4);
    std::vector<int> row_host(av_host.size());
    std::vector<int> col_host(av_host.size());
    std::vector<double> val_host(av_host.size());
    tiled.permutation.copyToHost(perm_host.data());
    tiled.tile_nnz_prefix.copyToHost(tile_nnz_prefix_host.data());
    tiled.tile_row_ind.copyToHost(tile_rows_host.data());
    tiled.tile_col_ind.copyToHost(tile_cols_host.data());
    tiled.row_ind.copyToHost(row_host.data());
    tiled.col_ind.copyToHost(col_host.data());
    tiled.values.copyToHost(val_host.data());

    const std::vector<int> expected_perm{0, 1, 2, 3, 5, 4};
    const std::vector<int> expected_tile_nnz_prefix{0, 1, 3, 5, 6};
    const std::vector<int> expected_tile_rows{0, 0, 1, 1};
    const std::vector<int> expected_tile_cols{0, 1, 0, 1};
    const std::vector<int> expected_rows{1, 1, 2, 3, 4, 3};
    const std::vector<int> expected_cols{1, 4, 3, 1, 2, 4};
    const std::vector<double> expected_vals{10.0, 11.0, 12.0, 13.0, 15.0, 14.0};

    EXPECT_EQ(perm_host, expected_perm);
    EXPECT_EQ(tile_nnz_prefix_host, expected_tile_nnz_prefix);
    EXPECT_EQ(tile_rows_host, expected_tile_rows);
    EXPECT_EQ(tile_cols_host, expected_tile_cols);
    EXPECT_EQ(row_host, expected_rows);
    EXPECT_EQ(col_host, expected_cols);
    EXPECT_EQ(val_host, expected_vals);
}

TEST(CudaCSRUtils, TileKeysToCOOMeta)
{
    const int col_bits = 1;
    std::vector<uint64_t> keys_host{0, 1, 1, 2, 2, 3};
    thrust::device_vector<uint64_t> d_keys(keys_host);
    thrust::device_vector<uint64_t> d_unique_keys(keys_host.size(), 0);
    thrust::device_vector<int> d_tile_nnz(keys_host.size(), 0);
    thrust::device_vector<int> d_tile_rows(keys_host.size(), -1);
    thrust::device_vector<int> d_tile_cols(keys_host.size(), -1);

    const int n_tiles = cuda_utils::CountUniqueTileKeys<int>(
        static_cast<int>(keys_host.size()),
        thrust::raw_pointer_cast(d_keys.data()),
        nullptr);

    cuda_utils::TileKeysToCOOMeta<int, int, int>(
        static_cast<int>(keys_host.size()),
        n_tiles,
        thrust::raw_pointer_cast(d_keys.data()),
        col_bits,
        thrust::raw_pointer_cast(d_unique_keys.data()),
        thrust::raw_pointer_cast(d_tile_nnz.data()),
        thrust::raw_pointer_cast(d_tile_rows.data()),
        thrust::raw_pointer_cast(d_tile_cols.data()));

    std::vector<uint64_t> unique_keys(static_cast<size_t>(n_tiles));
    std::vector<int> tile_nnz(static_cast<size_t>(n_tiles));
    std::vector<int> tile_rows(static_cast<size_t>(n_tiles));
    std::vector<int> tile_cols(static_cast<size_t>(n_tiles));
    thrust::copy(d_unique_keys.begin(), d_unique_keys.begin() + n_tiles, unique_keys.begin());
    thrust::copy(d_tile_nnz.begin(), d_tile_nnz.begin() + n_tiles, tile_nnz.begin());
    thrust::copy(d_tile_rows.begin(), d_tile_rows.begin() + n_tiles, tile_rows.begin());
    thrust::copy(d_tile_cols.begin(), d_tile_cols.begin() + n_tiles, tile_cols.begin());

    const std::vector<uint64_t> expected_unique_keys{0, 1, 2, 3};
    const std::vector<int> expected_tile_nnz{1, 2, 2, 1};
    const std::vector<int> expected_rows{0, 0, 1, 1};
    const std::vector<int> expected_cols{0, 1, 0, 1};

    EXPECT_EQ(unique_keys, expected_unique_keys);
    EXPECT_EQ(tile_nnz, expected_tile_nnz);
    EXPECT_EQ(tile_rows, expected_rows);
    EXPECT_EQ(tile_cols, expected_cols);
}
