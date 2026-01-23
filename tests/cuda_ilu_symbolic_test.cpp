#include <gtest/gtest.h>
#include "cuda_ilu_symbolic.cuh"
#include "precond.hpp"
#include "matrix_utils.hpp"
#include "io.hpp"
#include "utils.h"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>

using namespace cuda_iterative_solver;
using namespace matrix_utils;

// Helper function to compare CSR structures
template <typename ROWTYPE, typename COLTYPE>
bool compare_csr_structure(
    int n,
    const ROWTYPE* ai1, const COLTYPE* aj1,
    const ROWTYPE* ai2, const COLTYPE* aj2)
{
    // Compare row pointers
    for (int i = 0; i <= n; ++i)
    {
        if (ai1[i] != ai2[i])
        {
            std::cout << "Row pointer mismatch at " << i 
                     << ": " << ai1[i] << " vs " << ai2[i] << std::endl;
            return false;
        }
    }
    
    // Compare column indices
    ROWTYPE nnz = ai1[n] - ai1[0];
    for (ROWTYPE i = 0; i < nnz; ++i)
    {
        if (aj1[i] != aj2[i])
        {
            std::cout << "Column index mismatch at " << i 
                     << ": " << aj1[i] << " vs " << aj2[i] << std::endl;
            return false;
        }
    }
    
    return true;
}

TEST(CudaILUSymbolic, SmallMatrix_Level0)
{
    // Test on ex5 matrix: small SPD matrix
    std::vector<int> csr_rows;
    std::vector<int> csr_cols;
    std::vector<double> csr_vals;
    
    std::ifstream f("data/ex5.mtx");
    utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
    f.close();
    
    int n = csr_rows.size() - 1;
    int nnz = csr_cols.size();
    int base = 0;
    
    std::cout << "Matrix size: " << n << "x" << n 
              << ", NNZ: " << nnz << std::endl;
    
    // CPU reference computation
    ILULevelSymbolicParallelU<CSRMatrix<int, int, double>, true> cpu_symbolic(1);
    CSRMatrix<int, int, double> U_cpu;
    bool cpu_success = cpu_symbolic(n, csr_rows.data(), csr_cols.data(), 0, U_cpu);
    ASSERT_TRUE(cpu_success);
    
    std::cout << "CPU result: U has " << U_cpu.NNZ() << " nonzeros" << std::endl;
    
    // Write CPU U pattern to SVG
    std::ofstream cpu_svg("U_cpu_level0.svg");
    matrix_utils::writeSVG(U_cpu.rows, U_cpu.cols, U_cpu.AI(), U_cpu.AJ(), cpu_svg);
    cpu_svg.close();
    std::cout << "CPU U pattern written to U_cpu_level0.svg" << std::endl;
    
    // Copy matrix to GPU
    int* d_ai;
    int* d_aj;
    cudaMalloc(&d_ai, (n + 1) * sizeof(int));
    cudaMalloc(&d_aj, nnz * sizeof(int));
    cudaMemcpy(d_ai, csr_rows.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_aj, csr_cols.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    
    // Allocate output on GPU
    int* d_u_ai;
    int* d_u_aj;
    int u_nnz;
    cudaMalloc(&d_u_ai, (n + 1) * sizeof(int));
    
    // Run CUDA version
    bool cuda_success = ILUSymbolicU_CUDA<int, int>(
        n, d_ai, d_aj, 0, base, true, d_u_ai, &d_u_aj, &u_nnz);
    ASSERT_TRUE(cuda_success);
    
    std::cout << "CUDA result: U has " << u_nnz << " nonzeros" << std::endl;
    
    // Copy results back to host
    std::vector<int> u_ai_host(n + 1);
    std::vector<int> u_aj_host(u_nnz);
    cudaMemcpy(u_ai_host.data(), d_u_ai, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(u_aj_host.data(), d_u_aj, u_nnz * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Write CUDA U pattern to SVG
    std::ofstream cuda_svg("U_cuda_level0.svg");
    matrix_utils::writeSVG(n, n, u_ai_host.data(), u_aj_host.data(), cuda_svg);
    cuda_svg.close();
    std::cout << "CUDA U pattern written to U_cuda_level0.svg" << std::endl;
    
    // Compare structures
    bool match = compare_csr_structure(n, 
                                       U_cpu.AI(), U_cpu.AJ(),
                                       u_ai_host.data(), u_aj_host.data());
    EXPECT_TRUE(match) << "CUDA and CPU results do not match!";
    
    // Cleanup
    cudaFree(d_ai);
    cudaFree(d_aj);
    cudaFree(d_u_ai);
    cudaFree(d_u_aj);
}

TEST(CudaILUSymbolic, SmallMatrix_Level1)
{
    std::vector<int> csr_rows;
    std::vector<int> csr_cols;
    std::vector<double> csr_vals;
    
    std::ifstream f("data/ex5.mtx");
    utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
    f.close();
    
    int n = csr_rows.size() - 1;
    int nnz = csr_cols.size();
    int base = 0;
    
    // CPU reference
    ILULevelSymbolicParallelU<CSRMatrix<int, int, double>, true> cpu_symbolic(1);
    CSRMatrix<int, int, double> U_cpu;
    bool cpu_success = cpu_symbolic(n, csr_rows.data(), csr_cols.data(), 1, U_cpu);
    ASSERT_TRUE(cpu_success);
    
    std::cout << "CPU result (level 1): U has" << U_cpu.NNZ() << " nonzeros" << std::endl;
    
    // Write CPU U pattern to SVG
    std::ofstream cpu_svg("U_cpu_level1.svg");
    matrix_utils::writeSVG(U_cpu.rows, U_cpu.cols, U_cpu.AI(), U_cpu.AJ(), cpu_svg);
    cpu_svg.close();
    std::cout << "CPU U pattern written to U_cpu_level1.svg" << std::endl;
    
    // Copy to GPU
    int* d_ai;
    int* d_aj;
    cudaMalloc(&d_ai, (n + 1) * sizeof(int));
    cudaMalloc(&d_aj, nnz * sizeof(int));
    cudaMemcpy(d_ai, csr_rows.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_aj, csr_cols.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    
    // Run CUDA
    int* d_u_ai;
    int* d_u_aj;
    int u_nnz;
    cudaMalloc(&d_u_ai, (n + 1) * sizeof(int));
    
    bool cuda_success = ILUSymbolicU_CUDA<int, int>(
        n, d_ai, d_aj, 1, base, true, d_u_ai, &d_u_aj, &u_nnz);
    ASSERT_TRUE(cuda_success);
    
    std::cout << "CUDA result (level 1): U has " << u_nnz << " nonzeros" << std::endl;
    
    // Copy and compare
    std::vector<int> u_ai_host(n + 1);
    std::vector<int> u_aj_host(u_nnz);
    cudaMemcpy(u_ai_host.data(), d_u_ai, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(u_aj_host.data(), d_u_aj, u_nnz * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Write CUDA U pattern to SVG
    std::ofstream cuda_svg("U_cuda_level1.svg");
    matrix_utils::writeSVG(n, n, u_ai_host.data(), u_aj_host.data(), cuda_svg);
    cuda_svg.close();
    std::cout << "CUDA U pattern written to U_cuda_level1.svg" << std::endl;
    
    bool match = compare_csr_structure(n,
                                       U_cpu.AI(), U_cpu.AJ(),
                                       u_ai_host.data(), u_aj_host.data());
    EXPECT_TRUE(match);
    
    // Cleanup
    cudaFree(d_ai);
    cudaFree(d_aj);
    cudaFree(d_u_ai);
    cudaFree(d_u_aj);
}

TEST(CudaILUSymbolic, MediumMatrix_Level10)
{
    const int level =10;
    std::vector<int> csr_rows;
    std::vector<int> csr_cols;
    std::vector<double> csr_vals;
    
    std::ifstream f("data/bcsstk17.mtx");
    utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
    f.close();
    
    int n = csr_rows.size() - 1;
    int nnz = csr_cols.size();
    int base = 0;
    
    std::cout << "Testing on bcsstk17: " << n << "x" << n 
              << ", NNZ: " << nnz << std::endl;
    
    // CPU reference
    ILULevelSymbolicParallelU<CSRMatrix<int, int, double>, true> cpu_symbolic(4);
    CSRMatrix<int, int, double> U_cpu;
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    bool cpu_success = cpu_symbolic(n, csr_rows.data(), csr_cols.data(), level, U_cpu);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration<double>(cpu_end - cpu_start).count();
    
    ASSERT_TRUE(cpu_success);
    std::cout << "CPU time: " << cpu_time << " seconds, "
              << "U has " << U_cpu.NNZ() << " nonzeros" << std::endl;
    
    // Write CPU U pattern to SVG
    std::ofstream cpu_svg("U_cpu_level2.svg");
    matrix_utils::writeSVG(U_cpu.rows, U_cpu.cols, U_cpu.AI(), U_cpu.AJ(), cpu_svg);
    cpu_svg.close();
    std::cout << "CPU U pattern written to U_cpu_level2.svg" << std::endl;
    
    // Copy to GPU
    int* d_ai;
    int* d_aj;
    cudaMalloc(&d_ai, (n + 1) * sizeof(int));
    cudaMalloc(&d_aj, nnz * sizeof(int));
    cudaMemcpy(d_ai, csr_rows.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_aj, csr_cols.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    
    // Run CUDA
    int* d_u_ai;
    int* d_u_aj;
    int u_nnz;
    cudaMalloc(&d_u_ai, (n + 1) * sizeof(int));
    
    auto cuda_start = std::chrono::high_resolution_clock::now();
    bool cuda_success = ILUSymbolicU_CUDA_Persistent<int, int>(
        n, d_ai, d_aj, level, base, true, d_u_ai, &d_u_aj, &u_nnz);
    cudaDeviceSynchronize();
    auto cuda_end = std::chrono::high_resolution_clock::now();
    auto cuda_time = std::chrono::duration<double>(cuda_end - cuda_start).count();
    
    ASSERT_TRUE(cuda_success);
    std::cout << "CUDA time: " << cuda_time << " seconds, "
              << "U has " << u_nnz << " nonzeros" << std::endl;
    std::cout << "Speedup: " << cpu_time / cuda_time << "x" << std::endl;
    
    // Copy and compare
    std::vector<int> u_ai_host(n + 1);
    std::vector<int> u_aj_host(u_nnz);
    cudaMemcpy(u_ai_host.data(), d_u_ai, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(u_aj_host.data(), d_u_aj, u_nnz * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Write CUDA U pattern to SVG
    std::ofstream cuda_svg("U_cuda_level2.svg");
    matrix_utils::writeSVG(n, n, u_ai_host.data(), u_aj_host.data(), cuda_svg);
    cuda_svg.close();
    std::cout << "CUDA U pattern written to U_cuda_level2.svg" << std::endl;
    
    bool match = compare_csr_structure(n,
                                       U_cpu.AI(), U_cpu.AJ(),
                                       u_ai_host.data(), u_aj_host.data());
    EXPECT_TRUE(match);
    
    // Cleanup
    cudaFree(d_ai);
    cudaFree(d_aj);
    cudaFree(d_u_ai);
    cudaFree(d_u_aj);
}
