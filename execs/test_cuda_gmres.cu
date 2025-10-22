#include "cuda_gmres.h"
#include "io.hpp"
#include "matrix_utils.hpp"
#include "precond.hpp"
#include "sparse_mat_traits.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <cxxopts.hpp>

using namespace cuda_iterative_solver;

/**
 * @brief CUDA GMRES solver with Matrix Market file support
 * 
 * This example shows how to:
 * 1. Read a Matrix Market file into CSR format
 * 2. Transfer data to GPU
 * 3. Configure and run the CUDA GMRES solver with various options
 * 4. Retrieve and verify the solution
 */
int main(int argc, char** argv)
{
    // Parse command-line arguments
    cxxopts::Options options("CUDA GMRES Test", "CUDA GMRES solver with Matrix Market file support");
    options.add_options()
        ("f,filename", "Matrix Market file to read", 
         cxxopts::value<std::string>()->default_value("../data/ex5.mtx"))
        ("l,level", "ILU level", 
         cxxopts::value<int>()->default_value("0"))
        ("r,restart", "GMRES restart parameter", 
         cxxopts::value<int>()->default_value("20"))
        ("m,maxiter", "Maximum number of GMRES iterations", 
         cxxopts::value<int>()->default_value("100"))
        ("t,reltol", "Relative tolerance for GMRES convergence", 
         cxxopts::value<double>()->default_value("1e-8"))
        ("a,abstol", "Absolute tolerance for GMRES convergence", 
         cxxopts::value<double>()->default_value("1e-12"))
        ("p,precond", "Preconditioner type: none, left, right", 
         cxxopts::value<std::string>()->default_value("none"))
        ("h,help", "Print usage");
    
    auto result = options.parse(argc, argv);
    
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }
    
    std::string filename = result["filename"].as<std::string>();
    int level = result["level"].as<int>();
    int restart = result["restart"].as<int>();
    int maxiter = result["maxiter"].as<int>();
    double reltol = result["reltol"].as<double>();
    double abstol = result["abstol"].as<double>();
    std::string precond_str = result["precond"].as<std::string>();
    
    // Parse preconditioner type
    PreconditionerType precond_type;
    if (precond_str == "none") {
        precond_type = PreconditionerType::NONE;
    } else if (precond_str == "left") {
        precond_type = PreconditionerType::LEFT;
    } else if (precond_str == "right") {
        precond_type = PreconditionerType::RIGHT;
    } else {
        std::cerr << "Invalid preconditioner type: " << precond_str 
                  << ". Valid options are: none, left, right" << std::endl;
        return 1;
    }
    
    // Print configuration
    std::cout << "CUDA GMRES Configuration:" << std::endl;
    std::cout << "  Matrix file: " << filename << std::endl;
    std::cout << "  ILU level: " << level << std::endl;
    std::cout << "  Restart: " << restart << std::endl;
    std::cout << "  Max iterations: " << maxiter << std::endl;
    std::cout << "  Relative tolerance: " << std::scientific << reltol << std::endl;
    std::cout << "  Absolute tolerance: " << std::scientific << abstol << std::endl;
    std::cout << "  Preconditioner: " << precond_str << std::endl;
    std::cout << std::endl;
    try {
        // Read Matrix Market file
        std::cout << "Reading matrix from file: " << filename << std::endl;
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return 1;
        }
        
        matrix_utils::CSRMatrix<int, int, double> csr_matrix;
        matrix_utils::readMatrixMarket(file, csr_matrix);
        file.close();
        
        const size_t n = csr_matrix.rows;
        const size_t nnz = csr_matrix.NNZ();
        
        std::cout << "Matrix loaded successfully:" << std::endl;
        std::cout << "  Size: " << n << " x " << csr_matrix.cols << std::endl;
        std::cout << "  Non-zeros: " << nnz << std::endl;
        std::cout << "  Density: " << std::fixed << std::setprecision(4) 
                  << (100.0 * nnz) / (n * n) << "%" << std::endl;
        
        // Perform ILU factorization if preconditioner is requested
        matrix_utils::CSRMatrix<int, int, double> ilu_matrix;
        matrix_utils::CSRMatrix<int, int, double> L_matrix, U_matrix;
        std::vector<double> D_vector;
        bool has_preconditioner = (precond_type != PreconditionerType::NONE);
        
        if (has_preconditioner) {
            std::cout << "\nPerforming ILU(" << level << ") factorization..." << std::endl;
            
            // Symbolic ILU factorization
            std::cout << "Symbolic ILU factorization..." << std::endl;
            matrix_utils::ILULevelSymbolic<decltype(ilu_matrix)> ilu;
            auto t1 = std::chrono::high_resolution_clock::now();
            bool success = ilu(csr_matrix.rows, csr_matrix.AI(), csr_matrix.AJ(), level, ilu_matrix);
            auto t2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = t2 - t1;
            std::cout << "Symbolic ILU factorization time: " << elapsed.count() << " s" << std::endl;
            
            if (!success) {
                std::cerr << "Symbolic ILU factorization failed." << std::endl;
                return 1;
            }
            std::cout << "Symbolic ILU factorization done. nnz: " << ilu_matrix.NNZ() << std::endl;
            
            // Numeric ILU factorization
            std::cout << "Numeric ILU factorization..." << std::endl;
            auto t3 = std::chrono::high_resolution_clock::now();
            success = matrix_utils::ILULevelNumeric(csr_matrix.rows, csr_matrix.AI(),
                                                   csr_matrix.AJ(), csr_matrix.AV(),
                                                   level, ilu_matrix);
            auto t4 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_numeric = t4 - t3;
            std::cout << "Numeric ILU factorization time: " << elapsed_numeric.count() << " s" << std::endl;
            
            if (!success) {
                std::cerr << "Numeric ILU factorization failed." << std::endl;
                return 1;
            }
            std::cout << "ILU factorization completed successfully." << std::endl;
            
            // Split ILU matrix into L, D, U components
            std::cout << "Splitting ILU matrix into L, D, U factors..." << std::endl;
            matrix_utils::SplitLDU<int, int, double>(n, ilu_matrix.AI()[0], ilu_matrix.AI(), ilu_matrix.AJ(),
                                                     ilu_matrix.AV(), L_matrix, D_vector, U_matrix);
            
            std::cout << "L factor nnz: " << L_matrix.NNZ() << std::endl;
            std::cout << "U factor nnz: " << U_matrix.NNZ() << std::endl;
        }
        
        // Generate right-hand side vector (all ones)
        std::vector<double> b_host(n, 1.0);
        
        // Generate random initial guess
        std::vector<double> x_host(n, 0.0);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-0.1, 0.1);
        for (size_t i = 0; i < n; ++i) {
            x_host[i] = dis(gen);
        }
        
        std::cout << "Generated RHS vector (all ones) and random initial guess" << std::endl;
        
        
        // Allocate device memory for matrix and vectors
        int *d_ia, *d_ja;
        double *d_va, *d_b, *d_x;
        
        // Allocate device memory for preconditioner factors (if needed)
        int *d_ia_L = nullptr, *d_ja_L = nullptr, *d_ia_U = nullptr, *d_ja_U = nullptr;
        double *d_va_L = nullptr, *d_va_U = nullptr;
        size_t nnz_L = 0, nnz_U = 0;
        
        cudaMalloc(&d_ia, (n + 1) * sizeof(int));
        cudaMalloc(&d_ja, nnz * sizeof(int));
        cudaMalloc(&d_va, nnz * sizeof(double));
        cudaMalloc(&d_b, n * sizeof(double));
        cudaMalloc(&d_x, n * sizeof(double));
        
        if (has_preconditioner) {
            nnz_L = L_matrix.NNZ();
            nnz_U = U_matrix.NNZ();
            
            cudaMalloc(&d_ia_L, (n + 1) * sizeof(int));
            cudaMalloc(&d_ja_L, nnz_L * sizeof(int));
            cudaMalloc(&d_va_L, nnz_L * sizeof(double));
            
            cudaMalloc(&d_ia_U, (n + 1) * sizeof(int));
            cudaMalloc(&d_ja_U, nnz_U * sizeof(int));
            cudaMalloc(&d_va_U, nnz_U * sizeof(double));
        }
        cudaMalloc(&d_x, n * sizeof(double));
        
        // Copy data to device
        cudaMemcpy(d_ia, csr_matrix.AI(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ja, csr_matrix.AJ(), nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_va, csr_matrix.AV(), nnz * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b_host.data(), n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x_host.data(), n * sizeof(double), cudaMemcpyHostToDevice);
        
        if (has_preconditioner) {
            // Copy preconditioner L and U factors to device
            cudaMemcpy(d_ia_L, L_matrix.AI(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_ja_L, L_matrix.AJ(), nnz_L * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_va_L, L_matrix.AV(), nnz_L * sizeof(double), cudaMemcpyHostToDevice);
            
            cudaMemcpy(d_ia_U, U_matrix.AI(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_ja_U, U_matrix.AJ(), nnz_U * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_va_U, U_matrix.AV(), nnz_U * sizeof(double), cudaMemcpyHostToDevice);
        }
        
        std::cout << "Data transferred to GPU successfully" << std::endl;
        
        
        // Create and configure GMRES solver
        CudaGMRES solver;
        solver.setMaxIter(maxiter);
        solver.setRelTol(reltol);
        solver.setAbsTol(abstol);
        solver.setRestart(restart);
        solver.setPreconditionerType(precond_type);
        
        std::cout << "\nStarting CUDA GMRES solver..." << std::endl;
        
        State result;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Setup matrix and preconditioner
        std::cout << "Setting up matrix and preconditioner..." << std::endl;
        if (has_preconditioner) {
            solver.setup(n, nnz, d_ia, d_ja, d_va,
                        nnz_L, d_ia_L, d_ja_L, d_va_L,    // L factor
                        nnz_U, d_ia_U, d_ja_U, d_va_U);   // U factor
        } else {
            solver.setup(n, nnz, d_ia, d_ja, d_va,
                        0, nullptr, nullptr, nullptr,    // No L factor
                        0, nullptr, nullptr, nullptr);   // No U factor
        }
        
        // Solve the system
        std::cout << "Solving linear system..." << std::endl;
        result = solver.solve(d_b, d_x);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "Solver completed in " << duration.count() << " ms" << std::endl;
        
        
        // Copy solution back to host
        cudaMemcpy(x_host.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost);
        
        // Print results
        std::cout << "\nSolver finished with state: ";
        switch (result) {
            case State::CONVERGED:
                std::cout << "CONVERGED" << std::endl;
                break;
            case State::MAX_ITER_REACHED:
                std::cout << "MAX_ITER_REACHED" << std::endl;
                break;
            case State::FAILED:
                std::cout << "FAILED" << std::endl;
                break;
            default:
                std::cout << "UNKNOWN" << std::endl;
                break;
        }
        
        // Compute and display solution statistics
        if (n <= 10) {
            std::cout << "Solution: [";
            for (size_t i = 0; i < n; ++i) {
                std::cout << std::scientific << std::setprecision(6) << x_host[i];
                if (i < n - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        } else {
            std::cout << "Solution (first 5 elements): [";
            for (size_t i = 0; i < 5; ++i) {
                std::cout << std::scientific << std::setprecision(6) << x_host[i];
                if (i < 4) std::cout << ", ";
            }
            std::cout << ", ...]" << std::endl;
        }
        
        // Compute residual: r = A*x - b
        std::vector<double> residual(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            double ax_i = 0.0;
            for (int j = csr_matrix.AI()[i]; j < csr_matrix.AI()[i + 1]; ++j) {
                ax_i += csr_matrix.AV()[j] * x_host[csr_matrix.AJ()[j]];
            }
            residual[i] = ax_i - b_host[i];
        }
        
        // Compute residual norms
        double residual_norm = 0.0;
        double b_norm = 0.0;
        for (size_t i = 0; i < n; ++i) {
            residual_norm += residual[i] * residual[i];
            b_norm += b_host[i] * b_host[i];
        }
        residual_norm = std::sqrt(residual_norm);
        b_norm = std::sqrt(b_norm);
        double relative_residual = (b_norm > 0.0) ? residual_norm / b_norm : residual_norm;
        
        std::cout << "\nResidual Analysis:" << std::endl;
        std::cout << "  Absolute L2 norm: " << std::scientific << std::setprecision(6) 
                  << residual_norm << std::endl;
        std::cout << "  Relative L2 norm: " << std::scientific << std::setprecision(6) 
                  << relative_residual << std::endl;
        std::cout << "  RHS L2 norm:      " << std::scientific << std::setprecision(6) 
                  << b_norm << std::endl;
        
        if (n <= 10) {
            std::cout << "Residual (A*x - b): [";
            for (size_t i = 0; i < n; ++i) {
                std::cout << std::scientific << std::setprecision(6) << residual[i];
                if (i < n - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        // Cleanup
        cudaFree(d_ia);
        cudaFree(d_ja);
        cudaFree(d_va);
        cudaFree(d_b);
        cudaFree(d_x);
        
        if (has_preconditioner) {
            cudaFree(d_ia_L);
            cudaFree(d_ja_L);
            cudaFree(d_va_L);
            cudaFree(d_ia_U);
            cudaFree(d_ja_U);
            cudaFree(d_va_U);
        }
        
        return (result == State::CONVERGED) ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}