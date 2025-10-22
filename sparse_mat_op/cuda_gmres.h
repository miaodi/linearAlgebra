#pragma once

#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

namespace cuda_iterative_solver
{

/**
 * @brief Enumeration of preconditioner types.
 * left: M^{-1} * A * x = M^{-1} * b
 * right: A * M^{-1} * y = b, x = M^{-1} * y
 * none: A * x = b
 */
enum class PreconditionerType
{
    NONE = 0,
    LEFT = 1,
    RIGHT = 2
};

enum class State : int
{
    CONVERGED = 0,
    RUNNING = 1,
    MAX_ITER_REACHED = 2,
    FAILED = 3
};

/**
 * @brief CUDA-based GMRES solver for sparse matrices in CSR format
 * 
 * This class implements the GMRES iterative solver using CUDA libraries
 * (cuBLAS, cuSPARSE) for solving linear systems Ax = b.
 * 
 * Key features:
 * - Matrix and preconditioner in CSR format (ia, ja, va)
 * - Self-contained (only depends on cuBLAS/cuSPARSE/std libraries)
 * - Fixed double precision (no templates)
 * - Givens rotations performed on host
 * - Hessenberg matrix in unified memory for host/device access
 * - Forward/backward substitution using cuSPARSE SpSV
 */
class CudaGMRES
{
public:
    /**
     * @brief Constructor
     */
    CudaGMRES();

    /**
     * @brief Destructor
     */
    ~CudaGMRES();

    // Configuration methods
    void setMaxIter(size_t max_iter) { _max_iter = max_iter; }
    void setAbsTol(double abs_tol) { _abs_tol = abs_tol; }
    void setRelTol(double rel_tol) { _rel_tol = rel_tol; }
    void setRestart(size_t restart) { _restart = restart; }
    void setPreconditionerType(PreconditionerType prec_type) { _prec_type = prec_type; }

    /**
     * @brief Setup matrix and preconditioner for subsequent solve operations
     * 
     * This method should be called once to setup the matrix and preconditioner.
     * After setup, multiple solve() calls can reuse the same operators.
     * 
     * @param n Matrix size
     * @param nnz Number of non-zeros in matrix A
     * @param d_ia_A Row pointers for matrix A (size n+1)
     * @param d_ja_A Column indices for matrix A (size nnz)
     * @param d_va_A Values for matrix A (size nnz)
     * @param nnz_L Number of non-zeros in L factor (0 if no preconditioner)
     * @param d_ia_L Row pointers for L factor (nullptr if no preconditioner)
     * @param d_ja_L Column indices for L factor (nullptr if no preconditioner)  
     * @param d_va_L Values for L factor (nullptr if no preconditioner)
     * @param nnz_U Number of non-zeros in U factor (0 if no preconditioner)
     * @param d_ia_U Row pointers for U factor (nullptr if no preconditioner)
     * @param d_ja_U Column indices for U factor (nullptr if no preconditioner)
     * @param d_va_U Values for U factor (nullptr if no preconditioner)
     */
    void setup(size_t n, size_t nnz,
               const int* d_ia_A, const int* d_ja_A, const double* d_va_A,
               size_t nnz_L, const int* d_ia_L, const int* d_ja_L, const double* d_va_L,
               size_t nnz_U, const int* d_ia_U, const int* d_ja_U, const double* d_va_U);

    /**
     * @brief Solve the linear system Ax = b using GMRES
     * 
     * Note: setup() must be called before the first solve() call.
     * Multiple solve() calls can reuse the same matrix/preconditioner setup.
     * 
     * @param d_b Right-hand side vector (device pointer)
     * @param d_x Solution vector (device pointer, input: initial guess, output: solution)
     * @return Convergence state
     */
    State solve(const double* d_b, double* d_x);

private:
    // CUDA handles
    cublasHandle_t _cublas_handle;
    cusparseHandle_t _cusparse_handle;
    cudaStream_t _stream;

    // Matrix descriptors
    cusparseSpMatDescr_t _mat_A;
    cusparseDnVecDescr_t _vec_x, _vec_y, _vec_tmp;
    cusparseSpSVDescr_t _spv_descr_L, _spv_descr_U;  // For triangular solves
    
    // Preconditioner descriptors
    cusparseSpMatDescr_t _mat_prec_L, _mat_prec_U;
    
    // Algorithm parameters
    size_t _max_iter;
    double _abs_tol;
    double _rel_tol;
    size_t _restart;
    PreconditionerType _prec_type;
    
    // Setup state tracking
    bool _is_setup;              // Whether setup() has been called

    // Workspace arrays (device memory)
    double* _d_Q;           // Krylov basis vectors: size n * (_restart + 1)
    double* _d_tmp;         // Temporary vector: size n
    double* _d_w;           // Work vector for SpMV: size n
    
    // Host workspace arrays
    double* _h_g;           // Residual vector for least squares: size _restart
    double* _h_c;           // Cosine values for Givens rotations: size _restart
    double* _h_s;           // Sine values for Givens rotations: size _restart
    double* _h_col_norms;   // Column norms buffer: size _restart
    
    // Unified memory for Hessenberg matrix (accessed by both host and device)
    double* _um_H;          // Hessenberg matrix: size _restart * _restart
    
    // Current problem size
    size_t _n;
    size_t _current_restart;
    
    // Buffer sizes for SpSV operations
    size_t _spv_buffer_size_L, _spv_buffer_size_U;
    void* _d_spv_buffer_L;
    void* _d_spv_buffer_U;
    
    // Buffer for SpMV operations
    size_t _spmv_buffer_size;
    void* _d_spmv_buffer;
    
    /**
     * @brief Initialize CUDA handles and resources
     */
    void initialize_cuda();

    /**
     * @brief Cleanup CUDA resources
     */
    void cleanup_cuda();

    /**
     * @brief Initialize workspace memory for given problem size
     */
    void initialize_workspace(size_t n);

    /**
     * @brief Setup matrix and preconditioner descriptors
     */
    void setup_matrix_descriptors(size_t n, size_t nnz,
                                  const int* d_ia_A, const int* d_ja_A, const double* d_va_A,
                                  size_t nnz_L, const int* d_ia_L, const int* d_ja_L, const double* d_va_L,
                                  size_t nnz_U, const int* d_ia_U, const int* d_ja_U, const double* d_va_U);

    /**
     * @brief Compute initial residual and setup first Krylov vector
     */
    double compute_initial_residual(const double* d_b, const double* d_x);

    /**
     * @brief Apply matrix-vector product with preconditioning
     */
    void apply_operator_with_preconditioning(const double* d_input, double* d_output);

    /**
     * @brief Apply preconditioner (triangular solve using SpSV)
     */
    void apply_preconditioner(const double* d_input, double* d_output);

    /**
     * @brief Perform one GMRES restart cycle
     */
    State perform_restart_cycle(const double* d_b, double* d_x, 
                               double init_resid, size_t& iter, 
                               double& resid, size_t& cycle_iterations);

    /**
     * @brief Arnoldi process: build Krylov subspace and Hessenberg matrix
     */
    double arnoldi_iteration(size_t j);

    /**
     * @brief Apply Givens rotation to Hessenberg matrix and residual vector (host operation)
     */
    void givens_rotation(double beta, size_t j, double& resid);

    /**
     * @brief Solve least squares problem using triangular solve
     */
    void solve_least_squares(size_t j);

    /**
     * @brief Update solution vector
     */
    void update_solution(double* d_x, size_t j);

    /**
     * @brief Check convergence criteria
     */
    bool check_convergence(double resid, double init_resid) const;

    /**
     * @brief Print iteration information
     */
    void print_iteration_info(size_t iter, double resid, double init_resid) const;

    /**
     * @brief Error checking macro for CUDA calls
     */
    void check_cuda_error(cudaError_t error, const char* message);
    void check_cublas_error(cublasStatus_t status, const char* message);
    void check_cusparse_error(cusparseStatus_t status, const char* message);
};

} // namespace cuda_iterative_solver