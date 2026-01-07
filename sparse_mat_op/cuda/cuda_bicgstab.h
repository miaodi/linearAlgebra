#pragma once

#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "cuda_memory.h"
#include "cuda_preconditioner.h"
#include "cuda_spmv.h"
#include "cuda_solver_types.h"

namespace cuda_iterative_solver
{

/**
 * @brief CUDA-based BiCGSTAB solver for sparse matrices
 * 
 * This class implements the BiCGSTAB (Biconjugate Gradient Stabilized) iterative solver
 * using CUDA libraries (cuBLAS, cuSPARSE) for solving linear systems Ax = b.
 * 
 * Key features:
 * - Uses SpMV operator interface for matrix-vector products
 * - Self-contained (only depends on cuBLAS/cuSPARSE/std libraries)
 * - Fixed double precision (no templates)
 * - Supports left and right preconditioning
 * - More memory efficient than GMRES (no restart needed)
 * 
 * Algorithm:
 * - BiCGSTAB is a variant of the BiCG method that uses a two-term recurrence
 * - Typically converges faster than GMRES for some problems
 * - Requires 7 vectors: r0, r, p, v, s, t, and temporary storage
 */
class CudaBiCGSTAB
{
public:
    /**
     * @brief Constructor
     */
    CudaBiCGSTAB();

    /**
     * @brief Destructor
     */
    ~CudaBiCGSTAB();

    // Configuration methods
    void setMaxIter(size_t max_iter) { _max_iter = max_iter; }
    void setAbsTol(double abs_tol) { _abs_tol = abs_tol; }
    void setRelTol(double rel_tol) { _rel_tol = rel_tol; }
    void setPreconditionerType(PreconditionerType prec_type) { _prec_type = prec_type; }
    
    /**
     * @brief Get the number of iterations from the last solve
     * @return Number of iterations performed in the last solve call
     */
    int getLastIterations() const { return _last_iterations; }
    
    /**
     * @brief Get the cuSPARSE handle for setting up preconditioners
     * @return cuSPARSE handle used by the solver
     */
    cusparseHandle_t getCusparseHandle() const { return _cusparse_handle; }

    /**
     * @brief Setup matrix operator for subsequent solve operations
     * 
     * This method sets the SpMV operator to be used by the solver.
     * The operator must be already initialized/preprocessed before being passed.
     * The solver does not take ownership - caller is responsible for keeping
     * the operator alive for the duration of solver usage.
     * The matrix size is obtained from the operator itself.
     * 
     * @param spmv_operator Pointer to an SpMV operator object (does not take ownership)
     */
    void setupOperator(matrix_utils::SpMVOperator<double>* spmv_operator);
    
    /**
     * @brief Set preconditioner for subsequent solve operations
     * 
     * The preconditioner must be already setup before passing to this method.
     * The solver does not take ownership - caller is responsible for keeping
     * the preconditioner alive for the duration of solver usage.
     * 
     * By default, NoPreconditioner is used (identity operation).
     * 
     * @param preconditioner Pointer to a preconditioner object (does not take ownership)
     */
    void setPreconditioner(Preconditioner* preconditioner);

    /**
     * @brief Solve the linear system Ax = b using BiCGSTAB (host interface)
     * 
     * This method provides a host-based interface for users without CUDA experience.
     * Data is automatically copied to/from device as needed.
     * 
     * Note: setupOperator() must be called before the first solve() call.
     * Multiple solve() calls can reuse the same matrix/preconditioner setup.
     * 
     * @tparam ZeroInitialGuess If true, assumes h_x is zero and skips copy, directly initializes device memory to zero
     * @param h_b Right-hand side vector (host pointer, size n)
     * @param h_x Solution vector (host pointer, size n, input: initial guess, output: solution)
     * @return Convergence state
     */
    template<bool ZeroInitialGuess = false>
    State solve(const double* h_b, double* h_x);

private:
    // CUDA handles
    cublasHandle_t _cublas_handle;
    cusparseHandle_t _cusparse_handle;

    // SpMV operator (not owned by solver)
    matrix_utils::SpMVOperator<double>* _spmv_operator;
    
    // Preconditioner (not owned by solver, except for default NoPreconditioner)
    Preconditioner* _preconditioner;
    NoPreconditioner _default_preconditioner;
    
    // DeviceVectorView objects for cuSPARSE operations
    DeviceVectorView _view_prec_x, _view_prec_y, _view_prec_tmp;
    DeviceVectorView _view_d_b, _view_d_x;
    
    // Algorithm parameters
    size_t _max_iter;
    double _abs_tol;
    double _rel_tol;
    PreconditionerType _prec_type;
    int _last_iterations;
    
    // Setup state
    bool _is_operator_setup;
    int _n;
    
    // Device memory arrays for BiCGSTAB algorithm
    // r0: initial residual (reference vector)
    // r: current residual
    // p: search direction
    // v: A * p (with preconditioning)
    // s: intermediate residual
    // t: A * s (with preconditioning)
    DeviceArray<double> _d_b, _d_x;
    DeviceArray<double> _d_r0, _d_r, _d_p, _d_v, _d_s, _d_t;
    DeviceArray<double> _d_x_hat; // Accumulated solution updates
    DeviceArray<double> _d_tmp;  // Temporary storage
    
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
     * @brief Solve the linear system Ax = b using BiCGSTAB (device interface)
     */
    State deviceSolve(const DeviceVectorView& d_b, DeviceVectorView& d_x);
    
    /**
     * @brief Compute initial residual r = b - Ax
     */
    double compute_initial_residual(const DeviceVectorView& d_b, const DeviceVectorView& d_x);

    /**
     * @brief Apply matrix-vector product with preconditioning
     */
    void apply_operator_with_preconditioning(const DeviceVectorView& d_input, DeviceVectorView& d_output);

    /**
     * @brief Update solution x = x + (M^{-1})_x_hat
     */
    void update_solution(DeviceVectorView& d_x);

    /**
     * @brief Check convergence criteria
     */
    bool check_convergence(double resid, double init_resid) const;

    /**
     * @brief Print iteration information
     */
    void print_iteration_info(int iter, double resid, double init_resid) const;

    /**
     * @brief Error checking functions for CUDA calls
     */
    void check_cuda_error(cudaError_t error, const char* message);
    void check_cublas_error(cublasStatus_t status, const char* message);
    void check_cusparse_error(cusparseStatus_t status, const char* message);
};

} // namespace cuda_iterative_solver

