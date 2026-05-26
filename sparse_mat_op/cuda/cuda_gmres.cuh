#pragma once

#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "cuda_memory.cuh"
#include "cuda_preconditioner.cuh"
#include "cuda_spmv.cuh"
#include "cuda_solver_types.cuh"

namespace matrix_utils::sparse_cuda
{
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
    void setMaxIter( size_t max_iter ) { _max_iter = max_iter; }
    void setAbsTol( double abs_tol ) { _abs_tol = abs_tol; }
    void setRelTol( double rel_tol ) { _rel_tol = rel_tol; }
    void setRestart( size_t restart ) { _restart = restart; }
    void setPreconditionerType( PreconditionerType prec_type ) { _prec_type = prec_type; }
    void setUseBatchOrthogonalization( bool enable ) { _use_batch_orthogonalization = enable; }

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
    void setupOperator( SpMVOperator<double>* spmv_operator );

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
    void setPreconditioner( Preconditioner* preconditioner );

    /**
     * @brief Solve the linear system Ax = b using GMRES (host interface)
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
    template <bool ZeroInitialGuess = false>
    State solve( const double* h_b, double* h_x );

private:
    // CUDA handles
    cublasHandle_t _cublas_handle;
    cusparseHandle_t _cusparse_handle;

    // SpMV operator (not owned by solver)
    SpMVOperator<double>* _spmv_operator;

    // Preconditioner (not owned by solver, except for default NoPreconditioner)
    Preconditioner* _preconditioner;
    NoPreconditioner _default_preconditioner;

    // DeviceVectorView objects for cuSPARSE operations
    DeviceVectorView _view_d_w;
    DeviceVectorView _view_q_j, _view_q_j_plus_1;
    DeviceVectorView _view_d_b, _view_d_x;

    // Algorithm parameters
    size_t _max_iter;
    double _abs_tol;
    double _rel_tol;
    size_t _restart;
    PreconditionerType _prec_type;
    bool _use_batch_orthogonalization;
    int _last_iterations;

    // Setup state
    bool _is_operator_setup;
    int _n;
    int _current_restart;

    // Device memory arrays
    DeviceArray<double> _d_b, _d_x;
    DeviceArray<double> _d_Q, _d_tmp, _d_w;
    DeviceArray<double> _d_g, _d_h_batch;

    // Host memory arrays
    std::vector<double> _h_c, _h_s;
    PinnedArray<double> _h_g, _h_H;

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
    void initialize_workspace( size_t n );

    /**
     * @brief Solve the linear system Ax = b using GMRES (device interface)
     */
    State deviceSolve( const DeviceVectorView& d_b, DeviceVectorView& d_x );

    /**
     * @brief Compute initial residual and setup first Krylov vector
     */
    double compute_initial_residual( const DeviceVectorView& d_b, const DeviceVectorView& d_x );

    /**
     * @brief Apply matrix-vector product with preconditioning
     */
    void apply_operator_with_preconditioning( const DeviceVectorView& d_input, DeviceVectorView& d_output );

    /**
     * @brief Perform one GMRES restart cycle
     */
    State perform_restart_cycle( const DeviceVectorView& d_b, DeviceVectorView& d_x, double init_resid, int& iter, double& resid );

    /**
     * @brief Arnoldi process: build Krylov subspace and Hessenberg matrix
     */
    double arnoldi_iteration( int j );

    /**
     * @brief Batch Modified Gram-Schmidt orthogonalization using GEMV operations
     */
    void batch_gram_schmidt( int j, double* d_q_j_plus_1 );

    /**
     * @brief Modified Gram-Schmidt orthogonalization using individual dot products
     */
    void gram_schmidt( int j, double* d_q_j_plus_1 );

    /**
     * @brief Apply Givens rotation to Hessenberg matrix and residual vector (host operation)
     */
    void givens_rotation( double beta, int j, double& resid );

    /**
     * @brief Solve least squares problem using triangular solve
     */
    void solve_least_squares( int j );

    /**
     * @brief Update solution vector
     */
    void update_solution( DeviceVectorView& d_x, int j );

    /**
     * @brief Check convergence criteria
     */
    bool check_convergence( double resid, double init_resid ) const;

    /**
     * @brief Print iteration information
     */
    void print_iteration_info( int iter, double resid, double init_resid ) const;

    /**
     * @brief Error checking functions for CUDA calls
     */
    void check_cuda_error( cudaError_t error, const char* message );
    void check_cublas_error( cublasStatus_t status, const char* message );
    void check_cusparse_error( cusparseStatus_t status, const char* message );
};

} // namespace matrix_utils::sparse_cuda
