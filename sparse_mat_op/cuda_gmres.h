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
 * @brief Helper class for managing device arrays with capacity tracking
 */
template<typename T>
class DeviceArray
{
public:
    DeviceArray() : _data(nullptr), _size(0), _capacity(0) {}
    
    ~DeviceArray() { release(); }
    
    void resize(size_t new_size) {
        if (new_size > _capacity) {
            if (_data) {
                cudaFree(_data);
            }
            cudaMalloc(&_data, new_size * sizeof(T));
            _capacity = new_size;
        }
        _size = new_size;
    }
    
    void copyFromHost(const T* host_data, size_t count) {
        resize(count);
        if (count > 0) {
            cudaMemcpy(_data, host_data, count * sizeof(T), cudaMemcpyHostToDevice);
        }
    }
    
    void release() {
        if (_data) {
            cudaFree(_data);
            _data = nullptr;
        }
        _size = 0;
        _capacity = 0;
    }
    
    T* data() { return _data; }
    const T* data() const { return _data; }
    size_t size() const { return _size; }
    size_t capacity() const { return _capacity; }
    
private:
    T* _data;
    size_t _size;
    size_t _capacity;
    
    // Disable copy and assignment
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

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
     * @brief Setup matrix operator for subsequent solve operations
     * 
     * This method should be called to setup the matrix operator.
     * Data is copied from host to device.
     * The number of non-zeros is calculated as ia_A[n] - ia_A[0].
     * The indexing base (0 or 1) is deduced from ia_A[0].
     * 
     * @param n Matrix size
     * @param h_ia_A Row pointers for matrix A (size n+1, host data)
     * @param h_ja_A Column indices for matrix A (host data)
     * @param h_va_A Values for matrix A (host data)
     */
    void setupOperator(size_t n,
                      const int* h_ia_A, const int* h_ja_A, const double* h_va_A);
    
    /**
     * @brief Setup ILU preconditioner for subsequent solve operations
     * 
     * This method should be called to setup the ILU preconditioner.
     * Data is copied from host to device.
     * The number of non-zeros is calculated as ia[n] - ia[0] for each factor.
     * 
     * @param n Matrix size  
     * @param h_ia_L Row pointers for L factor (size n+1, host data)
     * @param h_ja_L Column indices for L factor (host data)
     * @param h_va_L Values for L factor (host data)
     * @param h_ia_U Row pointers for U factor (size n+1, host data)
     * @param h_ja_U Column indices for U factor (host data)
     * @param h_va_U Values for U factor (host data)
     */
    void setupILU(size_t n,
                  const int* h_ia_L, const int* h_ja_L, const double* h_va_L,
                  const int* h_ia_U, const int* h_ja_U, const double* h_va_U);

    /**
     * @brief Solve the linear system Ax = b using GMRES (host interface)
     * 
     * This method provides a host-based interface for users without CUDA experience.
     * Data is automatically copied to/from device as needed.
     * 
     * Note: setupOperator() must be called before the first solve() call.
     * Multiple solve() calls can reuse the same matrix/preconditioner setup.
     * 
     * @param h_b Right-hand side vector (host pointer, size n)
     * @param h_x Solution vector (host pointer, size n, input: initial guess, output: solution)
     * @return Convergence state
     */
    State solve(const double* h_b, double* h_x);

private:
    // CUDA handles
    cublasHandle_t _cublas_handle;
    cusparseHandle_t _cusparse_handle;

    // Matrix descriptors
    cusparseSpMatDescr_t _mat_A;
    cusparseDnVecDescr_t _vec_x, _vec_y, _vec_tmp;
    cusparseSpSVDescr_t _spv_descr_L, _spv_descr_U;  // For triangular solves
    
    // Preconditioner descriptors
    cusparseSpMatDescr_t _mat_prec_L, _mat_prec_U;
    cusparseDnVecDescr_t _vec_prec_x, _vec_prec_y;  // Dedicated vectors for preconditioner
    
    // Algorithm parameters
    size_t _max_iter;
    double _abs_tol;
    double _rel_tol;
    size_t _restart;
    PreconditionerType _prec_type;
    
    // Setup state tracking
    bool _is_operator_setup;     // Whether setupOperator() has been called
    bool _is_ilu_setup;          // Whether setupILU() has been called
    
    // Device storage for matrix data
    DeviceArray<int> _d_ia_A, _d_ja_A;     // Matrix A structure
    DeviceArray<double> _d_va_A;           // Matrix A values
    DeviceArray<int> _d_ia_L, _d_ja_L;     // L factor structure
    DeviceArray<double> _d_va_L;           // L factor values
    DeviceArray<int> _d_ia_U, _d_ja_U;     // U factor structure
    DeviceArray<double> _d_va_U;           // U factor values
    
    // Device storage for RHS and solution vectors
    DeviceArray<double> _d_b;              // Right-hand side vector
    DeviceArray<double> _d_x;              // Solution vector
    
    // Matrix properties
    size_t _matrix_n, _matrix_nnz;
    size_t _ilu_nnz_L, _ilu_nnz_U;
    int _index_base;  // 0-based or 1-based indexing (deduced from ia_A[0])
    int _index_base_L, _index_base_U;  // Index bases for L and U factors

    // Workspace arrays (device memory)
    double* _d_Q;           // Krylov basis vectors: size n * (_restart + 1)
    double* _d_tmp;         // Temporary vector: size n
    double* _d_w;           // Work vector for SpMV: size n
    
    // Host workspace arrays
    double* _h_c;           // Cosine values for Givens rotations: size _restart
    double* _h_s;           // Sine values for Givens rotations: size _restart
    double* _h_col_norms;   // Column norms buffer: size _restart
    
    // Unified memory arrays (accessible from both host and device)
    double* _um_H;          // Hessenberg matrix: size _restart * _restart
    double* _um_g;          // Residual vector for least squares: size _restart
    
    // Current problem size
    size_t _n;
    size_t _current_restart;
    
    // Buffer sizes for SpSV operations
    size_t _spv_buffer_size_L, _spv_buffer_size_U;
    DeviceArray<char> _d_spv_buffer_L;
    DeviceArray<char> _d_spv_buffer_U;
    
    // Buffer for SpMV operations
    DeviceArray<char> _d_spmv_buffer;
    
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
     * @brief Setup matrix A descriptor
     */
    void setup_matrix_descriptor();
    
    /**
     * @brief Setup ILU preconditioner descriptors
     */
    void setup_ilu_descriptors();

    /**
     * @brief Solve the linear system Ax = b using GMRES (device interface)
     * 
     * Internal method that operates on device pointers.
     * 
     * @param d_b Right-hand side vector (device pointer)
     * @param d_x Solution vector (device pointer, input: initial guess, output: solution)
     * @return Convergence state
     */
    State deviceSolve(const double* d_b, double* d_x);
    
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
