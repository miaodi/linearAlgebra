#pragma once

#include "cuda_gmres.h"

namespace cuda_iterative_solver
{

/**
 * @brief CUDA GMRES implementation that keeps Krylov orthogonalization and
 *        Hessenberg updates on the device.
 *
 * The public interface mirrors the host-assisted CudaGMRES class, but the
 * Arnoldi process, Givens rotations, and least-squares solve are performed
 * using device memory and CUDA kernels. Small scalar values are only copied
 * back to the host when convergence checks or logging require them.
 */
class DeviceCudaGMRES
{
public:
    DeviceCudaGMRES();
    ~DeviceCudaGMRES();

    // Configuration
    void setMaxIter(size_t max_iter) { _max_iter = max_iter; }
    void setAbsTol(double abs_tol) { _abs_tol = abs_tol; }
    void setRelTol(double rel_tol) { _rel_tol = rel_tol; }
    void setRestart(size_t restart) { _restart = restart; }
    void setPreconditionerType(PreconditionerType prec_type) { _prec_type = prec_type; }
    void setUseBatchOrthogonalization(bool enable) { _use_batch_orthogonalization = enable; }

    void setupOperator(size_t n,
                       const int* h_ia_A, const int* h_ja_A, const double* h_va_A);

    void setupILU(size_t n,
                  const int* h_ia_L, const int* h_ja_L, const double* h_va_L,
                  const int* h_ia_U, const int* h_ja_U, const double* h_va_U);

    State solve(const double* h_b, double* h_x);

private:
    // CUDA handles
    cublasHandle_t _cublas_handle;
    cusparseHandle_t _cusparse_handle;

    // Matrix descriptors
    cusparseSpMatDescr_t _mat_A;
    cusparseDnVecDescr_t _vec_x, _vec_y, _vec_tmp;
    cusparseDnVecDescr_t _vec_prec_x, _vec_prec_y;
    cusparseSpSVDescr_t _spv_descr_L, _spv_descr_U;

    // Preconditioner matrices
    cusparseSpMatDescr_t _mat_prec_L, _mat_prec_U;

    // Algorithm parameters
    size_t _max_iter;
    double _abs_tol;
    double _rel_tol;
    size_t _restart;
    PreconditionerType _prec_type;
    bool _use_batch_orthogonalization;

    // Setup state
    bool _is_operator_setup;
    bool _is_ilu_setup;

    // Matrix storage (device)
    DeviceArray<int> _d_ia_A, _d_ja_A;
    DeviceArray<double> _d_va_A;
    DeviceArray<int> _d_ia_L, _d_ja_L;
    DeviceArray<double> _d_va_L;
    DeviceArray<int> _d_ia_U, _d_ja_U;
    DeviceArray<double> _d_va_U;

    // RHS / solution storage
    DeviceArray<double> _d_b;
    DeviceArray<double> _d_x;

    // Workspace
    DeviceArray<double> _d_Q;
    DeviceArray<double> _d_tmp;
    DeviceArray<double> _d_w;
    DeviceArray<double> _d_H;
    DeviceArray<double> _d_c;
    DeviceArray<double> _d_s;
    DeviceArray<double> _d_g;
    DeviceArray<double> _d_scalar_workspace;
    DeviceArray<double> _d_residual;

    // Problem metadata
    size_t _matrix_n;
    size_t _matrix_nnz;
    size_t _ilu_nnz_L;
    size_t _ilu_nnz_U;
    int _index_base;
    int _index_base_L;
    int _index_base_U;
    size_t _n;
    size_t _current_restart;

    // cuSPARSE buffers
    size_t _spv_buffer_size_L, _spv_buffer_size_U;
    DeviceArray<char> _d_spv_buffer_L;
    DeviceArray<char> _d_spv_buffer_U;
    size_t _spmv_buffer_size;
    DeviceArray<char> _d_spmv_buffer;

    // Initialization helpers
    void initialize_cuda();
    void cleanup_cuda();
    void initialize_workspace(size_t n);
    void setup_matrix_descriptor();
    void setup_ilu_descriptors();

    // Core algorithm
    State deviceSolve(const double* d_b, double* d_x);
    double compute_initial_residual(const double* d_b, const double* d_x);
    void apply_operator_with_preconditioning(const double* d_input, double* d_output);
    void apply_preconditioner(const double* d_input, double* d_output);
    State perform_restart_cycle( double* d_x,
                                 double init_resid,
                                 size_t& iter,
                                 double& resid,
                                 size_t& cycle_iterations );
    const double* arnoldi_iteration(size_t j);
    const double* run_modified_gram_schmidt(size_t j);
    void apply_givens_rotations(const double* d_beta, size_t j, double& resid);
    void solve_least_squares(size_t j);
    void update_solution(double* d_x, size_t j);
    bool check_convergence(double resid, double init_resid) const;
    void print_iteration_info(size_t iter, double resid, double init_resid) const;

    // Error handling helpers
    void check_cuda_error(cudaError_t error, const char* message);
    void check_cublas_error(cublasStatus_t status, const char* message);
    void check_cusparse_error(cusparseStatus_t status, const char* message);
};

} // namespace cuda_iterative_solver
