#include "cuda_gmres.cuh"
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <stdexcept>

extern "C" {
void dtrsv_(const char *uplo, const char *trans, const char *diag,
            const int *n, const double *a, const int *lda, double *x,
            const int *incx);
}

namespace matrix_utils::sparse_cuda
{

CudaGMRES::CudaGMRES()
    : _cublas_handle(nullptr)
    , _cusparse_handle(nullptr)
    , _spmv_operator(nullptr)
    , _preconditioner(&_default_preconditioner)
    , _default_preconditioner()
    , _max_iter(100)
    , _abs_tol(0.0)
    , _rel_tol(1e-8)
    , _restart(20)
    , _prec_type(PreconditionerType::LEFT)
    , _use_batch_orthogonalization(false)
    , _last_iterations(0)
    , _is_operator_setup(false)
    , _n(0)
    , _current_restart(0)
{
    initialize_cuda();
}

CudaGMRES::~CudaGMRES()
{
    cleanup_cuda();
}

void CudaGMRES::initialize_cuda()
{
    try {
        // Create cuBLAS handle
        check_cublas_error(cublasCreate(&_cublas_handle), "Failed to create cuBLAS handle");

        // Create cuSPARSE handle
        check_cusparse_error(cusparseCreate(&_cusparse_handle), "Failed to create cuSPARSE handle");
    } catch (...) {
        cleanup_cuda();
        throw;
    }
}

void CudaGMRES::cleanup_cuda()
{
    // Free workspace memory
    _d_Q.release();
    _d_tmp.release();
    _d_w.release();
    _d_h_batch.release();
    _h_H.release();
    _h_g.release();
    _d_g.release();
    
    // Destroy handles
    if (_cusparse_handle) cusparseDestroy(_cusparse_handle);
    if (_cublas_handle) cublasDestroy(_cublas_handle);
}

void CudaGMRES::initialize_workspace(size_t n)
{
    if (_n == n && _current_restart == _restart) {
        return; // Already initialized for this size
    }
    
    _n = n;
    _current_restart = std::min(_restart, n);
    
    // Allocate device memory using DeviceArray
    _d_Q.resize(n * (_current_restart + 1));
    _d_tmp.resize(n);
    _d_w.resize(n);
    
    // Allocate host memory using std::vector
    _h_c.resize(_current_restart);
    _h_s.resize(_current_restart);
    
    // Allocate pinned memory using PinnedArray
    _h_H.resize(_current_restart * _current_restart);
    _h_g.resize(_current_restart);
    
    // Initialize _h_H to zero
    std::fill(_h_H.data(), _h_H.data() + _h_H.size(), 0.0);
    
    // Allocate device memory for GMRES algorithm
    _d_g.resize(_current_restart);
    if (_use_batch_orthogonalization) {
        _d_h_batch.resize(_current_restart);
    }
    
    // Initialize arrays to zero
    std::fill(_h_g.data(), _h_g.data() + _h_g.size(), 0.0);
}

void CudaGMRES::setupOperator(SpMVOperator<double>* spmv_operator)
{
    if (!spmv_operator) {
        throw std::runtime_error("setupOperator: spmv_operator cannot be nullptr");
    }
    
    // Store operator pointer
    _spmv_operator = spmv_operator;
    
    // Get matrix size from operator
    size_t n = _spmv_operator->size();
    
    // Initialize workspace for this problem size
    initialize_workspace(n);
    
    // Create DeviceVectorView wrappers for temporary vectors
    _view_d_w.create(static_cast<size_t>(_n), nullptr);
    
    // Create DeviceVectorView wrappers for Krylov vectors
    _view_q_j.create(static_cast<size_t>(_n), nullptr);
    _view_q_j_plus_1.create(static_cast<size_t>(_n), nullptr);
    
    // Create DeviceVectorView wrappers for RHS and solution vectors
    _view_d_b.create(static_cast<size_t>(_n), nullptr);
    _view_d_x.create(static_cast<size_t>(_n), nullptr);
    
    // Set up DeviceVectorView wrappers to point to allocated memory
    _view_d_w.setData(_d_w.data());
    
    _is_operator_setup = true;
}

void CudaGMRES::setPreconditioner(Preconditioner* preconditioner)
{
    _preconditioner = preconditioner;
}

template<bool ZeroInitialGuess>
State CudaGMRES::solve(const double* h_b, double* h_x)
{
    // Check if setup has been called
    if (!_is_operator_setup) {
        throw std::runtime_error("setupOperator must be called before solve");
    }
    
    // Check if preconditioner is required but not setup
    if (_prec_type != PreconditionerType::NONE && !_preconditioner) {
        throw std::runtime_error("setPreconditioner must be called before solve when using preconditioner");
    }
    
    // Copy host data to device
    _d_b.copy<MemoryLocation::Host>(h_b, static_cast<size_t>(_n));
    
    if constexpr (ZeroInitialGuess) {
        // Initialize device memory to zero instead of copying
        _d_x.resize(static_cast<size_t>(_n)); // Ensure size
        cudaMemset(_d_x.data(), 0, static_cast<size_t>(_n) * sizeof(double));
    } else {
        // Copy initial guess from host
        _d_x.copy<MemoryLocation::Host>(h_x, static_cast<size_t>(_n));
    }
    
    // Set up DeviceVectorView wrappers
    _view_d_b.setData(_d_b.data());
    _view_d_x.setData(_d_x.data());
    
    // Solve on device
    State result = deviceSolve(_view_d_b, _view_d_x);
    
    // Copy solution back to host
    cudaMemcpy(h_x, _d_x.data(), static_cast<size_t>(_n) * sizeof(double), cudaMemcpyDeviceToHost);
    
    return result;
}

// Explicit template instantiations
template State CudaGMRES::solve<false>(const double* h_b, double* h_x);
template State CudaGMRES::solve<true>(const double* h_b, double* h_x);

State CudaGMRES::deviceSolve(const DeviceVectorView& d_b, DeviceVectorView& d_x)
{
    // Check if setup has been called
    if (!_is_operator_setup) {
        throw std::runtime_error("setupOperator must be called before solve");
    }
    
    // Check if preconditioner is required but not setup
    if (_prec_type != PreconditionerType::NONE && !_preconditioner) {
        throw std::runtime_error("setPreconditioner must be called before solve when using preconditioner");
    }
    
    // Initialize iteration counter
    int iter = 0;
    _last_iterations = 0;
    
    // Compute initial residual
    double init_resid = compute_initial_residual(d_b, d_x);
    if (init_resid < _abs_tol) {
        return State::CONVERGED;
    }
    
    double resid = init_resid;
    for (iter = 0; iter < _max_iter; ) {
        State restart_state = perform_restart_cycle(d_b, d_x, init_resid, iter, resid);
        
        if (restart_state == State::CONVERGED) {
            _last_iterations = iter;
            return State::CONVERGED;
        }
        
        if (restart_state == State::MAX_ITER_REACHED) {
            _last_iterations = iter;
            return State::MAX_ITER_REACHED;
        }
        
        if (restart_state != State::RUNNING) {
            break;
        }
        
        // Recompute residual for next restart cycle
        resid = compute_initial_residual(d_b, d_x);
    }
    
    _last_iterations = iter;
    return State::MAX_ITER_REACHED;
}

double CudaGMRES::compute_initial_residual(const DeviceVectorView& d_b, const DeviceVectorView& d_x)
{
    // Copy b to first Krylov vector: Q[:, 0] = b
    _d_Q.copy<MemoryLocation::Device>(d_b.data(), _n);

    // Set up _view_q_j to point to the first Krylov vector
    _view_q_j.setData(_d_Q.data());
    
    // Compute r = b - Ax: Q[:, 0] = Q[:, 0] - A * x = b - A * x
    // First compute Ax into a temporary vector
    _spmv_operator->operator()(d_x.data(), _d_tmp.data(), 1.0, 0.0);
    
    // Then compute r = b - Ax using axpy: r = b + (-1.0) * Ax
    const double neg_one = -1.0;
    check_cublas_error(cublasDaxpy(_cublas_handle, _n, &neg_one, _d_tmp.data(), 1, _d_Q.data(), 1),
                       "Failed to compute residual");

    // Apply left preconditioning if needed
    if (_prec_type == PreconditionerType::LEFT) {
        _preconditioner->operator()(_view_q_j, _view_q_j);
    }
    
    // Compute norm of residual
    double norm;
    check_cublas_error(cublasDnrm2(_cublas_handle, _n, _d_Q.data(), 1, &norm), 
                       "Failed to compute residual norm");
    return norm;
}

void CudaGMRES::apply_operator_with_preconditioning(const DeviceVectorView& d_input, DeviceVectorView& d_output)
{
    switch (_prec_type) {
    case PreconditionerType::RIGHT:
        // Right preconditioning: A * M^{-1} * input
        _preconditioner->operator()(d_input, _view_d_w);
        _spmv_operator->operator()(_view_d_w.data(), d_output.data(), 1.0, 0.0);
        break;
        
    case PreconditionerType::LEFT:
        // Left preconditioning: M^{-1} * A * input
        _spmv_operator->operator()(d_input.data(), d_output.data(), 1.0, 0.0);
        _preconditioner->operator()(d_output, d_output);
        break;
        
    case PreconditionerType::NONE:
        // No preconditioning: A * input
        _spmv_operator->operator()(d_input.data(), d_output.data(), 1.0, 0.0);
        break;
    }
}

State CudaGMRES::perform_restart_cycle(const DeviceVectorView& d_b, DeviceVectorView& d_x,
                                       double init_resid, int& iter,
                                       double& resid)
{
    // Normalize first Krylov vector
    const double inv_resid = 1.0 / resid;
    check_cublas_error(cublasDscal(_cublas_handle, _n, &inv_resid, _d_Q.data(), 1),
                       "Failed to normalize first Krylov vector");
    
    int j = 0;
    bool converged = false;

    while (j < _current_restart && iter < _max_iter)
    {
        double beta = arnoldi_iteration(j);
        givens_rotation(beta, j, resid);

        ++iter;
        ++j;

        print_iteration_info(iter, resid, init_resid);

        if (check_convergence(resid, init_resid))
        {
            converged = true;
            break;
        }
    }
    
    solve_least_squares(j);
    update_solution(d_x, j);
    
    if (converged)
    {
        return State::CONVERGED;
    }
    if (iter >= _max_iter)
    {
        return State::MAX_ITER_REACHED;
    }
    return State::RUNNING;
}

double CudaGMRES::arnoldi_iteration(int j)
{
    double* d_q_j = _d_Q.data() + j * _n;           // Current Krylov vector
    double* d_q_j_plus_1 = _d_Q.data() + (j + 1) * _n;  // Next Krylov vector
    
    // Set up DeviceVectorView wrappers for Krylov vectors
    _view_q_j.setData(d_q_j);
    _view_q_j_plus_1.setData(d_q_j_plus_1);
    
    // Apply operator: q_{j+1} = A * q_j (with preconditioning)
    apply_operator_with_preconditioning(_view_q_j, _view_q_j_plus_1);
    
    // Modified Gram-Schmidt orthogonalization
    if (_use_batch_orthogonalization) {
        batch_gram_schmidt(j, d_q_j_plus_1);
    } else {
        gram_schmidt(j, d_q_j_plus_1);
    }
    
    // Compute norm of q_{j+1}
    double beta;
    check_cublas_error(cublasDnrm2(_cublas_handle, _n, d_q_j_plus_1, 1, &beta),
                       "Failed to compute norm in Arnoldi iteration");
    
    // Normalize q_{j+1}
    if ( beta != 0.0 )
    { // Avoid division by zero
        const double inv_beta = 1.0 / beta;
        check_cublas_error(cublasDscal(_cublas_handle, _n, &inv_beta, d_q_j_plus_1, 1),
                           "Failed to normalize Krylov vector");
    }

    return beta;
}

void CudaGMRES::batch_gram_schmidt(int j, double* d_q_j_plus_1)
{
    int m = _n;
    int k = j + 1;
    double alpha = 1.0;
    double zero = 0.0;
    
    // Compute all dot products in a single GEMV call: h = Q^T * q_{j+1}
    check_cublas_error(
        cublasDgemv(_cublas_handle, CUBLAS_OP_T,
                    m, k,
                    &alpha, _d_Q.data(), m,
                    d_q_j_plus_1, 1,
                    &zero, _d_h_batch.data(), 1),
        "Failed to compute batched dot products in Gram-Schmidt");

    // Copy Hessenberg column from device to host
    check_cuda_error(
        cudaMemcpy(_h_H.data() + j * _current_restart,
                   _d_h_batch.data(), k * sizeof(double),
                   cudaMemcpyDeviceToHost),
        "Failed to copy Hessenberg column to host");

    // Zero out remaining entries in the Hessenberg column
    if (k < _current_restart) {
        double* start = _h_H.data() + j * _current_restart + k;
        double* end = _h_H.data() + (j + 1) * _current_restart;
        std::fill(start, end, 0.0);
    }

    // Update q_{j+1} = q_{j+1} - Q * h
    double neg_one = -1.0;
    double one = 1.0;
    check_cublas_error(
        cublasDgemv(_cublas_handle, CUBLAS_OP_N,
                    m, k,
                    &neg_one, _d_Q.data(), m,
                    _d_h_batch.data(), 1,
                    &one, d_q_j_plus_1, 1),
        "Failed to update vector in batched Gram-Schmidt");
}

void CudaGMRES::gram_schmidt(int j, double* d_q_j_plus_1)
{
    for (int i = 0; i <= j; ++i) {
        double* d_q_i = _d_Q.data() + i * _n;

        // Compute h_{i,j} = <q_i, q_{j+1}> and store directly in Hessenberg matrix
        check_cublas_error(
            cublasDdot(_cublas_handle, _n, d_q_i, 1, d_q_j_plus_1, 1,
                       &_h_H.data()[i + j * _current_restart]),
            "Failed to compute dot product in Gram-Schmidt");

        // q_{j+1} = q_{j+1} - h_{i,j} * q_i
        double neg_h_ij = -_h_H.data()[i + j * _current_restart];
        check_cublas_error(
            cublasDaxpy(_cublas_handle, _n, &neg_h_ij, d_q_i, 1, d_q_j_plus_1, 1),
            "Failed to update vector in Gram-Schmidt");
    }
}

void CudaGMRES::givens_rotation(double beta, int j, double& resid)
{
    // Access Hessenberg matrix column j (host memory)
    double* H_col_j = _h_H.data() + j * _current_restart;
    
    // Apply previous Givens rotations to H_col_j
    #pragma unroll(8)
    for (int i = 0; i < j; i++) {
        double tmp = _h_c[i] * H_col_j[i] - _h_s[i] * H_col_j[i + 1];
        H_col_j[i + 1] = _h_s[i] * H_col_j[i] + _h_c[i] * H_col_j[i + 1];
        H_col_j[i] = tmp;
    }
    
    // Compute new Givens rotation
    double div_r = 1.0 / std::hypot(H_col_j[j], beta);
    _h_c[j] = div_r * H_col_j[j];
    _h_s[j] = -div_r * beta;
    
    // Handle numerical stability
    if (std::abs(_h_s[j]) < 1e-16) {
        _h_c[j] = 1.0;
        _h_s[j] = 0.0;
    }
    
    // Update Hessenberg matrix entry
    H_col_j[j] = _h_c[j] * H_col_j[j] - _h_s[j] * beta;
    
    // Apply Givens rotation to residual vector
    if (j == 0) {
        _h_g.data()[0] = resid;  // Initialize residual vector
    }
    _h_g.data()[j] = _h_c[j] * resid;
    resid *= _h_s[j];
}

void CudaGMRES::solve_least_squares(int j)
{
    if (j == 0) return;
    
    // Solve upper triangular system H * y = g using CPU BLAS DTRSV
    // H is stored in column-major format in host memory
    // DTRSV parameters:
    // - UPLO: U (upper triangular)
    // - TRANS: N (no transpose)
    // - DIAG: N (non-unit diagonal)
    // - N: size of the system (j)
    // - A: Hessenberg matrix (_h_H)
    // - lda: leading dimension (_current_restart)
    // - X: right-hand side and solution vector (_h_g)
    // - incX: increment for X (1)
    const char uplo = 'U';
    const char trans = 'N';
    const char diag = 'N';
    const int incx = 1;

    dtrsv_(&uplo, &trans, &diag, &j, _h_H.data(), &_current_restart,
           _h_g.data(), &incx);
}

void CudaGMRES::update_solution(DeviceVectorView& d_x, int j)
{
    if (j == 0) return;
    
    const double one = 1.0, zero = 0.0;
    
    // Copy g from pinned memory to device for GEMV operation
    _d_g.copy<MemoryLocation::Host>(_h_g.data(), j);
    
    // Compute solution update: delta_x = Q[:, 0:j] * g[0:j]
    check_cublas_error(
        cublasDgemv(_cublas_handle, CUBLAS_OP_N,
                   _n, j,
                   &one, _d_Q.data(), _n,
                   _d_g.data(), 1,
                   &zero, _view_d_w.data(), 1),
        "Failed to compute solution update");
    
    if (_prec_type == PreconditionerType::RIGHT) {
        // For right preconditioning: x = x + M^{-1} * delta_x
        _preconditioner->operator()(_view_d_w, _view_d_w);
        check_cublas_error(cublasDaxpy(_cublas_handle, _n, &one, _d_w.data(), 1, d_x.data(), 1),
                           "Failed to update solution with right preconditioning");
    } else {
        // For left or no preconditioning: x = x + delta_x
        check_cublas_error(cublasDaxpy(_cublas_handle, _n, &one, _view_d_w.data(), 1, d_x.data(), 1),
                           "Failed to update solution");
    }
}

bool CudaGMRES::check_convergence(double resid, double init_resid) const
{
    return std::abs(resid) < _abs_tol || std::abs(resid) < _rel_tol * init_resid;
}

void CudaGMRES::print_iteration_info(int iter, double resid, double init_resid) const
{
    std::cout << "iter: " << std::setw(4) << iter 
              << " resid: " << std::scientific << std::setprecision(4) << std::abs(resid)
              << " relative resid: " << std::scientific << std::setprecision(4) << std::abs(resid) / init_resid 
              << std::endl;
}

void CudaGMRES::check_cuda_error(cudaError_t error, const char* message)
{
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " - " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error(message);
    }
}

void CudaGMRES::check_cublas_error(cublasStatus_t status, const char* message)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error: " << message << " - Status: " << status << std::endl;
        throw std::runtime_error(message);
    }
}

void CudaGMRES::check_cusparse_error(cusparseStatus_t status, const char* message)
{
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "cuSPARSE Error: " << message << " - Status: " << status << std::endl;
        throw std::runtime_error(message);
    }
}

} // namespace matrix_utils::sparse_cuda
