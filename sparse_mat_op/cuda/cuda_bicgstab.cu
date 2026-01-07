#include "cuda_bicgstab.h"
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <stdexcept>

namespace cuda_iterative_solver
{

CudaBiCGSTAB::CudaBiCGSTAB()
    : _cublas_handle(nullptr)
    , _cusparse_handle(nullptr)
    , _spmv_operator(nullptr)
    , _preconditioner(&_default_preconditioner)
    , _default_preconditioner()
    , _max_iter(100)
    , _abs_tol(0.0)
    , _rel_tol(1e-8)
    , _prec_type(PreconditionerType::LEFT)
    , _last_iterations(0)
    , _is_operator_setup(false)
    , _n(0)
{
    initialize_cuda();
}

CudaBiCGSTAB::~CudaBiCGSTAB()
{
    cleanup_cuda();
}

void CudaBiCGSTAB::initialize_cuda()
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

void CudaBiCGSTAB::cleanup_cuda()
{
    // Free workspace memory
    _d_r0.release();
    _d_r.release();
    _d_p.release();
    _d_v.release();
    _d_s.release();
    _d_t.release();
    _d_x_hat.release();
    _d_tmp.release();
    
    // Destroy handles
    if (_cusparse_handle) cusparseDestroy(_cusparse_handle);
    if (_cublas_handle) cublasDestroy(_cublas_handle);
}

void CudaBiCGSTAB::initialize_workspace(size_t n)
{
    if (_n == n) {
        return; // Already initialized for this size
    }
    
    _n = n;
    
    // Allocate device memory for BiCGSTAB vectors
    _d_r0.resize(n);  // Reference residual vector
    _d_r.resize(n);   // Current residual
    _d_p.resize(n);   // Search direction
    _d_v.resize(n);   // A * p (with preconditioning)
    _d_s.resize(n);   // Intermediate residual
    _d_t.resize(n);   // A * s (with preconditioning)
    _d_x_hat.resize(n); // Accumulated solution updates
    _d_tmp.resize(n); // Temporary storage
}

void CudaBiCGSTAB::setupOperator(matrix_utils::SpMVOperator<double>* spmv_operator)
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
    
    // Create vector descriptors using DeviceVectorView
    _view_prec_x.create(static_cast<size_t>(_n), nullptr);
    _view_prec_y.create(static_cast<size_t>(_n), nullptr);
    _view_prec_tmp.create(static_cast<size_t>(_n), nullptr);
    
    // Create DeviceVectorView wrappers for RHS and solution vectors
    _view_d_b.create(static_cast<size_t>(_n), nullptr);
    _view_d_x.create(static_cast<size_t>(_n), nullptr);
    
    // Set up DeviceVectorView wrappers to point to allocated memory
    _view_prec_tmp.setData(_d_tmp.data());
    
    _is_operator_setup = true;
}

void CudaBiCGSTAB::setPreconditioner(Preconditioner* preconditioner)
{
    _preconditioner = preconditioner;
}

template<bool ZeroInitialGuess>
State CudaBiCGSTAB::solve(const double* h_b, double* h_x)
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
template State CudaBiCGSTAB::solve<false>(const double* h_b, double* h_x);
template State CudaBiCGSTAB::solve<true>(const double* h_b, double* h_x);

State CudaBiCGSTAB::deviceSolve(const DeviceVectorView& d_b, DeviceVectorView& d_x)
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
    _last_iterations = 0;
    
    // Compute initial residual r = b - Ax (with preconditioning if LEFT)
    double init_resid = compute_initial_residual(d_b, d_x);
    if (init_resid < _abs_tol) {
        return State::CONVERGED;
    }
    
    // Choose arbitrary r_tilde (commonly r_tilde = r0)
    _d_r0.copy<MemoryLocation::Device>(_d_r.data(), _n);
    
    // Initialize p0 = r0
    _d_p.copy<MemoryLocation::Device>(_d_r.data(), _n);
    
    // Initialize _x_hat to zero
    cudaMemset(_d_x_hat.data(), 0, static_cast<size_t>(_n) * sizeof(double));
    
    // Initialize scalars
    double rho, alpha = 1.0, omega = 1.0;
    
    // Compute initial rho = <r_tilde, r>
    check_cublas_error(cublasDdot(_cublas_handle, _n, _d_r0.data(), 1, _d_r.data(), 1, &rho),
                       "Failed to compute initial rho");
    
    // BiCGSTAB main iteration loop
    for (size_t iter = 0; iter < _max_iter; ++iter)
    {
        // step 1: Compute alpha
        // step 1.1: Compute v = A * p (with preconditioning)
        _view_prec_x.setData(_d_p.data());
        _view_prec_y.setData(_d_v.data());
        apply_operator_with_preconditioning(_view_prec_x, _view_prec_y);
        
        // step 1.2: alpha = rho / (r_tilde, v)
        double rtilde_v;
        check_cublas_error(cublasDdot(_cublas_handle, _n, _d_r0.data(), 1, _d_v.data(), 1, &rtilde_v),
                           "Failed to compute <r_tilde, v>");
        
        // if (std::abs(rtilde_v) < 1e-14) {
        //     std::cerr << "BiCGSTAB breakdown: r_tilde_v = " << rtilde_v << std::endl;
        //     _last_iterations = iter;
        //     return State::FAILED;
        // }
        alpha = rho / rtilde_v;
        
        // step 2: Compute s = r - alpha * v
        _d_s.copy<MemoryLocation::Device>(_d_r.data(), _n);
        const double neg_alpha = -alpha;
        check_cublas_error(cublasDaxpy(_cublas_handle, _n, &neg_alpha, _d_v.data(), 1, _d_s.data(), 1),
                           "Failed to compute s");
        
        // step 3: Compute omega
        // step 3.1: Compute t = A * s (with preconditioning)
        _view_prec_x.setData(_d_s.data());
        _view_prec_y.setData(_d_t.data());
        apply_operator_with_preconditioning(_view_prec_x, _view_prec_y);
        
        // step 3.2: omega = (t, s) / (t, t)
        double t_s, t_t;
        check_cublas_error(cublasDdot(_cublas_handle, _n, _d_t.data(), 1, _d_s.data(), 1, &t_s),
                           "Failed to compute <t, s>");
        check_cublas_error(cublasDdot(_cublas_handle, _n, _d_t.data(), 1, _d_t.data(), 1, &t_t),
                           "Failed to compute <t, t>");
        omega = t_s / t_t;
        
        // step 4: Update solution x_hat = x_hat + alpha * p + omega * s
        check_cublas_error(cublasDaxpy(_cublas_handle, _n, &alpha, _d_p.data(), 1, _d_x_hat.data(), 1),
                           "Failed to update x_hat (alpha*p)");
        check_cublas_error(cublasDaxpy(_cublas_handle, _n, &omega, _d_s.data(), 1, _d_x_hat.data(), 1),
                           "Failed to update x_hat (omega*s)");
        
        // step 5: Update residual r = s - omega * t
        _d_r.copy<MemoryLocation::Device>(_d_s.data(), _n);
        const double neg_omega = -omega;
        check_cublas_error(cublasDaxpy(_cublas_handle, _n, &neg_omega, _d_t.data(), 1, _d_r.data(), 1),
                           "Failed to update residual");
        
        // step 6: Check convergence
        double resid;
        check_cublas_error(cublasDnrm2(_cublas_handle, _n, _d_r.data(), 1, &resid),
                           "Failed to compute residual norm");
        
        print_iteration_info(iter, resid, init_resid);
        
        if (check_convergence(resid, init_resid)) {
            _last_iterations = iter + 1;
            update_solution(d_x);
            return State::CONVERGED;
        }
        
        // step 7: Compute beta
        double rho_new;
        check_cublas_error(cublasDdot(_cublas_handle, _n, _d_r0.data(), 1, _d_r.data(), 1, &rho_new),
                           "Failed to compute rho_new");
        
        // if (std::abs(rho_new) < 1e-14) {
        //     std::cerr << "BiCGSTAB breakdown: rho_new = " << rho_new << std::endl;
        //     _last_iterations = iter + 1;
        //     return State::FAILED;
        // }
        
        const double beta = (rho_new / rho) * (alpha / omega);
        rho = rho_new;
        
        // step 8: Update search direction p = r + beta * (p - omega * v)
        // This implements: p = r + beta * p - beta * omega * v
        // First: p = p - omega * v
        check_cublas_error(cublasDaxpy(_cublas_handle, _n, &neg_omega, _d_v.data(), 1, _d_p.data(), 1),
                           "Failed to update p (subtract omega*v)");
        
        // Then: p = r + beta * p
        check_cublas_error(cublasDscal(_cublas_handle, _n, &beta, _d_p.data(), 1),
                           "Failed to scale p by beta");
        const double one = 1.0;
        check_cublas_error(cublasDaxpy(_cublas_handle, _n, &one, _d_r.data(), 1, _d_p.data(), 1),
                           "Failed to update p (add r)");
    }
    
    _last_iterations = _max_iter;
    update_solution(d_x);
    return State::MAX_ITER_REACHED;
}

double CudaBiCGSTAB::compute_initial_residual(const DeviceVectorView& d_b, const DeviceVectorView& d_x)
{
    // Compute r = b - Ax
    // First compute Ax into a temporary vector
    _spmv_operator->operator()(d_x.data(), _d_tmp.data(), 1.0, 0.0);
    
    // Then compute r = b - Ax using axpy: r = b + (-1.0) * Ax
    _d_r.copy<MemoryLocation::Device>(d_b.data(), _n);
    const double neg_one = -1.0;
    check_cublas_error(cublasDaxpy(_cublas_handle, _n, &neg_one, _d_tmp.data(), 1, _d_r.data(), 1),
                       "Failed to compute residual");
    
    // Apply left preconditioning if needed
    if (_prec_type == PreconditionerType::LEFT) {
        _view_prec_x.setData(_d_r.data());
        _view_prec_y.setData(_d_r.data());
        _preconditioner->operator()(_view_prec_x, _view_prec_y);
    }
    
    // Compute norm of residual
    double norm;
    check_cublas_error(cublasDnrm2(_cublas_handle, _n, _d_r.data(), 1, &norm), 
                       "Failed to compute residual norm");
    return norm;
}

void CudaBiCGSTAB::apply_operator_with_preconditioning(const DeviceVectorView& d_input, DeviceVectorView& d_output)
{
    switch (_prec_type) {
    case PreconditionerType::RIGHT:
        // Right preconditioning: A * M^{-1} * input
        _preconditioner->operator()(d_input, _view_prec_tmp);
        _spmv_operator->operator()(_view_prec_tmp.data(), d_output.data(), 1.0, 0.0);
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

void CudaBiCGSTAB::update_solution(DeviceVectorView& d_x)
{
    if (_prec_type == PreconditionerType::RIGHT) {
        // Right preconditioning: x = x + M^{-1} * x_hat
        _view_prec_x.setData(_d_x_hat.data());
        _view_prec_y.setData(_d_tmp.data());
        _preconditioner->operator()(_view_prec_x, _view_prec_y);
        const double one = 1.0;
        check_cublas_error(cublasDaxpy(_cublas_handle, _n, &one, _d_tmp.data(), 1, d_x.data(), 1),
                           "Failed to update solution with preconditioned x_hat");
    } else {
        // Left or no preconditioning: x = x + x_hat
        const double one = 1.0;
        check_cublas_error(cublasDaxpy(_cublas_handle, _n, &one, _d_x_hat.data(), 1, d_x.data(), 1),
                           "Failed to update solution with x_hat");
    }
}

bool CudaBiCGSTAB::check_convergence(double resid, double init_resid) const
{
    return std::abs(resid) < _abs_tol || std::abs(resid) < _rel_tol * init_resid;
}

void CudaBiCGSTAB::print_iteration_info(int iter, double resid, double init_resid) const
{
    std::cout << "iter: " << std::setw(4) << iter 
              << " resid: " << std::scientific << std::setprecision(4) << std::abs(resid)
              << " relative resid: " << std::scientific << std::setprecision(4) << std::abs(resid) / init_resid 
              << std::endl;
}

void CudaBiCGSTAB::check_cuda_error(cudaError_t error, const char* message)
{
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " - " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error(message);
    }
}

void CudaBiCGSTAB::check_cublas_error(cublasStatus_t status, const char* message)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error: " << message << " - Status: " << status << std::endl;
        throw std::runtime_error(message);
    }
}

void CudaBiCGSTAB::check_cusparse_error(cusparseStatus_t status, const char* message)
{
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "cuSPARSE Error: " << message << " - Status: " << status << std::endl;
        throw std::runtime_error(message);
    }
}

} // namespace cuda_iterative_solver

