#include "cuda_gmres.h"
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <mkl_cblas.h>  // Include MKL CBLAS for CPU BLAS routines

namespace cuda_iterative_solver
{

CudaGMRES::CudaGMRES()
    : _cublas_handle(nullptr)
    , _cusparse_handle(nullptr)
    , _stream(nullptr)
    , _mat_A(nullptr)
    , _vec_x(nullptr)
    , _vec_y(nullptr)
    , _vec_tmp(nullptr)
    , _spv_descr_L(nullptr)
    , _spv_descr_U(nullptr)
    , _mat_prec_L(nullptr)
    , _mat_prec_U(nullptr)
    , _max_iter(100)
    , _abs_tol(0.0)
    , _rel_tol(1e-8)
    , _restart(20)
    , _prec_type(PreconditionerType::LEFT)
    , _is_setup(false)
    , _d_Q(nullptr)
    , _d_tmp(nullptr)
    , _d_w(nullptr)
    , _h_g(nullptr)
    , _h_c(nullptr)
    , _h_s(nullptr)
    , _h_col_norms(nullptr)
    , _um_H(nullptr)
    , _n(0)
    , _current_restart(0)
    , _spv_buffer_size_L(0)
    , _spv_buffer_size_U(0)
    , _d_spv_buffer_L(nullptr)
    , _d_spv_buffer_U(nullptr)
    , _spmv_buffer_size(0)
    , _d_spmv_buffer(nullptr)
{
    initialize_cuda();
}

CudaGMRES::~CudaGMRES()
{
    cleanup_cuda();
}

void CudaGMRES::initialize_cuda()
{
    // Create CUDA stream
    check_cuda_error(cudaStreamCreate(&_stream), "Failed to create CUDA stream");
    
    // Create cuBLAS handle
    check_cublas_error(cublasCreate(&_cublas_handle), "Failed to create cuBLAS handle");
    check_cublas_error(cublasSetStream(_cublas_handle, _stream), "Failed to set cuBLAS stream");
    
    // Create cuSPARSE handle
    check_cusparse_error(cusparseCreate(&_cusparse_handle), "Failed to create cuSPARSE handle");
    check_cusparse_error(cusparseSetStream(_cusparse_handle, _stream), "Failed to set cuSPARSE stream");
}

void CudaGMRES::cleanup_cuda()
{
    // Free workspace memory
    if (_d_Q) { cudaFree(_d_Q); _d_Q = nullptr; }
    if (_d_tmp) { cudaFree(_d_tmp); _d_tmp = nullptr; }
    if (_d_w) { cudaFree(_d_w); _d_w = nullptr; }
    if (_h_g) { cudaFree(_h_g); _h_g = nullptr; }  // Changed from cudaFreeHost to cudaFree for unified memory
    if (_h_c) { cudaFreeHost(_h_c); _h_c = nullptr; }
    if (_h_s) { cudaFreeHost(_h_s); _h_s = nullptr; }
    if (_h_col_norms) { cudaFreeHost(_h_col_norms); _h_col_norms = nullptr; }
    if (_um_H) { cudaFree(_um_H); _um_H = nullptr; }
    if (_d_spv_buffer_L) { cudaFree(_d_spv_buffer_L); _d_spv_buffer_L = nullptr; }
    if (_d_spv_buffer_U) { cudaFree(_d_spv_buffer_U); _d_spv_buffer_U = nullptr; }
    if (_d_spmv_buffer) { cudaFree(_d_spmv_buffer); _d_spmv_buffer = nullptr; }
    
    // Free device constants
    
    // Destroy descriptors
    if (_mat_A) cusparseDestroySpMat(_mat_A);
    if (_vec_x) cusparseDestroyDnVec(_vec_x);
    if (_vec_y) cusparseDestroyDnVec(_vec_y);
    if (_vec_tmp) cusparseDestroyDnVec(_vec_tmp);
    if (_spv_descr_L) cusparseSpSV_destroyDescr(_spv_descr_L);
    if (_spv_descr_U) cusparseSpSV_destroyDescr(_spv_descr_U);
    if (_mat_prec_L) cusparseDestroySpMat(_mat_prec_L);
    if (_mat_prec_U) cusparseDestroySpMat(_mat_prec_U);
    
    // Destroy handles
    if (_cusparse_handle) cusparseDestroy(_cusparse_handle);
    if (_cublas_handle) cublasDestroy(_cublas_handle);
    if (_stream) cudaStreamDestroy(_stream);
}

void CudaGMRES::initialize_workspace(size_t n)
{
    if (_n == n && _current_restart == _restart) {
        return; // Already initialized for this size
    }
    
    // Free existing memory
    if (_d_Q) cudaFree(_d_Q);
    if (_d_tmp) cudaFree(_d_tmp);
    if (_d_w) cudaFree(_d_w);
    if (_h_g) cudaFree(_h_g);  // Changed from cudaFreeHost to cudaFree for unified memory
    if (_h_c) cudaFreeHost(_h_c);
    if (_h_s) cudaFreeHost(_h_s);
    if (_h_col_norms) cudaFreeHost(_h_col_norms);
    if (_um_H) cudaFree(_um_H);
    
    _n = n;
    _current_restart = std::min(_restart, n);
    
    // Allocate device memory
    check_cuda_error(cudaMalloc(&_d_Q, n * (_current_restart + 1) * sizeof(double)), 
                     "Failed to allocate Krylov basis memory");
    check_cuda_error(cudaMalloc(&_d_tmp, n * sizeof(double)), 
                     "Failed to allocate temporary vector memory");
    check_cuda_error(cudaMalloc(&_d_w, n * sizeof(double)), 
                     "Failed to allocate work vector memory");
    
    // Allocate host memory
    check_cuda_error(cudaMallocManaged(&_h_g, _current_restart * sizeof(double)), 
                     "Failed to allocate residual vector unified memory");
    check_cuda_error(cudaMallocHost(&_h_c, _current_restart * sizeof(double)), 
                     "Failed to allocate cosine values memory");
    check_cuda_error(cudaMallocHost(&_h_s, _current_restart * sizeof(double)), 
                     "Failed to allocate sine values memory");
    check_cuda_error(cudaMallocHost(&_h_col_norms, _current_restart * sizeof(double)), 
                     "Failed to allocate column norms memory");
    
    // Allocate unified memory for Hessenberg matrix
    check_cuda_error(cudaMallocManaged(&_um_H, _current_restart * _current_restart * sizeof(double)), 
                     "Failed to allocate Hessenberg matrix unified memory");
    
    // Initialize Hessenberg matrix to zero
    cudaMemset(_um_H, 0, _current_restart * _current_restart * sizeof(double));
}

void CudaGMRES::setup_matrix_descriptors(size_t n, size_t nnz,
                                         const int* d_ia_A, const int* d_ja_A, const double* d_va_A,
                                         size_t nnz_L, const int* d_ia_L, const int* d_ja_L, const double* d_va_L,
                                         size_t nnz_U, const int* d_ia_U, const int* d_ja_U, const double* d_va_U)
{
    // Destroy existing descriptors
    if (_mat_A) cusparseDestroySpMat(_mat_A);
    if (_vec_x) cusparseDestroyDnVec(_vec_x);
    if (_vec_y) cusparseDestroyDnVec(_vec_y);
    if (_vec_tmp) cusparseDestroyDnVec(_vec_tmp);
    const double one = 1.0;

    // Create matrix A descriptor
    check_cusparse_error(
        cusparseCreateCsr(&_mat_A, n, n, nnz,
                         const_cast<int*>(d_ia_A), const_cast<int*>(d_ja_A), const_cast<double*>(d_va_A),
                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
        "Failed to create matrix A descriptor");
    
    // Create vector descriptors (will be updated with actual pointers during solve)
    check_cusparse_error(
        cusparseCreateDnVec(&_vec_x, n, nullptr, CUDA_R_64F),
        "Failed to create vector x descriptor");
    check_cusparse_error(
        cusparseCreateDnVec(&_vec_y, n, nullptr, CUDA_R_64F),
        "Failed to create vector y descriptor");
    check_cusparse_error(
        cusparseCreateDnVec(&_vec_tmp, n, nullptr, CUDA_R_64F),
        "Failed to create temporary vector descriptor");
    
    // Setup ILU preconditioner if provided
    if (_prec_type != PreconditionerType::NONE && nnz_L > 0 && nnz_U > 0) {
        // Destroy existing preconditioner descriptors
        if (_mat_prec_L) cusparseDestroySpMat(_mat_prec_L);
        if (_mat_prec_U) cusparseDestroySpMat(_mat_prec_U);
        if (_spv_descr_L) cusparseSpSV_destroyDescr(_spv_descr_L);
        if (_spv_descr_U) cusparseSpSV_destroyDescr(_spv_descr_U);
        
        // Create L factor (lower triangular with unit diagonal)
        check_cusparse_error(
            cusparseCreateCsr(&_mat_prec_L, n, n, nnz_L,
                             const_cast<int*>(d_ia_L), const_cast<int*>(d_ja_L), const_cast<double*>(d_va_L),
                             CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
            "Failed to create preconditioner L matrix descriptor");
        
        // Set matrix properties for lower triangular
        cusparseFillMode_t lower_fill = CUSPARSE_FILL_MODE_LOWER;
        cusparseDiagType_t unit_diag = CUSPARSE_DIAG_TYPE_UNIT;
        cusparseSpMatSetAttribute(_mat_prec_L, CUSPARSE_SPMAT_FILL_MODE, &lower_fill, sizeof(cusparseFillMode_t));
        cusparseSpMatSetAttribute(_mat_prec_L, CUSPARSE_SPMAT_DIAG_TYPE, &unit_diag, sizeof(cusparseDiagType_t));
        
        // Create U factor (upper triangular with non-unit diagonal)
        check_cusparse_error(
            cusparseCreateCsr(&_mat_prec_U, n, n, nnz_U,
                             const_cast<int*>(d_ia_U), const_cast<int*>(d_ja_U), const_cast<double*>(d_va_U),
                             CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
            "Failed to create preconditioner U matrix descriptor");
        
        // Set matrix properties for upper triangular
        cusparseFillMode_t upper_fill = CUSPARSE_FILL_MODE_UPPER;
        cusparseDiagType_t nonunit_diag = CUSPARSE_DIAG_TYPE_NON_UNIT;
        cusparseSpMatSetAttribute(_mat_prec_U, CUSPARSE_SPMAT_FILL_MODE, &upper_fill, sizeof(cusparseFillMode_t));
        cusparseSpMatSetAttribute(_mat_prec_U, CUSPARSE_SPMAT_DIAG_TYPE, &nonunit_diag, sizeof(cusparseDiagType_t));
        
        // Create SpSV descriptors for triangular solves
        check_cusparse_error(cusparseSpSV_createDescr(&_spv_descr_L), "Failed to create SpSV L descriptor");
        check_cusparse_error(cusparseSpSV_createDescr(&_spv_descr_U), "Failed to create SpSV U descriptor");
        
        check_cusparse_error(
            cusparseSpSV_bufferSize(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &one, _mat_prec_L, _vec_x, _vec_y, CUDA_R_64F,
                                   CUSPARSE_SPSV_ALG_DEFAULT, _spv_descr_L, &_spv_buffer_size_L),
            "Failed to get SpSV L buffer size");
        
        check_cusparse_error(
            cusparseSpSV_bufferSize(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &one, _mat_prec_U, _vec_x, _vec_y, CUDA_R_64F,
                                   CUSPARSE_SPSV_ALG_DEFAULT, _spv_descr_U, &_spv_buffer_size_U),
            "Failed to get SpSV U buffer size");
        
        // Allocate SpSV buffers
        if (_spv_buffer_size_L > 0) {
            check_cuda_error(cudaMalloc(&_d_spv_buffer_L, _spv_buffer_size_L), 
                           "Failed to allocate SpSV L buffer");
        }
        if (_spv_buffer_size_U > 0) {
            check_cuda_error(cudaMalloc(&_d_spv_buffer_U, _spv_buffer_size_U), 
                           "Failed to allocate SpSV U buffer");
        }
        
        // Analyze the sparsity patterns for optimal performance
        check_cusparse_error(
            cusparseSpSV_analysis(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &one, _mat_prec_L, _vec_x, _vec_y, CUDA_R_64F,
                                 CUSPARSE_SPSV_ALG_DEFAULT, _spv_descr_L, _d_spv_buffer_L),
            "Failed to analyze SpSV L pattern");
        
        check_cusparse_error(
            cusparseSpSV_analysis(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &one, _mat_prec_U, _vec_x, _vec_y, CUDA_R_64F,
                                 CUSPARSE_SPSV_ALG_DEFAULT, _spv_descr_U, _d_spv_buffer_U),
            "Failed to analyze SpSV U pattern");
    }
    
    // Setup SpMV buffer
    check_cusparse_error(
        cusparseSpMV_bufferSize(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &one, _mat_A, _vec_x, &one, _vec_y, CUDA_R_64F,
                               CUSPARSE_SPMV_ALG_DEFAULT, &_spmv_buffer_size),
        "Failed to get SpMV buffer size");
    
    if (_d_spmv_buffer) cudaFree(_d_spmv_buffer);
    if (_spmv_buffer_size > 0) {
        check_cuda_error(cudaMalloc(&_d_spmv_buffer, _spmv_buffer_size), 
                       "Failed to allocate SpMV buffer");
    }
}

void CudaGMRES::setup(size_t n, size_t nnz,
                      const int* d_ia_A, const int* d_ja_A, const double* d_va_A,
                      size_t nnz_L, const int* d_ia_L, const int* d_ja_L, const double* d_va_L,
                      size_t nnz_U, const int* d_ia_U, const int* d_ja_U, const double* d_va_U)
{
    // Initialize workspace for this problem size
    initialize_workspace(n);
    
    // Setup matrix and preconditioner descriptors
    setup_matrix_descriptors(n, nnz, d_ia_A, d_ja_A, d_va_A, nnz_L, d_ia_L, d_ja_L, d_va_L, nnz_U, d_ia_U, d_ja_U, d_va_U);
    
    // Mark as setup complete
    _is_setup = true;
}

State CudaGMRES::solve(const double* d_b, double* d_x)
{
    // Check if setup has been called
    if (!_is_setup) {
        std::cerr << "Error: setup() must be called before solve()" << std::endl;
        throw std::runtime_error("GMRES solver not properly initialized. Call setup() first.");
    }
    
    // Compute initial residual
    double init_resid = compute_initial_residual(d_b, d_x);
    if (init_resid < _abs_tol) {
        return State::CONVERGED;
    }
    
    double resid = init_resid;
    for (size_t iter = 0; iter < _max_iter; ) {
        size_t cycle_iterations;
        State restart_state = perform_restart_cycle(d_b, d_x, init_resid, iter, resid, cycle_iterations);
        
        if (restart_state == State::CONVERGED) {
            return State::CONVERGED;
        }
        
        if (restart_state == State::MAX_ITER_REACHED) {
            return State::MAX_ITER_REACHED;
        }
        
        if (restart_state != State::RUNNING) {
            break;
        }
        
        // Recompute residual for next restart cycle
        resid = compute_initial_residual(d_b, d_x);
    }
    
    return State::MAX_ITER_REACHED;
}

double CudaGMRES::compute_initial_residual(const double* d_b, const double* d_x)
{
    const double one = 1.0, neg_one = -1.0;
    
    // Copy b to first Krylov vector: Q[:, 0] = b
    check_cuda_error(cudaMemcpy(_d_Q, d_b, _n * sizeof(double), cudaMemcpyDeviceToDevice),
                     "Failed to copy b to Q");
    
    // Update vector descriptors
    check_cusparse_error(cusparseDnVecSetValues(_vec_x, const_cast<double*>(d_x)), "Failed to set vec_x");
    check_cusparse_error(cusparseDnVecSetValues(_vec_y, _d_Q), "Failed to set vec_y");
    
    // Compute r = b - Ax: Q[:, 0] = Q[:, 0] - A * x = b - A * x
    check_cusparse_error(
        cusparseSpMV(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &neg_one, _mat_A, _vec_x, &one, _vec_y, CUDA_R_64F,
                    CUSPARSE_SPMV_ALG_DEFAULT, _d_spmv_buffer),
        "Failed to compute SpMV for residual");
    
    // Apply left preconditioning if needed
    if (_prec_type == PreconditionerType::LEFT) {
        apply_preconditioner(_d_Q, _d_tmp);
        check_cuda_error(cudaMemcpy(_d_Q, _d_tmp, _n * sizeof(double), cudaMemcpyDeviceToDevice),
                         "Failed to copy preconditioned residual");
    }
    
    // Compute norm of residual
    double norm;
    check_cublas_error(cublasDnrm2(_cublas_handle, _n, _d_Q, 1, &norm), 
                       "Failed to compute residual norm");
    
    return norm;
}

void CudaGMRES::apply_operator_with_preconditioning(const double* d_input, double* d_output)
{
    const double one = 1.0, zero = 0.0;
    
    // Update vector descriptors
    check_cusparse_error(cusparseDnVecSetValues(_vec_x, const_cast<double*>(d_input)), "Failed to set input vector");
    check_cusparse_error(cusparseDnVecSetValues(_vec_y, d_output), "Failed to set output vector");
    
    switch (_prec_type) {
    case PreconditionerType::RIGHT:
        // Right preconditioning: A * M^{-1} * input
        apply_preconditioner(d_input, _d_tmp);
        check_cusparse_error(cusparseDnVecSetValues(_vec_x, _d_tmp), "Failed to set preconditioned input");
        check_cusparse_error(
            cusparseSpMV(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &one, _mat_A, _vec_x, &zero, _vec_y, CUDA_R_64F,
                        CUSPARSE_SPMV_ALG_DEFAULT, _d_spmv_buffer),
            "Failed to compute SpMV with right preconditioning");
        break;
        
    case PreconditionerType::LEFT:
        // Left preconditioning: M^{-1} * A * input
        check_cusparse_error(
            cusparseSpMV(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &one, _mat_A, _vec_x, &zero, _vec_y, CUDA_R_64F,
                        CUSPARSE_SPMV_ALG_DEFAULT, _d_spmv_buffer),
            "Failed to compute SpMV");
        apply_preconditioner(d_output, _d_tmp);
        check_cuda_error(cudaMemcpy(d_output, _d_tmp, _n * sizeof(double), cudaMemcpyDeviceToDevice),
                         "Failed to copy preconditioned result");
        break;
        
    case PreconditionerType::NONE:
        // No preconditioning: A * input
        check_cusparse_error(
            cusparseSpMV(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &one, _mat_A, _vec_x, &zero, _vec_y, CUDA_R_64F,
                        CUSPARSE_SPMV_ALG_DEFAULT, _d_spmv_buffer),
            "Failed to compute SpMV without preconditioning");
        break;
    }
}

void CudaGMRES::apply_preconditioner(const double* d_input, double* d_output)
{
    if (_prec_type == PreconditionerType::NONE || !_spv_descr_L || !_spv_descr_U) {
        // No preconditioner, just copy
        check_cuda_error(cudaMemcpy(d_output, d_input, _n * sizeof(double), cudaMemcpyDeviceToDevice),
                         "Failed to copy input to output");
        return;
    }
    
    const double one = 1.0;
    
    // Apply ILU preconditioning: M^{-1} = U^{-1} * L^{-1}
    // First solve: L * y = input (forward substitution)
    check_cusparse_error(cusparseDnVecSetValues(_vec_x, const_cast<double*>(d_input)), "Failed to set input vector");
    check_cusparse_error(cusparseDnVecSetValues(_vec_y, _d_tmp), "Failed to set intermediate vector");
    
    check_cusparse_error(
        cusparseSpSV_solve(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          &one, _mat_prec_L, _vec_x, _vec_y, CUDA_R_64F,
                          CUSPARSE_SPSV_ALG_DEFAULT, _spv_descr_L),
        "Failed to solve forward substitution (L * y = input)");
    
    // Second solve: U * output = y (backward substitution)
    check_cusparse_error(cusparseDnVecSetValues(_vec_x, _d_tmp), "Failed to set intermediate as input");
    check_cusparse_error(cusparseDnVecSetValues(_vec_y, d_output), "Failed to set output vector");
    
    check_cusparse_error(
        cusparseSpSV_solve(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          &one, _mat_prec_U, _vec_x, _vec_y, CUDA_R_64F,
                          CUSPARSE_SPSV_ALG_DEFAULT, _spv_descr_U),
        "Failed to solve backward substitution (U * output = y)");
}

State CudaGMRES::perform_restart_cycle(const double* d_b, double* d_x,
                                       double init_resid, size_t& iter,
                                       double& resid, size_t& cycle_iterations)
{
    // Normalize first Krylov vector
    const double inv_resid = 1.0 / resid;
    check_cublas_error(cublasDscal(_cublas_handle, _n, &inv_resid, _d_Q, 1),
                       "Failed to normalize first Krylov vector");
    
    size_t j;
    for (j = 0; j < _current_restart && iter < _max_iter; ++j, ++iter) {
        // Arnoldi iteration: generate next Krylov vector
        double beta = arnoldi_iteration(j);
        
        // Apply Givens rotation on host
        givens_rotation(beta, j, resid);
        
        print_iteration_info(iter, resid, init_resid);
        
        // Check convergence
        if (check_convergence(resid, init_resid)) {
            cycle_iterations = j + 1;
            solve_least_squares(j + 1);
            update_solution(d_x, j + 1);
            return State::CONVERGED;
        }
    }
    
    cycle_iterations = j;
    solve_least_squares(j);
    update_solution(d_x, j);
    return (iter >= _max_iter) ? State::MAX_ITER_REACHED : State::RUNNING;
}

double CudaGMRES::arnoldi_iteration(size_t j)
{
    double* q_j = _d_Q + j * _n;           // Current Krylov vector
    double* q_j_plus_1 = _d_Q + (j + 1) * _n;  // Next Krylov vector
    
    // Apply operator: q_{j+1} = A * q_j (with preconditioning)
    apply_operator_with_preconditioning(q_j, q_j_plus_1);
    
    // Modified Gram-Schmidt orthogonalization
    for (size_t i = 0; i <= j; ++i) {
        double* q_i = _d_Q + i * _n;
        
        // Compute h_{i,j} = <q_i, q_{j+1}> and store directly in Hessenberg matrix
        check_cublas_error(cublasDdot(_cublas_handle, _n, q_i, 1, q_j_plus_1, 1, &_um_H[i + j * _current_restart]),
                           "Failed to compute dot product in Gram-Schmidt");
        
        // q_{j+1} = q_{j+1} - h_{i,j} * q_i
        double neg_h_ij = -_um_H[i + j * _current_restart];
        check_cublas_error(cublasDaxpy(_cublas_handle, _n, &neg_h_ij, q_i, 1, q_j_plus_1, 1),
                           "Failed to update vector in Gram-Schmidt");
    }
    
    // Compute norm of q_{j+1}
    double beta;
    check_cublas_error(cublasDnrm2(_cublas_handle, _n, q_j_plus_1, 1, &beta),
                       "Failed to compute norm in Arnoldi iteration");
    
    // Normalize q_{j+1}
    if (beta > 1e-14) {  // Avoid division by zero
        const double inv_beta = 1.0 / beta;
        check_cublas_error(cublasDscal(_cublas_handle, _n, &inv_beta, q_j_plus_1, 1),
                           "Failed to normalize Krylov vector");
    }
    
    return beta;
}

void CudaGMRES::givens_rotation(double beta, size_t j, double& resid)
{
    // Access Hessenberg matrix column j (unified memory)
    double* H_col_j = _um_H + j * _current_restart;
    
    // Apply previous Givens rotations to H_col_j
    for (size_t i = 0; i < j; i++) {
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
        _h_g[0] = resid;  // Initialize residual vector
    }
    _h_g[j] = _h_c[j] * resid;
    resid *= _h_s[j];
}

void CudaGMRES::solve_least_squares(size_t j)
{
    if (j == 0) return;
    
    // Solve upper triangular system H * y = g using CPU BLAS DTRSV
    // H is stored in column-major format in unified memory (accessible from host)
    // DTRSV parameters:
    // - CBLAS_ORDER: CblasColMajor (column-major storage)
    // - CBLAS_UPLO: CblasUpper (upper triangular)
    // - CBLAS_TRANSPOSE: CblasNoTrans (no transpose)
    // - CBLAS_DIAG: CblasNonUnit (non-unit diagonal)
    // - N: size of the system (j)
    // - A: Hessenberg matrix (_um_H)
    // - lda: leading dimension (_current_restart)
    // - X: right-hand side and solution vector (_h_g)
    // - incX: increment for X (1)
    
    cblas_dtrsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                static_cast<int>(j), _um_H, static_cast<int>(_current_restart), _h_g, 1);
}

void CudaGMRES::update_solution(double* d_x, size_t j)
{
    if (j == 0) return;
    
    const double one = 1.0, zero = 0.0;
    
    // Compute solution update: delta_x = Q[:, 0:j] * g[0:j]
    // Use GEMV: y = alpha * A * x + beta * y
    // Here: _d_tmp = 1.0 * Q[:, 0:j] * g[0:j] + 0.0 * _d_tmp
    
    // Ensure unified memory is synchronized before use
    check_cuda_error(cudaDeviceSynchronize(), "Failed to synchronize before GEMV");
    
    check_cublas_error(
        cublasDgemv(_cublas_handle, CUBLAS_OP_N,
                   _n, j,
                   &one, _d_Q, _n,
                   _h_g, 1,
                   &zero, _d_tmp, 1),
        "Failed to compute solution update");
    
    if (_prec_type == PreconditionerType::RIGHT) {
        // For right preconditioning: x = x + M^{-1} * delta_x
        apply_preconditioner(_d_tmp, _d_w);
        check_cublas_error(cublasDaxpy(_cublas_handle, _n, &one, _d_w, 1, d_x, 1),
                           "Failed to update solution with right preconditioning");
    } else {
        // For left or no preconditioning: x = x + delta_x
        check_cublas_error(cublasDaxpy(_cublas_handle, _n, &one, _d_tmp, 1, d_x, 1),
                           "Failed to update solution");
    }
}

bool CudaGMRES::check_convergence(double resid, double init_resid) const
{
    return std::abs(resid) < _abs_tol || std::abs(resid) < _rel_tol * init_resid;
}

void CudaGMRES::print_iteration_info(size_t iter, double resid, double init_resid) const
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

} // namespace cuda_iterative_solver