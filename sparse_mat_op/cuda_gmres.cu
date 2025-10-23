#include "cuda_gmres.h"
#include <algorithm>
#include <iomanip>
#include <stdexcept>
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
    , _vec_prec_x(nullptr)
    , _vec_prec_y(nullptr)
    , _spv_descr_L(nullptr)
    , _spv_descr_U(nullptr)
    , _mat_prec_L(nullptr)
    , _mat_prec_U(nullptr)
    , _max_iter(100)
    , _abs_tol(0.0)
    , _rel_tol(1e-8)
    , _restart(20)
    , _prec_type(PreconditionerType::LEFT)
    , _is_operator_setup(false)
    , _is_ilu_setup(false)
    , _matrix_n(0)
    , _matrix_nnz(0)
    , _ilu_nnz_L(0)
    , _ilu_nnz_U(0)
    , _index_base(0)
    , _index_base_L(0)
    , _index_base_U(0)
    , _d_Q(nullptr)
    , _d_tmp(nullptr)
    , _d_w(nullptr)
    , _h_c(nullptr)
    , _h_s(nullptr)
    , _um_H(nullptr)
    , _um_g(nullptr)
    , _um_norm(nullptr)
    , _n(0)
    , _current_restart(0)
    , _spv_buffer_size_L(0)
    , _spv_buffer_size_U(0)
    , _d_spv_buffer_L(nullptr)
    , _d_spv_buffer_U(nullptr)
    , _spmv_buffer_size(0)
    , _d_spmv_buffer(nullptr)
    , _d_scalar_buffer(nullptr)
    , _d_neg_one(nullptr)
    , _d_one(nullptr)
    , _d_zero(nullptr)
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
    if (_h_c) { cudaFreeHost(_h_c); _h_c = nullptr; }
    if (_h_s) { cudaFreeHost(_h_s); _h_s = nullptr; }
    if (_um_H) { cudaFree(_um_H); _um_H = nullptr; }
    if (_um_g) { cudaFree(_um_g); _um_g = nullptr; }
    if (_um_norm) { cudaFree(_um_norm); _um_norm = nullptr; }
    if (_d_spv_buffer_L) { cudaFree(_d_spv_buffer_L); _d_spv_buffer_L = nullptr; }
    if (_d_spv_buffer_U) { cudaFree(_d_spv_buffer_U); _d_spv_buffer_U = nullptr; }
    if (_d_spmv_buffer) { cudaFree(_d_spmv_buffer); _d_spmv_buffer = nullptr; }
    if (_d_scalar_buffer) { cudaFree(_d_scalar_buffer); _d_scalar_buffer = nullptr; }
    if (_d_neg_one) { cudaFree(_d_neg_one); _d_neg_one = nullptr; }
    if (_d_one) { cudaFree(_d_one); _d_one = nullptr; }
    if (_d_zero) { cudaFree(_d_zero); _d_zero = nullptr; }
    
    // Destroy descriptors
    if (_mat_A) cusparseDestroySpMat(_mat_A);
    if (_vec_x) cusparseDestroyDnVec(_vec_x);
    if (_vec_y) cusparseDestroyDnVec(_vec_y);
    if (_vec_tmp) cusparseDestroyDnVec(_vec_tmp);
    if (_vec_prec_x) cusparseDestroyDnVec(_vec_prec_x);
    if (_vec_prec_y) cusparseDestroyDnVec(_vec_prec_y);
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
    if (_h_c) cudaFreeHost(_h_c);
    if (_h_s) cudaFreeHost(_h_s);
    if (_um_H) cudaFree(_um_H);
    if (_um_g) cudaFree(_um_g);
    if (_um_norm) cudaFree(_um_norm);
    
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
    check_cuda_error(cudaMallocHost(&_h_c, _current_restart * sizeof(double)), 
                     "Failed to allocate cosine values memory");
    check_cuda_error(cudaMallocHost(&_h_s, _current_restart * sizeof(double)), 
                     "Failed to allocate sine values memory");
    
    // Allocate unified memory for matrices/vectors accessible from both host and device
    check_cuda_error(cudaMallocManaged(&_um_H, _current_restart * _current_restart * sizeof(double)), 
                     "Failed to allocate Hessenberg matrix unified memory");
    check_cuda_error(cudaMallocManaged(&_um_g, _current_restart * sizeof(double)), 
                     "Failed to allocate residual vector unified memory");
    check_cuda_error(cudaMallocManaged(&_um_norm, sizeof(double)), 
                     "Failed to allocate norm unified memory");
    
    // Allocate device scalar buffer for device-only operations
    if (!_d_scalar_buffer) {
        check_cuda_error(cudaMalloc(&_d_scalar_buffer, sizeof(double)), 
                         "Failed to allocate device scalar buffer");
    }
    
    // Allocate and initialize device constants
    if (!_d_neg_one) {
        check_cuda_error(cudaMalloc(&_d_neg_one, sizeof(double)), 
                         "Failed to allocate device negative one constant");
        const double neg_one_host = -1.0;
        check_cuda_error(cudaMemcpy(_d_neg_one, &neg_one_host, sizeof(double), cudaMemcpyHostToDevice),
                         "Failed to initialize device negative one constant");
    }
    
    if (!_d_one) {
        check_cuda_error(cudaMalloc(&_d_one, sizeof(double)), 
                         "Failed to allocate device one constant");
        const double one_host = 1.0;
        check_cuda_error(cudaMemcpy(_d_one, &one_host, sizeof(double), cudaMemcpyHostToDevice),
                         "Failed to initialize device one constant");
    }
    
    if (!_d_zero) {
        check_cuda_error(cudaMalloc(&_d_zero, sizeof(double)), 
                         "Failed to allocate device zero constant");
        const double zero_host = 0.0;
        check_cuda_error(cudaMemcpy(_d_zero, &zero_host, sizeof(double), cudaMemcpyHostToDevice),
                         "Failed to initialize device zero constant");
    }
    
    // Initialize unified memory to zero
    cudaMemset(_um_H, 0, _current_restart * _current_restart * sizeof(double));
    cudaMemset(_um_g, 0, _current_restart * sizeof(double));
}

void CudaGMRES::setupOperator(size_t n,
                              const int* h_ia_A, const int* h_ja_A, const double* h_va_A)
{
    // Deduce indexing base from first row pointer
    _index_base = h_ia_A[0];
    
    // Calculate number of non-zeros
    size_t nnz = h_ia_A[n] - h_ia_A[0];
    
    // Store matrix properties
    _matrix_n = n;
    _matrix_nnz = nnz;
    
    // Copy matrix data from host to device
    _d_ia_A.copyFromHost(h_ia_A, n + 1);
    _d_ja_A.copyFromHost(h_ja_A, nnz);
    _d_va_A.copyFromHost(h_va_A, nnz);
    
    // Initialize workspace for this problem size
    initialize_workspace(n);
    
    // Setup matrix descriptor
    setup_matrix_descriptor();
    
    _is_operator_setup = true;
}

void CudaGMRES::setupILU(size_t n,
                         const int* h_ia_L, const int* h_ja_L, const double* h_va_L,
                         const int* h_ia_U, const int* h_ja_U, const double* h_va_U)
{
    if (!_is_operator_setup || _matrix_n != n) {
        throw std::runtime_error("setupOperator must be called first with the same matrix size");
    }
    
    // Calculate number of non-zeros for L and U factors and deduce index bases
    size_t nnz_L = 0;
    size_t nnz_U = 0;
    
    if (h_ia_L != nullptr) {
        _index_base_L = h_ia_L[0];
        nnz_L = h_ia_L[n] - h_ia_L[0];
    }
    
    if (h_ia_U != nullptr) {
        _index_base_U = h_ia_U[0];
        nnz_U = h_ia_U[n] - h_ia_U[0];
    }
    
    // Store ILU properties
    _ilu_nnz_L = nnz_L;
    _ilu_nnz_U = nnz_U;
    
    // Copy ILU data from host to device
    if (nnz_L > 0) {
        _d_ia_L.copyFromHost(h_ia_L, n + 1);
        _d_ja_L.copyFromHost(h_ja_L, nnz_L);
        _d_va_L.copyFromHost(h_va_L, nnz_L);
    }
    
    if (nnz_U > 0) {
        _d_ia_U.copyFromHost(h_ia_U, n + 1);
        _d_ja_U.copyFromHost(h_ja_U, nnz_U);
        _d_va_U.copyFromHost(h_va_U, nnz_U);
    }
    
    // Setup ILU descriptors
    setup_ilu_descriptors();
    
    _is_ilu_setup = true;
}

void CudaGMRES::setup_matrix_descriptor()
{
    // Destroy existing descriptors
    if (_mat_A) cusparseDestroySpMat(_mat_A);
    if (_vec_x) cusparseDestroyDnVec(_vec_x);
    if (_vec_y) cusparseDestroyDnVec(_vec_y);
    if (_vec_tmp) cusparseDestroyDnVec(_vec_tmp);
    if (_vec_prec_x) cusparseDestroyDnVec(_vec_prec_x);
    if (_vec_prec_y) cusparseDestroyDnVec(_vec_prec_y);
    
    const double one = 1.0;

    // Create matrix A descriptor
    cusparseIndexBase_t index_base = (_index_base == 0) ? CUSPARSE_INDEX_BASE_ZERO : CUSPARSE_INDEX_BASE_ONE;
    check_cusparse_error(
        cusparseCreateCsr(&_mat_A, _matrix_n, _matrix_n, _matrix_nnz,
                         _d_ia_A.data(), _d_ja_A.data(), _d_va_A.data(),
                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, index_base, CUDA_R_64F),
        "Failed to create matrix A descriptor");
    
    // Create vector descriptors (will be updated with actual pointers during solve)
    check_cusparse_error(
        cusparseCreateDnVec(&_vec_x, _matrix_n, nullptr, CUDA_R_64F),
        "Failed to create vector x descriptor");
    check_cusparse_error(
        cusparseCreateDnVec(&_vec_y, _matrix_n, nullptr, CUDA_R_64F),
        "Failed to create vector y descriptor");
    check_cusparse_error(
        cusparseCreateDnVec(&_vec_tmp, _matrix_n, nullptr, CUDA_R_64F),
        "Failed to create temporary vector descriptor");
    check_cusparse_error(
        cusparseCreateDnVec(&_vec_prec_x, _matrix_n, nullptr, CUDA_R_64F),
        "Failed to create preconditioner x vector descriptor");
    check_cusparse_error(
        cusparseCreateDnVec(&_vec_prec_y, _matrix_n, nullptr, CUDA_R_64F),
        "Failed to create preconditioner y vector descriptor");
    
    // Setup SpMV buffer (use host constants for setup phase)
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
    
    // Preprocess SpMV for optimal performance (use host constants for setup phase)
    check_cusparse_error(
        cusparseSpMV_preprocess(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &one, _mat_A, _vec_x, &one, _vec_y, CUDA_R_64F,
                               CUSPARSE_SPMV_ALG_DEFAULT, _d_spmv_buffer),
        "Failed to preprocess SpMV");
}

void CudaGMRES::setup_ilu_descriptors()
{
    if (_prec_type == PreconditionerType::NONE || _ilu_nnz_L == 0 || _ilu_nnz_U == 0) {
        return; // No preconditioner setup needed
    }
    
    // Destroy existing preconditioner descriptors
    if (_mat_prec_L) cusparseDestroySpMat(_mat_prec_L);
    if (_mat_prec_U) cusparseDestroySpMat(_mat_prec_U);
    if (_spv_descr_L) cusparseSpSV_destroyDescr(_spv_descr_L);
    if (_spv_descr_U) cusparseSpSV_destroyDescr(_spv_descr_U);
    
    const double one = 1.0;
    
    // Create L factor (lower triangular with unit diagonal)
    cusparseIndexBase_t index_base_L = (_index_base_L == 0) ? CUSPARSE_INDEX_BASE_ZERO : CUSPARSE_INDEX_BASE_ONE;
    check_cusparse_error(
        cusparseCreateCsr(&_mat_prec_L, _matrix_n, _matrix_n, _ilu_nnz_L,
                         _d_ia_L.data(), _d_ja_L.data(), _d_va_L.data(),
                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, index_base_L, CUDA_R_64F),
        "Failed to create preconditioner L matrix descriptor");
    
    // Set matrix properties for lower triangular
    cusparseFillMode_t lower_fill = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t unit_diag = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseSpMatSetAttribute(_mat_prec_L, CUSPARSE_SPMAT_FILL_MODE, &lower_fill, sizeof(cusparseFillMode_t));
    cusparseSpMatSetAttribute(_mat_prec_L, CUSPARSE_SPMAT_DIAG_TYPE, &unit_diag, sizeof(cusparseDiagType_t));
    
    // Create U factor (upper triangular with non-unit diagonal)
    cusparseIndexBase_t index_base_U = (_index_base_U == 0) ? CUSPARSE_INDEX_BASE_ZERO : CUSPARSE_INDEX_BASE_ONE;
    check_cusparse_error(
        cusparseCreateCsr(&_mat_prec_U, _matrix_n, _matrix_n, _ilu_nnz_U,
                         _d_ia_U.data(), _d_ja_U.data(), _d_va_U.data(),
                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, index_base_U, CUDA_R_64F),
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
                               &one, _mat_prec_L, _vec_prec_x, _vec_prec_y, CUDA_R_64F,
                               CUSPARSE_SPSV_ALG_DEFAULT, _spv_descr_L, &_spv_buffer_size_L),
        "Failed to get SpSV L buffer size");
    
    check_cusparse_error(
        cusparseSpSV_bufferSize(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &one, _mat_prec_U, _vec_prec_x, _vec_prec_y, CUDA_R_64F,
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
                             &one, _mat_prec_L, _vec_prec_x, _vec_prec_y, CUDA_R_64F,
                             CUSPARSE_SPSV_ALG_DEFAULT, _spv_descr_L, _d_spv_buffer_L),
        "Failed to analyze SpSV L pattern");
    
    check_cusparse_error(
        cusparseSpSV_analysis(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             &one, _mat_prec_U, _vec_prec_x, _vec_prec_y, CUDA_R_64F,
                             CUSPARSE_SPSV_ALG_DEFAULT, _spv_descr_U, _d_spv_buffer_U),
        "Failed to analyze SpSV U pattern");
}



State CudaGMRES::solve(const double* h_b, double* h_x)
{
    // Check if setup has been called
    if (!_is_operator_setup) {
        throw std::runtime_error("setupOperator must be called before solve");
    }
    
    // Check if ILU is required but not setup
    if (_prec_type != PreconditionerType::NONE && !_is_ilu_setup) {
        throw std::runtime_error("setupILU must be called before solve when using preconditioner");
    }
    
    // Copy host data to device
    _d_b.copyFromHost(h_b, _matrix_n);
    _d_x.copyFromHost(h_x, _matrix_n);
    
    // Solve on device
    State result = deviceSolve(_d_b.data(), _d_x.data());
    
    // Copy solution back to host
    cudaMemcpy(h_x, _d_x.data(), _matrix_n * sizeof(double), cudaMemcpyDeviceToHost);
    
    return result;
}

State CudaGMRES::deviceSolve(const double* d_b, double* d_x)
{
    // Check if setup has been called
    if (!_is_operator_setup) {
        throw std::runtime_error("setupOperator must be called before solve");
    }
    
    // Check if ILU is required but not setup
    if (_prec_type != PreconditionerType::NONE && !_is_ilu_setup) {
        throw std::runtime_error("setupILU must be called before solve when using preconditioner");
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
    // Set cuBLAS to device pointer mode for device-only operations
    cublasSetPointerMode(_cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
    
    // Copy b to first Krylov vector: Q[:, 0] = b
    check_cuda_error(cudaMemcpy(_d_Q, d_b, _n * sizeof(double), cudaMemcpyDeviceToDevice),
                     "Failed to copy b to Q");
    
    // Update vector descriptors
    check_cusparse_error(cusparseDnVecSetValues(_vec_x, const_cast<double*>(d_x)), "Failed to set vec_x");
    check_cusparse_error(cusparseDnVecSetValues(_vec_y, _d_Q), "Failed to set vec_y");
    
    // Set cuSPARSE to device pointer mode for SpMV operations
    cusparseSetPointerMode(_cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);
    
    // Compute r = b - Ax: Q[:, 0] = Q[:, 0] - A * x = b - A * x
    check_cusparse_error(
        cusparseSpMV(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    _d_neg_one, _mat_A, _vec_x, _d_one, _vec_y, CUDA_R_64F,
                    CUSPARSE_SPMV_ALG_DEFAULT, _d_spmv_buffer),
        "Failed to compute SpMV for residual");

    // Reset cuSPARSE to host pointer mode
    cusparseSetPointerMode(_cusparse_handle, CUSPARSE_POINTER_MODE_HOST);

    // Apply left preconditioning if needed
    if (_prec_type == PreconditionerType::LEFT) {
        apply_preconditioner( _d_Q, _d_tmp, true );
        check_cuda_error(cudaMemcpy(_d_Q, _d_tmp, _n * sizeof(double), cudaMemcpyDeviceToDevice),
                         "Failed to copy preconditioned residual");
    }
    
    // Compute norm of residual using device pointer mode
    check_cublas_error(cublasDnrm2(_cublas_handle, _n, _d_Q, 1, _um_norm), 
                       "Failed to compute residual norm");
    
    // Synchronize only the stream to access norm from host
    check_cuda_error(cudaStreamSynchronize(_stream), "Failed to synchronize stream before accessing norm");
    
    // Reset cuBLAS to host pointer mode
    cublasSetPointerMode(_cublas_handle, CUBLAS_POINTER_MODE_HOST);
    
    return *_um_norm;
}

void CudaGMRES::apply_operator_with_preconditioning(const double* d_input, double* d_output)
{
    // Set cuSPARSE to device pointer mode for fully device-resident operations
    cusparseSetPointerMode(_cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);
    
    // Update vector descriptors
    check_cusparse_error(cusparseDnVecSetValues(_vec_x, const_cast<double*>(d_input)), "Failed to set input vector");
    check_cusparse_error(cusparseDnVecSetValues(_vec_y, d_output), "Failed to set output vector");
    
    switch (_prec_type) {
    case PreconditionerType::RIGHT:
        // Right preconditioning: A * M^{-1} * input
        apply_preconditioner(d_input, _d_tmp, false);
        check_cusparse_error(cusparseDnVecSetValues(_vec_x, _d_tmp), "Failed to set preconditioned input");
        check_cusparse_error(
            cusparseSpMV(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        _d_one, _mat_A, _vec_x, _d_zero, _vec_y, CUDA_R_64F,
                        CUSPARSE_SPMV_ALG_DEFAULT, _d_spmv_buffer),
            "Failed to compute SpMV with right preconditioning");
        break;
        
    case PreconditionerType::LEFT:
        // Left preconditioning: M^{-1} * A * input
        check_cusparse_error(
            cusparseSpMV(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        _d_one, _mat_A, _vec_x, _d_zero, _vec_y, CUDA_R_64F,
                        CUSPARSE_SPMV_ALG_DEFAULT, _d_spmv_buffer),
            "Failed to compute SpMV");
        apply_preconditioner(d_output, _d_tmp, false);
        check_cuda_error(cudaMemcpy(d_output, _d_tmp, _n * sizeof(double), cudaMemcpyDeviceToDevice),
                         "Failed to copy preconditioned result");
        break;
        
    case PreconditionerType::NONE:
        // No preconditioning: A * input
        check_cusparse_error(
            cusparseSpMV(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        _d_one, _mat_A, _vec_x, _d_zero, _vec_y, CUDA_R_64F,
                        CUSPARSE_SPMV_ALG_DEFAULT, _d_spmv_buffer),
            "Failed to compute SpMV without preconditioning");
        break;
    }
    
    // Reset cuSPARSE to host pointer mode for other operations
    cusparseSetPointerMode(_cusparse_handle, CUSPARSE_POINTER_MODE_HOST);
}

void CudaGMRES::apply_preconditioner(const double* d_input, double* d_output, bool manage_pointer_mode)
{
    // Set cuSPARSE to device pointer mode if requested
    if (manage_pointer_mode) {
        cusparseSetPointerMode(_cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);
    }
    
    if (_prec_type == PreconditionerType::NONE || !_spv_descr_L || !_spv_descr_U) {
        // No preconditioner, just copy
        check_cuda_error(cudaMemcpy(d_output, d_input, _n * sizeof(double), cudaMemcpyDeviceToDevice),
                         "Failed to copy input to output");
        if (manage_pointer_mode) {
            cusparseSetPointerMode(_cusparse_handle, CUSPARSE_POINTER_MODE_HOST);
        }
        return;
    }

    // Apply ILU preconditioning: M^{-1} = U^{-1} * L^{-1}
    // First solve: L * y = input (forward substitution)
    check_cusparse_error(cusparseDnVecSetValues(_vec_prec_x, const_cast<double*>(d_input)), "Failed to set preconditioner input vector");
    check_cusparse_error(cusparseDnVecSetValues(_vec_prec_y, _d_tmp), "Failed to set preconditioner intermediate vector");

    check_cusparse_error(
        cusparseSpSV_solve(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          _d_one, _mat_prec_L, _vec_prec_x, _vec_prec_y, CUDA_R_64F,
                          CUSPARSE_SPSV_ALG_DEFAULT, _spv_descr_L),
        "Failed to solve forward substitution (L * y = input)");
    
    // Second solve: U * output = y (backward substitution)
    check_cusparse_error(cusparseDnVecSetValues(_vec_prec_x, _d_tmp), "Failed to set preconditioner intermediate as input");
    check_cusparse_error(cusparseDnVecSetValues(_vec_prec_y, d_output), "Failed to set preconditioner output vector");
    
    check_cusparse_error(
        cusparseSpSV_solve(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          _d_one, _mat_prec_U, _vec_prec_x, _vec_prec_y, CUDA_R_64F,
                          CUSPARSE_SPSV_ALG_DEFAULT, _spv_descr_U),
        "Failed to solve backward substitution (U * output = y)");
    
    // Reset cuSPARSE to host pointer mode if requested
    if (manage_pointer_mode) {
        cusparseSetPointerMode(_cusparse_handle, CUSPARSE_POINTER_MODE_HOST);
    }
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

    double* d_q_j = _d_Q + j * _n;           // Current Krylov vector
    double* d_q_j_plus_1 = _d_Q + (j + 1) * _n;  // Next Krylov vector
    
    // Apply operator: q_{j+1} = A * q_j (with preconditioning)
    apply_operator_with_preconditioning(d_q_j, d_q_j_plus_1);
    
    // Set cuBLAS to device pointer mode for device-only Gram-Schmidt
    cublasSetPointerMode(_cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
    
    // Modified Gram-Schmidt orthogonalization - device-only version
    for (size_t i = 0; i <= j; ++i) {
        double* d_q_i = _d_Q + i * _n;
        
        // Compute h_{i,j} = <q_i, q_{j+1}> and store on device
        double* d_h_ij = &_um_H[i + j * _current_restart];
        check_cublas_error(cublasDdot(_cublas_handle, _n, d_q_i, 1, d_q_j_plus_1, 1, d_h_ij),
                           "Failed to compute dot product in Gram-Schmidt");
        
        // q_{j+1} = q_{j+1} - h_{i,j} * q_i using device-only operations
        // Copy h_{i,j} to device scalar buffer and negate it
        check_cuda_error(cudaMemcpy(_d_scalar_buffer, d_h_ij, sizeof(double), cudaMemcpyDeviceToDevice),
                         "Failed to copy coefficient to scalar buffer");
        
        // Negate the coefficient using cuBLAS scal with device -1 constant
        check_cublas_error(cublasDscal(_cublas_handle, 1, _d_neg_one, _d_scalar_buffer, 1),
                           "Failed to negate coefficient");
        
        // Subtract projection: q_{j+1} = q_{j+1} + (-h_{i,j}) * q_i
        check_cublas_error(cublasDaxpy(_cublas_handle, _n, _d_scalar_buffer, d_q_i, 1, d_q_j_plus_1, 1),
                           "Failed to update vector in Gram-Schmidt");
    }
    
    // Compute norm of q_{j+1} and store in unified memory for host access
    check_cublas_error(cublasDnrm2(_cublas_handle, _n, d_q_j_plus_1, 1, _um_norm),
                       "Failed to compute norm in Arnoldi iteration");
    
    // Synchronize to access beta from host
    check_cuda_error(cudaDeviceSynchronize(), "Failed to synchronize before accessing beta");
    const double beta = *_um_norm;

    // Compute 1/beta and store in unified memory for device access
    *_um_norm = 1.0 / beta;

    // Normalize using device pointer to unified memory
    check_cublas_error( cublasDscal( _cublas_handle, _n, _um_norm, d_q_j_plus_1, 1 ),
                        "Failed to normalize Krylov vector" );

    // Reset cuBLAS to host pointer mode for subsequent operations
    cublasSetPointerMode(_cublas_handle, CUBLAS_POINTER_MODE_HOST);

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
        _um_g[0] = resid;  // Initialize residual vector
    }
    _um_g[j] = _h_c[j] * resid;
    resid *= _h_s[j];
}

void CudaGMRES::solve_least_squares(size_t j)
{
    if (j == 0) return;
    
    // Solve upper triangular system H * y = g using device cuBLAS DTRSV
    // H is stored in column-major format in unified memory (accessible from device)
    // cuBLAS DTRSV parameters:
    // - handle: cuBLAS handle
    // - uplo: CUBLAS_FILL_MODE_UPPER (upper triangular)
    // - trans: CUBLAS_OP_N (no transpose)
    // - diag: CUBLAS_DIAG_NON_UNIT (non-unit diagonal)
    // - n: size of the system (j)
    // - A: Hessenberg matrix (_um_H)
    // - lda: leading dimension (_current_restart)
    // - x: right-hand side and solution vector (_um_g)
    // - incx: increment for x (1)
    
    // Set cuBLAS to host pointer mode for scalar parameters
    cublasSetPointerMode(_cublas_handle, CUBLAS_POINTER_MODE_HOST);
    
    check_cublas_error(
        cublasDtrsv(_cublas_handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                   static_cast<int>(j), _um_H, static_cast<int>(_current_restart), _um_g, 1),
        "Failed to solve upper triangular system with cuBLAS");
}

void CudaGMRES::update_solution(double* d_x, size_t j)
{
    if (j == 0) return;
    
    // Set cuBLAS to device pointer mode for device-only operations
    cublasSetPointerMode(_cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
    
    // Compute solution update: delta_x = Q[:, 0:j] * g[0:j]
    // Use GEMV: y = alpha * A * x + beta * y
    // Here: _d_tmp = 1.0 * Q[:, 0:j] * g[0:j] + 0.0 * _d_tmp
    
    // No synchronization needed - unified memory accessible from device
    
    check_cublas_error(
        cublasDgemv(_cublas_handle, CUBLAS_OP_N,
                   _n, j,
                   _d_one, _d_Q, _n,
                   _um_g, 1,
                   _d_zero, _d_tmp, 1),
        "Failed to compute solution update");
    
    if (_prec_type == PreconditionerType::RIGHT) {
        // For right preconditioning: x = x + M^{-1} * delta_x
        apply_preconditioner(_d_tmp, _d_w, true);
        check_cublas_error(cublasDaxpy(_cublas_handle, _n, _d_one, _d_w, 1, d_x, 1),
                           "Failed to update solution with right preconditioning");
    } else {
        // For left or no preconditioning: x = x + delta_x
        check_cublas_error(cublasDaxpy(_cublas_handle, _n, _d_one, _d_tmp, 1, d_x, 1),
                           "Failed to update solution");
    }
    
    // Reset cuBLAS to host pointer mode for subsequent operations
    cublasSetPointerMode(_cublas_handle, CUBLAS_POINTER_MODE_HOST);
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