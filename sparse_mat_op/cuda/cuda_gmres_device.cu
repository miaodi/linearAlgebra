#include "cuda_gmres_device.h"

#include <algorithm>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace cuda_iterative_solver
{

namespace
{
__global__ void set_scalar_kernel(double value, double* dst)
{
    dst[0] = value;
}

__global__ void negate_value_kernel(const double* input, double* output)
{
    output[0] = -input[0];
}

__global__ void reciprocal_kernel(const double* input, double* output)
{
    double val = input[0];
    output[0] = (fabs(val) > 0.0) ? 1.0 / val : 0.0;
}

__global__ void givens_rotation_kernel(double* H_col,
                                       const double* beta_ptr,
                                       double* c,
                                       double* s,
                                       double* g,
                                       int j,
                                       double* resid_ptr)
{
    double beta = beta_ptr[0];
    double resid = resid_ptr[0];

    for (int i = 0; i < j; ++i) {
        double h_i = H_col[i];
        double h_ip1 = H_col[i + 1];
        double tmp = c[i] * h_i - s[i] * h_ip1;
        H_col[i + 1] = s[i] * h_i + c[i] * h_ip1;
        H_col[i] = tmp;
    }

    double Hjj = H_col[j];
    double denom = hypot(Hjj, beta);
    double cos_theta = 1.0;
    double sin_theta = 0.0;
    if (denom != 0.0) {
        cos_theta = Hjj / denom;
        sin_theta = -beta / denom;
        if (fabs(sin_theta) < 1e-16) {
            cos_theta = 1.0;
            sin_theta = 0.0;
        }
    }
    c[j] = cos_theta;
    s[j] = sin_theta;
    H_col[j] = cos_theta * H_col[j] - sin_theta * beta;

    if (j == 0) {
        g[0] = resid;
    }
    g[j] = cos_theta * resid;
    resid_ptr[0] = sin_theta * resid;
}

constexpr int kScalarOne = 0;
constexpr int kScalarZero = 1;
constexpr int kScalarNegOne = 2;
constexpr int kScalarTmp0 = 3;
constexpr int kScalarTmp1 = 4;
constexpr int kScalarTmp2 = 5;

} // namespace

DeviceCudaGMRES::DeviceCudaGMRES()
    : _cublas_handle(nullptr)
    , _cusparse_handle(nullptr)
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
    , _use_batch_orthogonalization(true)
    , _is_operator_setup(false)
    , _is_ilu_setup(false)
    , _matrix_n(0)
    , _matrix_nnz(0)
    , _ilu_nnz_L(0)
    , _ilu_nnz_U(0)
    , _index_base(0)
    , _index_base_L(0)
    , _index_base_U(0)
    , _n(0)
    , _current_restart(0)
    , _spv_buffer_size_L(0)
    , _spv_buffer_size_U(0)
    , _spmv_buffer_size(0)
{
    initialize_cuda();
}

DeviceCudaGMRES::~DeviceCudaGMRES()
{
    cleanup_cuda();
}

void DeviceCudaGMRES::initialize_cuda()
{
    check_cublas_error(cublasCreate(&_cublas_handle), "Failed to create cuBLAS handle");
    check_cublas_error(cublasSetPointerMode(_cublas_handle, CUBLAS_POINTER_MODE_DEVICE),
                       "Failed to set cuBLAS pointer mode");

    check_cusparse_error(cusparseCreate(&_cusparse_handle), "Failed to create cuSPARSE handle");
    check_cusparse_error(cusparseSetPointerMode(_cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE),
                         "Failed to set cuSPARSE pointer mode");
}

void DeviceCudaGMRES::cleanup_cuda()
{
    _d_Q.release();
    _d_tmp.release();
    _d_w.release();
    _d_H.release();
    _d_c.release();
    _d_s.release();
    _d_g.release();
    _d_scalar_workspace.release();
    _d_residual.release();

    _d_b.release();
    _d_x.release();

    _d_ia_A.release();
    _d_ja_A.release();
    _d_va_A.release();
    _d_ia_L.release();
    _d_ja_L.release();
    _d_va_L.release();
    _d_ia_U.release();
    _d_ja_U.release();
    _d_va_U.release();

    _d_spv_buffer_L.release();
    _d_spv_buffer_U.release();
    _d_spmv_buffer.release();

    if (_mat_A) { cusparseDestroySpMat(_mat_A); _mat_A = nullptr; }
    if (_vec_x) { cusparseDestroyDnVec(_vec_x); _vec_x = nullptr; }
    if (_vec_y) { cusparseDestroyDnVec(_vec_y); _vec_y = nullptr; }
    if (_vec_tmp) { cusparseDestroyDnVec(_vec_tmp); _vec_tmp = nullptr; }
    if (_vec_prec_x) { cusparseDestroyDnVec(_vec_prec_x); _vec_prec_x = nullptr; }
    if (_vec_prec_y) { cusparseDestroyDnVec(_vec_prec_y); _vec_prec_y = nullptr; }
    if (_spv_descr_L) { cusparseSpSV_destroyDescr(_spv_descr_L); _spv_descr_L = nullptr; }
    if (_spv_descr_U) { cusparseSpSV_destroyDescr(_spv_descr_U); _spv_descr_U = nullptr; }
    if (_mat_prec_L) { cusparseDestroySpMat(_mat_prec_L); _mat_prec_L = nullptr; }
    if (_mat_prec_U) { cusparseDestroySpMat(_mat_prec_U); _mat_prec_U = nullptr; }

    if (_cusparse_handle) { cusparseDestroy(_cusparse_handle); _cusparse_handle = nullptr; }
    if (_cublas_handle) { cublasDestroy(_cublas_handle); _cublas_handle = nullptr; }
}

void DeviceCudaGMRES::initialize_workspace(size_t n)
{
    if (_n == n && _current_restart == std::min(_restart, n)) {
        return;
    }

    _n = n;
    _current_restart = std::min(_restart, n);

    _d_Q.resize(n * (_current_restart + 1));
    _d_tmp.resize(n);
    _d_w.resize(n);
    _d_H.resize(_current_restart * _current_restart);
    _d_c.resize(_current_restart);
    _d_s.resize(_current_restart);
    _d_g.resize(_current_restart);
    _d_scalar_workspace.resize(std::max<size_t>(_current_restart + 6, 8));
    _d_residual.resize(1);

    check_cuda_error(cudaMemset(_d_Q.data(), 0, _d_Q.size() * sizeof(double)),
                     "Failed to zero Krylov basis memory");
    check_cuda_error(cudaMemset(_d_tmp.data(), 0, _d_tmp.size() * sizeof(double)),
                     "Failed to zero temporary vector");
    check_cuda_error(cudaMemset(_d_w.data(), 0, _d_w.size() * sizeof(double)),
                     "Failed to zero work vector");
    check_cuda_error(cudaMemset(_d_H.data(), 0, _d_H.size() * sizeof(double)),
                     "Failed to zero Hessenberg matrix");
    check_cuda_error(cudaMemset(_d_c.data(), 0, _d_c.size() * sizeof(double)),
                     "Failed to zero cosine array");
    check_cuda_error(cudaMemset(_d_s.data(), 0, _d_s.size() * sizeof(double)),
                     "Failed to zero sine array");
    check_cuda_error(cudaMemset(_d_g.data(), 0, _d_g.size() * sizeof(double)),
                     "Failed to zero residual vector");
    check_cuda_error(cudaMemset(_d_residual.data(), 0, sizeof(double)),
                     "Failed to zero residual storage");

    set_scalar_kernel<<<1, 1>>>(1.0, _d_scalar_workspace.data() + kScalarOne);
    set_scalar_kernel<<<1, 1>>>(0.0, _d_scalar_workspace.data() + kScalarZero);
    set_scalar_kernel<<<1, 1>>>(-1.0, _d_scalar_workspace.data() + kScalarNegOne);
    check_cuda_error(cudaGetLastError(), "Failed to initialize scalar workspace constants");
}

void DeviceCudaGMRES::setupOperator(size_t n,
                                    const int* h_ia_A, const int* h_ja_A, const double* h_va_A)
{
    _index_base = h_ia_A[0];
    size_t nnz = h_ia_A[n] - h_ia_A[0];

    _matrix_n = n;
    _matrix_nnz = nnz;

    _d_ia_A.copyFromHost(h_ia_A, n + 1);
    _d_ja_A.copyFromHost(h_ja_A, nnz);
    _d_va_A.copyFromHost(h_va_A, nnz);

    initialize_workspace(n);
    setup_matrix_descriptor();

    _is_operator_setup = true;
}

void DeviceCudaGMRES::setupILU(size_t n,
                               const int* h_ia_L, const int* h_ja_L, const double* h_va_L,
                               const int* h_ia_U, const int* h_ja_U, const double* h_va_U)
{
    if (!_is_operator_setup || _matrix_n != n) {
        throw std::runtime_error("setupOperator must be called first with the same matrix size");
    }

    _ilu_nnz_L = 0;
    _ilu_nnz_U = 0;

    if (h_ia_L) {
        _index_base_L = h_ia_L[0];
        _ilu_nnz_L = h_ia_L[n] - h_ia_L[0];
        _d_ia_L.copyFromHost(h_ia_L, n + 1);
        _d_ja_L.copyFromHost(h_ja_L, _ilu_nnz_L);
        _d_va_L.copyFromHost(h_va_L, _ilu_nnz_L);
    }

    if (h_ia_U) {
        _index_base_U = h_ia_U[0];
        _ilu_nnz_U = h_ia_U[n] - h_ia_U[0];
        _d_ia_U.copyFromHost(h_ia_U, n + 1);
        _d_ja_U.copyFromHost(h_ja_U, _ilu_nnz_U);
        _d_va_U.copyFromHost(h_va_U, _ilu_nnz_U);
    }

    setup_ilu_descriptors();
    _is_ilu_setup = true;
}

void DeviceCudaGMRES::setup_matrix_descriptor()
{
    if (_mat_A) { cusparseDestroySpMat(_mat_A); _mat_A = nullptr; }
    if (_vec_x) { cusparseDestroyDnVec(_vec_x); _vec_x = nullptr; }
    if (_vec_y) { cusparseDestroyDnVec(_vec_y); _vec_y = nullptr; }
    if (_vec_tmp) { cusparseDestroyDnVec(_vec_tmp); _vec_tmp = nullptr; }
    if (_vec_prec_x) { cusparseDestroyDnVec(_vec_prec_x); _vec_prec_x = nullptr; }
    if (_vec_prec_y) { cusparseDestroyDnVec(_vec_prec_y); _vec_prec_y = nullptr; }

    cusparseIndexBase_t index_base = (_index_base == 0) ? CUSPARSE_INDEX_BASE_ZERO : CUSPARSE_INDEX_BASE_ONE;
    check_cusparse_error(
        cusparseCreateCsr(&_mat_A, _matrix_n, _matrix_n, _matrix_nnz,
                          _d_ia_A.data(), _d_ja_A.data(), _d_va_A.data(),
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, index_base, CUDA_R_64F),
        "Failed to create matrix A descriptor");

    check_cusparse_error(cusparseCreateDnVec(&_vec_x, _matrix_n, nullptr, CUDA_R_64F),
                         "Failed to create vector x descriptor");
    check_cusparse_error(cusparseCreateDnVec(&_vec_y, _matrix_n, nullptr, CUDA_R_64F),
                         "Failed to create vector y descriptor");
    check_cusparse_error(cusparseCreateDnVec(&_vec_tmp, _matrix_n, nullptr, CUDA_R_64F),
                         "Failed to create temporary vector descriptor");
    check_cusparse_error(cusparseCreateDnVec(&_vec_prec_x, _matrix_n, nullptr, CUDA_R_64F),
                         "Failed to create preconditioner input descriptor");
    check_cusparse_error(cusparseCreateDnVec(&_vec_prec_y, _matrix_n, nullptr, CUDA_R_64F),
                         "Failed to create preconditioner output descriptor");

    double* d_one = _d_scalar_workspace.data() + kScalarOne;
    check_cusparse_error(
        cusparseSpMV_bufferSize(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                d_one, _mat_A, _vec_x, d_one, _vec_y, CUDA_R_64F,
                                CUSPARSE_SPMV_ALG_DEFAULT, &_spmv_buffer_size),
        "Failed to get SpMV buffer size");

    _d_spmv_buffer.resize(_spmv_buffer_size);
    void* spmv_buffer = (_spmv_buffer_size > 0) ? static_cast<void*>(_d_spmv_buffer.data()) : nullptr;

    // Preprocess SpMV for faster repeated calls
    check_cusparse_error(cusparseDnVecSetValues(_vec_x, _d_tmp.data()),
                         "Failed to bind temporary x vector for preprocessing");
    check_cusparse_error(cusparseDnVecSetValues(_vec_y, _d_w.data()),
                         "Failed to bind temporary y vector for preprocessing");
    check_cusparse_error(
        cusparseSpMV_preprocess(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                d_one, _mat_A, _vec_x, d_one, _vec_y, CUDA_R_64F,
                                CUSPARSE_SPMV_ALG_DEFAULT, spmv_buffer),
        "Failed to preprocess SpMV");
    check_cusparse_error(cusparseDnVecSetValues(_vec_x, _d_tmp.data()),
                         "Failed to reset x descriptor after preprocessing");
    check_cusparse_error(cusparseDnVecSetValues(_vec_y, _d_w.data()),
                         "Failed to reset y descriptor after preprocessing");
}

void DeviceCudaGMRES::setup_ilu_descriptors()
{
    if (_prec_type == PreconditionerType::NONE || _ilu_nnz_L == 0 || _ilu_nnz_U == 0) {
        return;
    }

    if (_mat_prec_L) { cusparseDestroySpMat(_mat_prec_L); _mat_prec_L = nullptr; }
    if (_mat_prec_U) { cusparseDestroySpMat(_mat_prec_U); _mat_prec_U = nullptr; }
    if (_spv_descr_L) { cusparseSpSV_destroyDescr(_spv_descr_L); _spv_descr_L = nullptr; }
    if (_spv_descr_U) { cusparseSpSV_destroyDescr(_spv_descr_U); _spv_descr_U = nullptr; }

    cusparseIndexBase_t index_base_L = (_index_base_L == 0) ? CUSPARSE_INDEX_BASE_ZERO : CUSPARSE_INDEX_BASE_ONE;
    cusparseIndexBase_t index_base_U = (_index_base_U == 0) ? CUSPARSE_INDEX_BASE_ZERO : CUSPARSE_INDEX_BASE_ONE;

    check_cusparse_error(
        cusparseCreateCsr(&_mat_prec_L, _matrix_n, _matrix_n, _ilu_nnz_L,
                          _d_ia_L.data(), _d_ja_L.data(), _d_va_L.data(),
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, index_base_L, CUDA_R_64F),
        "Failed to create preconditioner L descriptor");
    check_cusparse_error(
        cusparseCreateCsr(&_mat_prec_U, _matrix_n, _matrix_n, _ilu_nnz_U,
                          _d_ia_U.data(), _d_ja_U.data(), _d_va_U.data(),
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, index_base_U, CUDA_R_64F),
        "Failed to create preconditioner U descriptor");

    cusparseFillMode_t lower_fill = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t unit_diag = CUSPARSE_DIAG_TYPE_UNIT;
    check_cusparse_error(
        cusparseSpMatSetAttribute(_mat_prec_L, CUSPARSE_SPMAT_FILL_MODE, &lower_fill, sizeof(lower_fill)),
        "Failed to set L fill mode");
    check_cusparse_error(
        cusparseSpMatSetAttribute(_mat_prec_L, CUSPARSE_SPMAT_DIAG_TYPE, &unit_diag, sizeof(unit_diag)),
        "Failed to set L diagonal type");

    cusparseFillMode_t upper_fill = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t nonunit_diag = CUSPARSE_DIAG_TYPE_NON_UNIT;
    check_cusparse_error(
        cusparseSpMatSetAttribute(_mat_prec_U, CUSPARSE_SPMAT_FILL_MODE, &upper_fill, sizeof(upper_fill)),
        "Failed to set U fill mode");
    check_cusparse_error(
        cusparseSpMatSetAttribute(_mat_prec_U, CUSPARSE_SPMAT_DIAG_TYPE, &nonunit_diag, sizeof(nonunit_diag)),
        "Failed to set U diagonal type");

    check_cusparse_error(cusparseSpSV_createDescr(&_spv_descr_L), "Failed to create SpSV L descriptor");
    check_cusparse_error(cusparseSpSV_createDescr(&_spv_descr_U), "Failed to create SpSV U descriptor");

    double* d_one = _d_scalar_workspace.data() + kScalarOne;
    check_cusparse_error(
        cusparseSpSV_bufferSize(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                d_one, _mat_prec_L, _vec_prec_x, _vec_prec_y, CUDA_R_64F,
                                CUSPARSE_SPSV_ALG_DEFAULT, _spv_descr_L, &_spv_buffer_size_L),
        "Failed to get SpSV L buffer size");
    check_cusparse_error(
        cusparseSpSV_bufferSize(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                d_one, _mat_prec_U, _vec_prec_x, _vec_prec_y, CUDA_R_64F,
                                CUSPARSE_SPSV_ALG_DEFAULT, _spv_descr_U, &_spv_buffer_size_U),
        "Failed to get SpSV U buffer size");

    _d_spv_buffer_L.resize(_spv_buffer_size_L);
    _d_spv_buffer_U.resize(_spv_buffer_size_U);

    void* spv_buffer_L = (_spv_buffer_size_L > 0) ? static_cast<void*>(_d_spv_buffer_L.data()) : nullptr;
    void* spv_buffer_U = (_spv_buffer_size_U > 0) ? static_cast<void*>(_d_spv_buffer_U.data()) : nullptr;

    check_cusparse_error(
        cusparseSpSV_analysis(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              d_one, _mat_prec_L, _vec_prec_x, _vec_prec_y, CUDA_R_64F,
                              CUSPARSE_SPSV_ALG_DEFAULT, _spv_descr_L, spv_buffer_L),
        "Failed to analyze SpSV L pattern");
    check_cusparse_error(
        cusparseSpSV_analysis(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              d_one, _mat_prec_U, _vec_prec_x, _vec_prec_y, CUDA_R_64F,
                              CUSPARSE_SPSV_ALG_DEFAULT, _spv_descr_U, spv_buffer_U),
        "Failed to analyze SpSV U pattern");
}

State DeviceCudaGMRES::solve(const double* h_b, double* h_x)
{
    if (!_is_operator_setup) {
        throw std::runtime_error("setupOperator must be called before solve");
    }

    if (_prec_type != PreconditionerType::NONE && !_is_ilu_setup) {
        throw std::runtime_error("setupILU must be called before solve when using preconditioner");
    }

    _d_b.copyFromHost(h_b, _matrix_n);
    _d_x.copyFromHost(h_x, _matrix_n);

    State result = deviceSolve(_d_b.data(), _d_x.data());

    check_cuda_error(cudaMemcpy(h_x, _d_x.data(), _matrix_n * sizeof(double), cudaMemcpyDeviceToHost),
                     "Failed to copy solution to host");
    return result;
}

State DeviceCudaGMRES::deviceSolve(const double* d_b, double* d_x)
{
    double init_resid = compute_initial_residual(d_b, d_x);
    if (std::abs(init_resid) < _abs_tol) {
        return State::CONVERGED;
    }

    double resid = init_resid;
    size_t iter = 0;
    while (iter < _max_iter) {
        size_t cycle_iterations = 0;
        State state = perform_restart_cycle(d_x, init_resid, iter, resid, cycle_iterations);

        if (state == State::CONVERGED || state == State::FAILED) {
            return state;
        }

        if (state == State::MAX_ITER_REACHED) {
            return State::MAX_ITER_REACHED;
        }

        resid = compute_initial_residual(d_b, d_x);
    }

    return State::MAX_ITER_REACHED;
}

double DeviceCudaGMRES::compute_initial_residual(const double* d_b, const double* d_x)
{
    double* d_one = _d_scalar_workspace.data() + kScalarOne;
    double* d_neg_one = _d_scalar_workspace.data() + kScalarNegOne;

    check_cuda_error(cudaMemcpy(_d_Q.data(), d_b, _n * sizeof(double), cudaMemcpyDeviceToDevice),
                     "Failed to copy b to Krylov basis");

    check_cusparse_error(cusparseDnVecSetValues(_vec_x, const_cast<double*>(d_x)),
                         "Failed to bind vec_x");
    check_cusparse_error(cusparseDnVecSetValues(_vec_y, _d_Q.data()),
                         "Failed to bind vec_y");

    void* spmv_buffer = (_spmv_buffer_size > 0) ? static_cast<void*>(_d_spmv_buffer.data()) : nullptr;
    check_cusparse_error(
        cusparseSpMV(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     d_neg_one, _mat_A, _vec_x, d_one, _vec_y, CUDA_R_64F,
                     CUSPARSE_SPMV_ALG_DEFAULT, spmv_buffer),
        "Failed to compute residual SpMV");

    if (_prec_type == PreconditionerType::LEFT) {
        apply_preconditioner(_d_Q.data(), _d_tmp.data());
        check_cuda_error(cudaMemcpy(_d_Q.data(), _d_tmp.data(), _n * sizeof(double), cudaMemcpyDeviceToDevice),
                         "Failed to copy preconditioned residual");
    }

    double* d_beta = _d_scalar_workspace.data() + kScalarTmp0;
    check_cublas_error(cublasDnrm2(_cublas_handle, _n, _d_Q.data(), 1, d_beta),
                       "Failed to compute residual norm");

    double beta = 0.0;
    check_cuda_error(cudaMemcpy(&beta, d_beta, sizeof(double), cudaMemcpyDeviceToHost),
                     "Failed to copy residual norm to host");
    return beta;
}

void DeviceCudaGMRES::apply_operator_with_preconditioning(const double* d_input, double* d_output)
{
    double* d_one = _d_scalar_workspace.data() + kScalarOne;
    double* d_zero = _d_scalar_workspace.data() + kScalarZero;

    check_cusparse_error(cusparseDnVecSetValues(_vec_x, const_cast<double*>(d_input)),
                         "Failed to bind input vector to vec_x");
    check_cusparse_error(cusparseDnVecSetValues(_vec_y, d_output),
                         "Failed to bind output vector to vec_y");

    void* spmv_buffer = (_spmv_buffer_size > 0) ? static_cast<void*>(_d_spmv_buffer.data()) : nullptr;

    switch (_prec_type) {
    case PreconditionerType::RIGHT:
        apply_preconditioner(d_input, _d_tmp.data());
        check_cusparse_error(cusparseDnVecSetValues(_vec_x, _d_tmp.data()),
                             "Failed to bind right-preconditioned input");
        check_cusparse_error(
            cusparseSpMV(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         d_one, _mat_A, _vec_x, d_zero, _vec_y, CUDA_R_64F,
                         CUSPARSE_SPMV_ALG_DEFAULT, spmv_buffer),
            "Failed to execute SpMV with right preconditioning");
        break;

    case PreconditionerType::LEFT:
        check_cusparse_error(
            cusparseSpMV(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         d_one, _mat_A, _vec_x, d_zero, _vec_y, CUDA_R_64F,
                         CUSPARSE_SPMV_ALG_DEFAULT, spmv_buffer),
            "Failed to execute SpMV");
        apply_preconditioner(d_output, _d_tmp.data());
        check_cuda_error(cudaMemcpy(d_output, _d_tmp.data(), _n * sizeof(double), cudaMemcpyDeviceToDevice),
                         "Failed to copy left-preconditioned result");
        break;

    case PreconditionerType::NONE:
        check_cusparse_error(
            cusparseSpMV(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         d_one, _mat_A, _vec_x, d_zero, _vec_y, CUDA_R_64F,
                         CUSPARSE_SPMV_ALG_DEFAULT, spmv_buffer),
            "Failed to execute SpMV without preconditioning");
        break;
    }
}

void DeviceCudaGMRES::apply_preconditioner(const double* d_input, double* d_output)
{
    if (_prec_type == PreconditionerType::NONE || !_spv_descr_L || !_spv_descr_U) {
        check_cuda_error(cudaMemcpy(d_output, d_input, _n * sizeof(double), cudaMemcpyDeviceToDevice),
                         "Failed to copy vector without preconditioning");
        return;
    }

    double* d_one = _d_scalar_workspace.data() + kScalarOne;

    check_cusparse_error(cusparseDnVecSetValues(_vec_prec_x, const_cast<double*>(d_input)),
                         "Failed to bind preconditioner input");
    check_cusparse_error(cusparseDnVecSetValues(_vec_prec_y, _d_tmp.data()),
                         "Failed to bind intermediate vector");

    check_cusparse_error(
        cusparseSpSV_solve(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           d_one, _mat_prec_L, _vec_prec_x, _vec_prec_y, CUDA_R_64F,
                           CUSPARSE_SPSV_ALG_DEFAULT, _spv_descr_L),
        "Failed to solve L system");

    check_cusparse_error(cusparseDnVecSetValues(_vec_prec_x, _d_tmp.data()),
                         "Failed to bind intermediate as input for U solve");
    check_cusparse_error(cusparseDnVecSetValues(_vec_prec_y, d_output),
                         "Failed to bind output vector for U solve");

    check_cusparse_error(
        cusparseSpSV_solve(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           d_one, _mat_prec_U, _vec_prec_x, _vec_prec_y, CUDA_R_64F,
                           CUSPARSE_SPSV_ALG_DEFAULT, _spv_descr_U),
        "Failed to solve U system");
}

State DeviceCudaGMRES::perform_restart_cycle(
    double* d_x, double init_resid, size_t& iter, double& resid, size_t& cycle_iterations )
{
    // check_cuda_error(cudaMemset(_d_H.data(), 0, _d_H.size() * sizeof(double)),
    //                  "Failed to reset Hessenberg matrix");
    // check_cuda_error(cudaMemset(_d_c.data(), 0, _d_c.size() * sizeof(double)),
    //                  "Failed to reset cosine array");
    // check_cuda_error(cudaMemset(_d_s.data(), 0, _d_s.size() * sizeof(double)),
    //                  "Failed to reset sine array");
    // check_cuda_error(cudaMemset(_d_g.data(), 0, _d_g.size() * sizeof(double)),
    //                  "Failed to reset g vector");

    if (resid != 0.0) {
        double inv_resid = 1.0 / resid;
        double* d_inv_resid = _d_scalar_workspace.data() + kScalarTmp0;
        set_scalar_kernel<<<1, 1>>>(inv_resid, d_inv_resid);
        check_cuda_error(cudaGetLastError(), "Failed to set inverse residual on device");
        check_cublas_error(cublasDscal(_cublas_handle, _n, d_inv_resid, _d_Q.data(), 1),
                           "Failed to normalize first Krylov vector");
    }

    size_t j = 0;
    for (; j < _current_restart && iter < _max_iter; ++j, ++iter) {
        const double* d_beta = arnoldi_iteration(j);
        apply_givens_rotations(d_beta, j, resid);

        print_iteration_info(iter, resid, init_resid);

        if (check_convergence(resid, init_resid)) {
            cycle_iterations = j + 1;
            solve_least_squares(j + 1);
            update_solution(d_x, j + 1);
            return State::CONVERGED;
        }
    }

    cycle_iterations = j;
    solve_least_squares(cycle_iterations);
    update_solution(d_x, cycle_iterations);

    if (iter >= _max_iter) {
        return State::MAX_ITER_REACHED;
    }
    return State::RUNNING;
}

const double* DeviceCudaGMRES::arnoldi_iteration(size_t j)
{
    double* d_q_j = _d_Q.data() + j * _n;
    double* d_q_j_plus_1 = _d_Q.data() + (j + 1) * _n;

    apply_operator_with_preconditioning(d_q_j, d_q_j_plus_1);
    return run_modified_gram_schmidt(j);
}

const double* DeviceCudaGMRES::run_modified_gram_schmidt(size_t j)
{
    double* d_q_j_plus_1 = _d_Q.data() + (j + 1) * _n;
    double* d_neg_alpha = _d_scalar_workspace.data() + kScalarTmp0;
    double* d_beta = _d_scalar_workspace.data() + kScalarTmp1;
    double* d_inv_beta = _d_scalar_workspace.data() + kScalarTmp2;
    double* d_h_col = _d_H.data() + j * _current_restart;
    double* d_one = _d_scalar_workspace.data() + kScalarOne;
    double* d_zero = _d_scalar_workspace.data() + kScalarZero;
    double* d_neg_one = _d_scalar_workspace.data() + kScalarNegOne;

    if (_use_batch_orthogonalization) {
        int m = static_cast<int>(_n);
        int k = static_cast<int>(j + 1);

        check_cublas_error(
            cublasDgemv(_cublas_handle, CUBLAS_OP_T,
                        m, k,
                        d_one, _d_Q.data(), m,
                        d_q_j_plus_1, 1,
                        d_zero, d_h_col, 1),
            "Failed to compute batched dot products in Gram-Schmidt");

        check_cublas_error(
            cublasDgemv(_cublas_handle, CUBLAS_OP_N,
                        m, k,
                        d_neg_one, _d_Q.data(), m,
                        d_h_col, 1,
                        d_one, d_q_j_plus_1, 1),
            "Failed to update vector in batched Gram-Schmidt");

        if (k < static_cast<int>(_current_restart)) {
            size_t remaining = _current_restart - static_cast<size_t>(k);
            check_cuda_error(
                cudaMemset(d_h_col + k, 0, remaining * sizeof(double)),
                "Failed to zero unused Hessenberg entries");
        }
    } else {
        for (size_t i = 0; i <= j; ++i) {
            double* d_q_i = _d_Q.data() + i * _n;
            double* h_ij = d_h_col + i;

            check_cublas_error(
                cublasDdot(_cublas_handle, _n, d_q_i, 1, d_q_j_plus_1, 1, h_ij),
                "Failed to compute dot product in Gram-Schmidt");

            negate_value_kernel<<<1, 1>>>(h_ij, d_neg_alpha);
            check_cuda_error(cudaGetLastError(), "Failed to launch negate kernel");

            check_cublas_error(
                cublasDaxpy(_cublas_handle, _n, d_neg_alpha, d_q_i, 1, d_q_j_plus_1, 1),
                "Failed to update vector in Gram-Schmidt");
        }
    }

    check_cublas_error(cublasDnrm2(_cublas_handle, _n, d_q_j_plus_1, 1, d_beta),
                       "Failed to compute norm in Gram-Schmidt");

    reciprocal_kernel<<<1, 1>>>(d_beta, d_inv_beta);
    check_cuda_error(cudaGetLastError(), "Failed to launch reciprocal kernel");
    check_cublas_error(cublasDscal(_cublas_handle, _n, d_inv_beta, d_q_j_plus_1, 1),
                       "Failed to normalize Krylov vector");

    return d_beta;
}

void DeviceCudaGMRES::apply_givens_rotations(const double* d_beta, size_t j, double& resid)
{
    check_cuda_error(cudaMemcpy(_d_residual.data(), &resid, sizeof(double), cudaMemcpyHostToDevice),
                     "Failed to copy residual to device");

    double* H_col = _d_H.data() + j * _current_restart;
    givens_rotation_kernel<<<1, 1>>>(H_col, d_beta, _d_c.data(), _d_s.data(),
                                     _d_g.data(), static_cast<int>(j),
                                     _d_residual.data());
    check_cuda_error(cudaGetLastError(), "Failed to launch Givens rotation kernel");

    check_cuda_error(cudaMemcpy(&resid, _d_residual.data(), sizeof(double), cudaMemcpyDeviceToHost),
                     "Failed to copy residual from device");
}

void DeviceCudaGMRES::solve_least_squares(size_t j)
{
    if (j == 0) {
        return;
    }

    int n_sys = static_cast<int>(j);
    check_cublas_error(
        cublasDtrsv(_cublas_handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                    CUBLAS_DIAG_NON_UNIT, n_sys,
                    _d_H.data(), static_cast<int>(_current_restart),
                    _d_g.data(), 1),
        "Failed to solve triangular system for least squares");
}

void DeviceCudaGMRES::update_solution(double* d_x, size_t j)
{
    if (j == 0) {
        return;
    }

    int cols = static_cast<int>(j);
    double* alpha = _d_scalar_workspace.data() + kScalarOne;
    double* beta = _d_scalar_workspace.data() + kScalarZero;

    check_cublas_error(
        cublasDgemv(_cublas_handle, CUBLAS_OP_N,
                    _n, cols,
                    alpha, _d_Q.data(), _n,
                    _d_g.data(), 1,
                    beta, _d_tmp.data(), 1),
        "Failed to compute solution update");

    double* daxpy_alpha = _d_scalar_workspace.data() + kScalarOne;
    if (_prec_type == PreconditionerType::RIGHT) {
        apply_preconditioner(_d_tmp.data(), _d_w.data());
        check_cublas_error(cublasDaxpy(_cublas_handle, _n, daxpy_alpha, _d_w.data(), 1, d_x, 1),
                           "Failed to update solution with right preconditioning");
    } else {
        check_cublas_error(cublasDaxpy(_cublas_handle, _n, daxpy_alpha, _d_tmp.data(), 1, d_x, 1),
                           "Failed to update solution");
    }
}

bool DeviceCudaGMRES::check_convergence(double resid, double init_resid) const
{
    return std::abs(resid) < _abs_tol || std::abs(resid) < _rel_tol * std::abs(init_resid);
}

void DeviceCudaGMRES::print_iteration_info(size_t iter, double resid, double init_resid) const
{
    std::cout << "iter: " << std::setw(4) << iter
              << " resid: " << std::scientific << std::setprecision(4) << std::abs(resid)
              << " relative resid: " << std::scientific << std::setprecision(4)
              << (std::abs(init_resid) > 0.0 ? std::abs(resid) / std::abs(init_resid) : 0.0)
              << std::endl;
}

void DeviceCudaGMRES::check_cuda_error(cudaError_t error, const char* message)
{
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " - " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error(message);
    }
}

void DeviceCudaGMRES::check_cublas_error(cublasStatus_t status, const char* message)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error: " << message << " - Status: " << status << std::endl;
        throw std::runtime_error(message);
    }
}

void DeviceCudaGMRES::check_cusparse_error(cusparseStatus_t status, const char* message)
{
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "cuSPARSE Error: " << message << " - Status: " << status << std::endl;
        throw std::runtime_error(message);
    }
}

} // namespace cuda_iterative_solver
