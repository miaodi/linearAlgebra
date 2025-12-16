#include "cuda_preconditioner.h"
#include "cuda_gmres.h"
#include <stdexcept>
#include <cstring>

namespace cuda_iterative_solver
{

// Helper function for error checking
static void check_cusparse_error(cusparseStatus_t status, const char* message)
{
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuSPARSE error: ") + message);
    }
}

static void check_cuda_error(cudaError_t error, const char* message)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + message + " - " + cudaGetErrorString(error));
    }
}

// ============================================================================
// NoPreconditioner Implementation
// ============================================================================

void NoPreconditioner::operator()(cusparseHandle_t handle,
                                  const DeviceVectorView& d_input,
                                  DeviceVectorView& d_output)
{
    // Simply copy input to output
    check_cuda_error(
        cudaMemcpy(d_output.data(), d_input.data(), 
                   d_input.size() * sizeof(double), cudaMemcpyDeviceToDevice),
        "Failed to copy input to output in NoPreconditioner");
}

// ============================================================================
// ILUPreconditionerBase Implementation
// ============================================================================

ILUPreconditionerBase::ILUPreconditionerBase()
    : _d_tmp(new DeviceArray<double>())
    , _view_tmp(new DeviceVectorView())
    , _mat_L(nullptr)
    , _mat_U(nullptr)
    , _spv_descr_L(nullptr)
    , _spv_descr_U(nullptr)
    , _is_setup(false)
{
}

ILUPreconditionerBase::~ILUPreconditionerBase()
{
    delete _d_tmp;
    delete _view_tmp;
}

void ILUPreconditionerBase::operator()(cusparseHandle_t handle,
                                       const DeviceVectorView& d_input,
                                       DeviceVectorView& d_output)
{
    if (!_is_setup) {
        throw std::runtime_error("ILU preconditioner operator() called before setup");
    }
    
    const double alpha = 1.0;
    
    // Solve L * tmp = input
    check_cusparse_error(
        cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                          _mat_L, d_input.descriptor(), _view_tmp->descriptor(),
                          CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, _spv_descr_L),
        "Failed to solve L system in ILU preconditioner");
    
    // Solve U * output = tmp
    check_cusparse_error(
        cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                          _mat_U, _view_tmp->descriptor(), d_output.descriptor(),
                          CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, _spv_descr_U),
        "Failed to solve U system in ILU preconditioner");
}

// ============================================================================
// ILUPreconditioner Implementation
// ============================================================================

ILUPreconditioner::ILUPreconditioner()
    : ILUPreconditionerBase()
    , _d_ia_L(new DeviceArray<int>())
    , _d_ja_L(new DeviceArray<int>())
    , _d_va_L(new DeviceArray<double>())
    , _d_ia_U(new DeviceArray<int>())
    , _d_ja_U(new DeviceArray<int>())
    , _d_va_U(new DeviceArray<double>())
    , _d_buffer_L(new DeviceArray<char>())
    , _d_buffer_U(new DeviceArray<char>())
    , _n(0)
    , _nnz_L(0)
    , _nnz_U(0)
    , _index_base_L(0)
    , _index_base_U(0)
{
}

ILUPreconditioner::~ILUPreconditioner()
{
    cleanup();
    delete _d_ia_L;
    delete _d_ja_L;
    delete _d_va_L;
    delete _d_ia_U;
    delete _d_ja_U;
    delete _d_va_U;
    delete _d_buffer_L;
    delete _d_buffer_U;
}

void ILUPreconditioner::cleanup()
{
    _d_ia_L->release();
    _d_ja_L->release();
    _d_va_L->release();
    _d_ia_U->release();
    _d_ja_U->release();
    _d_va_U->release();
    _d_tmp->release();
    _d_buffer_L->release();
    _d_buffer_U->release();
    
    if (_mat_L) {
        cusparseDestroySpMat(_mat_L);
        _mat_L = nullptr;
    }
    if (_mat_U) {
        cusparseDestroySpMat(_mat_U);
        _mat_U = nullptr;
    }
    if (_spv_descr_L) {
        cusparseSpSV_destroyDescr(_spv_descr_L);
        _spv_descr_L = nullptr;
    }
    if (_spv_descr_U) {
        cusparseSpSV_destroyDescr(_spv_descr_U);
        _spv_descr_U = nullptr;
    }
    
    _is_setup = false;
}

void ILUPreconditioner::setup(cusparseHandle_t handle, size_t n,
                              const int* h_ia_L, const int* h_ja_L, const double* h_va_L,
                              const int* h_ia_U, const int* h_ja_U, const double* h_va_U)
{
    cleanup();
    
    _n = n;
    
    // Deduce index bases and calculate nnz
    if (h_ia_L != nullptr) {
        _index_base_L = h_ia_L[0];
        _nnz_L = h_ia_L[n] - h_ia_L[0];
    }
    
    if (h_ia_U != nullptr) {
        _index_base_U = h_ia_U[0];
        _nnz_U = h_ia_U[n] - h_ia_U[0];
    }
    
    // Copy L factor to device
    if (_nnz_L > 0) {
        _d_ia_L->copyFromHost(h_ia_L, n + 1);
        _d_ja_L->copyFromHost(h_ja_L, _nnz_L);
        _d_va_L->copyFromHost(h_va_L, _nnz_L);
    }
    
    // Copy U factor to device
    if (_nnz_U > 0) {
        _d_ia_U->copyFromHost(h_ia_U, n + 1);
        _d_ja_U->copyFromHost(h_ja_U, _nnz_U);
        _d_va_U->copyFromHost(h_va_U, _nnz_U);
    }
    
    // Allocate temporary storage
    _d_tmp->resize(n);
    _view_tmp->create(n, _d_tmp->data());
    
    // Create L matrix descriptor (lower triangular with unit diagonal)
    cusparseIndexBase_t index_base_L = (_index_base_L == 0) ? CUSPARSE_INDEX_BASE_ZERO : CUSPARSE_INDEX_BASE_ONE;
    check_cusparse_error(
        cusparseCreateCsr(&_mat_L, n, n, _nnz_L,
                         _d_ia_L->data(), _d_ja_L->data(), _d_va_L->data(),
                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                         index_base_L, CUDA_R_64F),
        "Failed to create L matrix descriptor");
    
    cusparseFillMode_t lower_fill = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t unit_diag = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseSpMatSetAttribute(_mat_L, CUSPARSE_SPMAT_FILL_MODE, &lower_fill, sizeof(cusparseFillMode_t));
    cusparseSpMatSetAttribute(_mat_L, CUSPARSE_SPMAT_DIAG_TYPE, &unit_diag, sizeof(cusparseDiagType_t));
    
    // Create U matrix descriptor (upper triangular with non-unit diagonal)
    cusparseIndexBase_t index_base_U = (_index_base_U == 0) ? CUSPARSE_INDEX_BASE_ZERO : CUSPARSE_INDEX_BASE_ONE;
    check_cusparse_error(
        cusparseCreateCsr(&_mat_U, n, n, _nnz_U,
                         _d_ia_U->data(), _d_ja_U->data(), _d_va_U->data(),
                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                         index_base_U, CUDA_R_64F),
        "Failed to create U matrix descriptor");
    
    cusparseFillMode_t upper_fill = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t nonunit_diag = CUSPARSE_DIAG_TYPE_NON_UNIT;
    cusparseSpMatSetAttribute(_mat_U, CUSPARSE_SPMAT_FILL_MODE, &upper_fill, sizeof(cusparseFillMode_t));
    cusparseSpMatSetAttribute(_mat_U, CUSPARSE_SPMAT_DIAG_TYPE, &nonunit_diag, sizeof(cusparseDiagType_t));
    
    // Create SpSV descriptors
    check_cusparse_error(cusparseSpSV_createDescr(&_spv_descr_L), "Failed to create SpSV L descriptor");
    check_cusparse_error(cusparseSpSV_createDescr(&_spv_descr_U), "Failed to create SpSV U descriptor");
    
    // Allocate buffers
    const double alpha = 1.0;
    size_t buffer_size_L, buffer_size_U;
    
    DeviceVectorView dummy_input, dummy_output;
    dummy_input.create(n, nullptr);
    dummy_output.create(n, nullptr);
    
    check_cusparse_error(
        cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                               _mat_L, dummy_input.descriptor(), dummy_output.descriptor(),
                               CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, _spv_descr_L, &buffer_size_L),
        "Failed to get SpSV L buffer size");
    _d_buffer_L->resize(buffer_size_L);
    
    check_cusparse_error(
        cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                               _mat_U, dummy_input.descriptor(), dummy_output.descriptor(),
                               CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, _spv_descr_U, &buffer_size_U),
        "Failed to get SpSV U buffer size");
    _d_buffer_U->resize(buffer_size_U);
    
    // Analyze phase
    check_cusparse_error(
        cusparseSpSV_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                             _mat_L, dummy_input.descriptor(), dummy_output.descriptor(),
                             CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, _spv_descr_L, _d_buffer_L->data()),
        "Failed to analyze L solve");
    
    check_cusparse_error(
        cusparseSpSV_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                             _mat_U, dummy_input.descriptor(), dummy_output.descriptor(),
                             CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, _spv_descr_U, _d_buffer_U->data()),
        "Failed to analyze U solve");
    
    _is_setup = true;
}

// ============================================================================
// GPUILU0Preconditioner Implementation
// ============================================================================

GPUILU0Preconditioner::GPUILU0Preconditioner()
    : ILUPreconditionerBase()
    , _d_ia(new DeviceArray<int>())
    , _d_ja(new DeviceArray<int>())
    , _d_va(new DeviceArray<double>())
    , _d_ilu0_buffer(new DeviceArray<char>())
    , _d_buffer_L(new DeviceArray<char>())
    , _d_buffer_U(new DeviceArray<char>())
    , _ilu0_info(nullptr)
    , _n(0)
    , _nnz(0)
    , _index_base(0)
{
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    check_cusparse_error(cusparseCreateCsrilu02Info(&_ilu0_info), 
                        "Failed to create ILU0 info");
    #pragma GCC diagnostic pop
}

GPUILU0Preconditioner::~GPUILU0Preconditioner()
{
    cleanup();
    
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    if (_ilu0_info) cusparseDestroyCsrilu02Info(_ilu0_info);
    #pragma GCC diagnostic pop
    
    delete _d_ia;
    delete _d_ja;
    delete _d_va;
    delete _d_ilu0_buffer;
    delete _d_buffer_L;
    delete _d_buffer_U;
}

void GPUILU0Preconditioner::cleanup()
{
    _d_ia->release();
    _d_ja->release();
    _d_va->release();
    _d_ilu0_buffer->release();
    _d_buffer_L->release();
    _d_buffer_U->release();
    
    if (_mat_L) {
        cusparseDestroySpMat(_mat_L);
        _mat_L = nullptr;
    }
    if (_mat_U) {
        cusparseDestroySpMat(_mat_U);
        _mat_U = nullptr;
    }
    if (_spv_descr_L) {
        cusparseSpSV_destroyDescr(_spv_descr_L);
        _spv_descr_L = nullptr;
    }
    if (_spv_descr_U) {
        cusparseSpSV_destroyDescr(_spv_descr_U);
        _spv_descr_U = nullptr;
    }
    
    _is_setup = false;
}

void GPUILU0Preconditioner::setup(cusparseHandle_t handle, size_t n,
                                  const int* h_ia, const int* h_ja, const double* h_va)
{
    cleanup();
    
    _n = n;
    _index_base = h_ia[0];
    _nnz = h_ia[n] - h_ia[0];
    
    // Copy matrix to device
    _d_ia->copyFromHost(h_ia, n + 1);
    _d_ja->copyFromHost(h_ja, _nnz);
    _d_va->copyFromHost(h_va, _nnz);
    
    // Allocate temporary storage
    _d_tmp->resize(n);
    _view_tmp->create(n, _d_tmp->data());
    
    // Perform factorization
    performFactorization(handle);
    
    // Setup SpSV descriptors
    setupSpSVDescriptors(handle);
    
    _is_setup = true;
}

void GPUILU0Preconditioner::setupFromDevice(cusparseHandle_t handle, size_t n, size_t nnz,
                                           const int* d_ia, const int* d_ja, const double* d_va,
                                           int index_base)
{
    cleanup();
    
    _n = n;
    _nnz = nnz;
    _index_base = index_base;
    
    // Copy matrix from device
    _d_ia->resize(n + 1);
    _d_ja->resize(nnz);
    _d_va->resize(nnz);
    
    check_cuda_error(cudaMemcpy(_d_ia->data(), d_ia, (n + 1) * sizeof(int), cudaMemcpyDeviceToDevice),
                     "Failed to copy IA array");
    check_cuda_error(cudaMemcpy(_d_ja->data(), d_ja, nnz * sizeof(int), cudaMemcpyDeviceToDevice),
                     "Failed to copy JA array");
    check_cuda_error(cudaMemcpy(_d_va->data(), d_va, nnz * sizeof(double), cudaMemcpyDeviceToDevice),
                     "Failed to copy VA array");
    
    // Allocate temporary storage
    _d_tmp->resize(n);
    _view_tmp->create(n, _d_tmp->data());
    
    // Perform factorization
    performFactorization(handle);
    
    // Setup SpSV descriptors
    setupSpSVDescriptors(handle);
    
    _is_setup = true;
}

void GPUILU0Preconditioner::performFactorization(cusparseHandle_t handle)
{
    // Create matrix descriptor for deprecated API
    cusparseMatDescr_t descr_A;
    check_cusparse_error(cusparseCreateMatDescr(&descr_A), "Failed to create matrix descriptor");
    check_cusparse_error(cusparseSetMatType(descr_A, CUSPARSE_MATRIX_TYPE_GENERAL), "Failed to set matrix type");
    check_cusparse_error(cusparseSetMatIndexBase(descr_A, 
                        (_index_base == 0) ? CUSPARSE_INDEX_BASE_ZERO : CUSPARSE_INDEX_BASE_ONE),
                        "Failed to set index base");
    
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    
    // Get buffer size
    int buffer_size;
    check_cusparse_error(
        cusparseDcsrilu02_bufferSize(handle, _n, _nnz,
                                     descr_A, _d_va->data(), _d_ia->data(), _d_ja->data(),
                                     _ilu0_info, &buffer_size),
        "Failed to get ILU0 buffer size");
    
    _d_ilu0_buffer->resize(buffer_size);
    
    // Analysis phase
    check_cusparse_error(
        cusparseDcsrilu02_analysis(handle, _n, _nnz,
                                   descr_A, _d_va->data(), _d_ia->data(), _d_ja->data(),
                                   _ilu0_info, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                   _d_ilu0_buffer->data()),
        "Failed to analyze ILU0");
    
    // Check for structural singularity
    int structural_zero;
    cusparseStatus_t status = cusparseXcsrilu02_zeroPivot(handle, _ilu0_info, &structural_zero);
    if (status == CUSPARSE_STATUS_ZERO_PIVOT) {
        throw std::runtime_error("ILU0 factorization: structural zero at position " + 
                                std::to_string(structural_zero));
    }
    
    // Factorization phase
    check_cusparse_error(
        cusparseDcsrilu02(handle, _n, _nnz,
                         descr_A, _d_va->data(), _d_ia->data(), _d_ja->data(),
                         _ilu0_info, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                         _d_ilu0_buffer->data()),
        "Failed to perform ILU0 factorization");
    
    // Check for numerical singularity
    int numerical_zero;
    status = cusparseXcsrilu02_zeroPivot(handle, _ilu0_info, &numerical_zero);
    if (status == CUSPARSE_STATUS_ZERO_PIVOT) {
        throw std::runtime_error("ILU0 factorization: numerical zero at position " + 
                                std::to_string(numerical_zero));
    }
    
    #pragma GCC diagnostic pop
    
    cusparseDestroyMatDescr(descr_A);
}

void GPUILU0Preconditioner::setupSpSVDescriptors(cusparseHandle_t handle)
{
    cusparseIndexBase_t index_base = (_index_base == 0) ? CUSPARSE_INDEX_BASE_ZERO : CUSPARSE_INDEX_BASE_ONE;
    
    // Create L matrix descriptor (lower triangular with unit diagonal)
    check_cusparse_error(
        cusparseCreateCsr(&_mat_L, _n, _n, _nnz,
                         _d_ia->data(), _d_ja->data(), _d_va->data(),
                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                         index_base, CUDA_R_64F),
        "Failed to create ILU0 L matrix descriptor");
    
    cusparseFillMode_t lower_fill = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t unit_diag = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseSpMatSetAttribute(_mat_L, CUSPARSE_SPMAT_FILL_MODE, &lower_fill, sizeof(cusparseFillMode_t));
    cusparseSpMatSetAttribute(_mat_L, CUSPARSE_SPMAT_DIAG_TYPE, &unit_diag, sizeof(cusparseDiagType_t));
    
    // Create U matrix descriptor (upper triangular with non-unit diagonal)
    check_cusparse_error(
        cusparseCreateCsr(&_mat_U, _n, _n, _nnz,
                         _d_ia->data(), _d_ja->data(), _d_va->data(),
                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                         index_base, CUDA_R_64F),
        "Failed to create ILU0 U matrix descriptor");
    
    cusparseFillMode_t upper_fill = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t nonunit_diag = CUSPARSE_DIAG_TYPE_NON_UNIT;
    cusparseSpMatSetAttribute(_mat_U, CUSPARSE_SPMAT_FILL_MODE, &upper_fill, sizeof(cusparseFillMode_t));
    cusparseSpMatSetAttribute(_mat_U, CUSPARSE_SPMAT_DIAG_TYPE, &nonunit_diag, sizeof(cusparseDiagType_t));
    
    // Create SpSV descriptors
    check_cusparse_error(cusparseSpSV_createDescr(&_spv_descr_L), "Failed to create SpSV L descriptor");
    check_cusparse_error(cusparseSpSV_createDescr(&_spv_descr_U), "Failed to create SpSV U descriptor");
    
    const double alpha = 1.0;
    size_t buffer_size_L, buffer_size_U;
    
    DeviceVectorView dummy_input, dummy_output;
    dummy_input.create(_n, nullptr);
    dummy_output.create(_n, nullptr);
    
    // Get buffer size for L solve
    check_cusparse_error(
        cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                               _mat_L, dummy_input.descriptor(), dummy_output.descriptor(),
                               CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, _spv_descr_L, &buffer_size_L),
        "Failed to get SpSV L buffer size");
    _d_buffer_L->resize(buffer_size_L);
    
    // Get buffer size for U solve
    check_cusparse_error(
        cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                               _mat_U, dummy_input.descriptor(), dummy_output.descriptor(),
                               CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, _spv_descr_U, &buffer_size_U),
        "Failed to get SpSV U buffer size");
    _d_buffer_U->resize(buffer_size_U);
    
    // Analyze L solve
    check_cusparse_error(
        cusparseSpSV_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                             _mat_L, dummy_input.descriptor(), dummy_output.descriptor(),
                             CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, _spv_descr_L, _d_buffer_L->data()),
        "Failed to analyze L solve");
    
    // Analyze U solve
    check_cusparse_error(
        cusparseSpSV_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                             _mat_U, dummy_input.descriptor(), dummy_output.descriptor(),
                             CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, _spv_descr_U, _d_buffer_U->data()),
        "Failed to analyze U solve");
}

} // namespace cuda_iterative_solver
