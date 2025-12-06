#include "cuda_spmv.h"
#include <cstring>
#include <iostream>

namespace matrix_utils
{

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
CudaSPMV<ROWTYPE, COLTYPE, VALTYPE>::CudaSPMV()
    : _handle(nullptr)
    , _mat_A(nullptr)
    , _vec_x(nullptr)
    , _vec_y(nullptr)
    , _d_ia(nullptr)
    , _d_ja(nullptr)
    , _d_av(nullptr)
    , _d_x(nullptr)
    , _d_y(nullptr)
    , _d_buffer(nullptr)
    , _buffer_size(0)
    , _n(0)
    , _nnz(0)
    , _index_base(0)
    , _is_initialized(false)
{
    check_cusparse_error(cusparseCreate(&_handle), "Failed to create cuSPARSE handle");
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
CudaSPMV<ROWTYPE, COLTYPE, VALTYPE>::~CudaSPMV()
{
    cleanup();
    if (_handle) {
        cusparseDestroy(_handle);
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void CudaSPMV<ROWTYPE, COLTYPE, VALTYPE>::cleanup()
{
    // Destroy descriptors
    if (_mat_A) cusparseDestroySpMat(_mat_A);
    if (_vec_x) cusparseDestroyDnVec(_vec_x);
    if (_vec_y) cusparseDestroyDnVec(_vec_y);
    
    // Free device memory
    if (_d_ia) cudaFree(_d_ia);
    if (_d_ja) cudaFree(_d_ja);
    if (_d_av) cudaFree(_d_av);
    if (_d_x) cudaFree(_d_x);
    if (_d_y) cudaFree(_d_y);
    if (_d_buffer) cudaFree(_d_buffer);
    
    // Reset pointers
    _mat_A = nullptr;
    _vec_x = nullptr;
    _vec_y = nullptr;
    _d_ia = nullptr;
    _d_ja = nullptr;
    _d_av = nullptr;
    _d_x = nullptr;
    _d_y = nullptr;
    _d_buffer = nullptr;
    _buffer_size = 0;
    _is_initialized = false;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void CudaSPMV<ROWTYPE, COLTYPE, VALTYPE>::preprocess(COLTYPE n,
                                                      const ROWTYPE* h_ia,
                                                      const COLTYPE* h_ja,
                                                      const VALTYPE* h_av)
{
    // Clean up any previous initialization
    if (_is_initialized) {
        cleanup();
    }
    
    // Store matrix properties
    _n = n;
    _index_base = h_ia[0];
    _nnz = h_ia[n] - h_ia[0];
    
    // Allocate and copy matrix data to device
    check_cuda_error(cudaMalloc(&_d_ia, (n + 1) * sizeof(ROWTYPE)), "Failed to allocate d_ia");
    check_cuda_error(cudaMalloc(&_d_ja, _nnz * sizeof(COLTYPE)), "Failed to allocate d_ja");
    check_cuda_error(cudaMalloc(&_d_av, _nnz * sizeof(VALTYPE)), "Failed to allocate d_av");
    
    check_cuda_error(cudaMemcpy(_d_ia, h_ia, (n + 1) * sizeof(ROWTYPE), cudaMemcpyHostToDevice),
                     "Failed to copy ia to device");
    check_cuda_error(cudaMemcpy(_d_ja, h_ja, _nnz * sizeof(COLTYPE), cudaMemcpyHostToDevice),
                     "Failed to copy ja to device");
    check_cuda_error(cudaMemcpy(_d_av, h_av, _nnz * sizeof(VALTYPE), cudaMemcpyHostToDevice),
                     "Failed to copy av to device");
    
    // Allocate device memory for vectors
    check_cuda_error(cudaMalloc(&_d_x, n * sizeof(VALTYPE)), "Failed to allocate d_x");
    check_cuda_error(cudaMalloc(&_d_y, n * sizeof(VALTYPE)), "Failed to allocate d_y");
    
    // Create matrix descriptor
    cusparseIndexBase_t index_base = (_index_base == 0) ? CUSPARSE_INDEX_BASE_ZERO : CUSPARSE_INDEX_BASE_ONE;
    
    check_cusparse_error(
        cusparseCreateCsr(&_mat_A, n, n, _nnz,
                         _d_ia, _d_ja, _d_av,
                         get_index_type(), get_index_type(),
                         index_base, get_cuda_data_type()),
        "Failed to create CSR matrix descriptor");
    
    // Create vector descriptors
    check_cusparse_error(
        cusparseCreateDnVec(&_vec_x, n, _d_x, get_cuda_data_type()),
        "Failed to create vector x descriptor");
    
    check_cusparse_error(
        cusparseCreateDnVec(&_vec_y, n, _d_y, get_cuda_data_type()),
        "Failed to create vector y descriptor");
    
    // Query buffer size for SpMV
    VALTYPE alpha = 1.0, beta = 0.0;
    check_cusparse_error(
        cusparseSpMV_bufferSize(_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, _mat_A, _vec_x, &beta, _vec_y,
                               get_cuda_data_type(), CUSPARSE_SPMV_ALG_DEFAULT,
                               &_buffer_size),
        "Failed to query SpMV buffer size");
    
    // Allocate buffer
    if (_buffer_size > 0) {
        check_cuda_error(cudaMalloc(&_d_buffer, _buffer_size), "Failed to allocate SpMV buffer");
    }
    
    // Preprocess SpMV
    check_cusparse_error(
        cusparseSpMV_preprocess(_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, _mat_A, _vec_x, &beta, _vec_y,
                               get_cuda_data_type(), CUSPARSE_SPMV_ALG_DEFAULT,
                               _d_buffer),
        "Failed to preprocess SpMV");
    
    _is_initialized = true;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void CudaSPMV<ROWTYPE, COLTYPE, VALTYPE>::operator()(const VALTYPE* h_x,
                                                      VALTYPE* h_y,
                                                      VALTYPE alpha,
                                                      VALTYPE beta)
{
    if (!_is_initialized) {
        throw std::runtime_error("CudaSPMV: preprocess() must be called before operator()");
    }
    
    // Copy input vector to device
    check_cuda_error(cudaMemcpy(_d_x, h_x, _n * sizeof(VALTYPE), cudaMemcpyHostToDevice),
                     "Failed to copy x to device");
    
    // Copy output vector to device (needed if beta != 0)
    if (beta != static_cast<VALTYPE>(0)) {
        check_cuda_error(cudaMemcpy(_d_y, h_y, _n * sizeof(VALTYPE), cudaMemcpyHostToDevice),
                         "Failed to copy y to device");
    }
    
    // Perform SpMV: y = alpha * A * x + beta * y
    check_cusparse_error(
        cusparseSpMV(_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, _mat_A, _vec_x, &beta, _vec_y,
                    get_cuda_data_type(), CUSPARSE_SPMV_ALG_DEFAULT,
                    _d_buffer),
        "Failed to execute SpMV");
    
    // Copy result back to host
    check_cuda_error(cudaMemcpy(h_y, _d_y, _n * sizeof(VALTYPE), cudaMemcpyDeviceToHost),
                     "Failed to copy y to host");
    
    // Synchronize to ensure completion
    check_cuda_error(cudaDeviceSynchronize(), "Failed to synchronize device");
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
cudaDataType CudaSPMV<ROWTYPE, COLTYPE, VALTYPE>::get_cuda_data_type()
{
    if constexpr (std::is_same_v<VALTYPE, double>) {
        return CUDA_R_64F;
    } else if constexpr (std::is_same_v<VALTYPE, float>) {
        return CUDA_R_32F;
    } else {
        throw std::runtime_error("Unsupported value type for CudaSPMV");
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
cusparseIndexType_t CudaSPMV<ROWTYPE, COLTYPE, VALTYPE>::get_index_type()
{
    if constexpr (sizeof(ROWTYPE) == sizeof(int32_t)) {
        return CUSPARSE_INDEX_32I;
    } else if constexpr (sizeof(ROWTYPE) == sizeof(int64_t)) {
        return CUSPARSE_INDEX_64I;
    } else {
        throw std::runtime_error("Unsupported index type for CudaSPMV");
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void CudaSPMV<ROWTYPE, COLTYPE, VALTYPE>::check_cusparse_error(cusparseStatus_t status, const char* message)
{
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuSPARSE error: ") + message);
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void CudaSPMV<ROWTYPE, COLTYPE, VALTYPE>::check_cuda_error(cudaError_t error, const char* message)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + message + " - " + cudaGetErrorString(error));
    }
}

// Explicit template instantiations for common types
template class CudaSPMV<int, int, double>;
template class CudaSPMV<int, int, float>;
template class CudaSPMV<int64_t, int64_t, double>;
template class CudaSPMV<int64_t, int64_t, float>;

} // namespace matrix_utils
