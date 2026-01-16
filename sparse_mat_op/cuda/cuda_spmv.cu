#include "cuda_spmv.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

namespace matrix_utils
{

namespace {
inline void cuda_check(cudaError_t error, const char* message)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + message + " - " +
                                 cudaGetErrorString(error));
    }
}
} // namespace

template <typename VALTYPE>
__inline__ __device__ VALTYPE warp_reduce_sum(VALTYPE val)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void copy_csr_host_to_device(COLTYPE n,
                             const ROWTYPE* h_ia,
                             const COLTYPE* h_ja,
                             const VALTYPE* h_av,
                             ROWTYPE** d_ia,
                             COLTYPE** d_ja,
                             VALTYPE** d_av,
                             ROWTYPE* nnz_out)
{
    if (!h_ia || !d_ia || !d_ja || !d_av) {
        throw std::runtime_error("copy_csr_host_to_device: invalid pointer");
    }

    const ROWTYPE base = h_ia[0];
    const ROWTYPE nnz = h_ia[n] - base;
    if (nnz_out) {
        *nnz_out = nnz;
    }
    if (nnz > 0 && (!h_ja || !h_av)) {
        throw std::runtime_error("copy_csr_host_to_device: invalid pointer");
    }

    cuda_check(cudaMalloc(d_ia, (static_cast<size_t>(n) + 1) * sizeof(ROWTYPE)),
               "Failed to allocate device row pointers");
    cuda_check(cudaMemcpy(*d_ia, h_ia, (static_cast<size_t>(n) + 1) * sizeof(ROWTYPE),
                          cudaMemcpyHostToDevice),
               "Failed to copy row pointers to device");

    if (nnz > 0) {
        cuda_check(cudaMalloc(d_ja, static_cast<size_t>(nnz) * sizeof(COLTYPE)),
                   "Failed to allocate device column indices");
        cuda_check(cudaMalloc(d_av, static_cast<size_t>(nnz) * sizeof(VALTYPE)),
                   "Failed to allocate device values");
        cuda_check(cudaMemcpy(*d_ja, h_ja, static_cast<size_t>(nnz) * sizeof(COLTYPE),
                              cudaMemcpyHostToDevice),
                   "Failed to copy column indices to device");
        cuda_check(cudaMemcpy(*d_av, h_av, static_cast<size_t>(nnz) * sizeof(VALTYPE),
                              cudaMemcpyHostToDevice),
                   "Failed to copy values to device");
    } else {
        *d_ja = nullptr;
        *d_av = nullptr;
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void copy_csr_device_to_host(COLTYPE n,
                             const ROWTYPE* d_ia,
                             const COLTYPE* d_ja,
                             const VALTYPE* d_av,
                             ROWTYPE* h_ia,
                             COLTYPE* h_ja,
                             VALTYPE* h_av,
                             ROWTYPE nnz)
{
    if (!d_ia || !h_ia) {
        throw std::runtime_error("copy_csr_device_to_host: invalid pointer");
    }

    cuda_check(cudaMemcpy(h_ia, d_ia, (static_cast<size_t>(n) + 1) * sizeof(ROWTYPE),
                          cudaMemcpyDeviceToHost),
               "Failed to copy row pointers to host");
    if (nnz > 0) {
        if (!d_ja || !d_av || !h_ja || !h_av) {
            throw std::runtime_error("copy_csr_device_to_host: invalid pointer");
        }
        cuda_check(cudaMemcpy(h_ja, d_ja, static_cast<size_t>(nnz) * sizeof(COLTYPE),
                              cudaMemcpyDeviceToHost),
                   "Failed to copy column indices to host");
        cuda_check(cudaMemcpy(h_av, d_av, static_cast<size_t>(nnz) * sizeof(VALTYPE),
                              cudaMemcpyDeviceToHost),
                   "Failed to copy values to host");
    }
}

// One thread per row CSR SpMV kernel: y = alpha * A * x + beta * y.
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
__global__ void csr_scalar_spmv_kernel(const COLTYPE n,
                                       ROWTYPE const* __restrict ia,
                                       COLTYPE const* __restrict ja,
                                       VALTYPE const* __restrict av,
                                       VALTYPE const* __restrict x,
                                       VALTYPE* __restrict y,
                                       const VALTYPE alpha,
                                       const VALTYPE beta,
                                       const int base)
{
    const COLTYPE row = static_cast<COLTYPE>(blockIdx.x) * static_cast<COLTYPE>(blockDim.x) +
                        static_cast<COLTYPE>(threadIdx.x);
    if (row >= n) {
        return;
    }

    const ROWTYPE start = ia[row] - base;
    const ROWTYPE end = ia[row + 1] - base;
    VALTYPE sum = static_cast<VALTYPE>(0);
    for (ROWTYPE idx = start; idx < end; ++idx) {
        const COLTYPE col = ja[idx] - base;
        sum += av[idx] * x[col];
    }

    const VALTYPE ax = alpha * sum;
    if (beta == static_cast<VALTYPE>(0)) {
        y[row] = ax;
    } else if (beta == static_cast<VALTYPE>(1)) {
        y[row] = ax + y[row];
    } else {
        y[row] = ax + beta * y[row];
    }
}

// One warp per row CSR SpMV kernel: y = alpha * A * x + beta * y.
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
__global__ void csr_vector_spmv_kernel(const COLTYPE n,
                                       ROWTYPE const* __restrict ia,
                                       COLTYPE const* __restrict ja,
                                       VALTYPE const* __restrict av,
                                       VALTYPE const* __restrict x,
                                       VALTYPE* __restrict y,
                                       const VALTYPE alpha,
                                       const VALTYPE beta,
                                       const int base)
{
    constexpr int warp_size = 32;
    const int warp_in_block = static_cast<int>(threadIdx.x) / warp_size;
    const int lane = static_cast<int>(threadIdx.x) & (warp_size - 1);
    const int warps_per_block = static_cast<int>(blockDim.x) / warp_size;
    const COLTYPE row = static_cast<COLTYPE>(blockIdx.x * warps_per_block + warp_in_block);

    if (row >= n) {
        return;
    }

    const ROWTYPE start = ia[row] - base;
    const ROWTYPE end = ia[row + 1] - base;
    VALTYPE sum = static_cast<VALTYPE>(0);

    for (ROWTYPE idx = start + lane; idx < end; idx += warp_size) {
        const COLTYPE col = ja[idx] - base;
        sum += av[idx] * x[col];
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        const VALTYPE ax = alpha * sum;
        if (beta == static_cast<VALTYPE>(0)) {
            y[row] = ax;
        } else if (beta == static_cast<VALTYPE>(1)) {
            y[row] = ax + y[row];
        } else {
            y[row] = ax + beta * y[row];
        }
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
CSRScalarSPMV<ROWTYPE, COLTYPE, VALTYPE>::CSRScalarSPMV()
    : SpMVOperator<VALTYPE>()
{
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
CSRScalarSPMV<ROWTYPE, COLTYPE, VALTYPE>::~CSRScalarSPMV()
{
    cleanup();
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
CSRScalarSPMV<ROWTYPE, COLTYPE, VALTYPE>::CSRScalarSPMV(CSRScalarSPMV&& other) noexcept
    : SpMVOperator<VALTYPE>(std::move(other))
    , _d_ia(other._d_ia)
    , _d_ja(other._d_ja)
    , _d_av(other._d_av)
    , _nnz(other._nnz)
    , _index_base(other._index_base)
    , _is_initialized(other._is_initialized)
    , _rows(other._rows)
{
    other._d_ia = nullptr;
    other._d_ja = nullptr;
    other._d_av = nullptr;
    other._nnz = 0;
    other._index_base = 0;
    other._is_initialized = false;
    other._rows = 0;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
CSRScalarSPMV<ROWTYPE, COLTYPE, VALTYPE>&
CSRScalarSPMV<ROWTYPE, COLTYPE, VALTYPE>::operator=(CSRScalarSPMV&& other) noexcept
{
    if (this != &other) {
        cleanup();

        SpMVOperator<VALTYPE>::operator=(std::move(other));
        _d_ia = other._d_ia;
        _d_ja = other._d_ja;
        _d_av = other._d_av;
        _nnz = other._nnz;
        _index_base = other._index_base;
        _is_initialized = other._is_initialized;
        _rows = other._rows;

        other._d_ia = nullptr;
        other._d_ja = nullptr;
        other._d_av = nullptr;
        other._nnz = 0;
        other._index_base = 0;
        other._is_initialized = false;
        other._rows = 0;
    }
    return *this;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void CSRScalarSPMV<ROWTYPE, COLTYPE, VALTYPE>::cleanup()
{
    _d_ia = nullptr;
    _d_ja = nullptr;
    _d_av = nullptr;
    _nnz = 0;
    _index_base = 0;
    _is_initialized = false;
    _rows = 0;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void CSRScalarSPMV<ROWTYPE, COLTYPE, VALTYPE>::preprocess(COLTYPE n,
                                                          const ROWTYPE* d_ia,
                                                          const COLTYPE* d_ja,
                                                          const VALTYPE* d_av,
                                                          ROWTYPE base,
                                                          ROWTYPE nnz)
{
    if (_is_initialized) {
        cleanup();
    }

    _rows = n;
    this->_n = static_cast<size_t>(n);
    if (n <= 0) {
        _is_initialized = true;
        return;
    }

    if (!d_ia) {
        throw std::runtime_error("CSRScalarSPMV: d_ia cannot be nullptr");
    }

    if (nnz < static_cast<ROWTYPE>(0)) {
        throw std::runtime_error("CSRScalarSPMV: nnz cannot be negative");
    }
    _index_base = static_cast<int>(base);
    _nnz = static_cast<size_t>(nnz);
    if (_nnz > 0 && (!d_ja || !d_av)) {
        throw std::runtime_error("CSRScalarSPMV: d_ja and d_av cannot be nullptr when nnz > 0");
    }
    _d_ia = d_ia;
    _d_ja = d_ja;
    _d_av = d_av;

    _is_initialized = true;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void CSRScalarSPMV<ROWTYPE, COLTYPE, VALTYPE>::operator()(const VALTYPE* d_x,
                                                          VALTYPE* d_y,
                                                          VALTYPE alpha,
                                                          VALTYPE beta)
{
    if (!_is_initialized) {
        throw std::runtime_error("CSRScalarSPMV: preprocess() must be called before operator()");
    }
    if (_rows <= 0) {
        return;
    }

    constexpr int block_size = 256;
    const size_t n_rows = static_cast<size_t>(_rows);
    const int num_blocks = static_cast<int>((n_rows + block_size - 1) / block_size);

    csr_scalar_spmv_kernel<<<num_blocks, block_size>>>(_rows, _d_ia, _d_ja, _d_av,
                                                       d_x, d_y, alpha, beta, _index_base);
    check_cuda_error(cudaGetLastError(), "CSRScalarSPMV kernel launch failed");
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void CSRScalarSPMV<ROWTYPE, COLTYPE, VALTYPE>::check_cuda_error(cudaError_t error,
                                                                const char* message)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + message + " - " +
                                 cudaGetErrorString(error));
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
CSRVectorSPMV<ROWTYPE, COLTYPE, VALTYPE>::CSRVectorSPMV()
    : SpMVOperator<VALTYPE>()
{
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
CSRVectorSPMV<ROWTYPE, COLTYPE, VALTYPE>::~CSRVectorSPMV()
{
    cleanup();
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
CSRVectorSPMV<ROWTYPE, COLTYPE, VALTYPE>::CSRVectorSPMV(CSRVectorSPMV&& other) noexcept
    : SpMVOperator<VALTYPE>(std::move(other))
    , _d_ia(other._d_ia)
    , _d_ja(other._d_ja)
    , _d_av(other._d_av)
    , _nnz(other._nnz)
    , _index_base(other._index_base)
    , _is_initialized(other._is_initialized)
    , _rows(other._rows)
{
    other._d_ia = nullptr;
    other._d_ja = nullptr;
    other._d_av = nullptr;
    other._nnz = 0;
    other._index_base = 0;
    other._is_initialized = false;
    other._rows = 0;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
CSRVectorSPMV<ROWTYPE, COLTYPE, VALTYPE>&
CSRVectorSPMV<ROWTYPE, COLTYPE, VALTYPE>::operator=(CSRVectorSPMV&& other) noexcept
{
    if (this != &other) {
        cleanup();

        SpMVOperator<VALTYPE>::operator=(std::move(other));
        _d_ia = other._d_ia;
        _d_ja = other._d_ja;
        _d_av = other._d_av;
        _nnz = other._nnz;
        _index_base = other._index_base;
        _is_initialized = other._is_initialized;
        _rows = other._rows;

        other._d_ia = nullptr;
        other._d_ja = nullptr;
        other._d_av = nullptr;
        other._nnz = 0;
        other._index_base = 0;
        other._is_initialized = false;
        other._rows = 0;
    }
    return *this;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void CSRVectorSPMV<ROWTYPE, COLTYPE, VALTYPE>::cleanup()
{
    _d_ia = nullptr;
    _d_ja = nullptr;
    _d_av = nullptr;
    _nnz = 0;
    _index_base = 0;
    _is_initialized = false;
    _rows = 0;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void CSRVectorSPMV<ROWTYPE, COLTYPE, VALTYPE>::preprocess(COLTYPE n,
                                                          const ROWTYPE* d_ia,
                                                          const COLTYPE* d_ja,
                                                          const VALTYPE* d_av,
                                                          ROWTYPE base,
                                                          ROWTYPE nnz)
{
    if (_is_initialized) {
        cleanup();
    }

    _rows = n;
    this->_n = static_cast<size_t>(n);
    if (n <= 0) {
        _is_initialized = true;
        return;
    }
    if (!d_ia) {
        throw std::runtime_error("CSRVectorSPMV: d_ia cannot be nullptr");
    }
    if (nnz < static_cast<ROWTYPE>(0)) {
        throw std::runtime_error("CSRVectorSPMV: nnz cannot be negative");
    }

    _index_base = static_cast<int>(base);
    _nnz = static_cast<size_t>(nnz);
    if (_nnz > 0 && (!d_ja || !d_av)) {
        throw std::runtime_error("CSRVectorSPMV: d_ja and d_av cannot be nullptr when nnz > 0");
    }
    _d_ia = d_ia;
    _d_ja = d_ja;
    _d_av = d_av;
    _is_initialized = true;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void CSRVectorSPMV<ROWTYPE, COLTYPE, VALTYPE>::operator()(const VALTYPE* d_x,
                                                          VALTYPE* d_y,
                                                          VALTYPE alpha,
                                                          VALTYPE beta)
{
    if (!_is_initialized) {
        throw std::runtime_error("CSRVectorSPMV: preprocess() must be called before operator()");
    }
    if (_rows <= 0) {
        return;
    }

    constexpr int block_size = 256;
    constexpr int warp_size = 32;
    const int warps_per_block = block_size / warp_size;
    const int num_blocks = static_cast<int>((static_cast<size_t>(_rows) + warps_per_block - 1) /
                                            warps_per_block);

    csr_vector_spmv_kernel<<<num_blocks, block_size>>>(_rows, _d_ia, _d_ja, _d_av,
                                                       d_x, d_y, alpha, beta, _index_base);
    check_cuda_error(cudaGetLastError(), "CSRVectorSPMV kernel launch failed");
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void CSRVectorSPMV<ROWTYPE, COLTYPE, VALTYPE>::check_cuda_error(cudaError_t error,
                                                                const char* message)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + message + " - " +
                                 cudaGetErrorString(error));
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
CuSparseSPMV<ROWTYPE, COLTYPE, VALTYPE>::CuSparseSPMV(cusparseHandle_t handle)
    : SpMVOperator<VALTYPE>()  // Initialize base class (_n = 0 by default)
    , _handle(handle)
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
    , _nnz(0)
    , _index_base(0)
    , _is_initialized(false)
{
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
CuSparseSPMV<ROWTYPE, COLTYPE, VALTYPE>::~CuSparseSPMV()
{
    cleanup();
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
CuSparseSPMV<ROWTYPE, COLTYPE, VALTYPE>::CuSparseSPMV(CuSparseSPMV&& other) noexcept
    : SpMVOperator<VALTYPE>(std::move(other))  // Move base class (_n)
    , _handle(other._handle)
    , _mat_A(other._mat_A)
    , _vec_x(other._vec_x)
    , _vec_y(other._vec_y)
    , _d_ia(other._d_ia)
    , _d_ja(other._d_ja)
    , _d_av(other._d_av)
    , _d_x(other._d_x)
    , _d_y(other._d_y)
    , _d_buffer(other._d_buffer)
    , _buffer_size(other._buffer_size)
    , _nnz(other._nnz)
    , _index_base(other._index_base)
    , _is_initialized(other._is_initialized)
{
    // Nullify the other object's pointers so it won't delete them
    other._handle = nullptr;
    other._mat_A = nullptr;
    other._vec_x = nullptr;
    other._vec_y = nullptr;
    other._d_ia = nullptr;
    other._d_ja = nullptr;
    other._d_av = nullptr;
    other._d_x = nullptr;
    other._d_y = nullptr;
    other._d_buffer = nullptr;
    other._buffer_size = 0;
    other._is_initialized = false;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
CuSparseSPMV<ROWTYPE, COLTYPE, VALTYPE>& CuSparseSPMV<ROWTYPE, COLTYPE, VALTYPE>::operator=(CuSparseSPMV&& other) noexcept
{
    if (this != &other) {
        // Clean up existing resources
        cleanup();
        // Note: _handle is not owned by this class, so we don't destroy it
        
        // Move base class (_n)
        SpMVOperator<VALTYPE>::operator=(std::move(other));
        
        // Move resources from other
        _handle = other._handle;
        _mat_A = other._mat_A;
        _vec_x = other._vec_x;
        _vec_y = other._vec_y;
        _d_ia = other._d_ia;
        _d_ja = other._d_ja;
        _d_av = other._d_av;
        _d_x = other._d_x;
        _d_y = other._d_y;
        _d_buffer = other._d_buffer;
        _buffer_size = other._buffer_size;
        _nnz = other._nnz;
        _index_base = other._index_base;
        _is_initialized = other._is_initialized;
        
        // Nullify other's pointers
        other._handle = nullptr;
        other._mat_A = nullptr;
        other._vec_x = nullptr;
        other._vec_y = nullptr;
        other._d_ia = nullptr;
        other._d_ja = nullptr;
        other._d_av = nullptr;
        other._d_x = nullptr;
        other._d_y = nullptr;
        other._d_buffer = nullptr;
        other._buffer_size = 0;
        other._is_initialized = false;
    }
    return *this;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void CuSparseSPMV<ROWTYPE, COLTYPE, VALTYPE>::cleanup()
{
    // Destroy descriptors
    if (_mat_A) cusparseDestroySpMat(_mat_A);
    if (_vec_x) cusparseDestroyDnVec(_vec_x);
    if (_vec_y) cusparseDestroyDnVec(_vec_y);
    
    // Free owned device memory
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
void CuSparseSPMV<ROWTYPE, COLTYPE, VALTYPE>::preprocess(COLTYPE n,
                                                      const ROWTYPE* d_ia,
                                                      const COLTYPE* d_ja,
                                                      const VALTYPE* d_av,
                                                      ROWTYPE base,
                                                      ROWTYPE nnz)
{
    // Clean up any previous initialization
    if (_is_initialized) {
        cleanup();
    }
    
    // Store matrix properties
    this->_n = static_cast<size_t>(n);  // Set base class member
    if (n <= 0) {
        _is_initialized = true;
        return;
    }
    if (!d_ia) {
        throw std::runtime_error("CuSparseSPMV: d_ia cannot be nullptr");
    }
    if (nnz < static_cast<ROWTYPE>(0)) {
        throw std::runtime_error("CuSparseSPMV: nnz cannot be negative");
    }
    _index_base = static_cast<int>(base);
    _nnz = static_cast<size_t>(nnz);
    if (_nnz > 0 && (!d_ja || !d_av)) {
        throw std::runtime_error("CuSparseSPMV: d_ja and d_av cannot be nullptr when nnz > 0");
    }

    _d_ia = d_ia;
    _d_ja = d_ja;
    _d_av = d_av;

    // Allocate device memory for vectors (used by cuSPARSE descriptors)
    check_cuda_error(cudaMalloc(&_d_x, n * sizeof(VALTYPE)), "Failed to allocate d_x");
    check_cuda_error(cudaMalloc(&_d_y, n * sizeof(VALTYPE)), "Failed to allocate d_y");
    
    // Create matrix descriptor
    cusparseIndexBase_t index_base = (_index_base == 0) ? CUSPARSE_INDEX_BASE_ZERO : CUSPARSE_INDEX_BASE_ONE;
    
    check_cusparse_error(
        cusparseCreateCsr(&_mat_A, n, n, _nnz,
                         const_cast<ROWTYPE*>(_d_ia),
                         const_cast<COLTYPE*>(_d_ja),
                         const_cast<VALTYPE*>(_d_av),
                         get_index_type(), get_index_type(),
                         index_base, get_cuda_data_type()),
        "Failed to create CSR matrix descriptor");
    
    // Create vector descriptors with device buffers (values replaced in operator())
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
void CuSparseSPMV<ROWTYPE, COLTYPE, VALTYPE>::operator()(const VALTYPE* d_x,
                                                 VALTYPE* d_y,
                                                 VALTYPE alpha,
                                                 VALTYPE beta)
{
    if (!_is_initialized) {
        throw std::runtime_error("CuSparseSPMV: preprocess() must be called before operator()");
    }
    if (this->_n == 0) {
        return;
    }
    
    // Update vector descriptors to point to the provided device memory
    check_cusparse_error(
        cusparseDnVecSetValues(_vec_x, const_cast<VALTYPE*>(d_x)),
        "Failed to set vector x values");
    
    check_cusparse_error(
        cusparseDnVecSetValues(_vec_y, d_y),
        "Failed to set vector y values");
    
    // Perform SpMV: y = alpha * A * x + beta * y
    check_cusparse_error(
        cusparseSpMV(_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, _mat_A, _vec_x, &beta, _vec_y,
                    get_cuda_data_type(), CUSPARSE_SPMV_ALG_DEFAULT,
                    _d_buffer),
        "Failed to execute SpMV");
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
cudaDataType CuSparseSPMV<ROWTYPE, COLTYPE, VALTYPE>::get_cuda_data_type()
{
    if constexpr (std::is_same_v<VALTYPE, double>) {
        return CUDA_R_64F;
    } else if constexpr (std::is_same_v<VALTYPE, float>) {
        return CUDA_R_32F;
    } else {
        throw std::runtime_error("Unsupported value type for CuSparseSPMV");
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
cusparseIndexType_t CuSparseSPMV<ROWTYPE, COLTYPE, VALTYPE>::get_index_type()
{
    if constexpr (sizeof(ROWTYPE) == sizeof(int32_t)) {
        return CUSPARSE_INDEX_32I;
    } else if constexpr (sizeof(ROWTYPE) == sizeof(int64_t)) {
        return CUSPARSE_INDEX_64I;
    } else {
        throw std::runtime_error("Unsupported index type for CuSparseSPMV");
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void CuSparseSPMV<ROWTYPE, COLTYPE, VALTYPE>::check_cusparse_error(cusparseStatus_t status, const char* message)
{
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuSPARSE error: ") + message);
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void CuSparseSPMV<ROWTYPE, COLTYPE, VALTYPE>::check_cuda_error(cudaError_t error, const char* message)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + message + " - " + cudaGetErrorString(error));
    }
}

// Explicit template instantiations for common types
template void copy_csr_host_to_device<int, int, double>(int, const int*, const int*, const double*, int**, int**, double**, int*);
template void copy_csr_host_to_device<int, int, float>(int, const int*, const int*, const float*, int**, int**, float**, int*);
template void copy_csr_host_to_device<int64_t, int64_t, double>(int64_t, const int64_t*, const int64_t*, const double*,
                                                                int64_t**, int64_t**, double**, int64_t*);
template void copy_csr_host_to_device<int64_t, int64_t, float>(int64_t, const int64_t*, const int64_t*, const float*,
                                                               int64_t**, int64_t**, float**, int64_t*);

template void copy_csr_device_to_host<int, int, double>(int, const int*, const int*, const double*, int*, int*, double*, int);
template void copy_csr_device_to_host<int, int, float>(int, const int*, const int*, const float*, int*, int*, float*, int);
template void copy_csr_device_to_host<int64_t, int64_t, double>(int64_t, const int64_t*, const int64_t*, const double*,
                                                                int64_t*, int64_t*, double*, int64_t);
template void copy_csr_device_to_host<int64_t, int64_t, float>(int64_t, const int64_t*, const int64_t*, const float*,
                                                               int64_t*, int64_t*, float*, int64_t);

template class CuSparseSPMV<int, int, double>;
template class CuSparseSPMV<int, int, float>;
template class CuSparseSPMV<int64_t, int64_t, double>;
template class CuSparseSPMV<int64_t, int64_t, float>;

template class CSRScalarSPMV<int, int, double>;
template class CSRScalarSPMV<int, int, float>;
template class CSRScalarSPMV<int64_t, int64_t, double>;
template class CSRScalarSPMV<int64_t, int64_t, float>;

template class CSRVectorSPMV<int, int, double>;
template class CSRVectorSPMV<int, int, float>;
template class CSRVectorSPMV<int64_t, int64_t, double>;
template class CSRVectorSPMV<int64_t, int64_t, float>;

} // namespace matrix_utils
