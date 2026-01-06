#pragma once

#include <cusparse.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace matrix_utils
{

/**
 * @brief cuSPARSE-based SpMV for y = alpha * A * x + beta * y
 * 
 * This class provides a CUDA-accelerated sparse matrix-vector multiplication
 * that is compatible with the matrix_utils::SPMV wrapper interface.
 * 
 * Key features:
 * - Uses cuSPARSE SpMV for optimal GPU performance
 * - Automatic memory management with device arrays
 * - Support for alpha/beta scaling (y = alpha * A * x + beta * y)
 * - Compatible with CSR matrices in 0-based or 1-based indexing
 * 
 * Usage:
 *   cusparseHandle_t handle;
 *   cusparseCreate(&handle);
 *   CuSparseSPMV<int, int, double> spmv(handle);
 *   spmv.preprocess(n, ia, ja, av);
 *   spmv(x, y, alpha, beta);
 *   cusparseDestroy(handle);
 */
template <typename ROWTYPE = int, typename COLTYPE = int, typename VALTYPE = double>
class CuSparseSPMV
{
public:
    explicit CuSparseSPMV(cusparseHandle_t handle);
    ~CuSparseSPMV();
    
    // Delete copy constructor and copy assignment to prevent double-free
    CuSparseSPMV(const CuSparseSPMV&) = delete;
    CuSparseSPMV& operator=(const CuSparseSPMV&) = delete;
    
    // Move constructor and move assignment
    CuSparseSPMV(CuSparseSPMV&& other) noexcept;
    CuSparseSPMV& operator=(CuSparseSPMV&& other) noexcept;

    /**
     * @brief Preprocess the matrix structure for subsequent SpMV operations
     * 
     * This sets up cuSPARSE descriptors and allocates device memory.
     * Must be called before the first SpMV operation.
     * 
     * @param n Matrix dimension (number of rows)
     * @param h_ia Row pointers (size n+1, host memory)
     * @param h_ja Column indices (host memory)
     * @param h_av Matrix values (host memory)
     */
    void preprocess(COLTYPE n,
                   const ROWTYPE* h_ia,
                   const COLTYPE* h_ja,
                   const VALTYPE* h_av);

    /**
     * @brief Perform SpMV: y = alpha * A * x + beta * y
     * 
     * Expects device memory pointers for x and y.
     * 
     * @param d_x Input vector x (device memory, size n)
     * @param d_y Output vector y (device memory, size n)
     * @param alpha Scalar multiplier for A*x
     * @param beta Scalar multiplier for y
     */
    void operator()(const VALTYPE* d_x, VALTYPE* d_y, VALTYPE alpha = 1.0, VALTYPE beta = 0.0);

    /**
     * @brief Get device memory pointers for vectors (for direct access)
     */
    VALTYPE* get_device_x() { return _d_x; }
    VALTYPE* get_device_y() { return _d_y; }
    const VALTYPE* get_device_x() const { return _d_x; }
    const VALTYPE* get_device_y() const { return _d_y; }
    
    /**
     * @brief Get matrix size
     */
    COLTYPE size() const { return _n; }
    
    using VALTYPE_ALIAS = VALTYPE;

private:
    // CUDA handles
    cusparseHandle_t _handle;
    
    // Matrix descriptor
    cusparseSpMatDescr_t _mat_A;
    
    // Vector descriptors
    cusparseDnVecDescr_t _vec_x;
    cusparseDnVecDescr_t _vec_y;
    
    // Device memory for matrix
    ROWTYPE* _d_ia;
    COLTYPE* _d_ja;
    VALTYPE* _d_av;
    
    // Device memory for vectors
    VALTYPE* _d_x;
    VALTYPE* _d_y;
    
    // cuSPARSE buffer for SpMV
    void* _d_buffer;
    size_t _buffer_size;
    
    // Matrix properties
    COLTYPE _n;
    size_t _nnz;
    int _index_base;
    bool _is_initialized;
    
    // Helper functions
    void cleanup();
    void check_cusparse_error(cusparseStatus_t status, const char* message);
    void check_cuda_error(cudaError_t error, const char* message);
    
    // Get cuSPARSE data type
    cudaDataType get_cuda_data_type();
    cusparseIndexType_t get_index_type();
};

} // namespace matrix_utils
