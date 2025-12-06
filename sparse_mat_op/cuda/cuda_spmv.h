#pragma once

#include <cusparse.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace matrix_utils
{

/**
 * @brief CUDA-based SpMV using cuSPARSE for y = alpha * A * x + beta * y
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
 *   CudaSPMV<int, int, double> spmv;
 *   spmv.preprocess(n, ia, ja, av);
 *   spmv(n, base, ia, ja, av, x, y, alpha, beta);
 */
template <typename ROWTYPE = int, typename COLTYPE = int, typename VALTYPE = double>
class CudaSPMV
{
public:
    CudaSPMV();
    ~CudaSPMV();

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
     * @param h_x Input vector x (host memory, size n)
     * @param h_y Output vector y (host memory, size n)
     * @param alpha Scalar multiplier for A*x
     * @param beta Scalar multiplier for y
     */
    void operator()(const VALTYPE* h_x,
                   VALTYPE* h_y,
                   VALTYPE alpha = 1.0,
                   VALTYPE beta = 0.0);

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
