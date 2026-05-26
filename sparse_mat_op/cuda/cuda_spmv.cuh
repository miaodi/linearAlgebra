#pragma once

#include <cusparse.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace matrix_utils::sparse_cuda
{

/**
 * @brief Virtual interface for SpMV operations: y = alpha * A * x + beta * y
 *
 * This interface allows decoupling SpMV implementations from solvers,
 * enabling different SpMV backends to be used interchangeably.
 */
template <typename VALTYPE = double>
class SpMVOperator
{
public:
    virtual ~SpMVOperator() = default;

    /**
     * @brief Perform SpMV: y = alpha * A * x + beta * y
     *
     * @param d_x Input vector x (device memory)
     * @param d_y Output vector y (device memory)
     * @param alpha Scalar multiplier for A*x (default: 1.0)
     * @param beta Scalar multiplier for y (default: 0.0)
     */
    virtual void operator()( const VALTYPE* d_x, VALTYPE* d_y, VALTYPE alpha = 1.0, VALTYPE beta = 0.0 ) = 0;

    /**
     * @brief Get matrix size (number of rows/columns)
     *
     * @return Matrix dimension
     */
    size_t size() const { return _n; }

protected:
    /**
     * @brief Matrix dimension (number of rows/columns)
     * Derived classes should set this during initialization/preprocessing
     */
    size_t _n = 0;
};

/**
 * @brief Allocate device CSR arrays and copy from host.
 *
 * Caller owns the returned device pointers and must cudaFree them.
 */
template <typename ROWTYPE = int, typename COLTYPE = int, typename VALTYPE = double>
void copy_csr_host_to_device( COLTYPE n,
                              const ROWTYPE* h_ia,
                              const COLTYPE* h_ja,
                              const VALTYPE* h_av,
                              ROWTYPE** d_ia,
                              COLTYPE** d_ja,
                              VALTYPE** d_av,
                              ROWTYPE* nnz_out = nullptr );

/**
 * @brief Copy CSR arrays from device to host.
 *
 * Caller must provide host buffers sized for (n + 1) row pointers and nnz entries.
 */
template <typename ROWTYPE = int, typename COLTYPE = int, typename VALTYPE = double>
void copy_csr_device_to_host( COLTYPE n,
                              const ROWTYPE* d_ia,
                              const COLTYPE* d_ja,
                              const VALTYPE* d_av,
                              ROWTYPE* h_ia,
                              COLTYPE* h_ja,
                              VALTYPE* h_av,
                              ROWTYPE nnz );

/**
 * @brief CSR-scalar SpMV: one thread per row on the GPU
 *
 * This kernel is a simple CSR implementation intended for correctness
 * and baseline performance comparisons.
 */
template <typename ROWTYPE = int, typename COLTYPE = int, typename VALTYPE = double>
class CSRScalarSPMV : public SpMVOperator<VALTYPE>
{
public:
    CSRScalarSPMV();
    ~CSRScalarSPMV();

    CSRScalarSPMV( const CSRScalarSPMV& ) = delete;
    CSRScalarSPMV& operator=( const CSRScalarSPMV& ) = delete;

    CSRScalarSPMV( CSRScalarSPMV&& other ) noexcept;
    CSRScalarSPMV& operator=( CSRScalarSPMV&& other ) noexcept;

    /**
     * @brief Preprocess the matrix structure for subsequent SpMV operations
     *
     * @param n Matrix dimension (number of rows)
     * @param d_ia Row pointers (size n+1, device memory; non-owning)
     * @param d_ja Column indices (device memory; non-owning)
     * @param d_av Matrix values (device memory; non-owning)
     * @param base CSR index base (0 or 1)
     * @param nnz Number of non-zeros in the matrix
     */
    void preprocess( COLTYPE n, const ROWTYPE* d_ia, const COLTYPE* d_ja, const VALTYPE* d_av, ROWTYPE base, ROWTYPE nnz );

    /**
     * @brief Perform SpMV: y = alpha * A * x + beta * y
     *
     * Expects device memory pointers for x and y.
     */
    void operator()( const VALTYPE* d_x, VALTYPE* d_y, VALTYPE alpha = 1.0, VALTYPE beta = 0.0 ) override;

    using VALTYPE_ALIAS = VALTYPE;

private:
    const ROWTYPE* _d_ia = nullptr;
    const COLTYPE* _d_ja = nullptr;
    const VALTYPE* _d_av = nullptr;
    ROWTYPE _nnz = 0;
    ROWTYPE _index_base = 0;
    bool _is_initialized = false;
    COLTYPE _rows = 0;

    void cleanup();
    void check_cuda_error( cudaError_t error, const char* message );
};

/**
 * @brief CSR-vector SpMV: one warp per row on the GPU
 *
 * Based on "Efficient sparse matrix-vector multiplication on cache-based GPUs"
 * (Bell and Garland). Each warp cooperatively processes a row and reduces
 * the partial sums.
 */
template <typename ROWTYPE = int, typename COLTYPE = int, typename VALTYPE = double>
class CSRVectorSPMV : public SpMVOperator<VALTYPE>
{
public:
    CSRVectorSPMV();
    ~CSRVectorSPMV();

    CSRVectorSPMV( const CSRVectorSPMV& ) = delete;
    CSRVectorSPMV& operator=( const CSRVectorSPMV& ) = delete;

    CSRVectorSPMV( CSRVectorSPMV&& other ) noexcept;
    CSRVectorSPMV& operator=( CSRVectorSPMV&& other ) noexcept;

    /**
     * @brief Preprocess the matrix structure for subsequent SpMV operations
     *
     * @param n Matrix dimension (number of rows)
     * @param d_ia Row pointers (size n+1, device memory; non-owning)
     * @param d_ja Column indices (device memory; non-owning)
     * @param d_av Matrix values (device memory; non-owning)
     * @param base CSR index base (0 or 1)
     * @param nnz Number of non-zeros in the matrix
     */
    void preprocess( COLTYPE n, const ROWTYPE* d_ia, const COLTYPE* d_ja, const VALTYPE* d_av, ROWTYPE base, ROWTYPE nnz );

    /**
     * @brief Perform SpMV: y = alpha * A * x + beta * y
     *
     * Expects device memory pointers for x and y.
     */
    void operator()( const VALTYPE* d_x, VALTYPE* d_y, VALTYPE alpha = 1.0, VALTYPE beta = 0.0 ) override;

    using VALTYPE_ALIAS = VALTYPE;

private:
    const ROWTYPE* _d_ia = nullptr;
    const COLTYPE* _d_ja = nullptr;
    const VALTYPE* _d_av = nullptr;
    ROWTYPE _nnz = 0;
    ROWTYPE _index_base = 0;
    bool _is_initialized = false;
    COLTYPE _rows = 0;

    void cleanup();
    void check_cuda_error( cudaError_t error, const char* message );
};

/**
 * @brief cuSPARSE-based SpMV for y = alpha * A * x + beta * y
 *
 * This class provides a CUDA-accelerated sparse matrix-vector multiplication
 * that is compatible with the matrix_utils::SPMV wrapper interface.
 *
 * Key features:
 * - Uses cuSPARSE SpMV for optimal GPU performance
 * - Automatic management of cuSPARSE descriptors and buffers
 * - Support for alpha/beta scaling (y = alpha * A * x + beta * y)
 * - Compatible with CSR matrices in 0-based or 1-based indexing
 *
 * Usage:
 *   cusparseHandle_t handle;
 *   cusparseCreate(&handle);
 *   copy_csr_host_to_device(n, ia, ja, av, &d_ia, &d_ja, &d_av);
 *   CuSparseSPMV<int, int, double> spmv(handle);
 *   spmv.preprocess(n, d_ia, d_ja, d_av, ia[0], ia[n] - ia[0]);
 *   spmv(x, y, alpha, beta);
 *   cusparseDestroy(handle);
 */
template <typename ROWTYPE = int, typename COLTYPE = int, typename VALTYPE = double>
class CuSparseSPMV : public SpMVOperator<VALTYPE>
{
public:
    explicit CuSparseSPMV( cusparseHandle_t handle );
    ~CuSparseSPMV();

    // Delete copy constructor and copy assignment to prevent double-free
    CuSparseSPMV( const CuSparseSPMV& ) = delete;
    CuSparseSPMV& operator=( const CuSparseSPMV& ) = delete;

    // Move constructor and move assignment
    CuSparseSPMV( CuSparseSPMV&& other ) noexcept;
    CuSparseSPMV& operator=( CuSparseSPMV&& other ) noexcept;

    /**
     * @brief Preprocess the matrix structure for subsequent SpMV operations
     *
     * This sets up cuSPARSE descriptors and auxiliary buffers.
     * Must be called before the first SpMV operation.
     *
     * @param n Matrix dimension (number of rows)
     * @param d_ia Row pointers (size n+1, device memory; non-owning)
     * @param d_ja Column indices (device memory; non-owning)
     * @param d_av Matrix values (device memory; non-owning)
     * @param base CSR index base (0 or 1)
     * @param nnz Number of non-zeros in the matrix
     */
    void preprocess( COLTYPE n, const ROWTYPE* d_ia, const COLTYPE* d_ja, const VALTYPE* d_av, ROWTYPE base, ROWTYPE nnz );

    /**
     * @brief Perform SpMV: y = alpha * A * x + beta * y
     *
     * Expects device memory pointers for x and y.
     * Implements the SpMVOperator interface.
     *
     * @param d_x Input vector x (device memory, size n)
     * @param d_y Output vector y (device memory, size n)
     * @param alpha Scalar multiplier for A*x
     * @param beta Scalar multiplier for y
     */
    void operator()( const VALTYPE* d_x, VALTYPE* d_y, VALTYPE alpha = 1.0, VALTYPE beta = 0.0 ) override;

    using VALTYPE_ALIAS = VALTYPE;

private:
    // CUDA handles
    cusparseHandle_t _handle;

    // Matrix descriptor
    cusparseSpMatDescr_t _mat_A;

    // Vector descriptors
    cusparseDnVecDescr_t _vec_x;
    cusparseDnVecDescr_t _vec_y;

    // Device memory for vectors (descriptor placeholders)
    VALTYPE* _d_x;
    VALTYPE* _d_y;

    // Device memory for matrix
    const ROWTYPE* _d_ia;
    const COLTYPE* _d_ja;
    const VALTYPE* _d_av;

    // cuSPARSE buffer for SpMV
    void* _d_buffer;
    size_t _buffer_size;

    // Matrix properties
    ROWTYPE _nnz;
    ROWTYPE _index_base;
    bool _is_initialized;

    // Helper functions
    void cleanup();
    void check_cusparse_error( cusparseStatus_t status, const char* message );
    void check_cuda_error( cudaError_t error, const char* message );

    // Get cuSPARSE data type
    cudaDataType get_cuda_data_type();
    cusparseIndexType_t get_index_type();
};

/**
 * @brief Merge-based CSR SpMV implementation
 *
 * Based on "Merge-Based Parallel Sparse Matrix-Vector Multiplication"
 * by Merrill & Garland (2016). Uses merge path partitioning for load balancing.
 *
 * Key features:
 * - Merge path decomposition for balanced workload distribution
 * - Better performance on irregular sparsity patterns
 * - Cooperative thread block processing
 *
 * Algorithm overview:
 * 1. Treat matrix as linearized array of (row_id, col_id, value) tuples
 * 2. Use merge path to partition work among thread blocks
 * 3. Each block cooperatively processes its assigned nonzeros
 * 4. Use shared memory for partial sums and atomic reduction
 */
template <typename ROWTYPE = int, typename COLTYPE = int, typename VALTYPE = double>
class CSRMergeSPMV : public SpMVOperator<VALTYPE>
{
public:
    CSRMergeSPMV();
    ~CSRMergeSPMV();

    CSRMergeSPMV( const CSRMergeSPMV& ) = delete;
    CSRMergeSPMV& operator=( const CSRMergeSPMV& ) = delete;

    CSRMergeSPMV( CSRMergeSPMV&& other ) noexcept;
    CSRMergeSPMV& operator=( CSRMergeSPMV&& other ) noexcept;

    void preprocess( COLTYPE n, const ROWTYPE* d_ia, const COLTYPE* d_ja, const VALTYPE* d_av, ROWTYPE base, ROWTYPE nnz );

    void operator()( const VALTYPE* d_x, VALTYPE* d_y, VALTYPE alpha = 1.0, VALTYPE beta = 0.0 ) override;

private:
    const ROWTYPE* _d_ia;
    const COLTYPE* _d_ja;
    const VALTYPE* _d_av;
    ROWTYPE _nnz;
    ROWTYPE _index_base;
    bool _is_initialized;
    COLTYPE _rows;

    // Precomputed merge path boundaries for each block
    ROWTYPE* _d_merge_path_boundaries;
    int _num_blocks;

    void cleanup();
    void check_cuda_error( cudaError_t error, const char* message );
    void compute_merge_path_boundaries();
};

} // namespace matrix_utils::sparse_cuda
