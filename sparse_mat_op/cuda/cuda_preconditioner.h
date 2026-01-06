#pragma once

#include "cuda_memory.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <memory>

namespace cuda_iterative_solver
{
/**
 * @brief Abstract base class for preconditioners
 *
 * All preconditioner implementations must inherit from this class and implement
 * the operator() method to apply the preconditioner.
 */
class Preconditioner
{
public:
    virtual ~Preconditioner() = default;

    /**
     * @brief Apply preconditioner: solve M * y = x for y
     *
     * @param d_input Input vector x (device memory)
     * @param d_output Output vector y = M^{-1} * x (device memory)
     */
    virtual void operator()(const DeviceVectorView& d_input, DeviceVectorView& d_output) = 0;

    /**
     * @brief Check if preconditioner is properly initialized
     * @return true if preconditioner is ready to use
     */
    virtual bool isSetup() const = 0;
};

/**
 * @brief No-op preconditioner (identity operation)
 *
 * This preconditioner simply copies the input to output, effectively doing nothing.
 * Used when no preconditioning is desired.
 */
class NoPreconditioner : public Preconditioner
{
public:
    NoPreconditioner() = default;
    ~NoPreconditioner() override = default;

    void operator()(const DeviceVectorView& d_input, DeviceVectorView& d_output) override;

    bool isSetup() const override { return true; }
};

/**
 * @brief Jacobi preconditioner using diagonal scaling
 *
 * This preconditioner applies diagonal scaling: M^{-1} = D^{-1}
 * where D is the diagonal of the matrix. Efficient for diagonally dominant matrices.
 */
class JacobiPreconditioner : public Preconditioner
{
public:
    JacobiPreconditioner();
    ~JacobiPreconditioner() override;

    /**
     * @brief Setup Jacobi preconditioner with diagonal values from host
     *
     * @param n Vector size
     * @param h_diag Diagonal values (size n, host data)
     */
    void setup(size_t n, const double* h_diag);

    /**
     * @brief Setup Jacobi preconditioner by extracting diagonal from CSR matrix
     *
     * @param n Matrix size
     * @param h_ia Row pointers (size n+1, host data)
     * @param h_ja Column indices (host data)
     * @param h_va Values (host data)
     */
    void setupFromMatrix(size_t n, const int* h_ia, const int* h_ja, const double* h_va);

    void operator()(const DeviceVectorView& d_input, DeviceVectorView& d_output) override;

    bool isSetup() const override { return _is_setup; }

private:
    // Device memory for inverse diagonal
    DeviceArray<double> _d_inv_diag;

    size_t _n;
    bool _is_setup;

    void cleanup();
};

/**
 * @brief Base class for cuSPARSE-based ILU preconditioners
 *
 * Provides common implementation for applying ILU preconditioner using
 * separate L and U matrix descriptors with cuSPARSE.
 */
class CuSparseILUPrecBase : public Preconditioner
{
protected:
    // Temporary storage for intermediate results
    DeviceArray<double> _d_tmp;
    DeviceVectorView _view_tmp;

    // cuSPARSE handle (stored for use in operator())
    cusparseHandle_t _cusparse_handle;

    // cuSPARSE descriptors (to be set by derived classes)
    cusparseSpMatDescr_t _mat_L;
    cusparseSpMatDescr_t _mat_U;
    cusparseSpSVDescr_t _spv_descr_L;
    cusparseSpSVDescr_t _spv_descr_U;

    bool _is_setup;

    explicit CuSparseILUPrecBase(cusparseHandle_t handle);
    ~CuSparseILUPrecBase() override;

public:
    void operator()(const DeviceVectorView& d_input, DeviceVectorView& d_output) override;

    bool isSetup() const override { return _is_setup; }
};

/**
 * @brief cuSPARSE-based ILU preconditioner using L and U factors from host
 *
 * This preconditioner applies incomplete LU factorization where the L and U factors
 * are computed externally (on host) and provided during setup.
 * Solves M^{-1} * x by solving L * U * y = x using two triangular solves.
 */
class CuSparseILUPrec : public CuSparseILUPrecBase
{
public:
    explicit CuSparseILUPrec(cusparseHandle_t handle);
    ~CuSparseILUPrec() override;

    /**
     * @brief Setup ILU preconditioner with L and U factors from host
     *
     * @param n Matrix size
     * @param h_ia_L Row pointers for L factor (size n+1, host data)
     * @param h_ja_L Column indices for L factor (host data)
     * @param h_va_L Values for L factor (host data)
     * @param h_ia_U Row pointers for U factor (size n+1, host data)
     * @param h_ja_U Column indices for U factor (host data)
     * @param h_va_U Values for U factor (host data)
     */
    void setup(size_t n, const int* h_ia_L, const int* h_ja_L, const double* h_va_L,
               const int* h_ia_U, const int* h_ja_U, const double* h_va_U);

private:
    // Device memory arrays for L and U factors
    DeviceArray<int> _d_ia_L;
    DeviceArray<int> _d_ja_L;
    DeviceArray<double> _d_va_L;
    DeviceArray<int> _d_ia_U;
    DeviceArray<int> _d_ja_U;
    DeviceArray<double> _d_va_U;

    // Buffers for SpSV operations
    DeviceArray<char> _d_buffer_L;
    DeviceArray<char> _d_buffer_U;

    // Matrix properties
    size_t _n;
    size_t _nnz_L;
    size_t _nnz_U;
    int _index_base_L;
    int _index_base_U;

    void cleanup();
};

/**
 * @brief cuSPARSE-based ILU0 preconditioner with GPU factorization
 *
 * This preconditioner computes ILU0 factorization directly on GPU using cuSPARSE.
 * The factorization is performed in-place on a copy of the input matrix.
 */
class CuSparseILU0Prec : public CuSparseILUPrecBase
{
public:
    explicit CuSparseILU0Prec(cusparseHandle_t handle);
    ~CuSparseILU0Prec() override;

    /**
     * @brief Setup ILU0 preconditioner by factorizing matrix on GPU
     *
     * @param n Matrix size
     * @param h_ia Row pointers for matrix (size n+1, host data)
     * @param h_ja Column indices for matrix (host data)
     * @param h_va Values for matrix (host data)
     */
    void setup(size_t n, const int* h_ia, const int* h_ja, const double* h_va);

    /**
     * @brief Setup ILU0 preconditioner using matrix already on device
     *
     * @param n Matrix size
     * @param nnz Number of non-zeros
     * @param d_ia Row pointers for matrix (size n+1, device data)
     * @param d_ja Column indices for matrix (device data)
     * @param d_va Values for matrix (device data)
     * @param index_base Index base (0 or 1)
     */
    void setupFromDevice(size_t n, size_t nnz, const int* d_ia, const int* d_ja, const double* d_va, int index_base);

private:
    // Device memory for ILU0 factorization (stored in-place)
    DeviceArray<int> _d_ia;
    DeviceArray<int> _d_ja;
    DeviceArray<double> _d_va;

// ILU0-specific descriptors (using deprecated API)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    csrilu02Info_t _ilu0_info;
#pragma GCC diagnostic pop

    // Buffers for ILU0 factorization and SpSV operations
    DeviceArray<char> _d_ilu0_buffer;
    DeviceArray<char> _d_buffer_L;
    DeviceArray<char> _d_buffer_U;

    void cleanup();
    void performFactorization(size_t n, size_t nnz, int index_base, int* d_ia, int* d_ja, double* d_va);
    void setupSpSVDescriptors(size_t n, size_t nnz, int index_base, const int* d_ia,
                              const int* d_ja, const double* d_va);
};

} // namespace cuda_iterative_solver
