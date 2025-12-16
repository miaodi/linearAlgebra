#pragma once

#include <cusparse.h>
#include <cuda_runtime.h>
#include <memory>

namespace cuda_iterative_solver
{

// Forward declarations
template<typename T, typename Allocator>
class Array;

struct DeviceAllocator;

template<typename T>
using DeviceArray = Array<T, DeviceAllocator>;

class DeviceVectorView;

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
     * @param handle cuSPARSE handle for sparse operations
     * @param d_input Input vector x (device memory)
     * @param d_output Output vector y = M^{-1} * x (device memory)
     */
    virtual void operator()(cusparseHandle_t handle, 
                          const DeviceVectorView& d_input, 
                          DeviceVectorView& d_output) = 0;
    
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
    
    void operator()(cusparseHandle_t handle, 
                   const DeviceVectorView& d_input, 
                   DeviceVectorView& d_output) override;
    
    bool isSetup() const override { return true; }
};

/**
 * @brief Base class for ILU-based preconditioners
 * 
 * Provides common implementation for applying ILU preconditioner using
 * separate L and U matrix descriptors.
 */
class ILUPreconditionerBase : public Preconditioner
{
protected:
    // Temporary storage for intermediate results
    DeviceArray<double>* _d_tmp;
    DeviceVectorView* _view_tmp;
    
    // cuSPARSE descriptors (to be set by derived classes)
    cusparseSpMatDescr_t _mat_L;
    cusparseSpMatDescr_t _mat_U;
    cusparseSpSVDescr_t _spv_descr_L;
    cusparseSpSVDescr_t _spv_descr_U;
    
    bool _is_setup;
    
    ILUPreconditionerBase();
    ~ILUPreconditionerBase() override;
    
public:
    void operator()(cusparseHandle_t handle,
                   const DeviceVectorView& d_input,
                   DeviceVectorView& d_output) override;
    
    bool isSetup() const override { return _is_setup; }
};

/**
 * @brief ILU preconditioner using L and U factors provided from host
 * 
 * This preconditioner applies incomplete LU factorization where the L and U factors
 * are computed externally (on host) and provided during setup.
 * Solves M^{-1} * x by solving L * U * y = x using two triangular solves.
 */
class ILUPreconditioner : public ILUPreconditionerBase
{
public:
    ILUPreconditioner();
    ~ILUPreconditioner() override;
    
    /**
     * @brief Setup ILU preconditioner with L and U factors from host
     * 
     * @param handle cuSPARSE handle for setup operations
     * @param n Matrix size
     * @param h_ia_L Row pointers for L factor (size n+1, host data)
     * @param h_ja_L Column indices for L factor (host data)
     * @param h_va_L Values for L factor (host data)
     * @param h_ia_U Row pointers for U factor (size n+1, host data)
     * @param h_ja_U Column indices for U factor (host data)
     * @param h_va_U Values for U factor (host data)
     */
    void setup(cusparseHandle_t handle, size_t n,
              const int* h_ia_L, const int* h_ja_L, const double* h_va_L,
              const int* h_ia_U, const int* h_ja_U, const double* h_va_U);
    
private:
    // Device memory arrays for L and U factors
    DeviceArray<int>* _d_ia_L;
    DeviceArray<int>* _d_ja_L;
    DeviceArray<double>* _d_va_L;
    DeviceArray<int>* _d_ia_U;
    DeviceArray<int>* _d_ja_U;
    DeviceArray<double>* _d_va_U;
    
    // Buffers for SpSV operations
    DeviceArray<char>* _d_buffer_L;
    DeviceArray<char>* _d_buffer_U;
    
    // Matrix properties
    size_t _n;
    size_t _nnz_L;
    size_t _nnz_U;
    int _index_base_L;
    int _index_base_U;
    
    void cleanup();
};

/**
 * @brief ILU0 preconditioner using GPU-based factorization
 * 
 * This preconditioner computes ILU0 factorization directly on GPU using cuSPARSE.
 * The factorization is performed in-place on a copy of the input matrix.
 */
class GPUILU0Preconditioner : public ILUPreconditionerBase
{
public:
    GPUILU0Preconditioner();
    ~GPUILU0Preconditioner() override;
    
    /**
     * @brief Setup ILU0 preconditioner by factorizing matrix on GPU
     * 
     * @param handle cuSPARSE handle for factorization and solve operations
     * @param n Matrix size
     * @param h_ia Row pointers for matrix (size n+1, host data)
     * @param h_ja Column indices for matrix (host data)
     * @param h_va Values for matrix (host data)
     */
    void setup(cusparseHandle_t handle, size_t n,
              const int* h_ia, const int* h_ja, const double* h_va);
    
    /**
     * @brief Setup ILU0 preconditioner using matrix already on device
     * 
     * @param handle cuSPARSE handle for factorization and solve operations
     * @param n Matrix size
     * @param nnz Number of non-zeros
     * @param d_ia Row pointers for matrix (size n+1, device data)
     * @param d_ja Column indices for matrix (device data)
     * @param d_va Values for matrix (device data)
     * @param index_base Index base (0 or 1)
     */
    void setupFromDevice(cusparseHandle_t handle, size_t n, size_t nnz,
                        const int* d_ia, const int* d_ja, const double* d_va,
                        int index_base);
    
private:
    // Device memory for ILU0 factorization (stored in-place)
    DeviceArray<int>* _d_ia;
    DeviceArray<int>* _d_ja;
    DeviceArray<double>* _d_va;
    
    // ILU0-specific descriptors (using deprecated API)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    csrilu02Info_t _ilu0_info;
    #pragma GCC diagnostic pop
    
    // Buffers for ILU0 factorization and SpSV operations
    DeviceArray<char>* _d_ilu0_buffer;
    DeviceArray<char>* _d_buffer_L;
    DeviceArray<char>* _d_buffer_U;
    
    // Matrix properties
    size_t _n;
    size_t _nnz;
    int _index_base;
    
    void cleanup();
    void performFactorization(cusparseHandle_t handle);
    void setupSpSVDescriptors(cusparseHandle_t handle);
};

} // namespace cuda_iterative_solver
