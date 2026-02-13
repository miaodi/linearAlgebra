#pragma once

#include <cusparse.h>
#include <cuda_runtime.h>
#include <string>
#include <stdexcept>

namespace matrix_utils::sparse_cuda
{

/**
 * @brief Memory location descriptor
 */
enum class MemoryLocation
{
    Host,
    Device
};

/**
 * @brief Memory allocator traits for different memory types
 */
struct DeviceAllocator
{
    static constexpr MemoryLocation location = MemoryLocation::Device;
    
    template<typename T>
    static cudaError_t allocate(T** ptr, size_t count) {
        return cudaMalloc(ptr, count * sizeof(T));
    }
    
    template<typename T>
    static cudaError_t deallocate(T* ptr) {
        return cudaFree(ptr);
    }
};

struct PinnedAllocator
{
    static constexpr MemoryLocation location = MemoryLocation::Host;
    
    template<typename T>
    static cudaError_t allocate(T** ptr, size_t count) {
        return cudaMallocHost(ptr, count * sizeof(T));
    }
    
    template<typename T>
    static cudaError_t deallocate(T* ptr) {
        return cudaFreeHost(ptr);
    }
};

/**
 * @brief Generic array class for managing CUDA memory with capacity tracking
 */
template<typename T, typename Allocator>
class Array
{
public:
    Array() : _data(nullptr), _size(0), _capacity(0) {}
    
    ~Array() { release(); }
    
    // Move constructor
    Array(Array&& other) noexcept 
        : _data(other._data), _size(other._size), _capacity(other._capacity) {
        other._data = nullptr;
        other._size = 0;
        other._capacity = 0;
    }
    
    // Move assignment operator
    Array& operator=(Array&& other) noexcept {
        if (this != &other) {
            release();
            _data = other._data;
            _size = other._size;
            _capacity = other._capacity;
            other._data = nullptr;
            other._size = 0;
            other._capacity = 0;
        }
        return *this;
    }
    
    void resize(size_t new_size) {
        if (new_size > _capacity) {
            if (_data) {
                Allocator::deallocate(_data);
            }
            Allocator::allocate(&_data, new_size);
            _capacity = new_size;
        }
        _size = new_size;
    }
    
    template<MemoryLocation SrcLocation>
    void copy(const T* src_data, size_t count) {
        resize(count);
        if (count > 0) {
            constexpr cudaMemcpyKind kind = getCopyKind<SrcLocation, Allocator::location>();
            cudaMemcpy(_data, src_data, count * sizeof(T), kind);
        }
    }
    
    void copyFromHost(const T* host_data, size_t count) {
        copy<MemoryLocation::Host>(host_data, count);
    }
    
    void copyFromDevice(const T* device_data, size_t count) {
        copy<MemoryLocation::Device>(device_data, count);
    }
    
    void copyToHost(T* host_data) const {
        if (_size > 0 && host_data != nullptr) {
            constexpr cudaMemcpyKind kind = getCopyKind<Allocator::location, MemoryLocation::Host>();
            cudaMemcpy(host_data, _data, _size * sizeof(T), kind);
        }
    }
    
private:
    template<MemoryLocation Src, MemoryLocation Dst>
    static constexpr cudaMemcpyKind getCopyKind() {
        if constexpr (Src == MemoryLocation::Host && Dst == MemoryLocation::Host) {
            return cudaMemcpyHostToHost;
        } else if constexpr (Src == MemoryLocation::Host && Dst == MemoryLocation::Device) {
            return cudaMemcpyHostToDevice;
        } else if constexpr (Src == MemoryLocation::Device && Dst == MemoryLocation::Host) {
            return cudaMemcpyDeviceToHost;
        } else { // Device to Device
            return cudaMemcpyDeviceToDevice;
        }
    }
    
public:
    
    void release() {
        if (_data) {
            Allocator::deallocate(_data);
            _data = nullptr;
        }
        _size = 0;
        _capacity = 0;
    }
    
    T* data() { return _data; }
    const T* data() const { return _data; }
    size_t size() const { return _size; }
    size_t capacity() const { return _capacity; }
    
    MemoryLocation getLocation() const { return Allocator::location; }
    
private:
    T* _data;
    size_t _size;
    size_t _capacity;
    
    // Disable copy and assignment
    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;
};

// Type aliases for convenience
template<typename T>
using DeviceArray = Array<T, DeviceAllocator>;

template<typename T>
using PinnedArray = Array<T, PinnedAllocator>;

/**
 * @brief View wrapper for cuSPARSE dense vector descriptor with device memory pointer
 */
class DeviceVectorView
{
public:
    DeviceVectorView() : _descriptor(nullptr), _data(nullptr), _size(0) {}
    
    ~DeviceVectorView() {
        if (_descriptor) {
            cusparseDestroyDnVec(_descriptor);
        }
    }
    
    void create(size_t size, double* data) {
        if (_descriptor) {
            cusparseDestroyDnVec(_descriptor);
        }
        _data = data;
        _size = size;
        cusparseCreateDnVec(&_descriptor, size, data, CUDA_R_64F);
    }
    
    void setData(double* data) {
        _data = data;
        if (_descriptor) {
            cusparseDnVecSetValues(_descriptor, data);
        }
    }
    
    cusparseDnVecDescr_t descriptor() const { return _descriptor; }
    double* data() const { return _data; }
    size_t size() const { return _size; }
    
private:
    cusparseDnVecDescr_t _descriptor;
    double* _data;
    size_t _size;
    
    // Disable copy and assignment
    DeviceVectorView(const DeviceVectorView&) = delete;
    DeviceVectorView& operator=(const DeviceVectorView&) = delete;
};

/**
 * @brief Helper function to check and report CUDA errors
 * 
 * @param error CUDA error code to check
 * @param message Descriptive message about the operation that failed
 * @throws std::runtime_error if error is not cudaSuccess
 */
inline void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + message + " - " + 
                                cudaGetErrorString(error));
    }
}

} // namespace matrix_utils::sparse_cuda
