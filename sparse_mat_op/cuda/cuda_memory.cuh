#pragma once

#include <cusparse.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace matrix_utils::sparse_cuda
{

inline void checkCudaMemoryOp( cudaError_t error, const char* op )
{
    if ( error != cudaSuccess )
    {
        char message[512];
        std::snprintf( message, sizeof( message ), "%s: %s", op, cudaGetErrorString( error ) );
        throw std::runtime_error( message );
    }
}

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

    template <typename T>
    static cudaError_t allocate( T** ptr, size_t count )
    {
        return cudaMalloc( ptr, count * sizeof( T ) );
    }

    template <typename T>
    static cudaError_t deallocate( T* ptr )
    {
        return cudaFree( ptr );
    }
};

struct PinnedAllocator
{
    static constexpr MemoryLocation location = MemoryLocation::Host;

    template <typename T>
    static cudaError_t allocate( T** ptr, size_t count )
    {
        return cudaMallocHost( ptr, count * sizeof( T ) );
    }

    template <typename T>
    static cudaError_t deallocate( T* ptr )
    {
        return cudaFreeHost( ptr );
    }
};

/**
 * @brief Generic array class for managing CUDA memory with capacity tracking
 *
 * The Allocator is held as an instance so stateful allocators (e.g.
 * AsyncDeviceAllocator carrying a cudaStream_t) pair alloc/free on the same
 * state. Stateless allocators (DeviceAllocator, PinnedAllocator) cost nothing
 * extra thanks to empty-base / zero-size-member optimisation.
 */
template <typename T, typename Allocator>
class Array
{
public:
    Array() : _data( nullptr ), _size( 0 ), _capacity( 0 ), _alloc{} {}

    explicit Array( Allocator alloc )
        : _data( nullptr ), _size( 0 ), _capacity( 0 ), _alloc( alloc )
    {
    }

    ~Array() { release(); }

    // Move constructor
    Array( Array&& other ) noexcept
        : _data( other._data ), _size( other._size ), _capacity( other._capacity ), _alloc( other._alloc )
    {
        other._data = nullptr;
        other._size = 0;
        other._capacity = 0;
    }

    // Move assignment operator
    Array& operator=( Array&& other ) noexcept
    {
        if ( this != &other )
        {
            release();
            _data = other._data;
            _size = other._size;
            _capacity = other._capacity;
            _alloc = other._alloc;
            other._data = nullptr;
            other._size = 0;
            other._capacity = 0;
        }
        return *this;
    }

    void resize( size_t new_size )
    {
        if ( new_size > _capacity )
        {
            if ( _data )
            {
                checkCudaMemoryOp( _alloc.deallocate( _data ), "cuda deallocate failed" );
            }

            if ( new_size > ( std::numeric_limits<size_t>::max() / sizeof( T ) ) )
            {
                throw std::runtime_error(
                    "cuda allocate failed: requested allocation size overflow" );
            }

            const size_t requested_bytes = new_size * sizeof( T );
#ifdef CUDA_ALLOC_TRACE
            size_t free_before_bytes = 0;
            size_t total_before_bytes = 0;
            if constexpr ( Allocator::location == MemoryLocation::Device )
            {
                (void)cudaMemGetInfo( &free_before_bytes, &total_before_bytes );
            }
#endif

            const cudaError_t alloc_status = _alloc.allocate( &_data, new_size );
            if ( alloc_status != cudaSuccess )
            {
                size_t free_bytes = 0;
                size_t total_bytes = 0;
                const cudaError_t mem_info_status = cudaMemGetInfo( &free_bytes, &total_bytes );
                constexpr double kBytesPerGB = 1024 * 1024 * 1024;
                const double requested_gb = static_cast<double>( requested_bytes ) / kBytesPerGB;
                const double free_gb = static_cast<double>( free_bytes ) / kBytesPerGB;
                const double total_gb = static_cast<double>( total_bytes ) / kBytesPerGB;

                char message[768];
                if ( mem_info_status == cudaSuccess )
                {
                    std::snprintf( message, sizeof( message ),
                                   "cuda allocate failed in Array::resize: requested %.3f GB (%zu "
                                   "elements x %zu bytes), "
                                   "capacity before resize %zu elements, device free %.3f GB / "
                                   "total %.3f GB: %s",
                                   requested_gb, new_size, sizeof( T ), _capacity, free_gb,
                                   total_gb, cudaGetErrorString( alloc_status ) );
                }
                else
                {
                    std::snprintf(
                        message, sizeof( message ),
                        "cuda allocate failed in Array::resize: requested %.3f GB (%zu elements x "
                        "%zu bytes), "
                        "capacity before resize %zu elements; memory stats unavailable: %s",
                        requested_gb, new_size, sizeof( T ), _capacity, cudaGetErrorString( alloc_status ) );
                }
                throw std::runtime_error( message );
            }

#ifdef CUDA_ALLOC_TRACE
            if constexpr ( Allocator::location == MemoryLocation::Device )
            {
                size_t free_after_bytes = 0;
                size_t total_after_bytes = 0;
                const cudaError_t after_status = cudaMemGetInfo( &free_after_bytes, &total_after_bytes );
                constexpr double kBytesPerGB = 1024 * 1024 * 1024;
                const double requested_gb = static_cast<double>( requested_bytes ) / kBytesPerGB;
                const double used_before_gb =
                    static_cast<double>( total_before_bytes - free_before_bytes ) / kBytesPerGB;
                const double free_before_gb = static_cast<double>( free_before_bytes ) / kBytesPerGB;

                if ( after_status == cudaSuccess )
                {
                    const double used_after_gb =
                        static_cast<double>( total_after_bytes - free_after_bytes ) / kBytesPerGB;
                    const double free_after_gb = static_cast<double>( free_after_bytes ) / kBytesPerGB;
                    const double used_delta_gb = used_after_gb - used_before_gb;
                    std::fprintf(
                        stderr,
                        "[CUDA_ALLOC] request=%.3f GB (%zu elements x %zu bytes), cap %zu->%zu, "
                        "used %.3f->%.3f GB (delta %.3f GB), free %.3f->%.3f GB, ptr=%p\n",
                        requested_gb, new_size, sizeof( T ), _capacity, new_size, used_before_gb, used_after_gb,
                        used_delta_gb, free_before_gb, free_after_gb, static_cast<void*>( _data ) );
                }
                else
                {
                    std::fprintf(
                        stderr,
                        "[CUDA_ALLOC] request=%.3f GB (%zu elements x %zu bytes), cap %zu->%zu, "
                        "used(before)=%.3f GB, free(before)=%.3f GB, ptr=%p (post-allocation mem "
                        "stats unavailable)\n",
                        requested_gb, new_size, sizeof( T ), _capacity, new_size, used_before_gb,
                        free_before_gb, static_cast<void*>( _data ) );
                }
            }
#endif

            _capacity = new_size;
        }
        _size = new_size;
    }

    template <MemoryLocation SrcLocation>
    void copy( const T* src_data, size_t count )
    {
        resize( count );
        if ( count > 0 )
        {
            constexpr cudaMemcpyKind kind = getCopyKind<SrcLocation, Allocator::location>();
            checkCudaMemoryOp( cudaMemcpy( _data, src_data, count * sizeof( T ), kind ),
                               "cudaMemcpy failed" );
        }
    }

    void copyFromHost( const T* host_data, size_t count )
    {
        copy<MemoryLocation::Host>( host_data, count );
    }

    void copyFromDevice( const T* device_data, size_t count )
    {
        copy<MemoryLocation::Device>( device_data, count );
    }

    void copyToHost( T* host_data ) const
    {
        if ( _size > 0 && host_data != nullptr )
        {
            constexpr cudaMemcpyKind kind = getCopyKind<Allocator::location, MemoryLocation::Host>();
            checkCudaMemoryOp( cudaMemcpy( host_data, _data, _size * sizeof( T ), kind ),
                               "cudaMemcpy to host failed" );
        }
    }

private:
    template <MemoryLocation Src, MemoryLocation Dst>
    static constexpr cudaMemcpyKind getCopyKind()
    {
        if constexpr ( Src == MemoryLocation::Host && Dst == MemoryLocation::Host )
        {
            return cudaMemcpyHostToHost;
        }
        else if constexpr ( Src == MemoryLocation::Host && Dst == MemoryLocation::Device )
        {
            return cudaMemcpyHostToDevice;
        }
        else if constexpr ( Src == MemoryLocation::Device && Dst == MemoryLocation::Host )
        {
            return cudaMemcpyDeviceToHost;
        }
        else
        { // Device to Device
            return cudaMemcpyDeviceToDevice;
        }
    }

public:
    void release()
    {
        if ( _data )
        {
#ifdef CUDA_ALLOC_TRACE
            if constexpr ( Allocator::location == MemoryLocation::Device )
            {
                size_t free_before = 0, total_before = 0;
                (void)cudaMemGetInfo( &free_before, &total_before );
                checkCudaMemoryOp( _alloc.deallocate( _data ), "cuda deallocate failed" );
                size_t free_after = 0, total_after = 0;
                const cudaError_t after_status = cudaMemGetInfo( &free_after, &total_after );
                constexpr double kBytesPerGB = 1024.0 * 1024 * 1024;
                const double released_gb = static_cast<double>( _capacity * sizeof( T ) ) / kBytesPerGB;
                if ( after_status == cudaSuccess )
                {
                    const double free_before_gb = static_cast<double>( free_before ) / kBytesPerGB;
                    const double free_after_gb = static_cast<double>( free_after ) / kBytesPerGB;
                    std::fprintf( stderr,
                                  "[CUDA_FREE] released=%.3f GB (%zu elements x %zu bytes), "
                                  "free %.3f->%.3f GB, ptr=%p\n",
                                  released_gb, _capacity, sizeof( T ), free_before_gb,
                                  free_after_gb, static_cast<void*>( _data ) );
                }
                else
                {
                    std::fprintf(
                        stderr, "[CUDA_FREE] released=%.3f GB (%zu elements x %zu bytes), ptr=%p\n",
                        released_gb, _capacity, sizeof( T ), static_cast<void*>( _data ) );
                }
            }
            else
            {
                checkCudaMemoryOp( _alloc.deallocate( _data ), "cuda deallocate failed" );
            }
#else
            checkCudaMemoryOp( _alloc.deallocate( _data ), "cuda deallocate failed" );
#endif
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

    Allocator& allocator() { return _alloc; }
    const Allocator& allocator() const { return _alloc; }

private:
    T* _data;
    size_t _size;
    size_t _capacity;
    Allocator _alloc;

    // Disable copy and assignment
    Array( const Array& ) = delete;
    Array& operator=( const Array& ) = delete;
};

// Type aliases for convenience
template <typename T>
using DeviceArray = Array<T, DeviceAllocator>;

template <typename T>
using PinnedArray = Array<T, PinnedAllocator>;

/**
 * @brief Stream-ordered device allocator using CUDA's built-in memory pool
 *        (cudaMallocAsync / cudaFreeAsync, CUDA 11.2+).
 *
 * Freed blocks are returned to the device's default memory pool and can be
 * reused by later allocations on the same stream without a round-trip through
 * the OS/driver. No custom caching logic is needed.
 *
 * This is a stateful allocator: each instance carries a cudaStream_t so that
 * allocations and deallocations are always paired on the same stream.
 */
struct AsyncDeviceAllocator
{
    static constexpr MemoryLocation location = MemoryLocation::Device;

    cudaStream_t stream = nullptr;

    template <typename T>
    cudaError_t allocate( T** ptr, size_t count ) const
    {
        return cudaMallocAsync( ptr, count * sizeof( T ), stream );
    }

    template <typename T>
    cudaError_t deallocate( T* ptr ) const
    {
        return cudaFreeAsync( ptr, stream );
    }
};

template <typename T>
using AsyncDeviceArray = Array<T, AsyncDeviceAllocator>;

/**
 * @brief View wrapper for cuSPARSE dense vector descriptor with device memory pointer
 */
class DeviceVectorView
{
public:
    DeviceVectorView() : _descriptor( nullptr ), _data( nullptr ), _size( 0 ) {}

    ~DeviceVectorView()
    {
        if ( _descriptor )
        {
            cusparseDestroyDnVec( _descriptor );
        }
    }

    void create( size_t size, double* data )
    {
        if ( _descriptor )
        {
            cusparseDestroyDnVec( _descriptor );
        }
        _data = data;
        _size = size;
        cusparseCreateDnVec( &_descriptor, size, data, CUDA_R_64F );
    }

    void setData( double* data )
    {
        _data = data;
        if ( _descriptor )
        {
            cusparseDnVecSetValues( _descriptor, data );
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
    DeviceVectorView( const DeviceVectorView& ) = delete;
    DeviceVectorView& operator=( const DeviceVectorView& ) = delete;
};

/**
 * @brief Helper function to check and report CUDA errors
 *
 * @param error CUDA error code to check
 * @param message Descriptive message about the operation that failed
 * @throws std::runtime_error if error is not cudaSuccess
 */
inline void checkCudaError( cudaError_t error, const char* message )
{
    if ( error != cudaSuccess )
    {
        char full_message[512];
        std::snprintf( full_message, sizeof( full_message ), "CUDA error: %s - %s", message,
                       cudaGetErrorString( error ) );
        throw std::runtime_error( full_message );
    }
}

} // namespace matrix_utils::sparse_cuda
