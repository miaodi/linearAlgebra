#include "cuda_csr_utils.cuh"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <iostream>
namespace matrix_utils::sparse_cuda
{

//==============================================================================
// Detail namespace: internal helpers and kernels
//==============================================================================
namespace detail
{

/// @brief Check CUDA status and throw on error.
inline void check_cuda( cudaError_t status, const char* msg )
{
    if ( status != cudaSuccess )
    {
        throw std::runtime_error( std::string( msg ) + ": " + cudaGetErrorString( status ) );
    }
}

/// @brief Functor to add a base offset to a value.
template <typename T>
struct AddBase
{
    T base;
    __host__ __device__ T operator()( T value ) const { return value + base; }
};

/// @brief Functor to subtract a base offset from a value.
template <typename T>
struct SubBase
{
    T base;
    __host__ __device__ T operator()( T value ) const { return value - base; }
};

/// @brief Functor to check if a flag is nonzero (keep).
template <typename FLAG>
struct KeepFlag
{
    __host__ __device__ bool operator()( FLAG flag ) const { return flag != FLAG( 0 ); }
};

inline int CeilLog2U64( uint64_t value )
{
    int bits = 0;
    uint64_t x = 1;
    while ( x < value )
    {
        x <<= 1;
        ++bits;
    }
    return bits;
}

/// @brief Kernel: convert CSR row pointers to COO row indices (one thread per row).
template <typename ROWTYPE, typename COLTYPE>
__global__ void CSRPtrToCOORowKernel( COLTYPE rows, const ROWTYPE* __restrict__ d_ai, COLTYPE* __restrict__ d_coo_rows, ROWTYPE base )
{
    COLTYPE row = static_cast<COLTYPE>( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( row >= rows )
        return;

    const ROWTYPE row_start = d_ai[row] - base;
    const ROWTYPE row_end = d_ai[row + 1] - base;
    const COLTYPE row_out = row + static_cast<COLTYPE>( base );
    for ( ROWTYPE k = row_start; k < row_end; ++k )
    {
        d_coo_rows[k] = row_out;
    }
}

/// @brief Device function: find row index for a given entry index using binary search.
/// @param entry_idx The entry index (0-based relative to base)
/// @param d_ai Row pointer array
/// @param rows Number of rows
/// @param base CSR base offset (0 or 1)
/// @return Row index for the entry
template <typename ROWTYPE, typename COLTYPE>
__device__ COLTYPE CSREntryToRow( ROWTYPE entry_idx, const ROWTYPE* __restrict__ d_ai, COLTYPE rows, ROWTYPE base )
{
    // Binary search to find row: find largest row i such that d_ai[i] <= entry_idx + base
    const ROWTYPE* upper = thrust::upper_bound( thrust::seq, d_ai, d_ai + rows + 1, entry_idx + base );
    return static_cast<COLTYPE>( upper - d_ai - 1 );
}

/// @brief Kernel: each thread handles one entry, uses binary search to find row, checks if diagonal.
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
__global__ void CSRFindDiagonalKernel( COLTYPE rows,
                                       const ROWTYPE* __restrict__ d_ai,
                                       const COLTYPE* __restrict__ d_aj,
                                       const VALTYPE* __restrict__ d_av,
                                       ROWTYPE* __restrict__ d_diag_pos,
                                       VALTYPE* __restrict__ d_diag_val,
                                       ROWTYPE base )
{
    ROWTYPE k = static_cast<ROWTYPE>( blockIdx.x ) * blockDim.x + threadIdx.x;
    ROWTYPE nnz = d_ai[rows] - base;
    if ( k >= nnz )
        return;

    // Find row index for entry k using binary search
    COLTYPE i = CSREntryToRow( k, d_ai, rows, base );
    COLTYPE j = d_aj[k];

    // Check if this is a diagonal entry (i is 0-based, j is in original base)
    if ( i + base == j )
    {
        d_diag_pos[i] = k + base;
        if ( d_diag_val && d_av )
        {
            d_diag_val[i] = d_av[k];
        }
    }
}

/// @brief Kernel: generate diagonal scaled prune mask (one thread per entry).
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, typename FLAGTYPE>
__global__ void CSRGenDiagScaledPruneMaskKernel( COLTYPE rows,
                                                 const ROWTYPE* __restrict__ d_ai,
                                                 const COLTYPE* __restrict__ d_aj,
                                                 const VALTYPE* __restrict__ d_av,
                                                 const VALTYPE* __restrict__ d_diag,
                                                 VALTYPE threshold,
                                                 FLAGTYPE* __restrict__ d_mask,
                                                 ROWTYPE base )
{
    ROWTYPE k = static_cast<ROWTYPE>( blockIdx.x ) * blockDim.x + threadIdx.x;
    ROWTYPE nnz = d_ai[rows] - base;
    if ( k >= nnz )
        return;

    // Find row index for entry k using binary search
    COLTYPE i = CSREntryToRow( k, d_ai, rows, base );
    COLTYPE j = d_aj[k] - base;
    VALTYPE aij = d_av[k];
    VALTYPE di = d_diag[i];
    VALTYPE dj = d_diag[j]; // j is in original base, but diag array is 0-indexed
    if ( i == j )
    {
        d_mask[k] = FLAGTYPE( 1 ); // Always keep diagonal
    }
    else
    {
        d_mask[k] = ( fabs( aij ) * fabs( aij ) >= fabs( di ) * fabs( dj ) * threshold ) ? FLAGTYPE( 1 )
                                                                                         : FLAGTYPE( 0 );
    }
}

} // namespace detail

//==============================================================================
// Public API implementations
//==============================================================================

template <typename ROWTYPE, typename COLTYPE>
void CSRPtrToCOORowDevice( COLTYPE rows, const ROWTYPE* d_ai, COLTYPE* d_coo_rows, cudaStream_t stream )
{
    if ( rows <= 0 )
        return;
    if ( !d_ai || !d_coo_rows )
    {
        throw std::invalid_argument( "CSRPtrToCOORowDevice received null pointer" );
    }

    ROWTYPE base{}, last{};
    detail::check_cuda( cudaMemcpy( &base, d_ai, sizeof( ROWTYPE ), cudaMemcpyDeviceToHost ),
                        "load CSR base" );
    detail::check_cuda( cudaMemcpy( &last, d_ai + rows, sizeof( ROWTYPE ), cudaMemcpyDeviceToHost ),
                        "load CSR nnz bound" );
    const ROWTYPE nnz = last - base;
    if ( nnz <= 0 )
        return;

    constexpr int block = 256;
    const int grid = static_cast<int>( ( rows + block - 1 ) / block );
    detail::CSRPtrToCOORowKernel<<<grid, block, 0, stream>>>( rows, d_ai, d_coo_rows, base );
    detail::check_cuda( cudaGetLastError(), "CSRPtrToCOORowDevice kernel launch" );
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void CSRFindDiagonalDevice( COLTYPE rows,
                            const ROWTYPE* d_ai,
                            const COLTYPE* d_aj,
                            const VALTYPE* d_av,
                            ROWTYPE* d_diag_pos,
                            VALTYPE* d_diag_val,
                            cudaStream_t stream )
{
    if ( rows <= 0 )
        return;

    // Get base and nnz
    ROWTYPE base{}, last{};
    detail::check_cuda( cudaMemcpy( &base, d_ai, sizeof( ROWTYPE ), cudaMemcpyDeviceToHost ),
                        "load CSR base" );
    detail::check_cuda( cudaMemcpy( &last, d_ai + rows, sizeof( ROWTYPE ), cudaMemcpyDeviceToHost ),
                        "load CSR nnz bound" );
    const ROWTYPE nnz = last - base;
    if ( nnz <= 0 )
        return;

    // Initialize diagonal arrays to -1 and 0
    auto exec = thrust::cuda::par.on( stream );
    thrust::fill_n( exec, thrust::device_pointer_cast( d_diag_pos ), rows, static_cast<ROWTYPE>( -1 ) );
    if ( d_diag_val )
    {
        thrust::fill_n( exec, thrust::device_pointer_cast( d_diag_val ), rows, VALTYPE( 0 ) );
    }

    // Launch kernel: one thread per entry
    constexpr int block = 256;
    int grid = static_cast<int>( ( nnz + block - 1 ) / block );
    detail::CSRFindDiagonalKernel<<<grid, block, 0, stream>>>( rows, d_ai, d_aj, d_av, d_diag_pos,
                                                               d_diag_val, base );
    detail::check_cuda( cudaGetLastError(), "CSRFindDiagonalDevice kernel launch" );
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, typename FLAGTYPE>
ROWTYPE CSRSelectByMaskDevice( COLTYPE rows,
                               const ROWTYPE* d_ai_in,
                               const COLTYPE* d_aj_in,
                               const VALTYPE* d_av_in,
                               const FLAGTYPE* d_keep_mask,
                               ROWTYPE* d_ai_out,
                               COLTYPE* d_aj_out,
                               VALTYPE* d_av_out,
                               cudaStream_t stream )
{
    static_assert( std::is_integral_v<FLAGTYPE>, "Mask type must be integral" );

    if ( !d_ai_in || !d_aj_in || !d_keep_mask || !d_ai_out || !d_aj_out )
    {
        throw std::invalid_argument( "CSRSelectByMaskDevice received null pointer" );
    }
    if ( ( d_av_in == nullptr ) != ( d_av_out == nullptr ) )
    {
        throw std::invalid_argument( "Value pointers must be both null or both non-null" );
    }

    // Load base and nnz from device
    ROWTYPE base{};
    ROWTYPE last{};
    detail::check_cuda( cudaMemcpy( &base, d_ai_in, sizeof( ROWTYPE ), cudaMemcpyDeviceToHost ),
                        "load CSR base" );
    detail::check_cuda( cudaMemcpy( &last, d_ai_in + rows, sizeof( ROWTYPE ), cudaMemcpyDeviceToHost ),
                        "load CSR nnz bound" );
    const ROWTYPE old_nnz = last - base;

    auto exec = thrust::cuda::par.on( stream );

    // Early exit for empty matrix
    if ( rows <= COLTYPE( 0 ) || old_nnz <= ROWTYPE( 0 ) )
    {
        thrust::copy_n( exec, d_ai_in, static_cast<std::size_t>( rows ) + 1, d_ai_out );
        return ROWTYPE( 0 );
    }

    const std::size_t row_count = static_cast<std::size_t>( rows );
    const std::size_t nnz_count = static_cast<std::size_t>( old_nnz );

    // Copy and normalize IA to zero-based for segmented reduction
    thrust::device_vector<ROWTYPE> offsets( row_count + 1 );
    thrust::copy_n( exec, d_ai_in, row_count + 1, offsets.begin() );
    thrust::transform( exec, offsets.begin(), offsets.end(), offsets.begin(), detail::SubBase<ROWTYPE>{ base } );

    // Count kept entries per row using segmented reduction
    thrust::device_vector<ROWTYPE> row_counts( row_count, ROWTYPE( 0 ) );
    const int num_segments = static_cast<int>( rows );
    size_t segmented_bytes = 0;
    cub::DeviceSegmentedReduce::Sum( nullptr, segmented_bytes, d_keep_mask,
                                     thrust::raw_pointer_cast( row_counts.data() ), num_segments,
                                     thrust::raw_pointer_cast( offsets.data() ),
                                     thrust::raw_pointer_cast( offsets.data() ) + 1, stream );
    thrust::device_vector<std::uint8_t> segmented_storage( segmented_bytes == 0 ? 1 : segmented_bytes );
    cub::DeviceSegmentedReduce::Sum( segmented_storage.data().get(), segmented_bytes, d_keep_mask,
                                     thrust::raw_pointer_cast( row_counts.data() ), num_segments,
                                     thrust::raw_pointer_cast( offsets.data() ),
                                     thrust::raw_pointer_cast( offsets.data() ) + 1, stream );

    // Build output IA via exclusive scan, then restore base and copy to output in one step
    thrust::device_vector<ROWTYPE> row_prefix( row_count + 1, ROWTYPE( 0 ) );
    thrust::inclusive_scan( exec, row_counts.begin(), row_counts.end(), row_prefix.begin() + 1 );
    thrust::transform( exec, row_prefix.begin(), row_prefix.end(),
                       thrust::device_pointer_cast( d_ai_out ), detail::AddBase<ROWTYPE>{ base } );

    // Compute new nnz
    ROWTYPE new_bound{};
    detail::check_cuda( cudaMemcpy( &new_bound, d_ai_out + rows, sizeof( ROWTYPE ), cudaMemcpyDeviceToHost ),
                        "copy new nnz bound" );
    const ROWTYPE new_nnz = new_bound - base;

    // Filter and copy column indices according to the keep mask
    auto flags_begin = thrust::device_pointer_cast( d_keep_mask );
    auto col_in_begin = thrust::device_pointer_cast( d_aj_in );
    auto col_out_begin = thrust::device_pointer_cast( d_aj_out );
    auto col_result = thrust::copy_if( exec, col_in_begin, col_in_begin + nnz_count, flags_begin,
                                       col_out_begin, detail::KeepFlag<FLAGTYPE>{} );
    if ( static_cast<ROWTYPE>( col_result - col_out_begin ) != new_nnz )
    {
        std::cout << "Expected nnz: " << new_nnz
                  << ", copied nnz: " << ( col_result - col_out_begin ) << std::endl;
        throw std::runtime_error( "Column copy mismatched expected nnz" );
    }

    // If values are present, filter and copy them in the same way
    if ( d_av_in && d_av_out )
    {
        auto val_in_begin = thrust::device_pointer_cast( d_av_in );
        auto val_out_begin = thrust::device_pointer_cast( d_av_out );
        auto val_result = thrust::copy_if( exec, val_in_begin, val_in_begin + nnz_count, flags_begin,
                                           val_out_begin, detail::KeepFlag<FLAGTYPE>{} );
    }

    return old_nnz - new_nnz;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, typename FLAGTYPE>
void CSRGenDiagScaledPruneMask( COLTYPE rows,
                                const ROWTYPE* d_ai,
                                const COLTYPE* d_aj,
                                const VALTYPE* d_av,
                                VALTYPE threshold,
                                FLAGTYPE* d_mask,
                                cudaStream_t stream )
{
    if ( rows <= 0 )
        return;

    // Get base and nnz
    ROWTYPE base{}, last{};
    detail::check_cuda( cudaMemcpy( &base, d_ai, sizeof( ROWTYPE ), cudaMemcpyDeviceToHost ),
                        "load CSR base" );
    detail::check_cuda( cudaMemcpy( &last, d_ai + rows, sizeof( ROWTYPE ), cudaMemcpyDeviceToHost ),
                        "load CSR nnz bound" );
    const ROWTYPE nnz = last - base;
    if ( nnz <= 0 )
        return;

    // Extract diagonal values
    thrust::device_vector<VALTYPE> diag( rows );
    thrust::device_vector<ROWTYPE> diag_pos( rows );
    CSRFindDiagonalDevice( rows, d_ai, d_aj, d_av, thrust::raw_pointer_cast( diag_pos.data() ),
                           thrust::raw_pointer_cast( diag.data() ), stream );

    // Launch kernel: one thread per entry
    constexpr int block = 256;
    int grid = static_cast<int>( ( nnz + block - 1 ) / block );
    detail::CSRGenDiagScaledPruneMaskKernel<<<grid, block, 0, stream>>>(
        rows, d_ai, d_aj, d_av, thrust::raw_pointer_cast( diag.data() ), threshold, d_mask, base );
    detail::check_cuda( cudaGetLastError(), "CSRGenDiagScaledPruneMask kernel launch" );
}

//==============================================================================
// CSRDiagDevice
//==============================================================================

template <typename ROWTYPE, typename COLTYPE>
void CSRDiagDevice( COLTYPE n, ROWTYPE base, DeviceCSRMatrix<ROWTYPE, COLTYPE>& out )
{
    out.n_rows = n;
    out.base = base;
    out.ai.resize( static_cast<size_t>( n ) + 1 );
    out.aj.resize( static_cast<size_t>( n ) );
    thrust::sequence( thrust::device, out.ai.data(), out.ai.data() + ( static_cast<size_t>( n ) + 1 ),
                      static_cast<ROWTYPE>( base ), static_cast<ROWTYPE>( 1 ) );
    thrust::sequence( thrust::device, out.aj.data(), out.aj.data() + n,
                      static_cast<COLTYPE>( base ), static_cast<COLTYPE>( 1 ) );
}

//==============================================================================
// Explicit template instantiations
//==============================================================================

template void CSRFindDiagonalDevice<int, int, float>( int, const int*, const int*, const float*, int*, float*, cudaStream_t );
template void CSRFindDiagonalDevice<int, int, double>( int, const int*, const int*, const double*, int*, double*, cudaStream_t );
template void CSRFindDiagonalDevice<std::int64_t, int, float>( int,
                                                               const std::int64_t*,
                                                               const int*,
                                                               const float*,
                                                               std::int64_t*,
                                                               float*,
                                                               cudaStream_t );
template void CSRFindDiagonalDevice<std::int64_t, int, double>( int,
                                                                const std::int64_t*,
                                                                const int*,
                                                                const double*,
                                                                std::int64_t*,
                                                                double*,
                                                                cudaStream_t );

template int CSRSelectByMaskDevice<int, int, float, int>( int,
                                                          const int*,
                                                          const int*,
                                                          const float*,
                                                          const int*,
                                                          int*,
                                                          int*,
                                                          float*,
                                                          cudaStream_t );
template int CSRSelectByMaskDevice<int, int, double, int>( int,
                                                           const int*,
                                                           const int*,
                                                           const double*,
                                                           const int*,
                                                           int*,
                                                           int*,
                                                           double*,
                                                           cudaStream_t );
template std::int64_t CSRSelectByMaskDevice<std::int64_t, int, float, int>( int,
                                                                            const std::int64_t*,
                                                                            const int*,
                                                                            const float*,
                                                                            const int*,
                                                                            std::int64_t*,
                                                                            int*,
                                                                            float*,
                                                                            cudaStream_t );
template std::int64_t CSRSelectByMaskDevice<std::int64_t, int, double, int>( int,
                                                                             const std::int64_t*,
                                                                             const int*,
                                                                             const double*,
                                                                             const int*,
                                                                             std::int64_t*,
                                                                             int*,
                                                                             double*,
                                                                             cudaStream_t );

template void CSRGenDiagScaledPruneMask<int, int, float, int>( int, const int*, const int*, const float*, float, int*, cudaStream_t );
template void CSRGenDiagScaledPruneMask<int, int, double, int>( int, const int*, const int*, const double*, double, int*, cudaStream_t );
template void CSRGenDiagScaledPruneMask<std::int64_t, int, float, int>( int,
                                                                        const std::int64_t*,
                                                                        const int*,
                                                                        const float*,
                                                                        float,
                                                                        int*,
                                                                        cudaStream_t );
template void CSRGenDiagScaledPruneMask<std::int64_t, int, double, int>( int,
                                                                         const std::int64_t*,
                                                                         const int*,
                                                                         const double*,
                                                                         double,
                                                                         int*,
                                                                         cudaStream_t );

template void CSRDiagDevice<int, int>( int, int, DeviceCSRMatrix<int, int>& );
template void CSRDiagDevice<std::int64_t, int>( int, std::int64_t, DeviceCSRMatrix<std::int64_t, int>& );
template void CSRPtrToCOORowDevice<int, int>( int, const int*, int*, cudaStream_t );
template void CSRPtrToCOORowDevice<std::int64_t, int>( int, const std::int64_t*, int*, cudaStream_t );

} // namespace matrix_utils::sparse_cuda
