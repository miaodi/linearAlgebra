#include "spgemm/spgemm_sort.cuh"

#include <cub/cub.cuh>
#include <cstdint>
#include <utility>

namespace matrix_utils::sparse_cuda
{
namespace
{

template <typename COLTYPE>
__global__ void make_zero_based_offsets_kernel( COLTYPE rows,
                                                const ExpandedIndex* row_ptr,
                                                ExpandedIndex base,
                                                ExpandedIndex* zero_based_offsets )
{
    COLTYPE idx = static_cast<COLTYPE>( blockIdx.x * blockDim.x + threadIdx.x );
    if ( idx > rows )
    {
        return;
    }
    zero_based_offsets[idx] = row_ptr[idx] - base;
}

template <typename ROWTYPE, typename COLTYPE>
bool symbolicSortInputIsValid( const SpGEMMSymbolicResult<ROWTYPE, COLTYPE>& symbolic )
{
    return symbolic.n_rows >= 0 && symbolic.total_expanded_nnz >= 0 &&
           symbolic.expanded_row_ptr.data() != nullptr;
}

} // namespace

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
bool SpGEMMSortExpandedProductsByColumn( const SpGEMMSymbolicResult<ROWTYPE, COLTYPE>& symbolic,
                                         SpGEMMExpandedProducts<COLTYPE, VALTYPE>& expanded,
                                         SpGEMMExpandedProducts<COLTYPE, VALTYPE>& sorted,
                                         cudaStream_t stream )
{
    if ( !symbolicSortInputIsValid( symbolic ) )
    {
        return false;
    }
    const ExpandedIndex total_items = symbolic.total_expanded_nnz;
    sorted.col_ind.resize( static_cast<size_t>( total_items ) );
    sorted.values.resize( static_cast<size_t>( total_items ) );
    if ( total_items == 0 )
    {
        return true;
    }
    if ( expanded.col_ind.data() == nullptr || expanded.values.data() == nullptr )
    {
        return false;
    }

    const AsyncDeviceAllocator pool{ stream };
    const ExpandedIndex* segment_offsets = symbolic.expanded_row_ptr.data();
    AsyncDeviceArray<ExpandedIndex> zero_based_offsets( pool );
    if ( symbolic.base != 0 )
    {
        zero_based_offsets.resize( static_cast<size_t>( symbolic.n_rows + 1 ) );
        constexpr int threads = 256;
        const int blocks = ( static_cast<int>( symbolic.n_rows + 1 ) + threads - 1 ) / threads;
        make_zero_based_offsets_kernel<COLTYPE><<<blocks, threads, 0, stream>>>(
            symbolic.n_rows, symbolic.expanded_row_ptr.data(),
            static_cast<ExpandedIndex>( symbolic.base ), zero_based_offsets.data() );
        checkCudaError( cudaGetLastError(), "launch SpGEMM segment-offset normalization kernel" );
        segment_offsets = zero_based_offsets.data();
    }

    // Compute minimal end_bit from n_cols to reduce radix sort passes.
    int end_bit = static_cast<int>( sizeof( COLTYPE ) * 8 );
    if ( symbolic.n_cols > 0 )
    {
        COLTYPE max_col = symbolic.n_cols - 1 + symbolic.base;
        end_bit = 1;
        while ( ( static_cast<COLTYPE>( 1 ) << end_bit ) <= max_col )
        {
            ++end_bit;
        }
    }

    // Use DoubleBuffer so CUB ping-pongs between expanded and sorted directly,
    // eliminating the large internal temporary buffer (~24 GB for 2B+ elements).
    cub::DoubleBuffer<COLTYPE> d_keys( expanded.col_ind.data(), sorted.col_ind.data() );
    cub::DoubleBuffer<VALTYPE> d_vals( expanded.values.data(), sorted.values.data() );

    void* temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    checkCudaError( cub::DeviceSegmentedRadixSort::SortPairs(
                        temp_storage, temp_storage_bytes, d_keys, d_vals, static_cast<std::int64_t>( total_items ),
                        static_cast<std::int64_t>( symbolic.n_rows ), segment_offsets,
                        segment_offsets + 1, 0, end_bit, stream ),
                    "query SpGEMM segmented sort temporary storage" );

    AsyncDeviceArray<std::uint8_t> temp( pool );
    temp.resize( temp_storage_bytes );
    checkCudaError( cub::DeviceSegmentedRadixSort::SortPairs(
                        temp.data(), temp_storage_bytes, d_keys, d_vals, static_cast<std::int64_t>( total_items ),
                        static_cast<std::int64_t>( symbolic.n_rows ), segment_offsets,
                        segment_offsets + 1, 0, end_bit, stream ),
                    "run SpGEMM segmented sort" );
    checkCudaError( cudaStreamSynchronize( stream ), "synchronize SpGEMM segmented sort" );

    // If the final radix pass left the result in expanded's buffer, swap so
    // that sorted always holds the output on return.
    if ( d_keys.Current() == expanded.col_ind.data() )
    {
        std::swap( expanded.col_ind, sorted.col_ind );
        std::swap( expanded.values, sorted.values );
    }

    return true;
}

template bool SpGEMMSortExpandedProductsByColumn<int, int, float>( const SpGEMMSymbolicResult<int, int>&,
                                                                   SpGEMMExpandedProducts<int, float>&,
                                                                   SpGEMMExpandedProducts<int, float>&,
                                                                   cudaStream_t );

template bool SpGEMMSortExpandedProductsByColumn<int, int, double>( const SpGEMMSymbolicResult<int, int>&,
                                                                    SpGEMMExpandedProducts<int, double>&,
                                                                    SpGEMMExpandedProducts<int, double>&,
                                                                    cudaStream_t );

} // namespace matrix_utils::sparse_cuda
