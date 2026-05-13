#include "spgemm/spgemm_sort.cuh"

#include <cub/cub.cuh>
#include <cstdint>

namespace matrix_utils::sparse_cuda
{
namespace
{

template <typename ROWTYPE, typename COLTYPE>
__global__ void make_zero_based_offsets_kernel( COLTYPE rows, const ROWTYPE* row_ptr, ROWTYPE base, ROWTYPE* zero_based_offsets )
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
                                         const SpGEMMExpandedProducts<COLTYPE, VALTYPE>& expanded,
                                         SpGEMMExpandedProducts<COLTYPE, VALTYPE>& sorted,
                                         cudaStream_t stream )
{
    if ( !symbolicSortInputIsValid( symbolic ) )
    {
        return false;
    }
    const ROWTYPE total_items = symbolic.total_expanded_nnz;
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

    const ROWTYPE* segment_offsets = symbolic.expanded_row_ptr.data();
    DeviceArray<ROWTYPE> zero_based_offsets;
    if ( symbolic.base != 0 )
    {
        zero_based_offsets.resize( static_cast<size_t>( symbolic.n_rows + 1 ) );
        constexpr int threads = 256;
        const int blocks = ( static_cast<int>( symbolic.n_rows + 1 ) + threads - 1 ) / threads;
        make_zero_based_offsets_kernel<ROWTYPE, COLTYPE><<<blocks, threads, 0, stream>>>(
            symbolic.n_rows, symbolic.expanded_row_ptr.data(), symbolic.base, zero_based_offsets.data() );
        checkCudaError( cudaGetLastError(), "launch SpGEMM segment-offset normalization kernel" );
        segment_offsets = zero_based_offsets.data();
    }

    void* temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    checkCudaError( cub::DeviceSegmentedRadixSort::SortPairs(
                        temp_storage, temp_storage_bytes, expanded.col_ind.data(), sorted.col_ind.data(),
                        expanded.values.data(), sorted.values.data(), static_cast<std::int64_t>( total_items ),
                        static_cast<std::int64_t>( symbolic.n_rows ), segment_offsets,
                        segment_offsets + 1, 0, sizeof( COLTYPE ) * 8, stream ),
                    "query SpGEMM segmented sort temporary storage" );

    DeviceArray<std::uint8_t> temp;
    temp.resize( temp_storage_bytes );
    checkCudaError( cub::DeviceSegmentedRadixSort::SortPairs(
                        temp.data(), temp_storage_bytes, expanded.col_ind.data(), sorted.col_ind.data(),
                        expanded.values.data(), sorted.values.data(), static_cast<std::int64_t>( total_items ),
                        static_cast<std::int64_t>( symbolic.n_rows ), segment_offsets,
                        segment_offsets + 1, 0, sizeof( COLTYPE ) * 8, stream ),
                    "run SpGEMM segmented sort" );
    checkCudaError( cudaStreamSynchronize( stream ), "synchronize SpGEMM segmented sort" );

    return true;
}

template bool SpGEMMSortExpandedProductsByColumn<int, int, float>( const SpGEMMSymbolicResult<int, int>&,
                                                                   const SpGEMMExpandedProducts<int, float>&,
                                                                   SpGEMMExpandedProducts<int, float>&,
                                                                   cudaStream_t );

template bool SpGEMMSortExpandedProductsByColumn<int, int, double>( const SpGEMMSymbolicResult<int, int>&,
                                                                    const SpGEMMExpandedProducts<int, double>&,
                                                                    SpGEMMExpandedProducts<int, double>&,
                                                                    cudaStream_t );

template bool SpGEMMSortExpandedProductsByColumn<std::int64_t, int, float>(
    const SpGEMMSymbolicResult<std::int64_t, int>&,
    const SpGEMMExpandedProducts<int, float>&,
    SpGEMMExpandedProducts<int, float>&,
    cudaStream_t );

template bool SpGEMMSortExpandedProductsByColumn<std::int64_t, int, double>(
    const SpGEMMSymbolicResult<std::int64_t, int>&,
    const SpGEMMExpandedProducts<int, double>&,
    SpGEMMExpandedProducts<int, double>&,
    cudaStream_t );

} // namespace matrix_utils::sparse_cuda
