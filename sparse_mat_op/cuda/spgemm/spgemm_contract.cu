#include "spgemm/spgemm_contract.cuh"

#include <cub/cub.cuh>
#include <cuda/std/functional>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include <cstdint>
#include <limits>

namespace matrix_utils::sparse_cuda
{
namespace
{

template <typename ROWTYPE, typename COLTYPE>
__global__ void build_packed_row_col_keys_kernel( COLTYPE rows,
                                                  ROWTYPE total_items,
                                                  const ROWTYPE* row_ptr,
                                                  const COLTYPE* col_ind,
                                                  ROWTYPE base,
                                                  std::uint64_t* packed_keys )
{
    ROWTYPE item = static_cast<ROWTYPE>( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( item >= total_items )
    {
        return;
    }

    COLTYPE low = 0;
    COLTYPE high = rows;
    while ( low < high )
    {
        const COLTYPE mid = static_cast<COLTYPE>( ( low + high ) / 2 );
        if ( row_ptr[mid + 1] - base <= item )
        {
            low = static_cast<COLTYPE>( mid + 1 );
        }
        else
        {
            high = mid;
        }
    }

    const std::uint64_t row = static_cast<std::uint64_t>( low + static_cast<COLTYPE>( base ) );
    const std::uint64_t col = static_cast<std::uint64_t>( static_cast<std::uint32_t>( col_ind[item] ) );
    packed_keys[item] = ( row << 32 ) | col;
}

template <typename ROWTYPE, typename COLTYPE>
__global__ void unpack_unique_keys_kernel( ROWTYPE unique_nnz,
                                           const std::uint64_t* unique_keys,
                                           COLTYPE* row_ind,
                                           COLTYPE* col_ind )
{
    ROWTYPE idx = static_cast<ROWTYPE>( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( idx >= unique_nnz )
    {
        return;
    }

    row_ind[idx] = static_cast<COLTYPE>( unique_keys[idx] >> 32 );
    col_ind[idx] = static_cast<COLTYPE>( unique_keys[idx] & 0xFFFFFFFFu );
}

template <typename ROWTYPE, typename COLTYPE>
__global__ void scatter_row_run_counts_kernel( COLTYPE num_row_runs,
                                               const COLTYPE* run_rows,
                                               const ROWTYPE* run_counts,
                                               ROWTYPE base,
                                               ROWTYPE* row_counts )
{
    COLTYPE idx = static_cast<COLTYPE>( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( idx >= num_row_runs )
    {
        return;
    }

    row_counts[run_rows[idx] - base] = run_counts[idx];
}

template <typename ROWTYPE, typename COLTYPE>
bool contractionInputIsValid( const SpGEMMSymbolicResult<ROWTYPE, COLTYPE>& symbolic )
{
    return symbolic.n_rows >= 0 && symbolic.total_expanded_nnz >= 0 &&
           symbolic.expanded_row_ptr.data() != nullptr;
}

} // namespace

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
bool SpGEMMContractSortedProducts( const SpGEMMSymbolicResult<ROWTYPE, COLTYPE>& symbolic,
                                   const SpGEMMExpandedProducts<COLTYPE, VALTYPE>& sorted,
                                   SpGEMMReducedProducts<ROWTYPE, VALTYPE>& reduced,
                                   cudaStream_t stream )
{
    static_assert( sizeof( COLTYPE ) <= 4,
                   "SpGEMM contraction packs row/column ids into 64-bit keys." );

    if ( !contractionInputIsValid( symbolic ) )
    {
        return false;
    }

    reduced.nnz = 0;
    if ( symbolic.total_expanded_nnz == 0 )
    {
        reduced.row_col_keys.resize( 0 );
        reduced.values.resize( 0 );
        return true;
    }
    if ( sorted.col_ind.data() == nullptr || sorted.values.data() == nullptr )
    {
        return false;
    }

    const ROWTYPE total_items = symbolic.total_expanded_nnz;
    if ( total_items > static_cast<ROWTYPE>( std::numeric_limits<int>::max() ) )
    {
        return false;
    }
    DeviceArray<std::uint64_t> packed_keys;
    DeviceArray<ROWTYPE> unique_count;
    packed_keys.resize( static_cast<size_t>( total_items ) );
    reduced.row_col_keys.resize( static_cast<size_t>( total_items ) );
    reduced.values.resize( static_cast<size_t>( total_items ) );
    unique_count.resize( 1 );

    constexpr int threads = 256;
    const int build_blocks = static_cast<int>( ( total_items + threads - 1 ) / threads );
    build_packed_row_col_keys_kernel<ROWTYPE, COLTYPE><<<build_blocks, threads, 0, stream>>>(
        symbolic.n_rows, total_items, symbolic.expanded_row_ptr.data(), sorted.col_ind.data(),
        symbolic.base, packed_keys.data() );
    checkCudaError( cudaGetLastError(), "launch SpGEMM contraction key-build kernel" );

    void* temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    checkCudaError( cub::DeviceReduce::ReduceByKey(
                        temp_storage, temp_storage_bytes, packed_keys.data(), reduced.row_col_keys.data(),
                        sorted.values.data(), reduced.values.data(), unique_count.data(),
                        ::cuda::std::plus<VALTYPE>(), static_cast<int>( total_items ), stream ),
                    "query SpGEMM reduce-by-key temporary storage" );

    DeviceArray<std::uint8_t> reduce_storage;
    reduce_storage.resize( temp_storage_bytes );
    checkCudaError( cub::DeviceReduce::ReduceByKey( reduce_storage.data(), temp_storage_bytes,
                                                    packed_keys.data(), reduced.row_col_keys.data(),
                                                    sorted.values.data(), reduced.values.data(),
                                                    unique_count.data(), ::cuda::std::plus<VALTYPE>(),
                                                    static_cast<int>( total_items ), stream ),
                    "run SpGEMM reduce-by-key" );

    ROWTYPE h_unique_nnz = 0;
    checkCudaError( cudaMemcpyAsync( &h_unique_nnz, unique_count.data(), sizeof( ROWTYPE ),
                                     cudaMemcpyDeviceToHost, stream ),
                    "copy SpGEMM unique nnz" );
    checkCudaError( cudaStreamSynchronize( stream ), "synchronize SpGEMM unique nnz" );

    reduced.nnz = h_unique_nnz;
    reduced.row_col_keys.resize( static_cast<size_t>( h_unique_nnz ) );
    reduced.values.resize( static_cast<size_t>( h_unique_nnz ) );
    return true;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
bool SpGEMMConstructCSR( const SpGEMMSymbolicResult<ROWTYPE, COLTYPE>& symbolic,
                         const SpGEMMReducedProducts<ROWTYPE, VALTYPE>& reduced,
                         DeviceCSRMatrix<ROWTYPE, COLTYPE>& output,
                         DeviceArray<VALTYPE>& output_values,
                         cudaStream_t stream )
{
    static_assert( sizeof( COLTYPE ) <= 4,
                   "SpGEMM construct unpacks row/column ids from 64-bit keys." );

    if ( symbolic.n_rows < 0 || reduced.nnz < 0 )
    {
        return false;
    }
    if ( static_cast<std::int64_t>( symbolic.n_rows ) + 1 >
         static_cast<std::int64_t>( std::numeric_limits<int>::max() ) )
    {
        return false;
    }
    if ( reduced.nnz > static_cast<ROWTYPE>( std::numeric_limits<int>::max() ) )
    {
        return false;
    }

    output.n_rows = symbolic.n_rows;
    output.base = symbolic.base;
    output.ai.resize( static_cast<size_t>( symbolic.n_rows + 1 ) );
    output.aj.resize( static_cast<size_t>( reduced.nnz ) );
    output_values.resize( static_cast<size_t>( reduced.nnz ) );

    if ( reduced.nnz == 0 )
    {
        auto policy = thrust::cuda::par.on( stream );
        auto ai_begin = thrust::device_pointer_cast( output.ai.data() );
        thrust::fill( policy, ai_begin, ai_begin + symbolic.n_rows + 1, symbolic.base );
        checkCudaError( cudaStreamSynchronize( stream ), "synchronize empty SpGEMM construct" );
        return true;
    }
    if ( reduced.row_col_keys.data() == nullptr || reduced.values.data() == nullptr )
    {
        return false;
    }

    checkCudaError( cudaMemcpyAsync( output_values.data(), reduced.values.data(),
                                     static_cast<size_t>( reduced.nnz ) * sizeof( VALTYPE ),
                                     cudaMemcpyDeviceToDevice, stream ),
                    "copy SpGEMM constructed values" );

    constexpr int threads = 256;
    const int unique_blocks = static_cast<int>( ( reduced.nnz + threads - 1 ) / threads );
    DeviceArray<COLTYPE> unique_row_ind;
    unique_row_ind.resize( static_cast<size_t>( reduced.nnz ) );
    unpack_unique_keys_kernel<ROWTYPE, COLTYPE><<<unique_blocks, threads, 0, stream>>>(
        reduced.nnz, reduced.row_col_keys.data(), unique_row_ind.data(), output.aj.data() );
    checkCudaError( cudaGetLastError(), "launch SpGEMM construct key-unpack kernel" );

    DeviceArray<COLTYPE> row_run_rows;
    DeviceArray<ROWTYPE> row_run_counts;
    DeviceArray<COLTYPE> row_run_count;
    row_run_rows.resize( static_cast<size_t>( reduced.nnz ) );
    row_run_counts.resize( static_cast<size_t>( reduced.nnz ) );
    row_run_count.resize( 1 );

    void* rle_temp_storage = nullptr;
    size_t rle_temp_bytes = 0;
    checkCudaError( cub::DeviceRunLengthEncode::Encode(
                        rle_temp_storage, rle_temp_bytes, unique_row_ind.data(), row_run_rows.data(),
                        row_run_counts.data(), row_run_count.data(), static_cast<int>( reduced.nnz ), stream ),
                    "query SpGEMM construct row RLE storage" );

    DeviceArray<std::uint8_t> rle_storage;
    rle_storage.resize( rle_temp_bytes );
    checkCudaError( cub::DeviceRunLengthEncode::Encode(
                        rle_storage.data(), rle_temp_bytes, unique_row_ind.data(), row_run_rows.data(),
                        row_run_counts.data(), row_run_count.data(), static_cast<int>( reduced.nnz ), stream ),
                    "run SpGEMM construct row RLE" );

    COLTYPE h_row_runs = 0;
    checkCudaError( cudaMemcpyAsync( &h_row_runs, row_run_count.data(), sizeof( COLTYPE ),
                                     cudaMemcpyDeviceToHost, stream ),
                    "copy SpGEMM construct row-run count" );
    checkCudaError( cudaStreamSynchronize( stream ), "synchronize SpGEMM construct row-run count" );

    DeviceArray<ROWTYPE> row_counts;
    row_counts.resize( static_cast<size_t>( symbolic.n_rows + 1 ) );
    checkCudaError( cudaMemsetAsync( row_counts.data(), 0,
                                     static_cast<size_t>( symbolic.n_rows + 1 ) * sizeof( ROWTYPE ), stream ),
                    "initialize SpGEMM construct row counts" );

    const int row_run_blocks = static_cast<int>( ( h_row_runs + threads - 1 ) / threads );
    scatter_row_run_counts_kernel<ROWTYPE, COLTYPE><<<row_run_blocks, threads, 0, stream>>>(
        h_row_runs, row_run_rows.data(), row_run_counts.data(), symbolic.base, row_counts.data() );
    checkCudaError( cudaGetLastError(), "launch SpGEMM construct row-count scatter kernel" );

    void* scan_temp_storage = nullptr;
    size_t scan_temp_bytes = 0;
    checkCudaError(
        cub::DeviceScan::ExclusiveScan( scan_temp_storage, scan_temp_bytes, row_counts.data(),
                                        output.ai.data(), ::cuda::std::plus<ROWTYPE>(), symbolic.base,
                                        static_cast<int>( symbolic.n_rows + 1 ), stream ),
        "query SpGEMM construct row scan storage" );

    DeviceArray<std::uint8_t> scan_storage;
    scan_storage.resize( scan_temp_bytes );
    checkCudaError(
        cub::DeviceScan::ExclusiveScan( scan_storage.data(), scan_temp_bytes, row_counts.data(),
                                        output.ai.data(), ::cuda::std::plus<ROWTYPE>(), symbolic.base,
                                        static_cast<int>( symbolic.n_rows + 1 ), stream ),
        "run SpGEMM construct row scan" );
    checkCudaError( cudaStreamSynchronize( stream ), "synchronize SpGEMM construct" );

    return true;
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
bool SpGEMMContractSortedProducts( const SpGEMMSymbolicResult<ROWTYPE, COLTYPE>& symbolic,
                                   const SpGEMMExpandedProducts<COLTYPE, VALTYPE>& sorted,
                                   DeviceCSRMatrix<ROWTYPE, COLTYPE>& output,
                                   DeviceArray<VALTYPE>& output_values,
                                   cudaStream_t stream )
{
    SpGEMMReducedProducts<ROWTYPE, VALTYPE> reduced;
    if ( !SpGEMMContractSortedProducts( symbolic, sorted, reduced, stream ) )
    {
        return false;
    }
    return SpGEMMConstructCSR( symbolic, reduced, output, output_values, stream );
}

template bool SpGEMMContractSortedProducts<int, int, float>( const SpGEMMSymbolicResult<int, int>&,
                                                             const SpGEMMExpandedProducts<int, float>&,
                                                             SpGEMMReducedProducts<int, float>&,
                                                             cudaStream_t );

template bool SpGEMMContractSortedProducts<int, int, double>( const SpGEMMSymbolicResult<int, int>&,
                                                              const SpGEMMExpandedProducts<int, double>&,
                                                              SpGEMMReducedProducts<int, double>&,
                                                              cudaStream_t );

template bool SpGEMMContractSortedProducts<std::int64_t, int, float>(
    const SpGEMMSymbolicResult<std::int64_t, int>&,
    const SpGEMMExpandedProducts<int, float>&,
    SpGEMMReducedProducts<std::int64_t, float>&,
    cudaStream_t );

template bool SpGEMMContractSortedProducts<std::int64_t, int, double>(
    const SpGEMMSymbolicResult<std::int64_t, int>&,
    const SpGEMMExpandedProducts<int, double>&,
    SpGEMMReducedProducts<std::int64_t, double>&,
    cudaStream_t );

template bool SpGEMMConstructCSR<int, int, float>( const SpGEMMSymbolicResult<int, int>&,
                                                   const SpGEMMReducedProducts<int, float>&,
                                                   DeviceCSRMatrix<int, int>&,
                                                   DeviceArray<float>&,
                                                   cudaStream_t );

template bool SpGEMMConstructCSR<int, int, double>( const SpGEMMSymbolicResult<int, int>&,
                                                    const SpGEMMReducedProducts<int, double>&,
                                                    DeviceCSRMatrix<int, int>&,
                                                    DeviceArray<double>&,
                                                    cudaStream_t );

template bool SpGEMMConstructCSR<std::int64_t, int, float>( const SpGEMMSymbolicResult<std::int64_t, int>&,
                                                            const SpGEMMReducedProducts<std::int64_t, float>&,
                                                            DeviceCSRMatrix<std::int64_t, int>&,
                                                            DeviceArray<float>&,
                                                            cudaStream_t );

template bool SpGEMMConstructCSR<std::int64_t, int, double>( const SpGEMMSymbolicResult<std::int64_t, int>&,
                                                             const SpGEMMReducedProducts<std::int64_t, double>&,
                                                             DeviceCSRMatrix<std::int64_t, int>&,
                                                             DeviceArray<double>&,
                                                             cudaStream_t );

template bool SpGEMMContractSortedProducts<int, int, float>( const SpGEMMSymbolicResult<int, int>&,
                                                             const SpGEMMExpandedProducts<int, float>&,
                                                             DeviceCSRMatrix<int, int>&,
                                                             DeviceArray<float>&,
                                                             cudaStream_t );

template bool SpGEMMContractSortedProducts<int, int, double>( const SpGEMMSymbolicResult<int, int>&,
                                                              const SpGEMMExpandedProducts<int, double>&,
                                                              DeviceCSRMatrix<int, int>&,
                                                              DeviceArray<double>&,
                                                              cudaStream_t );

template bool SpGEMMContractSortedProducts<std::int64_t, int, float>( const SpGEMMSymbolicResult<std::int64_t, int>&,
                                                                      const SpGEMMExpandedProducts<int, float>&,
                                                                      DeviceCSRMatrix<std::int64_t, int>&,
                                                                      DeviceArray<float>&,
                                                                      cudaStream_t );

template bool SpGEMMContractSortedProducts<std::int64_t, int, double>(
    const SpGEMMSymbolicResult<std::int64_t, int>&,
    const SpGEMMExpandedProducts<int, double>&,
    DeviceCSRMatrix<std::int64_t, int>&,
    DeviceArray<double>&,
    cudaStream_t );

} // namespace matrix_utils::sparse_cuda
