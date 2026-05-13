#include <gtest/gtest.h>

#include "spgemm/spgemm_contract.cuh"
#include "spgemm/spgemm_expand.cuh"
#include "spgemm/spgemm_sort.cuh"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

using namespace matrix_utils::sparse_cuda;

namespace
{

bool HasCudaDevice( std::string& reason )
{
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount( &device_count );
    if ( status != cudaSuccess || device_count == 0 )
    {
        reason = cudaGetErrorString( status );
        return false;
    }
    return true;
}

} // namespace

TEST( CudaSpGEMMSymbolic, ComputesExpandedCountsPrefixPermutationAndBins )
{
    std::string cuda_skip_reason;
    if ( !HasCudaDevice( cuda_skip_reason ) )
    {
        GTEST_SKIP() << "CUDA device unavailable: " << cuda_skip_reason;
    }

    const std::vector<int> A_row_ptr = { 0, 2, 3, 5, 5 };
    const std::vector<int> A_col_ind = { 0, 2, 1, 3, 4 };
    const std::vector<int> B_row_ptr = { 0, 2, 2, 5, 6, 8 };

    DeviceArray<int> d_A_row_ptr;
    DeviceArray<int> d_A_col_ind;
    DeviceArray<int> d_B_row_ptr;
    d_A_row_ptr.copyFromHost( A_row_ptr.data(), A_row_ptr.size() );
    d_A_col_ind.copyFromHost( A_col_ind.data(), A_col_ind.size() );
    d_B_row_ptr.copyFromHost( B_row_ptr.data(), B_row_ptr.size() );

    SpGEMMSymbolicOptions options;
    options.thread_threshold = 0;
    options.warp_threshold = 3;
    options.cta_threshold = 4;

    SpGEMMSymbolicResult<int, int> symbolic;
    ASSERT_TRUE( ( SpGEMMSymbolicAnalyzeCSR<int, int>(
        4, 5, d_A_row_ptr.data(), d_A_col_ind.data(), 5, d_B_row_ptr.data(), 0, symbolic, options ) ) );

    EXPECT_EQ( symbolic.n_rows, 4 );
    EXPECT_EQ( symbolic.base, 0 );
    EXPECT_EQ( symbolic.total_expanded_nnz, 8 );

    std::vector<int> expanded_nnz( 4 );
    std::vector<int> expanded_row_ptr( 5 );
    std::vector<int> sorted_expanded_nnz( 4 );
    std::vector<int> row_perm( 4 );
    symbolic.expanded_nnz.copyToHost( expanded_nnz.data() );
    symbolic.expanded_row_ptr.copyToHost( expanded_row_ptr.data() );
    symbolic.sorted_expanded_nnz.copyToHost( sorted_expanded_nnz.data() );
    symbolic.row_perm.copyToHost( row_perm.data() );

    EXPECT_EQ( expanded_nnz, ( std::vector<int>{ 5, 0, 3, 0 } ) );
    EXPECT_EQ( expanded_row_ptr, ( std::vector<int>{ 0, 5, 5, 8, 8 } ) );
    EXPECT_EQ( sorted_expanded_nnz, ( std::vector<int>{ 0, 0, 3, 5 } ) );
    EXPECT_EQ( row_perm, ( std::vector<int>{ 1, 3, 2, 0 } ) );
    EXPECT_EQ( symbolic.row_class_offsets, ( std::array<int, 5>{ 0, 2, 3, 3, 4 } ) );
    EXPECT_EQ( symbolic.classBegin( SpGEMMRowClass::Warp ), 2 );
    EXPECT_EQ( symbolic.classEnd( SpGEMMRowClass::Warp ), 3 );
}

TEST( CudaSpGEMMSymbolic, ExpandsRawProductsInOriginalRowOrder )
{
    std::string cuda_skip_reason;
    if ( !HasCudaDevice( cuda_skip_reason ) )
    {
        GTEST_SKIP() << "CUDA device unavailable: " << cuda_skip_reason;
    }

    const std::vector<int> A_row_ptr = { 0, 2, 3, 5, 5 };
    const std::vector<int> A_col_ind = { 0, 2, 1, 3, 4 };
    const std::vector<double> A_values = { 2.0, 3.0, 7.0, 4.0, 5.0 };
    const std::vector<int> B_row_ptr = { 0, 2, 2, 5, 6, 8 };
    const std::vector<int> B_col_ind = { 1, 3, 0, 3, 4, 2, 1, 2 };
    const std::vector<double> B_values = { 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0 };

    DeviceArray<int> d_A_row_ptr;
    DeviceArray<int> d_A_col_ind;
    DeviceArray<double> d_A_values;
    DeviceArray<int> d_B_row_ptr;
    DeviceArray<int> d_B_col_ind;
    DeviceArray<double> d_B_values;
    d_A_row_ptr.copyFromHost( A_row_ptr.data(), A_row_ptr.size() );
    d_A_col_ind.copyFromHost( A_col_ind.data(), A_col_ind.size() );
    d_A_values.copyFromHost( A_values.data(), A_values.size() );
    d_B_row_ptr.copyFromHost( B_row_ptr.data(), B_row_ptr.size() );
    d_B_col_ind.copyFromHost( B_col_ind.data(), B_col_ind.size() );
    d_B_values.copyFromHost( B_values.data(), B_values.size() );

    SpGEMMSymbolicResult<int, int> symbolic;
    ASSERT_TRUE( ( SpGEMMSymbolicAnalyzeCSR<int, int>( 4, 5, d_A_row_ptr.data(), d_A_col_ind.data(),
                                                       5, d_B_row_ptr.data(), 0, symbolic ) ) );

    SpGEMMExpandedProducts<int, double> expanded;
    ASSERT_TRUE( ( SpGEMMExpandCSR<int, int, double>(
        4, 5, d_A_row_ptr.data(), d_A_col_ind.data(), d_A_values.data(), 5, d_B_row_ptr.data(),
        d_B_col_ind.data(), d_B_values.data(), 0, symbolic, expanded ) ) );

    std::vector<int> expanded_col_ind( symbolic.total_expanded_nnz );
    std::vector<double> expanded_values( symbolic.total_expanded_nnz );
    expanded.col_ind.copyToHost( expanded_col_ind.data() );
    expanded.values.copyToHost( expanded_values.data() );

    EXPECT_EQ( expanded_col_ind, ( std::vector<int>{ 1, 3, 0, 3, 4, 2, 1, 2 } ) );
    EXPECT_EQ( expanded_values, ( std::vector<double>{ 20.0, 40.0, 90.0, 120.0, 150.0, 240.0, 350.0, 400.0 } ) );
}

TEST( CudaSpGEMMSymbolic, SortsExpandedProductsByColumnWithinRows )
{
    std::string cuda_skip_reason;
    if ( !HasCudaDevice( cuda_skip_reason ) )
    {
        GTEST_SKIP() << "CUDA device unavailable: " << cuda_skip_reason;
    }

    const std::vector<int> row_ptr = { 0, 3, 3, 6 };
    const std::vector<int> col_ind = { 4, 1, 2, 3, 0, 2 };
    const std::vector<double> values = { 40.0, 10.0, 20.0, 30.0, 0.0, 21.0 };

    SpGEMMSymbolicResult<int, int> symbolic;
    symbolic.n_rows = 3;
    symbolic.base = 0;
    symbolic.total_expanded_nnz = 6;
    symbolic.expanded_row_ptr.copyFromHost( row_ptr.data(), row_ptr.size() );

    SpGEMMExpandedProducts<int, double> expanded;
    expanded.col_ind.copyFromHost( col_ind.data(), col_ind.size() );
    expanded.values.copyFromHost( values.data(), values.size() );

    SpGEMMExpandedProducts<int, double> sorted;
    ASSERT_TRUE( ( SpGEMMSortExpandedProductsByColumn<int, int, double>( symbolic, expanded, sorted ) ) );

    std::vector<int> sorted_col_ind( 6 );
    std::vector<double> sorted_values( 6 );
    sorted.col_ind.copyToHost( sorted_col_ind.data() );
    sorted.values.copyToHost( sorted_values.data() );

    EXPECT_EQ( sorted_col_ind, ( std::vector<int>{ 1, 2, 4, 0, 2, 3 } ) );
    EXPECT_EQ( sorted_values, ( std::vector<double>{ 10.0, 20.0, 40.0, 0.0, 21.0, 30.0 } ) );
}

TEST( CudaSpGEMMSymbolic, ContractsAndConstructsSortedProductsWithDuplicateColumns )
{
    std::string cuda_skip_reason;
    if ( !HasCudaDevice( cuda_skip_reason ) )
    {
        GTEST_SKIP() << "CUDA device unavailable: " << cuda_skip_reason;
    }

    const std::vector<int> row_ptr = { 0, 5, 5, 8 };
    const std::vector<int> col_ind = { 1, 1, 3, 3, 4, 0, 0, 2 };
    const std::vector<double> values = { 2.0, 5.0, 10.0, 20.0, 7.0, 1.0, 3.0, 9.0 };

    SpGEMMSymbolicResult<int, int> symbolic;
    symbolic.n_rows = 3;
    symbolic.base = 0;
    symbolic.total_expanded_nnz = 8;
    symbolic.expanded_row_ptr.copyFromHost( row_ptr.data(), row_ptr.size() );

    SpGEMMExpandedProducts<int, double> sorted;
    sorted.col_ind.copyFromHost( col_ind.data(), col_ind.size() );
    sorted.values.copyFromHost( values.data(), values.size() );

    SpGEMMReducedProducts<int, double> reduced;
    ASSERT_TRUE( ( SpGEMMContractSortedProducts<int, int, double>( symbolic, sorted, reduced ) ) );
    EXPECT_EQ( reduced.nnz, 5 );

    DeviceCSRMatrix<int, int> output;
    DeviceArray<double> output_values;
    ASSERT_TRUE( ( SpGEMMConstructCSR<int, int, double>( symbolic, reduced, output, output_values ) ) );

    std::vector<int> output_row_ptr( 4 );
    std::vector<int> output_col_ind( 5 );
    std::vector<double> output_vals( 5 );
    output.ai.copyToHost( output_row_ptr.data() );
    output.aj.copyToHost( output_col_ind.data() );
    output_values.copyToHost( output_vals.data() );

    EXPECT_EQ( output.n_rows, 3 );
    EXPECT_EQ( output.base, 0 );
    EXPECT_EQ( output_row_ptr, ( std::vector<int>{ 0, 3, 3, 5 } ) );
    EXPECT_EQ( output_col_ind, ( std::vector<int>{ 1, 3, 4, 0, 2 } ) );
    EXPECT_EQ( output_vals, ( std::vector<double>{ 7.0, 30.0, 7.0, 4.0, 9.0 } ) );
}

TEST( CudaSpGEMMSymbolic, PreservesOneBasedRowPointersAndPermutationIds )
{
    std::string cuda_skip_reason;
    if ( !HasCudaDevice( cuda_skip_reason ) )
    {
        GTEST_SKIP() << "CUDA device unavailable: " << cuda_skip_reason;
    }

    const std::vector<std::int64_t> A_row_ptr = { 1, 3, 4 };
    const std::vector<int> A_col_ind = { 1, 2, 3 };
    const std::vector<std::int64_t> B_row_ptr = { 1, 2, 4, 4 };

    DeviceArray<std::int64_t> d_A_row_ptr;
    DeviceArray<int> d_A_col_ind;
    DeviceArray<std::int64_t> d_B_row_ptr;
    d_A_row_ptr.copyFromHost( A_row_ptr.data(), A_row_ptr.size() );
    d_A_col_ind.copyFromHost( A_col_ind.data(), A_col_ind.size() );
    d_B_row_ptr.copyFromHost( B_row_ptr.data(), B_row_ptr.size() );

    SpGEMMSymbolicResult<std::int64_t, int> symbolic;
    ASSERT_TRUE( ( SpGEMMSymbolicAnalyzeCSR<std::int64_t, int>(
        2, 3, d_A_row_ptr.data(), d_A_col_ind.data(), 3, d_B_row_ptr.data(), 1, symbolic ) ) );

    EXPECT_EQ( symbolic.total_expanded_nnz, 3 );

    std::vector<std::int64_t> expanded_nnz( 2 );
    std::vector<std::int64_t> expanded_row_ptr( 3 );
    std::vector<int> row_perm( 2 );
    symbolic.expanded_nnz.copyToHost( expanded_nnz.data() );
    symbolic.expanded_row_ptr.copyToHost( expanded_row_ptr.data() );
    symbolic.row_perm.copyToHost( row_perm.data() );

    EXPECT_EQ( expanded_nnz, ( std::vector<std::int64_t>{ 3, 0 } ) );
    EXPECT_EQ( expanded_row_ptr, ( std::vector<std::int64_t>{ 1, 4, 4 } ) );
    EXPECT_EQ( row_perm, ( std::vector<int>{ 2, 1 } ) );
}
