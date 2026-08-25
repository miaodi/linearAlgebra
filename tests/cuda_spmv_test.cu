#include "cuda_spmv.cuh"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include <vector>

namespace cuda_utils = matrix_utils::sparse_cuda;

namespace
{

template <typename T>
T* device_data_or_null( thrust::device_vector<T>& values )
{
    return values.empty() ? nullptr : thrust::raw_pointer_cast( values.data() );
}

template <typename Index = int, typename Value = double>
void run_merge_spmv_case( const Index rows,
                          const Index base,
                          const std::vector<Index>& ia,
                          const std::vector<Index>& ja,
                          const std::vector<Value>& av,
                          const std::vector<Value>& x,
                          const std::vector<Value>& y_initial,
                          const Value alpha,
                          const Value beta )
{
    ASSERT_EQ( ia.size(), static_cast<size_t>( rows + 1 ) );
    ASSERT_EQ( ja.size(), av.size() );
    ASSERT_EQ( y_initial.size(), static_cast<size_t>( rows ) );

    const Index nnz = ia[static_cast<size_t>( rows )] - base;
    ASSERT_EQ( static_cast<size_t>( nnz ), ja.size() );

    std::vector<Value> expected = y_initial;
    for ( Index row = 0; row < rows; ++row )
    {
        Value sum = static_cast<Value>( 0 );
        for ( Index idx = ia[static_cast<size_t>( row )] - base;
              idx < ia[static_cast<size_t>( row + 1 )] - base; ++idx )
        {
            const Index col = ja[static_cast<size_t>( idx )] - base;
            sum += av[static_cast<size_t>( idx )] * x[static_cast<size_t>( col )];
        }
        expected[static_cast<size_t>( row )] = alpha * sum + beta * expected[static_cast<size_t>( row )];
    }

    thrust::device_vector<Index> d_ia( ia );
    thrust::device_vector<Index> d_ja( ja );
    thrust::device_vector<Value> d_av( av );
    thrust::device_vector<Value> d_x( x );
    thrust::device_vector<Value> d_y( y_initial );

    cuda_utils::CSRMergeSPMV<Index, Index, Value> spmv;
    spmv.preprocess( rows, thrust::raw_pointer_cast( d_ia.data() ), device_data_or_null( d_ja ),
                     device_data_or_null( d_av ), base, nnz );
    spmv( thrust::raw_pointer_cast( d_x.data() ), thrust::raw_pointer_cast( d_y.data() ), alpha, beta );

    const cudaError_t sync_status = cudaDeviceSynchronize();
    ASSERT_EQ( sync_status, cudaSuccess ) << cudaGetErrorString( sync_status );

    std::vector<Value> actual( static_cast<size_t>( rows ) );
    thrust::copy( d_y.begin(), d_y.end(), actual.begin() );

    constexpr double relative_tolerance = std::is_same_v<Value, float> ? 1e-5 : 1e-11;
    for ( Index row = 0; row < rows; ++row )
    {
        const double expected_value = static_cast<double>( expected[static_cast<size_t>( row )] );
        const double tolerance = relative_tolerance * std::max( 1.0, std::abs( expected_value ) );
        EXPECT_NEAR( static_cast<double>( actual[static_cast<size_t>( row )] ), expected_value, tolerance )
            << "row " << row;
    }
}

} // namespace

TEST( CSRMergeSPMV, HandlesEmptyRowsSplitRowsAndBeta )
{
    constexpr int rows = 6;
    constexpr int base = 0;

    std::vector<int> ia = { 0, 0, 3, 3, 23, 24, 24 };
    std::vector<int> ja = { 0, 1, 2 };
    std::vector<double> av = { 1.0, -2.0, 0.5 };
    for ( int k = 0; k < 20; ++k )
    {
        ja.push_back( k % rows );
        av.push_back( 0.25 + 0.125 * static_cast<double>( k ) );
    }
    ja.push_back( 5 );
    av.push_back( -3.0 );

    const std::vector<double> x = { 1.0, -1.0, 2.0, 0.5, -0.25, 4.0 };
    const std::vector<double> y_initial( rows, 2.0 );

    run_merge_spmv_case( rows, base, ia, ja, av, x, y_initial, 1.25, 0.5 );
}

TEST( CSRMergeSPMV, HandlesBaseOneCSR )
{
    constexpr int rows = 4;
    constexpr int base = 1;

    const std::vector<int> ia = { 1, 2, 2, 4, 5 };
    const std::vector<int> ja = { 1, 2, 4, 3 };
    const std::vector<double> av = { 2.0, -1.0, 3.0, 4.0 };
    const std::vector<double> x = { 0.5, -2.0, 1.5, 3.0 };
    const std::vector<double> y_initial = { -1.0, 2.0, -3.0, 4.0 };

    run_merge_spmv_case( rows, base, ia, ja, av, x, y_initial, -2.0, -0.25 );
}

TEST( CSRMergeSPMV, AllEmptyRowsOnlyScaleY )
{
    constexpr int rows = 5;
    constexpr int base = 0;

    const std::vector<int> ia( rows + 1, base );
    const std::vector<int> ja;
    const std::vector<double> av;
    const std::vector<double> x( rows, 1.0 );
    const std::vector<double> y_initial = { 1.0, -2.0, 3.0, -4.0, 5.0 };

    run_merge_spmv_case( rows, base, ia, ja, av, x, y_initial, 7.0, -0.5 );
}

TEST( CSRMergeSPMV, HandlesRowSplitAcrossMultipleThreadBlocks )
{
    constexpr int rows = 2;
    constexpr int base = 0;
    constexpr int long_row_nnz = 4097;

    const std::vector<int> ia = { 0, long_row_nnz, long_row_nnz };
    std::vector<int> ja( long_row_nnz, 0 );
    std::vector<double> av;
    av.reserve( long_row_nnz );
    for ( int idx = 0; idx < long_row_nnz; ++idx )
    {
        av.push_back( 0.001 * static_cast<double>( idx % 7 + 1 ) );
    }

    const std::vector<double> x = { 2.0, -1.0 };
    const std::vector<double> y_initial = { 3.0, -4.0 };

    run_merge_spmv_case( rows, base, ia, ja, av, x, y_initial, 1.5, 0.0 );
    run_merge_spmv_case( rows, base, ia, ja, av, x, y_initial, -0.5, 1.0 );
}

TEST( CSRMergeSPMV, MatchesCpuForIrregularPattern )
{
    constexpr int rows = 257;
    constexpr int base = 0;

    std::vector<int> ia( rows + 1, base );
    std::vector<int> ja;
    std::vector<double> av;
    for ( int row = 0; row < rows; ++row )
    {
        const int row_nnz = row % 7 == 0 ? 0 : ( row * 17 + 3 ) % 41;
        for ( int entry = 0; entry < row_nnz; ++entry )
        {
            ja.push_back( ( row * 13 + entry * 29 ) % rows );
            av.push_back( 0.01 * static_cast<double>( ( row + entry ) % 19 - 9 ) );
        }
        ia[static_cast<size_t>( row + 1 )] = static_cast<int>( ja.size() );
    }

    std::vector<double> x( rows );
    std::vector<double> y_initial( rows );
    for ( int row = 0; row < rows; ++row )
    {
        x[static_cast<size_t>( row )] = 0.125 * static_cast<double>( row % 11 - 5 );
        y_initial[static_cast<size_t>( row )] = 0.25 * static_cast<double>( row % 5 - 2 );
    }

    run_merge_spmv_case( rows, base, ia, ja, av, x, y_initial, 1.75, -0.375 );
}

TEST( CSRMergeSPMV, HandlesInt64FloatInstantiation )
{
    using Index = std::int64_t;
    using Value = float;

    constexpr Index rows = 5;
    constexpr Index base = 1;
    const std::vector<Index> ia = { 1, 1, 4, 4, 6, 7 };
    const std::vector<Index> ja = { 1, 3, 5, 2, 4, 5 };
    const std::vector<Value> av = { 2.0F, -1.0F, 0.5F, 3.0F, -4.0F, 1.25F };
    const std::vector<Value> x = { 1.0F, -2.0F, 3.0F, -4.0F, 0.5F };
    const std::vector<Value> y_initial = { 5.0F, 4.0F, 3.0F, 2.0F, 1.0F };

    run_merge_spmv_case( rows, base, ia, ja, av, x, y_initial, 0.75F, 0.0F );
}
