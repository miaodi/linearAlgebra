#include <gtest/gtest.h>

#include "csr5_convert.hpp"
#include "csr5_format.hpp"
#include "csr5_policy.hpp"
#include "csr5_spmv.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace matrix_utils;

namespace
{

using SmallPolicy = CSR5StaticPolicy<4, 4>;

template <typename T>
std::vector<T> expectedTransposedTile( const std::vector<T>& input )
{
    std::vector<T> out( input.size() );
    for ( int lane = 0; lane < SmallPolicy::OMEGA; ++lane )
    {
        for ( int i = 0; i < SmallPolicy::SIGMA; ++i )
        {
            const int old_idx = lane * SmallPolicy::SIGMA + i;
            const int new_idx = i * SmallPolicy::OMEGA + lane;
            out[new_idx] = input[old_idx];
        }
    }
    return out;
}

std::vector<int> rowPtrFromLengths( const std::vector<int>& row_lengths )
{
    std::vector<int> row_ptr( row_lengths.size() + 1, 0 );
    for ( std::size_t row = 0; row < row_lengths.size(); ++row )
    {
        row_ptr[row + 1] = row_ptr[row] + row_lengths[row];
    }
    return row_ptr;
}

void referenceSpmv( const std::vector<int>& row_ptr,
                    const std::vector<int>& col_idx,
                    const std::vector<double>& values,
                    const std::vector<double>& x,
                    std::vector<double>& y,
                    const double alpha,
                    const double beta )
{
    for ( std::size_t row = 0; row + 1 < row_ptr.size(); ++row )
    {
        double sum = 0;
        for ( int idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx )
        {
            sum += values[idx] * x[col_idx[idx]];
        }
        y[row] = alpha * sum + beta * y[row];
    }
}

} // namespace

TEST( CSR5PolicyTest, StaticPolicyComputesDescriptorBits )
{
    EXPECT_EQ( SmallPolicy::OMEGA, 4 );
    EXPECT_EQ( SmallPolicy::SIGMA, 4 );
    EXPECT_EQ( SmallPolicy::TILE_SIZE, 16 );
    EXPECT_EQ( SmallPolicy::BIT_Y_OFFSET, 4 );
    EXPECT_EQ( SmallPolicy::BIT_SEG_OFFSET, 2 );
    EXPECT_LE( SmallPolicy::DESCRIPTOR_BITS, 32 );
}

TEST( CSR5PolicyTest, AVX2PoliciesUseSingleDescriptorPacket )
{
    EXPECT_EQ( CSR5_AVX2_Policy<double>::OMEGA, 4 );
    EXPECT_EQ( CSR5_AVX2_Policy<double>::SIGMA, 16 );
    EXPECT_EQ( CSR5_AVX2_Policy<double>::TILE_SIZE, 64 );
    EXPECT_LE( CSR5_AVX2_Policy<double>::DESCRIPTOR_BITS, 32 );

    EXPECT_EQ( CSR5_AVX2_Policy<float>::OMEGA, 8 );
    EXPECT_EQ( CSR5_AVX2_Policy<float>::SIGMA, 16 );
    EXPECT_EQ( CSR5_AVX2_Policy<float>::TILE_SIZE, 128 );
    EXPECT_LE( CSR5_AVX2_Policy<float>::DESCRIPTOR_BITS, 32 );
}

TEST( CSR5ConvertTest, PackUnpackLaneDescriptor )
{
    const uint32_t bit_flags_in = 0b1010;
    const uint32_t y_offset_in = 7;
    const uint32_t seg_offset_in = 2;

    const uint32_t packed = packCSR5LaneDesc<SmallPolicy>( bit_flags_in, y_offset_in, seg_offset_in );

    uint32_t bit_flags_out = 0;
    uint32_t y_offset_out = 0;
    uint32_t seg_offset_out = 0;
    unpackCSR5LaneDesc<SmallPolicy>( packed, bit_flags_out, y_offset_out, seg_offset_out );

    EXPECT_EQ( bit_flags_out, bit_flags_in );
    EXPECT_EQ( y_offset_out, y_offset_in );
    EXPECT_EQ( seg_offset_out, seg_offset_in );
}

TEST( CSR5ConvertTest, FullTileAoSoATransposeAndDescriptor )
{
    std::vector<int> ai = { 0, 4, 8, 12, 16 };
    std::vector<int> aj( 16 );
    std::vector<double> av( 16 );
    for ( int i = 0; i < 16; ++i )
    {
        aj[i] = i;
        av[i] = 100.0 + i;
    }

    CSR5Data<int, int, double, SmallPolicy> data;
    convertCSRtoCSR5<int, int, double, SmallPolicy>( 4, ai.data(), aj.data(), av.data(), data, 2 );

    EXPECT_EQ( data._num_rows, 4 );
    EXPECT_EQ( data._nnz, 16 );
    EXPECT_EQ( data._num_full_tiles, 1 );
    EXPECT_EQ( data._num_tiles, 1 );
    EXPECT_EQ( data._tail_tile_length, 0 );
    EXPECT_EQ( data._row_ptr, ai );
    EXPECT_EQ( data._tile_ptr, ( std::vector<int>{ 0, 4 } ) );

    EXPECT_EQ( data._tile_col_idx, expectedTransposedTile( aj ) );
    EXPECT_EQ( data._tile_val, expectedTransposedTile( av ) );

    for ( int lane = 0; lane < SmallPolicy::OMEGA; ++lane )
    {
        uint32_t bit_flags = 0;
        uint32_t y_offset = 0;
        uint32_t seg_offset = 0;
        data.unpackTileDesc( 0, lane, bit_flags, y_offset, seg_offset );

        EXPECT_EQ( bit_flags, 0b0001u );
        EXPECT_EQ( y_offset, lane == 0 ? 0u : static_cast<uint32_t>( lane - 1 ) );
        EXPECT_EQ( seg_offset, 0u );
    }
}

TEST( CSR5ConvertTest, FastTrackLongRowKeepsOnlyBitFlags )
{
    std::vector<int> ai = { 0, 20, 24 };
    std::vector<int> aj( 24 );
    std::vector<double> av( 24 );
    for ( int i = 0; i < 24; ++i )
    {
        aj[i] = i;
        av[i] = static_cast<double>( i );
    }

    CSR5Data<int, int, double, SmallPolicy> data;
    convertCSRtoCSR5<int, int, double, SmallPolicy>( 2, ai.data(), aj.data(), av.data(), data, 3 );

    EXPECT_EQ( data._num_full_tiles, 1 );
    EXPECT_EQ( data._num_tiles, 2 );
    EXPECT_EQ( data._tail_tile_length, 8 );
    EXPECT_EQ( data._tile_ptr, ( std::vector<int>{ 0, 0, 2 } ) );

    uint32_t bit_flags = 0;
    uint32_t y_offset = 0;
    uint32_t seg_offset = 0;
    data.unpackTileDesc( 0, 0, bit_flags, y_offset, seg_offset );
    EXPECT_EQ( bit_flags, 0b0001u );
    EXPECT_EQ( y_offset, 0u );
    EXPECT_EQ( seg_offset, 0u );

    for ( int lane = 1; lane < SmallPolicy::OMEGA; ++lane )
    {
        data.unpackTileDesc( 0, lane, bit_flags, y_offset, seg_offset );
        EXPECT_EQ( bit_flags, 0u );
        EXPECT_EQ( seg_offset, 0u );
    }

    for ( int idx = 16; idx < 24; ++idx )
    {
        EXPECT_EQ( data._tile_col_idx[idx], aj[idx] );
        EXPECT_EQ( data._tile_val[idx], av[idx] );
    }
}

TEST( CSR5ConvertTest, NormalTileProducesCrossLaneSegmentOffset )
{
    std::vector<int> ai = { 0, 9, 20 };
    std::vector<int> aj( 20 );
    std::vector<double> av( 20 );
    for ( int i = 0; i < 20; ++i )
    {
        aj[i] = i % 2;
        av[i] = static_cast<double>( i );
    }

    CSR5Data<int, int, double, SmallPolicy> data;
    convertCSRtoCSR5<int, int, double, SmallPolicy>( 2, ai.data(), aj.data(), av.data(), data, 2 );

    EXPECT_EQ( data._num_full_tiles, 1 );
    EXPECT_EQ( data._tile_ptr, ( std::vector<int>{ 0, 1, 2 } ) );

    uint32_t bit_flags = 0;
    uint32_t y_offset = 0;
    uint32_t seg_offset = 0;
    data.unpackTileDesc( 0, 0, bit_flags, y_offset, seg_offset );
    EXPECT_EQ( bit_flags, 0b0001u );
    EXPECT_EQ( y_offset, 0u );
    EXPECT_EQ( seg_offset, 1u );

    data.unpackTileDesc( 0, 1, bit_flags, y_offset, seg_offset );
    EXPECT_EQ( bit_flags, 0u );
    EXPECT_EQ( y_offset, 0u );
    EXPECT_EQ( seg_offset, 0u );

    data.unpackTileDesc( 0, 2, bit_flags, y_offset, seg_offset );
    EXPECT_EQ( bit_flags, 0b0010u );
    EXPECT_EQ( y_offset, 0u );
    EXPECT_EQ( seg_offset, 1u );
}

TEST( CSR5ConvertTest, OneBasedInputIsNormalized )
{
    std::vector<int> ai = { 1, 5, 9, 13, 17 };
    std::vector<int> aj( 16 );
    std::vector<double> av( 16 );
    for ( int i = 0; i < 16; ++i )
    {
        aj[i] = i + 1;
        av[i] = static_cast<double>( i );
    }

    CSR5Data<int, int, double, SmallPolicy> data;
    convertCSRtoCSR5<int, int, double, SmallPolicy>( 4, ai.data(), aj.data(), av.data(), data, 2 );

    EXPECT_EQ( data._base, 1 );
    EXPECT_EQ( data._row_ptr, ( std::vector<int>{ 0, 4, 8, 12, 16 } ) );

    std::vector<int> normalized_aj( 16 );
    for ( int i = 0; i < 16; ++i )
    {
        normalized_aj[i] = i;
    }
    EXPECT_EQ( data._tile_col_idx, expectedTransposedTile( normalized_aj ) );
}

TEST( CSR5ConvertTest, EmptyRowsAreRejectedInFirstVersion )
{
    std::vector<int> ai = { 0, 2, 2, 4 };
    std::vector<int> aj = { 0, 1, 0, 1 };
    std::vector<double> av = { 1.0, 2.0, 3.0, 4.0 };

    CSR5Data<int, int, double, SmallPolicy> data;
    EXPECT_THROW(
        ( convertCSRtoCSR5<int, int, double, SmallPolicy>( 3, ai.data(), aj.data(), av.data(), data, 2 ) ),
        std::invalid_argument );
}

TEST( CSR5FormatTest, MemoryEstimationMatchesOwnedData )
{
    const int num_rows = 4;
    const int nnz = 16;
    const size_t estimated = CSR5Data<int, int, double, SmallPolicy>::estimateMemoryBytes( num_rows, nnz );

    const size_t expected = static_cast<size_t>( num_rows + 1 ) * sizeof( int ) +
                            static_cast<size_t>( 2 ) * sizeof( int ) +
                            static_cast<size_t>( nnz ) * ( sizeof( int ) + sizeof( double ) ) +
                            static_cast<size_t>( SmallPolicy::OMEGA ) * sizeof( uint32_t );
    EXPECT_EQ( estimated, expected );
}

TEST( CSR5SPMVTest, ConstructorControlsPreprocessThreads )
{
    std::vector<int> ai = { 0, 4, 8, 12, 16 };
    std::vector<int> aj( 16 );
    std::vector<double> av( 16 );
    for ( int i = 0; i < 16; ++i )
    {
        aj[i] = i;
        av[i] = static_cast<double>( i );
    }

    CSR5SPMV<int, int, double, SmallPolicy> spmv( 2 );
    EXPECT_EQ( spmv.numThreads(), 2 );
    spmv.preprocess( 4, ai.data(), aj.data(), av.data() );
    EXPECT_EQ( spmv.data()._num_full_tiles, 1 );

    spmv.setNumThreads( -5 );
    EXPECT_EQ( spmv.numThreads(), 1 );
}

TEST( CSR5SPMVTest, DoubleKernelMatchesReferenceAcrossTilesAndTail )
{
    const std::vector<int> row_lengths = { 7, 70, 5, 68, 16 };
    std::vector<int> ai = rowPtrFromLengths( row_lengths );
    const int nnz = ai.back();

    std::vector<int> aj( nnz );
    std::vector<double> av( nnz );
    for ( int idx = 0; idx < nnz; ++idx )
    {
        aj[idx] = ( 3 * idx + 1 ) % static_cast<int>( row_lengths.size() );
        av[idx] = 0.25 + static_cast<double>( ( idx % 11 ) - 5 ) * 0.125;
    }

    std::vector<double> x = { 1.0, -2.0, 0.5, 3.0, -1.5 };
    std::vector<double> expected( row_lengths.size(), 2.0 );
    std::vector<double> actual = expected;

    const double alpha = 1.75;
    const double beta = -0.25;
    referenceSpmv( ai, aj, av, x, expected, alpha, beta );

    CSR5SPMV<int, int, double> spmv( 4 );
    spmv.preprocess( static_cast<int>( row_lengths.size() ), ai.data(), aj.data(), av.data() );
    EXPECT_EQ( spmv.data()._num_full_tiles, 2 );
    EXPECT_EQ( spmv.data()._tail_tile_length, 38 );

    spmv( x.data(), actual.data(), alpha, beta );

    for ( std::size_t row = 0; row < expected.size(); ++row )
    {
        EXPECT_NEAR( actual[row], expected[row], 1e-12 );
    }
}

TEST( CSR5SPMVTest, DoubleKernelHandlesTailOnlyMatrix )
{
    const std::vector<int> row_lengths = { 3, 4, 5 };
    std::vector<int> ai = rowPtrFromLengths( row_lengths );
    const int nnz = ai.back();

    std::vector<int> aj( nnz );
    std::vector<double> av( nnz );
    for ( int idx = 0; idx < nnz; ++idx )
    {
        aj[idx] = idx % static_cast<int>( row_lengths.size() );
        av[idx] = static_cast<double>( idx + 1 ) * 0.5;
    }

    std::vector<double> x = { 2.0, -1.0, 0.25 };
    std::vector<double> expected( row_lengths.size(), 1.0 );
    std::vector<double> actual = expected;

    referenceSpmv( ai, aj, av, x, expected, 1.0, 0.5 );

    CSR5SPMV<int, int, double> spmv( 2 );
    spmv.preprocess( static_cast<int>( row_lengths.size() ), ai.data(), aj.data(), av.data() );
    EXPECT_EQ( spmv.data()._num_full_tiles, 0 );
    EXPECT_EQ( spmv.data()._tail_tile_length, nnz );

    spmv( x.data(), actual.data(), 1.0, 0.5 );

    for ( std::size_t row = 0; row < expected.size(); ++row )
    {
        EXPECT_NEAR( actual[row], expected[row], 1e-12 );
    }
}
