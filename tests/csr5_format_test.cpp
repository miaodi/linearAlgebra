#include <gtest/gtest.h>
#include "csr5_policy.hpp"
#include "csr5_format.hpp"
#include "csr5_convert.hpp"
#include <vector>

using namespace matrix_utils;

// Test CSR5 policy constants
TEST( CSR5PolicyTest, AVX2DoublePolicy )
{
    using Policy = CSR5_AVX2_Policy<double>;

    EXPECT_EQ( Policy::OMEGA, 4 );
    EXPECT_EQ( Policy::SIGMA, 32 );
    EXPECT_EQ( Policy::TILE_SIZE, 128 );
}

TEST( CSR5PolicyTest, AVX2FloatPolicy )
{
    using Policy = CSR5_AVX2_Policy<float>;

    EXPECT_EQ( Policy::OMEGA, 8 );
    EXPECT_EQ( Policy::SIGMA, 32 );
    EXPECT_EQ( Policy::TILE_SIZE, 256 );
}

// Test metadata packing/unpacking
TEST( CSR5ConvertTest, PackUnpackMetadata )
{
    // Test with OMEGA = 4
    {
        uint32_t bit_flag_in = 0b1010; // 4 bits
        uint32_t y_offset_in = 12345;
        uint16_t seg_offset_in = 7;

        uint64_t packed = packCSR5TileDesc( bit_flag_in, y_offset_in, seg_offset_in, 4 );

        uint32_t bit_flag_out, y_offset_out;
        uint16_t seg_offset_out;
        unpackCSR5TileDesc( packed, bit_flag_out, y_offset_out, seg_offset_out, 4 );

        EXPECT_EQ( bit_flag_out, bit_flag_in );
        EXPECT_EQ( y_offset_out, y_offset_in );
        EXPECT_EQ( seg_offset_out, seg_offset_in );
    }

    // Test with OMEGA = 8
    {
        uint32_t bit_flag_in = 0b10101010; // 8 bits
        uint32_t y_offset_in = 54321;
        uint16_t seg_offset_in = 15;

        uint64_t packed = packCSR5TileDesc( bit_flag_in, y_offset_in, seg_offset_in, 8 );

        uint32_t bit_flag_out, y_offset_out;
        uint16_t seg_offset_out;
        unpackCSR5TileDesc( packed, bit_flag_out, y_offset_out, seg_offset_out, 8 );

        EXPECT_EQ( bit_flag_out, bit_flag_in );
        EXPECT_EQ( y_offset_out, y_offset_in );
        EXPECT_EQ( seg_offset_out, seg_offset_in );
    }
}

// Test conversion with a small matrix
TEST( CSR5ConvertTest, SmallMatrixConversion )
{
    using Policy = CSR5_AVX2_Policy<double>;

    // Create a simple 3x3 matrix with 5 non-zeros (0-based indexing)
    // [1 0 2]
    // [0 3 0]
    // [4 5 0]
    std::vector<int> ai = { 0, 2, 3, 5 };                 // row pointers
    std::vector<int> aj = { 0, 2, 1, 0, 1 };              // column indices
    std::vector<double> av = { 1.0, 2.0, 3.0, 4.0, 5.0 }; // values

    int num_rows = 3;

    CSR5Data<int, int, double, Policy> csr5_data;

    convertCSRtoCSR5<int, int, double, Policy>( num_rows, ai.data(), aj.data(), av.data(), csr5_data );

    EXPECT_EQ( csr5_data._num_rows, 3 );
    EXPECT_EQ( csr5_data._nnz, 5 );
    EXPECT_EQ( csr5_data._num_tiles, 1 ); // 5 elements < 128, so 1 tile
    EXPECT_EQ( csr5_data._tail_tile_length, 5 );

    // Verify tile_ptr
    EXPECT_EQ( csr5_data._tile_ptr.size(), 2 ); // num_tiles + 1
    auto tile_start_row = CSR5Data<int, int, double, Policy>::getTileStartRow( csr5_data._tile_ptr[0] );
    EXPECT_EQ( tile_start_row, 0 );
    EXPECT_EQ( csr5_data._tile_ptr[1], 3 ); // Points to row after last element
    auto has_empty = CSR5Data<int, int, double, Policy>::hasEmptyRows( csr5_data._tile_ptr[0] );
    EXPECT_FALSE( has_empty );

    // Verify data is stored column-major
    // Element 0: lane 0, col 0, index = 0*4 + 0 = 0
    // Element 1: lane 1, col 0, index = 0*4 + 1 = 1
    // Element 2: lane 2, col 0, index = 0*4 + 2 = 2
    // Element 3: lane 3, col 0, index = 0*4 + 3 = 3
    // Element 4: lane 0, col 1, index = 1*4 + 0 = 4

    const int* tile_col = csr5_data.getTileColIdx( 0 );
    const double* tile_val = csr5_data.getTileVal( 0 );

    EXPECT_EQ( tile_col[0], 0 ); // element 0, lane 0
    EXPECT_EQ( tile_val[0], 1.0 );

    EXPECT_EQ( tile_col[1], 2 ); // element 1, lane 1
    EXPECT_EQ( tile_val[1], 2.0 );

    EXPECT_EQ( tile_col[2], 1 ); // element 2, lane 2
    EXPECT_EQ( tile_val[2], 3.0 );

    EXPECT_EQ( tile_col[3], 0 ); // element 3, lane 3
    EXPECT_EQ( tile_val[3], 4.0 );

    EXPECT_EQ( tile_col[4], 1 ); // element 4, lane 0, col 1
    EXPECT_EQ( tile_val[4], 5.0 );

    // Check metadata
    uint32_t bit_flag;
    int y_offset;
    uint16_t seg_offset;
    csr5_data.unpackTileDesc( 0, bit_flag, y_offset, seg_offset );

    EXPECT_EQ( y_offset, 0 );   // starts at row 0
    EXPECT_EQ( seg_offset, 0 ); // first segment

    // Bit flag should indicate new rows:
    // Lane 0: element 0 starts row 0 -> bit 0 = 1
    // Lane 1: element 1 is in row 0 -> bit 1 = 0
    // Lane 2: element 2 starts row 1 -> bit 2 = 1
    // Lane 3: element 3 starts row 2 -> bit 3 = 1
    EXPECT_EQ( bit_flag & 0b0001, 0b0001 ); // bit 0 set
    EXPECT_EQ( bit_flag & 0b0100, 0b0100 ); // bit 2 set
    EXPECT_EQ( bit_flag & 0b1000, 0b1000 ); // bit 3 set
}

// Test memory estimation
TEST( CSR5FormatTest, MemoryEstimation )
{
    using Policy = CSR5_AVX2_Policy<double>;

    int nnz = 1000;
    size_t estimated = CSR5Data<int, int, double, Policy>::estimateMemoryBytes( nnz );

    // Expected: ceil(1000/128) = 8 tiles
    // (8 + 1) * sizeof(int) tile pointers
    // + 8 * 128 * (sizeof(int) + sizeof(double)) tile data
    // + 8 * sizeof(uint64_t) descriptors

    int num_tiles = ( nnz + Policy::TILE_SIZE - 1 ) / Policy::TILE_SIZE;
    size_t expected = ( num_tiles + 1 ) * sizeof( int ) +
                      num_tiles * Policy::TILE_SIZE * ( sizeof( int ) + sizeof( double ) ) +
                      num_tiles * sizeof( uint64_t );

    EXPECT_EQ( estimated, expected );
}

// Test tail tile handling
TEST( CSR5ConvertTest, TailTileHandling )
{
    using Policy = CSR5_AVX2_Policy<double>;

    // Create matrix with 135 elements (1 full tile + 7 element tail)
    int nnz = 135;
    int num_rows = 10;

    std::vector<int> ai( num_rows + 1 );
    std::vector<int> aj( nnz );
    std::vector<double> av( nnz );

    // Simple pattern: each row has roughly nnz/num_rows elements
    for ( int i = 0; i <= num_rows; ++i )
    {
        ai[i] = ( i * nnz ) / num_rows;
    }

    for ( int i = 0; i < nnz; ++i )
    {
        aj[i] = i % num_rows;
        av[i] = static_cast<double>( i );
    }

    CSR5Data<int, int, double, Policy> csr5_data;

    convertCSRtoCSR5<int, int, double, Policy>( num_rows, ai.data(), aj.data(), av.data(), csr5_data );

    EXPECT_EQ( csr5_data._num_tiles, 2 );        // ceil(135/128) = 2
    EXPECT_EQ( csr5_data._tail_tile_length, 7 ); // 135 % 128 = 7

    // Tail tile should be padded with zeros
    const double* tile_val_tail = csr5_data.getTileVal( 1 );
    for ( int i = 7; i < Policy::TILE_SIZE; ++i )
    {
        EXPECT_EQ( tile_val_tail[i], 0.0 );
    }
}

// Test empty matrix
TEST( CSR5ConvertTest, EmptyMatrix )
{
    using Policy = CSR5_AVX2_Policy<double>;

    int num_rows = 5;
    std::vector<int> ai( num_rows + 1, 0 ); // All zeros -> empty matrix

    CSR5Data<int, int, double, Policy> csr5_data;

    convertCSRtoCSR5<int, int, double, Policy>( num_rows, ai.data(), nullptr, nullptr, csr5_data );

    EXPECT_EQ( csr5_data._num_rows, 5 );
    EXPECT_EQ( csr5_data._nnz, 0 );
    EXPECT_EQ( csr5_data._num_tiles, 0 );
}

// Test matrix with empty rows
TEST( CSR5ConvertTest, MatrixWithEmptyRows )
{
    using Policy = CSR5_AVX2_Policy<double>;

    // Create a 5x5 matrix with empty row 2 (0-based indexing)
    // Row 0: [1, 2]
    // Row 1: [3]
    // Row 2: []      <- empty
    // Row 3: [4, 5]
    // Row 4: [6]
    std::vector<int> ai = { 0, 2, 3, 3, 5, 6 }; // Note: ai[2] == ai[3]
    std::vector<int> aj = { 0, 1, 0, 0, 1, 0 };
    std::vector<double> av = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };

    int num_rows = 5;

    CSR5Data<int, int, double, Policy> csr5_data;

    convertCSRtoCSR5<int, int, double, Policy>( num_rows, ai.data(), aj.data(), av.data(), csr5_data );

    EXPECT_EQ( csr5_data._num_tiles, 1 );
    EXPECT_EQ( csr5_data._nnz, 6 );

    // Check that empty row flag is set (MSB = 1)
    auto has_empty_rows = CSR5Data<int, int, double, Policy>::hasEmptyRows( csr5_data._tile_ptr[0] );
    EXPECT_TRUE( has_empty_rows );
    auto start_row = CSR5Data<int, int, double, Policy>::getTileStartRow( csr5_data._tile_ptr[0] );
    EXPECT_EQ( start_row, 0 );
}
