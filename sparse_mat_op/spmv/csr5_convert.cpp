#include "csr5_convert.hpp"
#include <omp.h>
#include <algorithm>
#include <cstring>

namespace matrix_utils
{

uint64_t packCSR5TileDesc( uint32_t bit_flag, uint32_t y_offset, uint16_t seg_offset, int omega )
{
    // Packing layout:
    // bits [0:omega-1]:        bit_flag
    // bits [omega:omega+15]:   seg_offset (16 bits)
    // bits [omega+16:63]:      y_offset

    uint64_t desc = 0;

    // Pack bit_flag in lower omega bits
    uint32_t bit_flag_mask = ( 1u << omega ) - 1;
    desc |= ( bit_flag & bit_flag_mask );

    // Pack seg_offset in next 16 bits
    desc |= ( static_cast<uint64_t>( seg_offset ) << omega );

    // Pack y_offset in remaining upper bits
    desc |= ( static_cast<uint64_t>( y_offset ) << ( omega + 16 ) );

    return desc;
}

void unpackCSR5TileDesc( uint64_t desc, uint32_t& bit_flag, uint32_t& y_offset, uint16_t& seg_offset, int omega )
{
    // Extract bit_flag from lower omega bits
    uint32_t bit_flag_mask = ( 1u << omega ) - 1;
    bit_flag = static_cast<uint32_t>( desc & bit_flag_mask );

    // Extract seg_offset from next 16 bits
    seg_offset = static_cast<uint16_t>( ( desc >> omega ) & 0xFFFF );

    // Extract y_offset from remaining upper bits
    y_offset = static_cast<uint32_t>( desc >> ( omega + 16 ) );
}

// Helper: Find which row contains a given CSR element index
template <typename ROWTYPE, typename COLTYPE>
static COLTYPE findRowForElement( COLTYPE num_rows, const ROWTYPE* ai, ROWTYPE base, ROWTYPE elem_idx )
{
    COLTYPE row_lower = 0;
    COLTYPE row_upper = num_rows;
    while ( row_lower < row_upper )
    {
        COLTYPE row_mid = row_lower + ( row_upper - row_lower ) / 2;
        ROWTYPE row_start = ai[row_mid] - base;
        if ( row_start <= elem_idx )
        {
            row_lower = row_mid + 1;
        }
        else
        {
            row_upper = row_mid;
        }
    }
    return ( row_lower > 0 ) ? row_lower - 1 : 0;
}

// Helper: Check if a tile contains empty rows
template <typename ROWTYPE, typename COLTYPE>
static bool detectEmptyRows( const ROWTYPE* ai, COLTYPE start_row, COLTYPE end_row )
{
    for ( COLTYPE r = start_row; r < end_row; ++r )
    {
        if ( ai[r] == ai[r + 1] )
        {
            return true;
        }
    }
    return false;
}

// Helper: Process a single tile - transpose to column-major and generate metadata
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, int OMEGA>
static void processTile( const ROWTYPE* ai,
                         const COLTYPE* aj,
                         const VALTYPE* av,
                         ROWTYPE base,
                         COLTYPE num_rows,
                         ROWTYPE csr_start,
                         int tile_length,
                         COLTYPE* tile_col,
                         VALTYPE* tile_val,
                         uint32_t& bit_flag,
                         COLTYPE& current_row )
{
    bit_flag = 0;

    for ( int k = 0; k < tile_length; ++k )
    {
        const ROWTYPE csr_idx = csr_start + k;

        // Find which row this element belongs to
        while ( current_row < num_rows && ai[current_row + 1] - base <= csr_idx )
        {
            current_row++;
        }

        // Map to column-major position within tile
        const int lane = k % OMEGA; // Which SIMD lane (row in tile)
        const int col = k / OMEGA;  // Which column in tile
        const int storage_idx = col * OMEGA + lane;

        // Copy data in column-major order (convert to 0-based indexing)
        tile_col[storage_idx] = aj[csr_idx] - base;
        tile_val[storage_idx] = av[csr_idx];

        // Set bit-flag if this lane starts a new row
        if ( csr_idx == ai[current_row] - base )
        {
            bit_flag |= ( 1u << lane );
        }
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, typename Policy>
void convertCSRtoCSR5( COLTYPE num_rows,
                       const ROWTYPE* ai,
                       const COLTYPE* aj,
                       const VALTYPE* av,
                       CSR5Data<ROWTYPE, COLTYPE, VALTYPE, Policy>& csr5_data )
{
    constexpr int OMEGA = Policy::OMEGA;
    constexpr int SIGMA = Policy::SIGMA;
    constexpr int TILE_SIZE = Policy::TILE_SIZE;

    // Extract matrix info and detect index base
    const ROWTYPE base = ai[0];
    const ROWTYPE nnz = ai[num_rows] - base;

    if ( nnz == 0 )
    {
        csr5_data._num_rows = num_rows;
        csr5_data._nnz = 0;
        csr5_data._num_tiles = 0;
        csr5_data._tail_tile_length = 0;
        return;
    }

    // Phase 1: Calculate number of tiles and allocate memory
    const ROWTYPE num_tiles = ( nnz + TILE_SIZE - 1 ) / TILE_SIZE;
    const int tail_length = ( nnz % TILE_SIZE == 0 ) ? TILE_SIZE : ( nnz % TILE_SIZE );

    csr5_data._num_rows = num_rows;
    csr5_data._nnz = nnz;
    csr5_data._num_tiles = num_tiles;
    csr5_data._tail_tile_length = tail_length;

    // Allocate tile pointer array (size: num_tiles + 1)
    csr5_data._tile_ptr.resize( num_tiles + 1, 0 );

    // Allocate column-major tile arrays (with padding for tail tile)
    csr5_data._tile_col_idx.resize( num_tiles * TILE_SIZE, 0 );
    csr5_data._tile_val.resize( num_tiles * TILE_SIZE, 0 );
    csr5_data._tile_desc.resize( num_tiles, 0 );

// Phase 2: Convert each tile in parallel
#pragma omp parallel for schedule( dynamic, 1 )
    for ( ROWTYPE tile_idx = 0; tile_idx < num_tiles; ++tile_idx )
    {
        const ROWTYPE csr_start = tile_idx * TILE_SIZE;
        const ROWTYPE csr_end = std::min( csr_start + TILE_SIZE, nnz );
        const int tile_length = static_cast<int>( csr_end - csr_start );

        COLTYPE* tile_col = csr5_data.getTileColIdx( tile_idx );
        VALTYPE* tile_val = csr5_data.getTileVal( tile_idx );

        // Find the starting row for this tile
        COLTYPE current_row = findRowForElement( num_rows, ai, base, csr_start );
        COLTYPE y_offset = current_row;

        // Populate tile_ptr[tile_idx] with starting row
        csr5_data._tile_ptr[tile_idx] = y_offset;

        // Determine seg_offset (count how many tiles came before in same row)
        uint16_t seg_offset = 0;
        ROWTYPE row_start = ai[current_row] - base;
        if ( csr_start > row_start )
        {
            seg_offset = static_cast<uint16_t>( ( csr_start - row_start ) / TILE_SIZE );
        }

        // Transpose CSR elements to column-major tile layout and generate bit-flags
        uint32_t bit_flag = 0;
        processTile<ROWTYPE, COLTYPE, VALTYPE, OMEGA>( ai, aj, av, base, num_rows, csr_start, tile_length,
                                                       tile_col, tile_val, bit_flag, current_row );

        // Pack metadata
        csr5_data._tile_desc[tile_idx] = packCSR5TileDesc( bit_flag, y_offset, seg_offset, OMEGA );

        // Detect and mark empty rows
        COLTYPE last_row_in_tile = current_row;
        bool has_empty_rows = detectEmptyRows( ai, y_offset, last_row_in_tile );

        if ( has_empty_rows )
        {
            constexpr COLTYPE MSB_MASK = COLTYPE( 1 ) << ( sizeof( COLTYPE ) * 8 - 1 );
            csr5_data._tile_ptr[tile_idx] |= MSB_MASK;
        }
    }

    // Set final tile_ptr entry (points to row after last tile)
    COLTYPE final_row = findRowForElement( num_rows, ai, base, nnz - 1 );
    csr5_data._tile_ptr[num_tiles] = final_row + 1;
}

// Explicit template instantiations for CSR5Data::unpackTileDesc
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, typename Policy>
void CSR5Data<ROWTYPE, COLTYPE, VALTYPE, Policy>::unpackTileDesc( ROWTYPE tile_idx,
                                                                  uint32_t& bit_flag,
                                                                  COLTYPE& y_offset,
                                                                  uint16_t& seg_offset ) const
{
    uint32_t y_off_u32;
    unpackCSR5TileDesc( _tile_desc[tile_idx], bit_flag, y_off_u32, seg_offset, Policy::OMEGA );
    y_offset = static_cast<COLTYPE>( y_off_u32 );
}

// Explicit template instantiations for commonly used types

// int, int, double with AVX2
template void convertCSRtoCSR5<int, int, double, CSR5_AVX2_Policy<double>>(
    int,
    const int*,
    const int*,
    const double*,
    CSR5Data<int, int, double, CSR5_AVX2_Policy<double>>& );
template void CSR5Data<int, int, double, CSR5_AVX2_Policy<double>>::unpackTileDesc( int, uint32_t&, int&, uint16_t& ) const;

// int, int, float with AVX2
template void convertCSRtoCSR5<int, int, float, CSR5_AVX2_Policy<float>>(
    int,
    const int*,
    const int*,
    const float*,
    CSR5Data<int, int, float, CSR5_AVX2_Policy<float>>& );
template void CSR5Data<int, int, float, CSR5_AVX2_Policy<float>>::unpackTileDesc( int, uint32_t&, int&, uint16_t& ) const;

// int64_t, int64_t, double with AVX2
template void convertCSRtoCSR5<int64_t, int64_t, double, CSR5_AVX2_Policy<double>>(
    int64_t,
    const int64_t*,
    const int64_t*,
    const double*,
    CSR5Data<int64_t, int64_t, double, CSR5_AVX2_Policy<double>>& );
template void CSR5Data<int64_t, int64_t, double, CSR5_AVX2_Policy<double>>::unpackTileDesc( int64_t,
                                                                                            uint32_t&,
                                                                                            int64_t&,
                                                                                            uint16_t& ) const;

// int64_t, int64_t, float with AVX2
template void convertCSRtoCSR5<int64_t, int64_t, float, CSR5_AVX2_Policy<float>>(
    int64_t,
    const int64_t*,
    const int64_t*,
    const float*,
    CSR5Data<int64_t, int64_t, float, CSR5_AVX2_Policy<float>>& );
template void CSR5Data<int64_t, int64_t, float, CSR5_AVX2_Policy<float>>::unpackTileDesc( int64_t,
                                                                                          uint32_t&,
                                                                                          int64_t&,
                                                                                          uint16_t& ) const;

} // namespace matrix_utils
