#pragma once

#include "csr5_policy.hpp"
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cassert>

namespace matrix_utils
{

/**
 * @brief CSR5 data container with column-major tile storage
 *
 * CSR5 format organizes sparse matrix data into tiles of size OMEGA × SIGMA.
 * Each tile is stored in column-major order to enable efficient SIMD operations.
 *
 * Tile layout visualization (OMEGA=4, SIGMA=32):
 *
 *   Tile coordinates:  (row_in_tile, col_in_tile)
 *
 *        Col 0  Col 1  Col 2  ...  Col 31
 *   Row 0: [0]    [4]    [8]   ...  [124]
 *   Row 1: [1]    [5]    [9]   ...  [125]
 *   Row 2: [2]    [6]    [10]  ...  [126]
 *   Row 3: [3]    [7]    [11]  ...  [127]
 *
 *   Numbers in brackets [] indicate linear storage index within tile.
 *   Element at (row, col) is stored at index: col * OMEGA + row
 *
 * Metadata per tile:
 * - bit_flag: OMEGA-bit field indicating row boundaries within tile
 *             bit i = 1 if lane i starts a new matrix row
 * - y_offset: Matrix row index where the tile's first lane begins
 * - seg_offset: Segment counter for multi-tile rows (0 for first tile in row)
 *
 * @tparam ROWTYPE Integer type for row pointers (int, int64_t)
 * @tparam COLTYPE Integer type for column indices (int, int64_t)
 * @tparam VALTYPE Value type (double, float)
 * @tparam Policy CSR5 policy defining OMEGA, SIGMA, TILE_SIZE
 */
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, typename Policy>
struct CSR5Data
{
    static_assert( is_csr5_policy_v<Policy>, "Policy must satisfy CSR5 policy requirements" );

    // Matrix dimensions
    COLTYPE _num_rows;
    ROWTYPE _nnz;
    ROWTYPE _num_tiles;

    // Tile row pointer array (analogous to CSR's ai)
    // Size: _num_tiles + 1
    // tile_ptr[i] stores the starting row index of tile i
    // MSB (most significant bit) = 1 if tile i contains empty rows
    // Actual row index is obtained by masking off the MSB
    std::vector<COLTYPE> _tile_ptr;

    // Column-major tile data (aligned for SIMD efficiency)
    // Size: _num_tiles * Policy::TILE_SIZE
    std::vector<COLTYPE> _tile_col_idx;
    std::vector<VALTYPE> _tile_val;

    // Packed metadata per tile
    // Each uint64_t encodes: bit_flag, y_offset, seg_offset
    // Size: _num_tiles
    std::vector<uint64_t> _tile_desc;

    // Tail tile information
    // If nnz % TILE_SIZE != 0, the last tile is incomplete
    // _tail_tile_length stores the number of valid elements in the last tile
    int _tail_tile_length;

    CSR5Data() : _num_rows( 0 ), _nnz( 0 ), _num_tiles( 0 ), _tail_tile_length( 0 ) {}

    /**
     * @brief Get pointer to column indices for a specific tile
     * @param tile_idx Tile index (0-based)
     * @return Pointer to column-major column index array for this tile
     */
    const COLTYPE* getTileColIdx( ROWTYPE tile_idx ) const
    {
        assert( tile_idx >= 0 && tile_idx < _num_tiles );
        return _tile_col_idx.data() + tile_idx * Policy::TILE_SIZE;
    }

    COLTYPE* getTileColIdx( ROWTYPE tile_idx )
    {
        assert( tile_idx >= 0 && tile_idx < _num_tiles );
        return _tile_col_idx.data() + tile_idx * Policy::TILE_SIZE;
    }

    /**
     * @brief Get pointer to values for a specific tile
     * @param tile_idx Tile index (0-based)
     * @return Pointer to column-major value array for this tile
     */
    const VALTYPE* getTileVal( ROWTYPE tile_idx ) const
    {
        assert( tile_idx >= 0 && tile_idx < _num_tiles );
        return _tile_val.data() + tile_idx * Policy::TILE_SIZE;
    }

    VALTYPE* getTileVal( ROWTYPE tile_idx )
    {
        assert( tile_idx >= 0 && tile_idx < _num_tiles );
        return _tile_val.data() + tile_idx * Policy::TILE_SIZE;
    }

    /**
     * @brief Get tile pointer (starting row index with empty row flag)
     * @param tile_idx Tile index (0-based, or _num_tiles for end pointer)
     * @return Packed value: MSB = empty row flag, remaining bits = row index
     */
    COLTYPE getTilePtr( ROWTYPE tile_idx ) const
    {
        assert( tile_idx >= 0 && tile_idx <= _num_tiles );
        return _tile_ptr[tile_idx];
    }

    /**
     * @brief Extract starting row index from tile_ptr value
     * @param tile_ptr_value Packed tile_ptr value
     * @return Starting row index (MSB masked off)
     */
    static COLTYPE getTileStartRow( COLTYPE tile_ptr_value )
    {
        // Mask off the MSB to get actual row index
        constexpr COLTYPE MSB_MASK = COLTYPE( 1 ) << ( sizeof( COLTYPE ) * 8 - 1 );
        return tile_ptr_value & ~MSB_MASK;
    }

    /**
     * @brief Check if tile contains empty rows
     * @param tile_ptr_value Packed tile_ptr value
     * @return true if tile has empty rows (MSB = 1)
     */
    static bool hasEmptyRows( COLTYPE tile_ptr_value )
    {
        constexpr COLTYPE MSB_MASK = COLTYPE( 1 ) << ( sizeof( COLTYPE ) * 8 - 1 );
        return ( tile_ptr_value & MSB_MASK ) != 0;
    }

    /**
     * @brief Get packed metadata descriptor for a specific tile
     * @param tile_idx Tile index (0-based)
     * @return Packed descriptor containing bit_flag, y_offset, seg_offset
     */
    uint64_t getTileDesc( ROWTYPE tile_idx ) const
    {
        assert( tile_idx >= 0 && tile_idx < _num_tiles );
        return _tile_desc[tile_idx];
    }

    /**
     * @brief Unpack tile metadata descriptor
     * @param tile_idx Tile index (0-based)
     * @param[out] bit_flag OMEGA-bit field indicating row boundaries
     * @param[out] y_offset Starting matrix row index for this tile
     * @param[out] seg_offset Segment counter (0 for first tile in row)
     */
    void unpackTileDesc( ROWTYPE tile_idx, uint32_t& bit_flag, COLTYPE& y_offset, uint16_t& seg_offset ) const;

    /**
     * @brief Estimate memory required for CSR5 conversion
     * @param nnz Number of non-zeros in original CSR matrix
     * @return Estimated memory in bytes
     */
    static size_t estimateMemoryBytes( ROWTYPE nnz )
    {
        const ROWTYPE num_tiles = ( nnz + Policy::TILE_SIZE - 1 ) / Policy::TILE_SIZE;

        // Tile row pointers (num_tiles + 1)
        size_t ptr_size = static_cast<size_t>( num_tiles + 1 ) * sizeof( COLTYPE );

        // Column indices + values (both TILE_SIZE per tile)
        size_t data_size = static_cast<size_t>( num_tiles ) * Policy::TILE_SIZE *
                           ( sizeof( COLTYPE ) + sizeof( VALTYPE ) );

        // Packed descriptors (one uint64_t per tile)
        size_t desc_size = static_cast<size_t>( num_tiles ) * sizeof( uint64_t );

        return ptr_size + data_size + desc_size;
    }

    /**
     * @brief Get number of tiles
     */
    ROWTYPE numTiles() const { return _num_tiles; }

    /**
     * @brief Get tile size
     */
    static constexpr int tileSize() { return Policy::TILE_SIZE; }

    /**
     * @brief Get OMEGA (tile height / SIMD lanes)
     */
    static constexpr int omega() { return Policy::OMEGA; }

    /**
     * @brief Get SIGMA (tile width)
     */
    static constexpr int sigma() { return Policy::SIGMA; }
};

} // namespace matrix_utils
