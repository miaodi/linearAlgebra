#pragma once

#include "csr5_policy.hpp"
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <type_traits>

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
 * Metadata per full tile:
 * - one 32-bit descriptor per lane
 * - descriptor layout: [y_offset][seg_offset][SIGMA row-start bits][unused]
 * - empty rows are intentionally unsupported in this first version
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
    static_assert( Policy::TILE_SIZE == Policy::OMEGA * Policy::SIGMA, "Invalid CSR5 tile size" );
    static_assert( Policy::DESCRIPTOR_BITS <= 32,
                   "CSR5Data stores one uint32_t descriptor packet per lane" );

    // Matrix dimensions
    COLTYPE _num_rows;
    ROWTYPE _nnz;
    ROWTYPE _num_tiles;
    ROWTYPE _num_full_tiles;

    // Original CSR input base. CSR5-owned row pointers, column indices, and
    // tile pointers are normalized to 0-based indexing during preprocess.
    ROWTYPE _base;

    // Normalized 0-based row pointer copy. The first CSR5 version does not
    // support empty rows, so this array is strictly increasing.
    std::vector<ROWTYPE> _row_ptr;

    // tile_ptr[t] stores the row containing nonzero boundary t * TILE_SIZE.
    // Empty-row high-bit encoding is intentionally not implemented in v1.
    std::vector<COLTYPE> _tile_ptr;

    // Matrix data owned by CSR5. Full tiles are stored in AoSoA order:
    // old local index = lane * SIGMA + j, new local index = j * OMEGA + lane.
    // The final incomplete tail, if present, remains in CSR order.
    std::vector<COLTYPE> _tile_col_idx;
    std::vector<VALTYPE> _tile_val;

    // Packed descriptor per full tile and per lane. Layout is [tile][lane].
    // Each uint32_t stores [y_offset][seg_offset][SIGMA bit flags][unused].
    std::vector<uint32_t> _tile_desc;

    // Tail tile information
    // If nnz % TILE_SIZE != 0, the last tile is incomplete
    // _tail_tile_length stores the number of valid elements in the last tile
    int _tail_tile_length;

    CSR5Data()
        : _num_rows( 0 ), _nnz( 0 ), _num_tiles( 0 ), _num_full_tiles( 0 ), _base( 0 ), _tail_tile_length( 0 )
    {
    }

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
     * @return Row index containing the tile boundary
     */
    COLTYPE getTilePtr( ROWTYPE tile_idx ) const
    {
        assert( tile_idx >= 0 && tile_idx <= _num_tiles );
        return _tile_ptr[tile_idx];
    }

    /**
     * @brief Get packed metadata descriptor for a specific tile
     * @param tile_idx Tile index (0-based)
     * @return Packed lane descriptor containing bit flags, y_offset, and seg_offset
     */
    uint32_t getTileDesc( ROWTYPE tile_idx, int lane ) const
    {
        assert( tile_idx >= 0 && tile_idx < _num_full_tiles );
        assert( lane >= 0 && lane < Policy::OMEGA );
        return _tile_desc[tile_idx * Policy::OMEGA + lane];
    }

    /**
     * @brief Unpack tile metadata descriptor
     * @param tile_idx Tile index (0-based)
     * @param[out] bit_flags SIGMA-bit field indicating row starts in this lane
     * @param[out] y_offset Compact row-output offset used by CSR5 SpMV
     * @param[out] seg_offset Cross-lane segmented-sum offset
     */
    void unpackTileDesc( ROWTYPE tile_idx, int lane, uint32_t& bit_flags, uint32_t& y_offset, uint32_t& seg_offset ) const;

    /**
     * @brief Estimate memory required for CSR5 conversion
     * @param nnz Number of non-zeros in original CSR matrix
     * @return Estimated memory in bytes
     */
    static size_t estimateMemoryBytes( COLTYPE num_rows, ROWTYPE nnz )
    {
        const ROWTYPE num_tiles = ( nnz + Policy::TILE_SIZE - 1 ) / Policy::TILE_SIZE;
        const ROWTYPE num_full_tiles = nnz / Policy::TILE_SIZE;

        size_t row_ptr_size = static_cast<size_t>( num_rows + 1 ) * sizeof( ROWTYPE );
        size_t tile_ptr_size = static_cast<size_t>( num_tiles + 1 ) * sizeof( COLTYPE );

        size_t data_size = static_cast<size_t>( nnz ) * ( sizeof( COLTYPE ) + sizeof( VALTYPE ) );

        size_t desc_size = static_cast<size_t>( num_full_tiles ) * Policy::OMEGA * sizeof( uint32_t );

        return row_ptr_size + tile_ptr_size + data_size + desc_size;
    }

    /**
     * @brief Get number of tiles
     */
    ROWTYPE numTiles() const { return _num_tiles; }

    ROWTYPE numFullTiles() const { return _num_full_tiles; }

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
