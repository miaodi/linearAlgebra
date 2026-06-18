#pragma once

#include "csr5_format.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <omp.h>

namespace matrix_utils
{

inline int csr5NormalizeThreadCount( const int num_threads )
{
    return num_threads > 0 ? num_threads : 1;
}

template <typename Policy>
constexpr uint32_t csr5FieldMask( const int bits )
{
    return bits >= 32 ? 0xFFFFFFFFu : ( ( uint32_t( 1 ) << bits ) - 1u );
}

template <typename Policy>
constexpr uint32_t packCSR5LaneDesc( const uint32_t bit_flags, const uint32_t y_offset, const uint32_t seg_offset )
{
    static_assert( is_csr5_policy_v<Policy>, "Policy must satisfy CSR5 policy requirements" );
    static_assert( Policy::DESCRIPTOR_BITS <= 32, "CSR5 lane descriptor must fit in one uint32_t" );

    constexpr int bit_y = Policy::BIT_Y_OFFSET;
    constexpr int bit_seg = Policy::BIT_SEG_OFFSET;
    constexpr int sigma = Policy::SIGMA;
    constexpr int bitflag_shift = 32 - bit_y - bit_seg - sigma;

    uint32_t desc = 0;
    desc |= ( y_offset & csr5FieldMask<Policy>( bit_y ) ) << ( 32 - bit_y );
    desc |= ( seg_offset & csr5FieldMask<Policy>( bit_seg ) ) << ( 32 - bit_y - bit_seg );

    for ( int i = 0; i < sigma; ++i )
    {
        if ( ( bit_flags & ( uint32_t( 1 ) << i ) ) != 0 )
        {
            desc |= uint32_t( 1 ) << ( bitflag_shift + sigma - 1 - i );
        }
    }
    return desc;
}

template <typename Policy>
constexpr void unpackCSR5LaneDesc( const uint32_t desc, uint32_t& bit_flags, uint32_t& y_offset, uint32_t& seg_offset )
{
    static_assert( is_csr5_policy_v<Policy>, "Policy must satisfy CSR5 policy requirements" );

    constexpr int bit_y = Policy::BIT_Y_OFFSET;
    constexpr int bit_seg = Policy::BIT_SEG_OFFSET;
    constexpr int sigma = Policy::SIGMA;
    constexpr int bitflag_shift = 32 - bit_y - bit_seg - sigma;

    y_offset = ( desc >> ( 32 - bit_y ) ) & csr5FieldMask<Policy>( bit_y );
    seg_offset = ( desc >> ( 32 - bit_y - bit_seg ) ) & csr5FieldMask<Policy>( bit_seg );
    bit_flags = 0;
    for ( int i = 0; i < sigma; ++i )
    {
        const uint32_t bit = ( desc >> ( bitflag_shift + sigma - 1 - i ) ) & 1u;
        bit_flags |= bit << i;
    }
}

template <typename Policy>
std::array<uint32_t, Policy::OMEGA> makeCSR5TileBitFlags( const std::vector<typename Policy::ROWTYPE>& row_ptr,
                                                          const std::vector<typename Policy::COLTYPE>& tile_ptr,
                                                          const typename Policy::ROWTYPE tile_idx )
{
    using ROWTYPE = typename Policy::ROWTYPE;
    using COLTYPE = typename Policy::COLTYPE;

    constexpr int SIGMA = Policy::SIGMA;
    constexpr int TILE_SIZE = Policy::TILE_SIZE;

    const ROWTYPE tile_start = tile_idx * TILE_SIZE;
    const ROWTYPE tile_end = tile_start + TILE_SIZE;
    const COLTYPE first_row = tile_ptr[tile_idx];
    const COLTYPE last_candidate =
        std::min<COLTYPE>( tile_ptr[tile_idx + 1], static_cast<COLTYPE>( row_ptr.size() - 2 ) );

    std::array<uint32_t, Policy::OMEGA> lane_bit_flags{};
    for ( COLTYPE row = first_row; row <= last_candidate; ++row )
    {
        const ROWTYPE row_start = row_ptr[row];
        if ( row_start < tile_start || row_start >= tile_end )
        {
            continue;
        }

        const ROWTYPE local = row_start - tile_start;
        const int row_lane = static_cast<int>( local / SIGMA );
        const int row_i = static_cast<int>( local % SIGMA );
        lane_bit_flags[row_lane] |= uint32_t( 1 ) << row_i;
    }
    return lane_bit_flags;
}

template <typename Policy>
std::array<uint32_t, Policy::OMEGA> makeCSR5TileDesc( const std::array<uint32_t, Policy::OMEGA>& lane_bit_flags )
{
    constexpr int OMEGA = Policy::OMEGA;
    constexpr int SIGMA = Policy::SIGMA;

    std::array<int, OMEGA + 1> segment_scan{};
    // Effective segment-head presence used by CSR5 SpMV. Lane 0 is treated as
    // a virtual segment head because the kernel forces the first tile entry to
    // start a tile-local segment. The stored bit flags remain unchanged.
    std::array<bool, OMEGA> effective_present{};

    for ( int l = 0; l < OMEGA; ++l )
    {
        const uint32_t flags = lane_bit_flags[l] & csr5FieldMask<Policy>( SIGMA );
        const bool first_bit = ( flags & 1u ) != 0;
        const int virtual_lane0_head = ( l == 0 && !first_bit ) ? 1 : 0;
        segment_scan[l + 1] = segment_scan[l] + std::popcount( flags ) + virtual_lane0_head;
        effective_present[l] = segment_scan[l + 1] != segment_scan[l];
    }

    std::array<uint32_t, OMEGA> seg_offsets{};
    int next_present_lane = OMEGA;
    for ( int l = OMEGA - 1; l >= 0; --l )
    {
        if ( effective_present[l] )
        {
            seg_offsets[l] = static_cast<uint32_t>( next_present_lane - l - 1 );
            next_present_lane = l;
        }
    }

    std::array<uint32_t, OMEGA> tile_desc{};
    for ( int lane = 0; lane < OMEGA; ++lane )
    {
        const uint32_t y_offset = lane == 0 ? 0u : static_cast<uint32_t>( segment_scan[lane] - 1 );
        tile_desc[lane] = packCSR5LaneDesc<Policy>( lane_bit_flags[lane], y_offset, seg_offsets[lane] );
    }
    return tile_desc;
}

/**
 * @brief Convert CSR matrix to CSR5 format
 *
 * Conversion process:
 * 1. Partition CSR elements into tiles of size OMEGA × SIGMA
 * 2. For each full tile, copy CSR data into AoSoA tile order
 * 3. Generate per-lane bit flags encoding row starts within each tile
 * 4. Compute compact y_offset and cross-lane seg_offset values
 * 5. Pack one 32-bit descriptor per full tile lane
 *
 * Column-major mapping:
 *   Full-tile local CSR element lane * SIGMA + i maps to:
 *     storage_index = i * OMEGA + lane
 *
 * @tparam ROWTYPE Integer type for row pointers
 * @tparam COLTYPE Integer type for column indices
 * @tparam VALTYPE Value type (double, float)
 * @tparam Policy CSR5 policy defining OMEGA, SIGMA, TILE_SIZE
 *
 * @param num_rows Number of rows in matrix
 * @param ai Row pointers (size: num_rows + 1)
 * @param aj Column indices (size: nnz)
 * @param av Values (size: nnz)
 * @param[out] csr5_data Output CSR5 data structure
 */
template <typename Policy>
void convertCSRtoCSR5( typename Policy::COLTYPE num_rows,
                       const typename Policy::ROWTYPE* ai,
                       const typename Policy::COLTYPE* aj,
                       const typename Policy::VALTYPE* av,
                       CSR5Data<Policy>& csr5_data,
                       int num_threads = omp_get_max_threads() )
{
    static_assert( is_csr5_policy_v<Policy>, "Policy must satisfy CSR5 policy requirements" );
    static_assert( Policy::DESCRIPTOR_BITS <= 32, "CSR5 preprocess supports only num_packet == 1" );

    using ROWTYPE = typename Policy::ROWTYPE;
    using COLTYPE = typename Policy::COLTYPE;
    using VALTYPE = typename Policy::VALTYPE;

    constexpr int OMEGA = Policy::OMEGA;
    constexpr int SIGMA = Policy::SIGMA;
    constexpr int TILE_SIZE = Policy::TILE_SIZE;

    if ( ai == nullptr )
    {
        throw std::invalid_argument( "convertCSRtoCSR5 requires a non-null CSR row pointer" );
    }
    if constexpr ( std::is_signed_v<COLTYPE> )
    {
        if ( num_rows < 0 )
        {
            throw std::invalid_argument( "convertCSRtoCSR5 requires a non-negative row count" );
        }
    }

    const int threads = csr5NormalizeThreadCount( num_threads );
    const ROWTYPE base = ai[0];
    if ( ai[num_rows] < base )
    {
        throw std::invalid_argument( "convertCSRtoCSR5 requires a monotonic CSR row pointer" );
    }
    const ROWTYPE nnz = ai[num_rows] - base;

    if ( nnz > 0 && ( aj == nullptr || av == nullptr ) )
    {
        throw std::invalid_argument(
            "convertCSRtoCSR5 requires non-null column and value arrays when nnz > 0" );
    }

    csr5_data = CSR5Data<Policy>{};
    csr5_data._num_rows = num_rows;
    csr5_data._nnz = nnz;
    csr5_data._base = base;
    csr5_data._num_full_tiles = nnz / TILE_SIZE;
    csr5_data._tail_tile_length = static_cast<int>( nnz % TILE_SIZE );
    csr5_data._num_tiles = csr5_data._num_full_tiles + ( csr5_data._tail_tile_length ? 1 : 0 );

    csr5_data._row_ptr.resize( static_cast<size_t>( num_rows ) + 1 );
    bool has_empty_rows = false;
    bool invalid_row_ptr = false;
#pragma omp parallel for num_threads( threads ) reduction( || : has_empty_rows, invalid_row_ptr )
    for ( COLTYPE row = 0; row < num_rows; ++row )
    {
        const ROWTYPE row_start = ai[row] - base;
        const ROWTYPE row_end = ai[row + 1] - base;
        csr5_data._row_ptr[row] = row_start;
        invalid_row_ptr = invalid_row_ptr || row_end < row_start;
        has_empty_rows = has_empty_rows || row_end == row_start;
    }
    csr5_data._row_ptr[num_rows] = nnz;

    if ( invalid_row_ptr )
    {
        throw std::invalid_argument( "convertCSRtoCSR5 requires a monotonic CSR row pointer" );
    }
    if ( has_empty_rows )
    {
        throw std::invalid_argument( "CSR5 preprocess v1 does not support empty rows" );
    }

    csr5_data._tile_ptr.resize( static_cast<size_t>( csr5_data._num_tiles ) + 1 );
    csr5_data._tile_col_idx.resize( nnz );
    csr5_data._tile_val.resize( nnz );
    csr5_data._tile_desc.resize( static_cast<size_t>( csr5_data._num_full_tiles ) * OMEGA );

#pragma omp parallel for num_threads( threads )
    for ( ROWTYPE tile = 0; tile <= csr5_data._num_tiles; ++tile )
    {
        ROWTYPE boundary = tile * TILE_SIZE;
        if ( boundary > nnz ) [[unlikely]]
        {
            boundary = nnz;
        }
        const auto row_it = std::upper_bound( csr5_data._row_ptr.begin(), csr5_data._row_ptr.end(), boundary );
        csr5_data._tile_ptr[tile] =
            static_cast<COLTYPE>( std::distance( csr5_data._row_ptr.begin(), row_it ) - 1 );
    }

#pragma omp parallel for num_threads( threads )
    for ( ROWTYPE tile = 0; tile < csr5_data._num_full_tiles; ++tile )
    {
        const ROWTYPE tile_start = tile * TILE_SIZE;
        for ( int lane = 0; lane < OMEGA; ++lane )
        {
            for ( int i = 0; i < SIGMA; ++i )
            {
                const ROWTYPE old_idx = tile_start + lane * SIGMA + i;
                const ROWTYPE new_idx = tile_start + i * OMEGA + lane;
                csr5_data._tile_col_idx[new_idx] = static_cast<COLTYPE>( aj[old_idx] - base );
                csr5_data._tile_val[new_idx] = av[old_idx];
            }
        }

        const auto lane_bit_flags =
            makeCSR5TileBitFlags<Policy>( csr5_data._row_ptr, csr5_data._tile_ptr, tile );
        const bool fast_track = csr5_data._tile_ptr[tile] == csr5_data._tile_ptr[tile + 1];
        const auto tile_desc =
            fast_track ? std::array<uint32_t, OMEGA>{} : makeCSR5TileDesc<Policy>( lane_bit_flags );
        for ( int lane = 0; lane < OMEGA; ++lane )
        {
            // The CSR5 reference skips y_offset/seg_offset generation for
            // fast-track tiles fully contained in one row. The first bit flag is
            // still kept so SpMV can distinguish direct write vs accumulation.
            if ( fast_track )
            {
                csr5_data._tile_desc[tile * OMEGA + lane] =
                    packCSR5LaneDesc<Policy>( lane_bit_flags[lane], 0, 0 );
            }
            else
            {
                csr5_data._tile_desc[tile * OMEGA + lane] = tile_desc[lane];
            }
        }
    }

    const ROWTYPE tail_start = csr5_data._num_full_tiles * TILE_SIZE;
#pragma omp parallel for num_threads( threads )
    for ( ROWTYPE idx = tail_start; idx < nnz; ++idx )
    {
        csr5_data._tile_col_idx[idx] = static_cast<COLTYPE>( aj[idx] - base );
        csr5_data._tile_val[idx] = av[idx];
    }
}

template <typename Policy>
void CSR5Data<Policy>::unpackTileDesc( typename Policy::ROWTYPE tile_idx,
                                       int lane,
                                       uint32_t& bit_flags,
                                       uint32_t& y_offset,
                                       uint32_t& seg_offset ) const
{
    unpackCSR5LaneDesc<Policy>( getTileDesc( tile_idx, lane ), bit_flags, y_offset, seg_offset );
}

} // namespace matrix_utils
