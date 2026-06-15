#pragma once

#include "csr5_format.hpp"
#include <cstdint>

namespace matrix_utils
{

/**
 * @brief Pack CSR5 tile metadata into uint64_t descriptor
 *
 * Packing layout (variable based on OMEGA):
 *   bits [0:OMEGA-1]:        bit_flag (OMEGA bits for row boundaries)
 *   bits [OMEGA:OMEGA+15]:   seg_offset (16 bits, 0-65535)
 *   bits [OMEGA+16:63]:      y_offset (remaining bits for row index)
 *
 * Example for OMEGA=4:
 *   bits [0:3]:   bit_flag (4 bits)
 *   bits [4:19]:  seg_offset (16 bits)
 *   bits [20:63]: y_offset (44 bits, supports up to 2^44 rows)
 *
 * @param bit_flag OMEGA-bit field indicating row boundaries in tile
 * @param y_offset Starting matrix row index for this tile
 * @param seg_offset Segment counter for multi-tile rows
 * @param omega Tile height (number of SIMD lanes)
 * @return Packed 64-bit descriptor
 */
uint64_t packCSR5TileDesc( uint32_t bit_flag, uint32_t y_offset, uint16_t seg_offset, int omega );

/**
 * @brief Unpack CSR5 tile metadata from uint64_t descriptor
 *
 * @param desc Packed 64-bit descriptor
 * @param[out] bit_flag OMEGA-bit field indicating row boundaries
 * @param[out] y_offset Starting matrix row index
 * @param[out] seg_offset Segment counter
 * @param omega Tile height (number of SIMD lanes)
 */
void unpackCSR5TileDesc( uint64_t desc, uint32_t& bit_flag, uint32_t& y_offset, uint16_t& seg_offset, int omega );

/**
 * @brief Convert CSR matrix to CSR5 format
 *
 * Conversion process:
 * 1. Partition CSR elements into tiles of size OMEGA × SIGMA
 * 2. For each tile, transpose from row-major CSR to column-major tile layout
 * 3. Generate bit-flags encoding row boundaries within each tile
 * 4. Compute y_offset (starting row) and seg_offset (continuation counter)
 * 5. Pack metadata and store in CSR5Data structure
 *
 * Column-major mapping:
 *   CSR element at position k maps to tile position:
 *     lane = k % OMEGA
 *     col = k / OMEGA
 *     storage_index = col * OMEGA + lane
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
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, typename Policy>
void convertCSRtoCSR5( COLTYPE num_rows,
                       const ROWTYPE* ai,
                       const COLTYPE* aj,
                       const VALTYPE* av,
                       CSR5Data<ROWTYPE, COLTYPE, VALTYPE, Policy>& csr5_data );

// Template implementations are in csr5_convert.cpp with explicit instantiations

} // namespace matrix_utils
