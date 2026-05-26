#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <list>
#include <unordered_map>
#include <vector>

namespace matrix_utils
{

/// @brief Compute cache-aware workload model for SpMV (CAMLB-SpMV approach)
/// @reference "CAMLB-SpMV: A Cache-Aware Memory Load Balance Strategy for SpMV on Many-Core Architectures"
///            Xin He, Miao Wang, Haipeng Jia, Yunquan Zhang
///            IEEE Transactions on Parallel and Distributed Systems (TPDS), 2019
///            DOI: 10.1109/TPDS.2018.2878777
/// @details Models memory access costs for y = A*x in CSR format by simulating:
///          - Streaming access for ai (row_ptr), aj (col_ind), ax (values), y (output)
///          - LRU cache simulation for x (input vector) with limited capacity
/// @tparam ROWTYPE Integer type for row pointers (e.g., int, int64_t)
/// @tparam COLTYPE Integer type for column indices (e.g., int, int64_t)
/// @tparam VALTYPE Value type for matrix elements and vectors (e.g., double, float)
/// @param nrows Number of rows in the sparse matrix
/// @param row_ptr CSR row pointer array (size nrows+1) - REQUIRED
/// @param col_ind CSR column index array (size nnz) - REQUIRED
/// @param val CSR value array (size nnz) - OPTIONAL (nullptr = assume base address 0 for estimation)
/// @param xvec Input vector x - OPTIONAL (nullptr = assume base address 0 for estimation)
/// @param yvec Output vector y - OPTIONAL (nullptr = assume base address 0 for estimation)
/// @param cache_line_bytes Size of cache line in bytes (typically 64)
/// @param swindow_lines LRU cache capacity for x-vector in cache lines (e.g., L1 size / cache_line_bytes)
/// @param prefix Output: prefix sum of cache line loads per element (size nnz+1)
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void compute_element_workload_prefix_hw( COLTYPE nrows,
                                         const ROWTYPE* row_ptr,
                                         const COLTYPE* col_ind,
                                         const VALTYPE* val,
                                         const VALTYPE* xvec,
                                         const VALTYPE* yvec,
                                         std::size_t cache_line_bytes,
                                         std::size_t swindow_lines,
                                         std::size_t* prefix )
{
    // Handle base indexing (0-based or 1-based)
    const ROWTYPE base = row_ptr[0];
    const std::size_t nnz = static_cast<std::size_t>( row_ptr[nrows] - base );

    // Initialize prefix array with zeros
    std::memset( prefix, 0, ( nnz + 1 ) * sizeof( std::size_t ) );

    // Early exit for empty or invalid inputs
    if ( nrows <= 0 || cache_line_bytes == 0 || nnz == 0 )
        return;

    // Helper lambda: compute cache line ID from memory address
    auto line_id_of = [cache_line_bytes]( std::uintptr_t addr ) -> std::size_t
    { return static_cast<std::size_t>( addr / cache_line_bytes ); };

    // Base addresses for each array (use 0 if array is nullptr for estimation purposes)
    const std::uintptr_t base_ai = reinterpret_cast<std::uintptr_t>( row_ptr );
    const std::uintptr_t base_aj = reinterpret_cast<std::uintptr_t>( col_ind );
    const std::uintptr_t base_ax = val ? reinterpret_cast<std::uintptr_t>( val ) : 0;
    const std::uintptr_t base_x = xvec ? reinterpret_cast<std::uintptr_t>( xvec ) : 0;
    const std::uintptr_t base_y = yvec ? reinterpret_cast<std::uintptr_t>( yvec ) : 0;

    // Element sizes
    const std::size_t idx_bytes = sizeof( ROWTYPE );
    const std::size_t col_bytes = sizeof( COLTYPE );
    const std::size_t val_bytes = sizeof( VALTYPE );

    // Track streaming behavior for ai, aj, ax, y arrays
    // These benefit from spatial locality in sequential access
    bool have_ai = false, have_aj = false, have_ax = false, have_y = false;
    std::size_t last_ai = 0, last_aj = 0, last_ax = 0, last_y = 0;

    // LRU cache simulation for x-vector (random access pattern)
    std::list<std::size_t> lru;
    std::unordered_map<std::size_t, typename std::list<std::size_t>::iterator> lru_map;
    if ( swindow_lines > 0 )
    {
        lru_map.reserve( swindow_lines * 2 );
    }

    // LRU touch function: returns true if cache hit, false if cache miss
    auto lru_touch_x = [&]( std::size_t line_id ) -> bool
    {
        auto it = lru_map.find( line_id );
        if ( it != lru_map.end() )
        {
            // Cache hit: move to front (most recently used)
            lru.splice( lru.begin(), lru, it->second );
            it->second = lru.begin();
            return true;
        }

        // Cache miss: add to front
        lru.push_front( line_id );
        lru_map[line_id] = lru.begin();

        // Evict least recently used if capacity exceeded
        if ( swindow_lines > 0 && lru.size() > swindow_lines )
        {
            auto last = std::prev( lru.end() );
            lru_map.erase( *last );
            lru.pop_back();
        }
        return false;
    };

    std::size_t k = 0; // Current element index in flattened arrays

    // Process each row
    for ( COLTYPE r = 0; r < nrows; ++r )
    {
        const ROWTYPE start = row_ptr[r] - base;
        const ROWTYPE end = row_ptr[r + 1] - base;

        if ( start >= end )
        {
            continue; // Skip empty rows
        }

        // ===================================================================
        // Compute row header cost (ai + y) - assigned to first element of row
        // ===================================================================
        std::size_t header_cost = 0;

        // Cost for accessing row_ptr[r] and row_ptr[r+1]
        {
            const std::uintptr_t addr_r = base_ai + static_cast<std::uintptr_t>( r ) * idx_bytes;
            const std::uintptr_t addr_r1 = base_ai + static_cast<std::uintptr_t>( r + 1 ) * idx_bytes;

            const std::size_t line_r = line_id_of( addr_r );
            const std::size_t line_r1 = line_id_of( addr_r1 );

            // Check if row_ptr[r] is on a new cache line
            if ( !have_ai || line_r != last_ai )
            {
                ++header_cost;
                last_ai = line_r;
                have_ai = true;
            }

            // Check if row_ptr[r+1] is on a new cache line
            if ( line_r1 != last_ai )
            {
                ++header_cost;
                last_ai = line_r1;
            }
        }

        // Cost for accessing y[r] (output vector)
        // Always compute cost even if yvec is nullptr (assume address 0)
        {
            const std::uintptr_t addr_y = base_y + static_cast<std::uintptr_t>( r ) * val_bytes;
            const std::size_t line_y = line_id_of( addr_y );

            if ( !have_y || line_y != last_y )
            {
                ++header_cost;
                last_y = line_y;
                have_y = true;
            }
        }

        // ===================================================================
        // Process elements in this row
        // ===================================================================
        for ( ROWTYPE p = start; p < end; ++p, ++k )
        {
            std::size_t cost = 0;

            // First element of row inherits row header cost
            if ( p == start )
            {
                cost += header_cost;
            }

            // Cost for accessing col_ind[p]
            {
                const std::uintptr_t addr_aj = base_aj + static_cast<std::uintptr_t>( p ) * col_bytes;
                const std::size_t line_aj = line_id_of( addr_aj );

                if ( !have_aj || line_aj != last_aj )
                {
                    ++cost;
                    last_aj = line_aj;
                    have_aj = true;
                }
            }

            // Cost for accessing val[p] (matrix values)
            // Always compute cost even if val is nullptr (assume address 0)
            {
                const std::uintptr_t addr_ax = base_ax + static_cast<std::uintptr_t>( p ) * val_bytes;
                const std::size_t line_ax = line_id_of( addr_ax );

                if ( !have_ax || line_ax != last_ax )
                {
                    ++cost;
                    last_ax = line_ax;
                    have_ax = true;
                }
            }

            // Cost for accessing x[col_ind[p]] (input vector) - LRU simulation
            // Always compute cost even if xvec is nullptr (assume address 0)
            {
                const COLTYPE col = col_ind[p]; // Column index (may be 0-based or 1-based)
                // For x-vector access, we need to account for potential 1-based indexing
                const COLTYPE col_offset = ( base == 0 ) ? col : ( col - base );
                const std::uintptr_t addr_x = base_x + static_cast<std::uintptr_t>( col_offset ) * val_bytes;
                const std::size_t line_x = line_id_of( addr_x );

                const bool hit = lru_touch_x( line_x );
                if ( !hit )
                {
                    ++cost; // Cache miss
                }
            }

            // Update prefix sum with accumulated cost
            prefix[k + 1] = prefix[k] + cost;
        }
    }
}

} // namespace matrix_utils