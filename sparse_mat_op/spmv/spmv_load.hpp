#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <unordered_set>
#include <vector>

namespace matrix_utils
{

/// @brief Compute cache-aware workload model for SpMV (CAMLB-SpMV approach)
/// @reference "CAMLB-SpMV: An Efficient Cache-Aware Memory Load-Balancing SpMV on CPU"
///            Jihu Guo, Rui Xia, Jie Liu, Xiaoxiong Zhu, Xiang Zhang
///            ICPP 2024, DOI: 10.1145/3673038.3673042
/// @details Models memory access costs for y = A*x in CSR format by simulating:
///          - Streaming access for ai (row_ptr), aj (col_ind), ax (values), y (output)
///          - FIFO sliding-window cache-line history for x (input vector)
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
/// @param swindow_lines FIFO sliding-window capacity for x-vector cache lines
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

    // FIFO sliding window for x-vector cache-line accesses, matching CAMLB Algorithm 2.
    std::deque<std::size_t> swindow;
    std::unordered_set<std::size_t> swindow_set;
    if ( swindow_lines > 0 )
    {
        swindow_set.reserve( swindow_lines * 2 );
    }

    // Returns true for a recorded cache-line hit. Hits do not refresh FIFO order.
    auto swindow_touch_x = [&]( std::size_t line_id ) -> bool
    {
        if ( swindow_set.find( line_id ) != swindow_set.end() )
        {
            return true;
        }

        if ( swindow_lines == 0 )
        {
            return false;
        }

        if ( swindow.size() == swindow_lines )
        {
            swindow_set.erase( swindow.front() );
            swindow.pop_front();
        }
        swindow.push_back( line_id );
        swindow_set.insert( line_id );
        return false;
    };

    std::size_t cache_lines = 0;

    // Process each row
    for ( COLTYPE r = 0; r < nrows; ++r )
    {
        const ROWTYPE start = row_ptr[r] - base;
        const ROWTYPE end = row_ptr[r + 1] - base;

        // Cost for accessing row_ptr[r] and row_ptr[r+1]
        {
            const std::uintptr_t addr_r = base_ai + static_cast<std::uintptr_t>( r ) * idx_bytes;
            const std::uintptr_t addr_r1 = base_ai + static_cast<std::uintptr_t>( r + 1 ) * idx_bytes;

            const std::size_t line_r = line_id_of( addr_r );
            const std::size_t line_r1 = line_id_of( addr_r1 );

            // Check if row_ptr[r] is on a new cache line
            if ( !have_ai || line_r != last_ai )
            {
                ++cache_lines;
                last_ai = line_r;
                have_ai = true;
            }

            // Check if row_ptr[r+1] is on a new cache line
            if ( line_r1 != last_ai )
            {
                ++cache_lines;
                last_ai = line_r1;
            }
        }

        if ( start >= end )
        {
            continue; // Empty-row Ap cost is carried into the next nonzero workload entry.
        }

        // ===================================================================
        // Process elements in this row
        // ===================================================================
        for ( ROWTYPE p = start; p < end; ++p )
        {
            // Cost for accessing col_ind[p]
            {
                const std::uintptr_t addr_aj = base_aj + static_cast<std::uintptr_t>( p ) * col_bytes;
                const std::size_t line_aj = line_id_of( addr_aj );

                if ( !have_aj || line_aj != last_aj )
                {
                    ++cache_lines;
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
                    ++cache_lines;
                    last_ax = line_ax;
                    have_ax = true;
                }
            }

            // Cost for accessing x[col_ind[p]] (input vector) - FIFO sliding window
            // Always compute cost even if xvec is nullptr (assume address 0)
            {
                const COLTYPE col = col_ind[p]; // Column index (may be 0-based or 1-based)
                // For x-vector access, we need to account for potential 1-based indexing
                const COLTYPE col_offset = ( base == 0 ) ? col : ( col - base );
                const std::uintptr_t addr_x = base_x + static_cast<std::uintptr_t>( col_offset ) * val_bytes;
                const std::size_t line_x = line_id_of( addr_x );

                const bool hit = swindow_touch_x( line_x );
                if ( !hit )
                {
                    ++cache_lines;
                }
            }

            // Cost for writing y[r], charged when the last nonzero of the row is processed.
            if ( p + 1 == end )
            {
                const std::uintptr_t addr_y = base_y + static_cast<std::uintptr_t>( r ) * val_bytes;
                const std::size_t line_y = line_id_of( addr_y );

                if ( !have_y || line_y != last_y )
                {
                    ++cache_lines;
                    last_y = line_y;
                    have_y = true;
                }
            }

            // workload[j + 1] records the cumulative cache-line load through nonzero j.
            prefix[static_cast<std::size_t>( p ) + 1] = cache_lines;
        }
    }
}

} // namespace matrix_utils
