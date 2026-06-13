#include "MaximumMatching.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace reordering
{
template <typename ROWTYPE, typename COLTYPE>
COLTYPE MaximumMatching( const COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj, COLTYPE* matching_row, COLTYPE* matching_col )
{
    const COLTYPE INVALID = std::numeric_limits<COLTYPE>::max();
    const ROWTYPE base = ai[0];

    std::fill_n( matching_row, rows, INVALID );
    std::fill_n( matching_col, rows, INVALID );

    COLTYPE match_count = 0;

    std::vector<COLTYPE> column_seen( rows, 0 );
    std::vector<COLTYPE> parent( rows, INVALID );
    std::vector<COLTYPE> queue;
    queue.reserve( rows );
    COLTYPE search_token = 0;

    auto next_search_token = [&]()
    {
        if ( search_token == std::numeric_limits<COLTYPE>::max() )
        {
            std::fill( column_seen.begin(), column_seen.end(), 0 );
            search_token = 0;
        }
        ++search_token;
        return search_token;
    };

    auto augment_path = [&]( COLTYPE col )
    {
        while ( true )
        {
            const COLTYPE row = parent[col];
            const COLTYPE next_col = matching_row[row];
            matching_row[row] = col + base;
            matching_col[col] = row + base;
            if ( next_col == INVALID )
            {
                break;
            }
            col = next_col - base;
        }
    };

    auto find_augmenting_path = [&]( const COLTYPE root )
    {
        const COLTYPE token = next_search_token();
        queue.clear();
        queue.push_back( root );

        for ( std::size_t head = 0; head < queue.size(); ++head )
        {
            const COLTYPE row = queue[head];
            for ( ROWTYPE i = ai[row] - base; i < ai[row + 1] - base; ++i )
            {
                const COLTYPE col = aj[i] - base;
                if ( column_seen[col] == token )
                {
                    continue;
                }

                column_seen[col] = token;
                parent[col] = row;
                if ( matching_col[col] == INVALID )
                {
                    augment_path( col );
                    return true;
                }

                queue.push_back( matching_col[col] - base );
            }
        }

        return false;
    };

    for ( COLTYPE row = 0; row < rows; ++row )
    {
        if ( matching_row[row] == INVALID && find_augmenting_path( row ) )
        {
            ++match_count;
        }
    }

    return match_count;
}

template int MaximumMatching<int, int>( const int rows, int const* ai, int const* aj, int* matching_row, int* matching_col );
template std::int64_t MaximumMatching<std::int64_t, std::int64_t>( const std::int64_t rows,
                                                                   std::int64_t const* ai,
                                                                   std::int64_t const* aj,
                                                                   std::int64_t* matching_row,
                                                                   std::int64_t* matching_col );

} // namespace reordering
