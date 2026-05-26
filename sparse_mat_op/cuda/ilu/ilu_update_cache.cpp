#include "ilu_update_cache.hpp"

#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace matrix_utils::sparse_cuda
{
namespace
{
template <typename ROWTYPE, typename COLTYPE>
ROWTYPE count_update_intersections( const ROWTYPE row_end,
                                    const ROWTYPE k_pos,
                                    const ROWTYPE k_u_begin,
                                    const ROWTYPE k_u_end,
                                    const COLTYPE* cols )
{
    ROWTYPE row_pos = k_pos + 1;
    ROWTYPE j_pos = k_u_begin;
    ROWTYPE count = 0;
    while ( row_pos < row_end && j_pos < k_u_end )
    {
        const COLTYPE row_col = cols[row_pos];
        const COLTYPE u_col = cols[j_pos];
        if ( row_col < u_col )
        {
            ++row_pos;
        }
        else if ( u_col < row_col )
        {
            ++j_pos;
        }
        else
        {
            ++count;
            ++row_pos;
            ++j_pos;
        }
    }
    return count;
}

template <typename ROWTYPE, typename COLTYPE>
ROWTYPE fill_update_intersections( const ROWTYPE row_end,
                                   const ROWTYPE k_pos,
                                   const ROWTYPE k_u_begin,
                                   const ROWTYPE k_u_end,
                                   const COLTYPE* cols,
                                   std::vector<ROWTYPE>& update_jpos,
                                   std::vector<ROWTYPE>& update_pos,
                                   ROWTYPE write )
{
    ROWTYPE row_pos = k_pos + 1;
    ROWTYPE j_pos = k_u_begin;
    while ( row_pos < row_end && j_pos < k_u_end )
    {
        const COLTYPE row_col = cols[row_pos];
        const COLTYPE u_col = cols[j_pos];
        if ( row_col < u_col )
        {
            ++row_pos;
        }
        else if ( u_col < row_col )
        {
            ++j_pos;
        }
        else
        {
            update_jpos[static_cast<std::size_t>( write )] = j_pos;
            update_pos[static_cast<std::size_t>( write )] = row_pos;
            ++write;
            ++row_pos;
            ++j_pos;
        }
    }
    return write;
}
} // namespace

template <typename ROWTYPE, typename COLTYPE>
ILUUpdateCache<ROWTYPE> BuildILUUpdateCache( const COLTYPE n,
                                             const ROWTYPE* lu_ai,
                                             const COLTYPE* lu_aj,
                                             const ROWTYPE* lu_diag,
                                             const COLTYPE base,
                                             const int threads )
{
    if ( n <= 0 || lu_ai == nullptr || lu_aj == nullptr || lu_diag == nullptr || threads <= 0 )
    {
        throw std::invalid_argument( "BuildILUUpdateCache received invalid input" );
    }

    const auto build_start = std::chrono::steady_clock::now();
    const ROWTYPE row_base = static_cast<ROWTYPE>( base );
    const ROWTYPE nnz = lu_ai[n] - row_base;

    ILUUpdateCache<ROWTYPE> cache;
    cache.update_ptr.assign( static_cast<std::size_t>( nnz ) + 1, ROWTYPE( 0 ) );

#pragma omp parallel for schedule( dynamic ) num_threads( threads )
    for ( COLTYPE i = 0; i < n; ++i )
    {
        const ROWTYPE row_begin = lu_ai[i] - row_base;
        const ROWTYPE row_end = lu_ai[i + 1] - row_base;
        const ROWTYPE lower_end = lu_diag[i] - row_base;
        for ( ROWTYPE k_pos = row_begin; k_pos < lower_end; ++k_pos )
        {
            const COLTYPE k = lu_aj[k_pos] - base;
            const ROWTYPE k_u_begin = ( lu_diag[k] - row_base ) + 1;
            const ROWTYPE k_u_end = lu_ai[k + 1] - row_base;
            cache.update_ptr[static_cast<std::size_t>( k_pos )] =
                count_update_intersections( row_end, k_pos, k_u_begin, k_u_end, lu_aj );
        }
    }

    std::int64_t total_updates = 0;
    for ( ROWTYPE pos = 0; pos < nnz; ++pos )
    {
        const ROWTYPE count = cache.update_ptr[static_cast<std::size_t>( pos )];
        if ( total_updates > std::numeric_limits<ROWTYPE>::max() )
        {
            throw std::runtime_error( "ILU update cache exceeds row index range" );
        }
        cache.update_ptr[static_cast<std::size_t>( pos )] = static_cast<ROWTYPE>( total_updates );
        total_updates += count;
    }
    if ( total_updates > std::numeric_limits<ROWTYPE>::max() )
    {
        throw std::runtime_error( "ILU update cache exceeds row index range" );
    }
    cache.update_ptr[static_cast<std::size_t>( nnz )] = static_cast<ROWTYPE>( total_updates );
    cache.update_jpos.resize( static_cast<std::size_t>( total_updates ) );
    cache.update_pos.resize( static_cast<std::size_t>( total_updates ) );

#pragma omp parallel for schedule( dynamic ) num_threads( threads )
    for ( COLTYPE i = 0; i < n; ++i )
    {
        const ROWTYPE row_begin = lu_ai[i] - row_base;
        const ROWTYPE row_end = lu_ai[i + 1] - row_base;
        const ROWTYPE lower_end = lu_diag[i] - row_base;
        for ( ROWTYPE k_pos = row_begin; k_pos < lower_end; ++k_pos )
        {
            const COLTYPE k = lu_aj[k_pos] - base;
            const ROWTYPE k_u_begin = ( lu_diag[k] - row_base ) + 1;
            const ROWTYPE k_u_end = lu_ai[k + 1] - row_base;
            fill_update_intersections( row_end, k_pos, k_u_begin, k_u_end, lu_aj, cache.update_jpos,
                                       cache.update_pos, cache.update_ptr[static_cast<std::size_t>( k_pos )] );
        }
    }

    const auto build_end = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> build_ms = build_end - build_start;
    cache.build_ms = build_ms.count();
    return cache;
}

template ILUUpdateCache<int> BuildILUUpdateCache<int, int>( int, const int*, const int*, const int*, int, int );

template ILUUpdateCache<std::int64_t> BuildILUUpdateCache<std::int64_t, int>( int,
                                                                              const std::int64_t*,
                                                                              const int*,
                                                                              const std::int64_t*,
                                                                              int,
                                                                              int );

} // namespace matrix_utils::sparse_cuda
