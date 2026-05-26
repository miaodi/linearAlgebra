#include "BFS.h"
#include "BitVector.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <omp.h>

namespace reordering
{
template <bool LASTLEVEL, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
bool BFS_Fn( COLTYPE rows,
             ROWTYPE const* ai,
             COLTYPE const* aj,
             [[maybe_unused]] VALTYPE const* av,
             COLTYPE source,
             COLTYPE shortCutWidth,
             COLTYPE& height,
             COLTYPE& width,
             std::vector<COLTYPE>& levels,
             std::vector<COLTYPE>& lastLevel )
{
    levels.resize( rows );
    lastLevel.resize( 0 );
    height = 0;
    std::fill_n( levels.begin(), levels.size(), -1 );
    const COLTYPE base = static_cast<COLTYPE>( ai[0] );

    utils::CircularBuffer<COLTYPE> cb( std::max<COLTYPE>( 1, static_cast<COLTYPE>( rows * .2 ) ) );
    cb.push_back( source - base );
    levels[source - base] = 0;
    if constexpr ( LASTLEVEL )
        lastLevel.push_back( source );
    COLTYPE widthCounter = 1;
    while ( !cb.empty() )
    {
        auto u = cb.first();
        cb.pop_front();
        for ( ROWTYPE i = ai[u] - base; i < ai[u + 1] - base; i++ )
        {
            auto v = aj[i] - base;
            if ( levels[v] == -1 )
            {
                if ( height < levels[u] + 1 )
                {
                    height = levels[u] + 1;
                    width = std::max( width, widthCounter );
                    widthCounter = 0;
                    if constexpr ( LASTLEVEL )
                    {
                        lastLevel.resize( 0 );
                    }
                }
                levels[v] = height;
                if constexpr ( LASTLEVEL )
                    lastLevel.push_back( v + base );
                if ( !cb.available() )
                    cb.resize( cb.size() * 2 );
                cb.push_back( v );
                if ( ++widthCounter >= shortCutWidth )
                    return false;
                ;
            }
        }
    }
    height++;
    return true;
}

template <bool LASTLEVEL, bool RECORDLEVEL, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
bool PBFS_Fn( COLTYPE rows,
              ROWTYPE const* ai,
              COLTYPE const* aj,
              [[maybe_unused]] VALTYPE const* av,
              COLTYPE source,
              COLTYPE shortCutWidth,
              COLTYPE& height,
              COLTYPE& width,
              std::vector<COLTYPE>& levels,
              std::vector<COLTYPE>& lastLevel )
{
    if constexpr ( RECORDLEVEL )
    {
        levels.resize( rows );
#pragma omp parallel for
        for ( COLTYPE i = 0; i < rows; ++i )
        {
            levels[i] = -1;
        }
    }
    const COLTYPE base = static_cast<COLTYPE>( ai[0] );
    bool stat = true;
    int max_threads = omp_get_max_threads();
    static std::vector<std::vector<COLTYPE>> bvc;
    static std::vector<std::vector<COLTYPE>> bvn;
    bvc.resize( max_threads );
    bvn.resize( max_threads );

    // std::vector<bool> visited(rows, false);
    utils::BitVector visited( rows );
    std::vector<COLTYPE> count_per_thread( max_threads + 1, 0 );
    std::vector<COLTYPE> count_per_thread_prev( max_threads + 1, 0 );
    height = 0;
    if constexpr ( RECORDLEVEL )
    {
        levels[source - base] = 0;
    }
    bvn[0].push_back( source - base );
    // visited[source - base] = true;
    visited.set( source - base );
    count_per_thread[1] = 1;
    COLTYPE total_work;
    COLTYPE total_work_prev;
    int nthreads;
    std::vector<std::pair<int, int>> chunck_pos_pairs( max_threads + 1 );
    chunck_pos_pairs[0] = std::make_pair( 0, 0 );
#pragma omp parallel shared( total_work, nthreads )
    {
        nthreads = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        while ( true )
        {
#pragma omp barrier
#pragma omp master
            {
                std::swap( bvn, bvc );
                std::inclusive_scan( count_per_thread.begin(), count_per_thread.end(),
                                     count_per_thread.begin() );
                if constexpr ( LASTLEVEL )
                {
                    total_work_prev = total_work;
                }
                total_work = count_per_thread[nthreads];
                width = std::max( width, total_work );
                int pos = 0;
                COLTYPE target = 0;
                for ( int i = 0; i < nthreads; i++ )
                {
                    target += total_work / nthreads + ( ( total_work % nthreads ) > i ? 1 : 0 );
                    while ( count_per_thread[pos + 1] < target )
                        pos++;
                    chunck_pos_pairs[i + 1] = std::make_pair( pos, target - count_per_thread[pos] );
                }
                if constexpr ( LASTLEVEL )
                {
                    if ( total_work == 0 )
                    {
                        lastLevel.resize( total_work_prev );
                    }
                    else
                    {
                        std::swap( count_per_thread, count_per_thread_prev );
                    }
                }
                height++;
            }
#pragma omp barrier
            if ( total_work == 0 )
            {
                if constexpr ( LASTLEVEL )
                {
                    for ( size_t i = 0; i < bvn[tid].size(); i++ )
                    {
                        *( lastLevel.data() + count_per_thread_prev[tid] + i ) = bvn[tid][i] + base;
                    }
                }
                bvn[tid].resize( 0 );
                break;
            }
            else if ( total_work >= shortCutWidth )
            {
                stat = false;
                bvn[tid].resize( 0 );
                break;
            }
            bvn[tid].resize( 0 );
            // bvn[tid].clear();
            for ( int i = chunck_pos_pairs[tid].first; i <= chunck_pos_pairs[tid + 1].first; i++ )
            {
                int start = ( i == chunck_pos_pairs[tid].first ) ? chunck_pos_pairs[tid].second : 0;
                int end = ( i == chunck_pos_pairs[tid + 1].first ) ? chunck_pos_pairs[tid + 1].second
                                                                   : bvc[i].size();
                for ( int j = start; j < end; j++ )
                {
                    for ( ROWTYPE k = ai[bvc[i][j]] - base; k < ai[bvc[i][j] + 1] - base; k++ )
                    {
                        auto v = aj[k] - base;
                        if constexpr ( RECORDLEVEL )
                        {
                            if ( !visited.get( v ) )
                            {
                                // visited[v] = true;
                                visited.set( v );
                                if ( levels[v] == -1 )
                                {
                                    levels[v] = height;
                                    bvn[tid].push_back( v );
                                }
                            }
                        }
                        else
                        {
                            if ( visited.testAndSet( v ) )
                            {
                                bvn[tid].push_back( v );
                            }
                        }
                    }
                }
            }
            count_per_thread[tid + 1] = bvn[tid].size();
        }
    }
    height--;
    return stat;
}

template bool BFS_Fn<true, int, int, double>( int,
                                              int const*,
                                              int const*,
                                              double const*,
                                              int,
                                              int,
                                              int&,
                                              int&,
                                              std::vector<int>&,
                                              std::vector<int>& );

template bool BFS_Fn<false, int, int, double>( int,
                                               int const*,
                                               int const*,
                                               double const*,
                                               int,
                                               int,
                                               int&,
                                               int&,
                                               std::vector<int>&,
                                               std::vector<int>& );

template bool PBFS_Fn<true, true, int, int, double>( int,
                                                     int const*,
                                                     int const*,
                                                     double const*,
                                                     int,
                                                     int,
                                                     int&,
                                                     int&,
                                                     std::vector<int>&,
                                                     std::vector<int>& );

template bool PBFS_Fn<true, false, int, int, double>( int,
                                                      int const*,
                                                      int const*,
                                                      double const*,
                                                      int,
                                                      int,
                                                      int&,
                                                      int&,
                                                      std::vector<int>&,
                                                      std::vector<int>& );

template bool PBFS_Fn<false, true, int, int, double>( int,
                                                      int const*,
                                                      int const*,
                                                      double const*,
                                                      int,
                                                      int,
                                                      int&,
                                                      int&,
                                                      std::vector<int>&,
                                                      std::vector<int>& );

template bool PBFS_Fn<false, false, int, int, double>( int,
                                                       int const*,
                                                       int const*,
                                                       double const*,
                                                       int,
                                                       int,
                                                       int&,
                                                       int&,
                                                       std::vector<int>&,
                                                       std::vector<int>& );

template bool BFS_Fn<true, int64_t, int64_t, double>( int64_t,
                                                      int64_t const*,
                                                      int64_t const*,
                                                      double const*,
                                                      int64_t,
                                                      int64_t,
                                                      int64_t&,
                                                      int64_t&,
                                                      std::vector<int64_t>&,
                                                      std::vector<int64_t>& );

template bool BFS_Fn<false, int64_t, int64_t, double>( int64_t,
                                                       int64_t const*,
                                                       int64_t const*,
                                                       double const*,
                                                       int64_t,
                                                       int64_t,
                                                       int64_t&,
                                                       int64_t&,
                                                       std::vector<int64_t>&,
                                                       std::vector<int64_t>& );

template bool PBFS_Fn<true, true, int64_t, int64_t, double>( int64_t,
                                                             int64_t const*,
                                                             int64_t const*,
                                                             double const*,
                                                             int64_t,
                                                             int64_t,
                                                             int64_t&,
                                                             int64_t&,
                                                             std::vector<int64_t>&,
                                                             std::vector<int64_t>& );

template bool PBFS_Fn<true, false, int64_t, int64_t, double>( int64_t,
                                                              int64_t const*,
                                                              int64_t const*,
                                                              double const*,
                                                              int64_t,
                                                              int64_t,
                                                              int64_t&,
                                                              int64_t&,
                                                              std::vector<int64_t>&,
                                                              std::vector<int64_t>& );

template bool PBFS_Fn<false, true, int64_t, int64_t, double>( int64_t,
                                                              int64_t const*,
                                                              int64_t const*,
                                                              double const*,
                                                              int64_t,
                                                              int64_t,
                                                              int64_t&,
                                                              int64_t&,
                                                              std::vector<int64_t>&,
                                                              std::vector<int64_t>& );

template bool PBFS_Fn<false, false, int64_t, int64_t, double>( int64_t,
                                                               int64_t const*,
                                                               int64_t const*,
                                                               double const*,
                                                               int64_t,
                                                               int64_t,
                                                               int64_t&,
                                                               int64_t&,
                                                               std::vector<int64_t>&,
                                                               std::vector<int64_t>& );
} // namespace reordering
