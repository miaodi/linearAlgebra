#include "precond.hpp"
#include "matrix_utils.hpp"
#include "utils.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <deque>
#include <iomanip>
#include <numeric>
#include <ranges>

namespace matrix_utils
{
template <typename OutVec, typename In1Iter, typename In2Iter>
void ICCMerge( OutVec& out_vec, In1Iter in1_begin, In1Iter in1_end, In2Iter in2_begin, In2Iter in2_end )
{
    while ( in1_begin != in1_end && in2_begin != in2_end )
    {
        if ( in1_begin->first < in2_begin->first )
        {
            out_vec.emplace_back( *in1_begin++ );
        }
        else if ( in1_begin->first > in2_begin->first )
        {
            out_vec.emplace_back( *in2_begin++ );
        }
        else
        {
            out_vec.emplace_back(
                std::make_pair( in1_begin->first, std::min( in1_begin->second, in2_begin->second ) ) );
            in1_begin++;
            in2_begin++;
        }
    }
    while ( in1_begin != in1_end )
    {
        out_vec.emplace_back( *in1_begin++ );
    }
    while ( in2_begin != in2_end )
    {
        out_vec.emplace_back( *in2_begin++ );
    }
}

template <typename OutVec, typename In1PosIter, typename In1LvlIter, typename In2PosIter, typename In2LvlIter,
          typename Level>
void ICCFirstMerge( OutVec& out_vec, In1PosIter in1p_begin, In1PosIter in1p_end, In1LvlIter in1l_iter,
                    In2PosIter in2p_begin, In2PosIter in2p_end, In2LvlIter in2l_iter, Level lvl )
{
    while ( in1p_begin != in1p_end && in2p_begin != in2p_end )
    {
        if ( *in1p_begin < *in2p_begin )
        {
            if ( *in1l_iter <= lvl )
                out_vec.emplace_back( std::make_pair( *in1p_begin, *in1l_iter ) );
            in1p_begin++;
            in1l_iter++;
        }
        else if ( *in1p_begin > *in2p_begin )
        {
            if ( *in2l_iter <= lvl )
                out_vec.emplace_back( std::make_pair( *in2p_begin, *in2l_iter ) );
            in2p_begin++;
            in2l_iter++;
        }
        else
        {
            auto l = std::min( *in1l_iter, *in2l_iter );
            if ( l <= lvl )
                out_vec.emplace_back( std::make_pair( *in1p_begin, l ) );
            in1p_begin++;
            in2p_begin++;
            in1l_iter++;
            in2l_iter++;
        }
    }
    while ( in1p_begin != in1p_end )
    {
        if ( *in1l_iter <= lvl )
            out_vec.emplace_back( std::make_pair( *in1p_begin, *in1l_iter ) );
        in1p_begin++;
        in1l_iter++;
    }
    while ( in2p_begin != in2p_end )
    {
        if ( *in2l_iter <= lvl )
            out_vec.emplace_back( std::make_pair( *in2p_begin, *in2l_iter ) );
        in2p_begin++;
        in2l_iter++;
    }
}

// NOTE: for level 0 ICC with Symmetric matrix input
template <ResizableCSR CSRMatrixType>
void ICCLevel0SymSymbolic( const typename CSRMatrixType::COLTYPE size, 
                           typename CSRMatrixType::ROWTYPE const* ai, 
                           typename CSRMatrixType::COLTYPE const* aj, 
                           CSRMatrixType& icc )
{
    using ROWTYPE = typename CSRMatrixType::ROWTYPE;
    using COLTYPE = typename CSRMatrixType::COLTYPE;
    static_assert( CSRMatrixFormat<ROWTYPE, COLTYPE, typename CSRMatrixType::VALTYPE, CSRMatrixType>::value == true );
    icc.rows = size;
    icc.cols = size;
    icc.ResizeAI( size + 1 );
    const auto base = ai[0];
    const ROWTYPE nnz = ai[size] - base;

    icc.ResizeAJ( nnz );
    icc.ResizeAV( nnz );
#pragma omp parallel for
    for ( COLTYPE i = 0; i < size + 1; i++ )
    {
        icc.ai[i] = ai[i];
    }
#pragma omp parallel for
    for ( ROWTYPE i = 0; i < nnz; i++ )
    {
        icc.aj[i] = aj[i];
    }
}

template <typename CSRMatrixType>
void ICCLevelSymbolic0( const typename CSRMatrixType::COLTYPE size, 
                        typename CSRMatrixType::ROWTYPE const* ai, 
                        typename CSRMatrixType::COLTYPE const* aj, 
                        typename CSRMatrixType::COLTYPE const* diag_pos,
                        const int lvl, 
                        CSRMatrixType& icc )
{
    using ROWTYPE = typename CSRMatrixType::ROWTYPE;
    using COLTYPE = typename CSRMatrixType::COLTYPE;
    icc.rows = size;
    icc.cols = size;
    icc.ResizeAI( size + 1 );
    const auto base = ai[0];
    const ROWTYPE nnz = ai[size] - base;
    const COLTYPE NONE = std::numeric_limits<COLTYPE>::max();
    std::vector<COLTYPE> llist( size, NONE );
    std::vector<ROWTYPE> jk( size );
    ROWTYPE nnz_icc = nnz;
    std::vector<COLTYPE> av_lvls( nnz_icc );
    icc.ResizeAJ( nnz_icc );
    std::forward_list<std::pair<COLTYPE, COLTYPE>> current_row; // <col, lvl>
    typename std::forward_list<std::pair<COLTYPE, COLTYPE>>::iterator cur_row_it;

    COLTYPE list_size, lidx, k, i, lik, next_i, level, llist_next;
    ROWTYPE i_idx, i_idx_end;
    icc.ai[0] = base;
    for ( COLTYPE j = 0; j < size; j++ )
    {
        i_idx = diag_pos[j];
        i_idx_end = ai[j + 1];
        cur_row_it = current_row.before_begin();
        list_size = i_idx_end - i_idx;

        // initialize the current row's nonzeros
        for ( ; i_idx != i_idx_end; i_idx++ )
        {
            cur_row_it = current_row.insert_after( cur_row_it, std::make_pair( aj[i_idx - base], 0 ) );
        }

        // use max as the list end to prevent from branch prediction
        current_row.insert_after( cur_row_it, std::make_pair( NONE, 0 ) );

        // iterate for k from 0 to j-1
        k = llist[j];
        while ( k < j )
        {
            i_idx = jk[k]++;
            i_idx_end = icc.ai[k + 1];

            llist_next = llist[k];
            //   update llist if necessary
            if ( i_idx + 1 < i_idx_end )
            {
                llist[k] = llist[icc.aj[i_idx + 1 - base] - base];
                llist[icc.aj[i_idx + 1 - base] - base] = k;
            }
            k = llist_next;

            lik = av_lvls[i_idx - base];
            cur_row_it = current_row.begin();
            next_i = std::next( cur_row_it )->first;
            // merge row k to row j
            for ( ; i_idx < i_idx_end; i_idx++ )
            {
                i = icc.aj[i_idx - base];

                while ( next_i <= i )
                {
                    cur_row_it = std::next( cur_row_it );
                    next_i = std::next( cur_row_it )->first;
                }
                level = lik + av_lvls[i_idx - base] + 1;
                if ( level <= lvl )
                {
                    if ( cur_row_it->first == i )
                    {
                        cur_row_it->second = std::min( cur_row_it->second, level );
                    }
                    else
                    {
                        cur_row_it = current_row.insert_after( cur_row_it, std::make_pair( i, level ) );
                        next_i = std::next( cur_row_it )->first;

                        list_size++;
                    }
                }
            }
        }
        icc.ai[j + 1] = icc.ai[j] + list_size;

        //   resize if needed
        if ( icc.ai[j + 1] - base > nnz_icc )
        {
            // estimate the new size
            if ( 2 * ( j - base ) >= size )
                nnz_icc *= 2;
            else
                nnz_icc = nnz_icc * std::ceil( size * 1. / ( j - base ) );
            av_lvls.resize( nnz_icc );
            icc.ResizeAJ( nnz_icc );
        }

        cur_row_it = current_row.begin();
        i_idx = icc.ai[j];

        //   copy the current row to icc
        for ( lidx = 0; lidx < list_size; lidx++ )
        {
            icc.aj[i_idx - base] = cur_row_it->first;
            av_lvls[i_idx - base] = cur_row_it->second;
            i_idx++;
            cur_row_it++;
        }

        //   update llist if necessary
        if ( icc.ai[j] + 1 < icc.ai[j + 1] )
        {
            llist[j] = llist[icc.aj[icc.ai[j] + 1 - base] - base];
            llist[icc.aj[icc.ai[j] + 1 - base] - base] = j;
            jk[j] = icc.ai[j] + 1;
        }
        current_row.clear();
    }
    icc.ResizeAV( icc.ai[size] - base );
}

template <ResizableCSR CSRMatrixType>
void ICCLevelSymbolic1( const typename CSRMatrixType::COLTYPE size, 
                        typename CSRMatrixType::ROWTYPE const* ai, 
                        typename CSRMatrixType::COLTYPE const* aj, 
                        typename CSRMatrixType::COLTYPE const* diag_pos,
                        const int lvl, 
                        CSRMatrixType& icc )
{
    using ROWTYPE = typename CSRMatrixType::ROWTYPE;
    using COLTYPE = typename CSRMatrixType::COLTYPE;
    icc.rows = size;
    icc.cols = size;
    icc.ResizeAI( size + 1 );
    const auto base = ai[0];
    const ROWTYPE nnz = ai[size] - base;
    const COLTYPE NONE = std::numeric_limits<COLTYPE>::max();
    std::vector<COLTYPE> llist( size, NONE );
    std::vector<ROWTYPE> jk( size );
    ROWTYPE nnz_icc = nnz;
    std::vector<COLTYPE> av_lvls( nnz_icc );
    icc.ResizeAJ( nnz_icc );
    std::vector<std::pair<COLTYPE, COLTYPE>> current_row; // <col, lvl>
    COLTYPE current_row_size_before, current_row_size_after, current_row_pos;
    typename std::vector<std::pair<COLTYPE, COLTYPE>>::iterator cur_row_it;
    current_row.reserve( size * .5 );

    COLTYPE lidx, k, i, lik, next_i, level, llist_next;
    ROWTYPE i_idx, i_idx_end;
    icc.ai[0] = base;
    for ( COLTYPE j = 0; j < size; j++ )
    {
        i_idx = diag_pos[j];
        i_idx_end = ai[j + 1];
        current_row_size_before = current_row_size_after = i_idx_end - i_idx;

        // initialize the current row's nonzeros
        for ( ; i_idx != i_idx_end; i_idx++ )
        {
            current_row.emplace_back( std::make_pair( aj[i_idx - base], 0 ) );
        }

        // iterate for k from 0 to j-1
        k = llist[j];
        while ( k < j )
        {
            i_idx = jk[k]++;
            i_idx_end = icc.ai[k + 1];

            llist_next = llist[k];
            //   update llist if necessary
            if ( i_idx + 1 < i_idx_end )
            {
                llist[k] = llist[icc.aj[i_idx + 1 - base] - base];
                llist[icc.aj[i_idx + 1 - base] - base] = k;
            }
            k = llist_next;

            lik = av_lvls[i_idx - base];
            current_row_pos = 0;
            next_i = current_row_pos + 1 == current_row_size_before ? NONE : current_row[current_row_pos + 1].first;
            // merge row k to row j
            for ( ; i_idx < i_idx_end; i_idx++ )
            {
                i = icc.aj[i_idx - base];

                while ( next_i <= i )
                {
                    current_row_pos += 1;
                    next_i = current_row_pos + 1 == current_row_size_before ? NONE
                                                                             : current_row[current_row_pos + 1].first;
                }
                level = lik + av_lvls[i_idx - base] + 1;
                if ( level <= lvl )
                {
                    if ( current_row[current_row_pos].first == i )
                    {
                        current_row[current_row_pos].second = std::min( current_row[current_row_pos].second, level );
                    }
                    else
                    {
                        current_row.emplace_back( std::make_pair( i, level ) );
                        current_row_size_after++;
                    }
                }
            }
            std::inplace_merge( current_row.begin(), current_row.begin() + current_row_size_before,
                                current_row.begin() + current_row_size_after );
            current_row_size_before = current_row_size_after;
        }
        icc.ai[j + 1] = icc.ai[j] + current_row_size_before;

        //   resize if needed
        if ( icc.ai[j + 1] - base > nnz_icc )
        {
            // estimate the new size
            if ( 2 * ( j - base ) >= size )
                nnz_icc *= 2;
            else
                nnz_icc = nnz_icc * std::ceil( size * 1. / ( j - base ) );
            av_lvls.resize( nnz_icc );
            icc.ResizeAJ( nnz_icc );
        }

        //   update llist if necessary
        if ( current_row_size_after > 1 )
        {
            llist[j] = llist[current_row[1].first - base];
            llist[current_row[1].first - base] = j;
            jk[j] = icc.ai[j] + 1;
        }

        i_idx = icc.ai[j];
        //   copy the current row to icc
        for ( const auto& p : current_row )
        {
            icc.aj[i_idx - base] = p.first;
            av_lvls[i_idx - base] = p.second;
            i_idx++;
        }

        current_row.clear();
    }
    icc.ResizeAV( icc.ai[size] - base );
}

template <ResizableCSR CSRMatrixType>
void ICCLevelSymbolic2( const typename CSRMatrixType::COLTYPE size, 
                        typename CSRMatrixType::ROWTYPE const* ai, 
                        typename CSRMatrixType::COLTYPE const* aj, 
                        typename CSRMatrixType::COLTYPE const* diag_pos,
                        const int lvl, 
                        CSRMatrixType& icc )
{
    using ROWTYPE = typename CSRMatrixType::ROWTYPE;
    using COLTYPE = typename CSRMatrixType::COLTYPE;
    icc.rows = size;
    icc.cols = size;
    icc.ResizeAI( size + 1 );
    const auto base = ai[0];
    const ROWTYPE nnz = ai[size] - base;
    const COLTYPE NONE = std::numeric_limits<COLTYPE>::max();
    std::vector<COLTYPE> llist( size, NONE );
    std::vector<ROWTYPE> jk( size );
    ROWTYPE nnz_icc = nnz;
    std::vector<COLTYPE> av_lvls( nnz_icc );
    icc.ResizeAJ( nnz_icc );
    std::vector<std::pair<COLTYPE, COLTYPE>> current_row1, current_row2; // <col, lvl>
    std::vector<ROWTYPE> span_prefix1, span_prefix2;
    current_row1.reserve( ai[size] - base );
    current_row2.reserve( ai[size] - base );
    span_prefix1.reserve( size );
    span_prefix2.reserve( size );

    COLTYPE lidx, k, i, lik, next_i, level, llist_next;
    ROWTYPE i_idx, i_idx_end;
    icc.ai[0] = base;
    for ( COLTYPE j = 0; j < size; j++ )
    {
        span_prefix1.push_back( 0 );
        i_idx = diag_pos[j];
        i_idx_end = ai[j + 1];

        // initialize the current row's nonzeros
        for ( ; i_idx != i_idx_end; i_idx++ )
        {
            current_row1.emplace_back( std::make_pair( aj[i_idx - base], 0 ) );
        }
        span_prefix1.push_back( current_row1.size() );

        // iterate for k from 0 to j-1
        k = llist[j];
        while ( k < j )
        {
            i_idx = jk[k]++;
            i_idx_end = icc.ai[k + 1];

            llist_next = llist[k];
            //   update llist if necessary
            if ( i_idx + 1 < i_idx_end )
            {
                llist[k] = llist[icc.aj[i_idx + 1 - base] - base];
                llist[icc.aj[i_idx + 1 - base] - base] = k;
            }
            k = llist_next;
            lik = av_lvls[i_idx - base];
            for ( ; i_idx < i_idx_end; i_idx++ )
            {
                level = lik + av_lvls[i_idx - base] + 1;
                if ( level <= lvl )
                {
                    current_row1.emplace_back( std::make_pair( icc.aj[i_idx - base], level ) );
                }
            }
            span_prefix1.push_back( current_row1.size() );
        }
        while ( span_prefix1.size() > 2 )
        {
            COLTYPE span_pos = 0;
            span_prefix2.push_back( 0 );
            for ( ; span_pos + 2 < span_prefix1.size(); span_pos += 2 )
            {
                ICCMerge( current_row2, current_row1.begin() + span_prefix1[span_pos],
                          current_row1.begin() + span_prefix1[span_pos + 1],
                          current_row1.begin() + span_prefix1[span_pos + 1],
                          current_row1.begin() + span_prefix1[span_pos + 2] );
                span_prefix2.push_back( current_row2.size() );
            }
            if ( span_pos + 1 < span_prefix1.size() )
            {
                current_row2.insert( current_row2.end(), current_row1.begin() + span_prefix1[span_pos],
                                     current_row1.begin() + span_prefix1[span_pos + 1] );
                span_prefix2.push_back( current_row2.size() );
            }

            std::swap( span_prefix1, span_prefix2 );
            std::swap( current_row1, current_row2 );
            span_prefix2.clear();
            current_row2.clear();
        }

        icc.ai[j + 1] = icc.ai[j] + current_row1.size();

        //   resize if needed
        if ( icc.ai[j + 1] - base > nnz_icc )
        {
            // estimate the new size
            if ( 2 * ( j - base ) >= size )
                nnz_icc *= 2;
            else
                nnz_icc = nnz_icc * std::ceil( size * 1. / ( j - base ) );
            av_lvls.resize( nnz_icc );
            icc.ResizeAJ( nnz_icc );
        }

        //   update llist if necessary
        if ( span_prefix1[1] > 1 )
        {
            llist[j] = llist[current_row1[1].first - base];
            llist[current_row1[1].first - base] = j;
            jk[j] = icc.ai[j] + 1;
        }

        i_idx = icc.ai[j];
        //   copy the current row to icc
        for ( const auto& p : current_row1 )
        {
            icc.aj[i_idx - base] = p.first;
            av_lvls[i_idx - base] = p.second;
            i_idx++;
        }
        span_prefix1.clear();
        current_row1.clear();
    }
    icc.ResizeAV( icc.ai[size] - base );
}

template <ResizableCSR CSRMatrixType>
void ICCLevelSymbolic3( const typename CSRMatrixType::COLTYPE size, 
                        typename CSRMatrixType::ROWTYPE const* ai, 
                        typename CSRMatrixType::COLTYPE const* aj, 
                        typename CSRMatrixType::COLTYPE const* diag_pos,
                        const int lvl, 
                        CSRMatrixType& icc )
{
    using ROWTYPE = typename CSRMatrixType::ROWTYPE;
    using COLTYPE = typename CSRMatrixType::COLTYPE;
    icc.rows = size;
    icc.cols = size;
    icc.ResizeAI( size + 1 );
    const auto base = ai[0];
    const ROWTYPE nnz = ai[size] - base;
    const COLTYPE NONE = std::numeric_limits<COLTYPE>::max();
    std::vector<COLTYPE> llist( size, NONE );
    std::vector<ROWTYPE> jk( size );
    std::vector<std::pair<ROWTYPE, size_t>> merge_spans;
    merge_spans.reserve( size );

    ROWTYPE nnz_icc = nnz;
    std::vector<COLTYPE> av_lvls( nnz_icc );
    icc.ResizeAJ( nnz_icc );
    std::vector<std::pair<COLTYPE, COLTYPE>> current_row1, current_row2; // <col, lvl>
    std::vector<ROWTYPE> span_prefix1, span_prefix2;
    current_row1.reserve( ai[size] - base );
    current_row2.reserve( ai[size] - base );
    span_prefix1.reserve( size );
    span_prefix2.reserve( size );

    COLTYPE lidx, k, i, lik, next_i, level, llist_next, lik1, lik2;
    ROWTYPE i_idx, i_idx_end;

    auto lvlTransform1 = std::views::transform( [&lik1]( const COLTYPE i ) { return i + lik1 + 1; } );

    auto lvlTransform2 = std::views::transform( [&lik2]( const COLTYPE i ) { return i + lik2 + 1; } );

    icc.ai[0] = base;
    for ( COLTYPE j = 0; j < size; j++ )
    {
        span_prefix1.push_back( 0 );
        i_idx = diag_pos[j];
        i_idx_end = ai[j + 1];

        // initialize the current row's nonzeros
        for ( ; i_idx != i_idx_end; i_idx++ )
        {
            current_row1.emplace_back( std::make_pair( aj[i_idx - base], 0 ) );
        }
        span_prefix1.push_back( current_row1.size() );

        // iterate for k from 0 to j-1
        k = llist[j];
        while ( k < j )
        {
            i_idx = jk[k]++;
            i_idx_end = icc.ai[k + 1];

            llist_next = llist[k];
            //   update llist if necessary
            if ( i_idx + 1 < i_idx_end )
            {
                llist[k] = llist[icc.aj[i_idx + 1 - base] - base];
                llist[icc.aj[i_idx + 1 - base] - base] = k;
            }
            k = llist_next;
            merge_spans.emplace_back( i_idx, i_idx_end - i_idx );
            // span_prefix1.push_back(current_row1.size());
        }

        COLTYPE span_pos = 0;
        for ( ; span_pos + 1 < merge_spans.size(); span_pos += 2 )
        {
            lik1 = av_lvls[merge_spans[span_pos].first - base];
            lik2 = av_lvls[merge_spans[span_pos + 1].first - base];
            std::span<COLTYPE> sp1{av_lvls.begin() + merge_spans[span_pos].first - base,
                                   merge_spans[span_pos].second};
            std::span<COLTYPE> sp2{av_lvls.begin() + merge_spans[span_pos + 1].first - base,
                                   merge_spans[span_pos + 1].second};
            auto tsp1 = lvlTransform1( sp1 );
            auto tsp2 = lvlTransform2( sp2 );
            ICCFirstMerge( current_row1, icc.aj.get() + merge_spans[span_pos].first - base,
                           icc.aj.get() + merge_spans[span_pos].first + merge_spans[span_pos].second - base,
                           tsp1.begin(), icc.aj.get() + merge_spans[span_pos + 1].first - base,
                           icc.aj.get() + merge_spans[span_pos + 1].first + merge_spans[span_pos + 1].second - base,
                           tsp2.begin(), lvl );
            span_prefix1.push_back( current_row1.size() );
        }

        if ( span_pos < merge_spans.size() )
        {
            lik1 = av_lvls[merge_spans[span_pos].first - base];
            std::span<COLTYPE> sp1{av_lvls.begin() + merge_spans[span_pos].first - base,
                                   merge_spans[span_pos].second};
            auto tsp1 = lvlTransform1( ( sp1 ) );

            ICCFirstMerge( current_row1, icc.aj.get() + merge_spans[span_pos].first - base,
                           icc.aj.get() + merge_spans[span_pos].first + merge_spans[span_pos].second - base,
                           tsp1.begin(), icc.aj.get() + merge_spans[span_pos].first - base,
                           icc.aj.get() + merge_spans[span_pos].first - base, tsp1.begin(), lvl );
            span_prefix1.push_back( current_row1.size() );
        }
        merge_spans.clear();

        while ( span_prefix1.size() > 2 )
        {
            span_pos = 0;
            span_prefix2.push_back( 0 );
            for ( ; span_pos + 2 < span_prefix1.size(); span_pos += 2 )
            {
                ICCMerge( current_row2, current_row1.begin() + span_prefix1[span_pos],
                          current_row1.begin() + span_prefix1[span_pos + 1],
                          current_row1.begin() + span_prefix1[span_pos + 1],
                          current_row1.begin() + span_prefix1[span_pos + 2] );
                span_prefix2.push_back( current_row2.size() );
            }
            if ( span_pos + 1 < span_prefix1.size() )
            {
                current_row2.insert( current_row2.end(), current_row1.begin() + span_prefix1[span_pos],
                                     current_row1.begin() + span_prefix1[span_pos + 1] );
                span_prefix2.push_back( current_row2.size() );
            }

            std::swap( span_prefix1, span_prefix2 );
            std::swap( current_row1, current_row2 );
            span_prefix2.clear();
            current_row2.clear();
        }

        icc.ai[j + 1] = icc.ai[j] + current_row1.size();

        //   resize if needed
        if ( icc.ai[j + 1] - base > nnz_icc )
        {
            // estimate the new size
            if ( 2 * ( j - base ) >= size )
                nnz_icc *= 2;
            else
                nnz_icc = nnz_icc * std::ceil( size * 1. / ( j - base ) );
            av_lvls.resize( nnz_icc );
            icc.ResizeAJ( nnz_icc );
        }

        //   update llist if necessary
        if ( span_prefix1[1] > 1 )
        {
            llist[j] = llist[current_row1[1].first - base];
            llist[current_row1[1].first - base] = j;
            jk[j] = icc.ai[j] + 1;
        }

        i_idx = icc.ai[j];
        //   copy the current row to icc
        for ( const auto& p : current_row1 )
        {
            icc.aj[i_idx - base] = p.first;
            av_lvls[i_idx - base] = p.second;
            i_idx++;
        }
        span_prefix1.clear();
        current_row1.clear();
    }
    icc.ResizeAV( icc.ai[size] - base );
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
bool ICCLevelNumeric( const COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj, VALTYPE const* av,
                      COLTYPE const* diag_pos, const int lvl, const VALTYPE omega, ROWTYPE const* icc_ai,
                      COLTYPE const* icc_aj, VALTYPE* icc_av )
{
    const auto base = ai[0];
    const ROWTYPE nnz = ai[size] - base;
    const COLTYPE NONE = std::numeric_limits<COLTYPE>::max();
    std::vector<COLTYPE> llist( size, NONE );
    std::vector<ROWTYPE> jk( size );

    COLTYPE lidx, k, i, lik, next_i, level, llist_next;
    ROWTYPE i_idx, i_idx_end, prec_i_idx_start, prec_i_idx, prec_i_idx_end;

    for ( COLTYPE j = 0; j < size; j++ )
    {
        i_idx = diag_pos[j];
        i_idx_end = ai[j + 1];
        prec_i_idx = prec_i_idx_start = icc_ai[j];
        prec_i_idx_end = icc_ai[j + 1];

        // NOTE: assume diagonal always exists
        icc_av[prec_i_idx++ - base] = av[i_idx++ - base] * ( static_cast<VALTYPE>( 1 ) + omega ); // shift the diagonal

        // copy initial value to current jth row
        for ( ; prec_i_idx < prec_i_idx_end; prec_i_idx++ )
        {
            if ( i_idx == i_idx_end || icc_aj[prec_i_idx - base] != aj[i_idx - base] )
                icc_av[prec_i_idx - base] = 0;
            else
                icc_av[prec_i_idx - base] = av[i_idx++ - base];
        }

        // iterate for k from 0 to j-1
        k = llist[j];
        while ( k < j )
        {
            i_idx = jk[k]++; // here i_idx become i index for k = 0:j
            i_idx_end = icc_ai[k + 1];

            // jump k to the next row
            llist_next = llist[k];
            //   update llist if necessary
            if ( i_idx + 1 < i_idx_end )
            {
                llist[k] = llist[icc_aj[i_idx + 1 - base] - base];
                llist[icc_aj[i_idx + 1 - base] - base] = k;
            }
            k = llist_next;

            // a_ij = a_ij - a_ik * a_kj
            const VALTYPE ajk = icc_av[i_idx - base];
            for ( prec_i_idx = prec_i_idx_start; i_idx < i_idx_end && prec_i_idx < prec_i_idx_end; )
            {
                if ( icc_aj[i_idx - base] == icc_aj[prec_i_idx - base] )
                {
                    icc_av[prec_i_idx++ - base] -= ajk * icc_av[i_idx++ - base];
                }
                else if ( icc_aj[i_idx - base] < icc_aj[prec_i_idx - base] )
                {
                    i_idx++;
                }
                else
                {
                    prec_i_idx++;
                }
            }
        }

        //   update llist if necessary
        if ( prec_i_idx_end - prec_i_idx_start > 1 )
        {
            llist[j] = llist[icc_aj[prec_i_idx_start + 1 - base] - base];
            llist[icc_aj[prec_i_idx_start + 1 - base] - base] = j;
            jk[j] = prec_i_idx_start + 1;
        }
        // negative diagonal detected, needs more shift
        if ( icc_av[prec_i_idx_start - base] < 0 )
            return false;

        const VALTYPE aii = std::sqrt( icc_av[prec_i_idx_start - base] );
        icc_av[prec_i_idx_start++ - base] = aii;
        for ( ; prec_i_idx_start < prec_i_idx_end; prec_i_idx_start++ )
            icc_av[prec_i_idx_start - base] /= aii;
    }
    return true;
}

template <ResizableDiagonal CSRMatrixType>
bool ILULevel0Symbolic( const typename CSRMatrixType::COLTYPE size, typename CSRMatrixType::ROWTYPE const* ai,
                        typename CSRMatrixType::COLTYPE const* aj, CSRMatrixType& ilu )
{
    ilu.rows = size;
    ilu.cols = size;
    ilu.ResizeAI( size + 1 );
    ilu.ResizeDiagonal( size );
    const auto base = ai[0];
    const typename CSRMatrixType::ROWTYPE nnz = ai[size] - base;

    ilu.ResizeAJ( nnz );
    ilu.ResizeAV( nnz );

#pragma omp parallel for
    for ( typename CSRMatrixType::COLTYPE i = 0; i < size + 1; i++ )
    {
        ilu.ai[i] = ai[i];
    }
#pragma omp parallel for
    for ( typename CSRMatrixType::ROWTYPE i = 0; i < nnz; i++ )
    {
        ilu.aj[i] = aj[i];
    }

    bool success = true;

#pragma omp parallel for
    for ( typename CSRMatrixType::COLTYPE i = 0; i < size; i++ )
    {
        auto it = std::find( ilu.aj.get() + ilu.ai[i] - base, ilu.aj.get() + ilu.ai[i + 1] - base, i + base );
        if ( it == ilu.aj.get() + ilu.ai[i + 1] - base )
        {
            success = false;
            ilu.diagonal[i] = std::numeric_limits<typename CSRMatrixType::ROWTYPE>::max();
        }
        else
        {
            ilu.diagonal[i] = std::distance( ilu.aj.get(), it ) + base;
        }
    }
    return success;
}

template <ResizableDiagonal CSRMatrixType>
bool ILULevelSymbolic<CSRMatrixType>::operator()( const typename CSRMatrixType::COLTYPE size,
                                                  typename CSRMatrixType::ROWTYPE const* ai,
                                                  typename CSRMatrixType::COLTYPE const* aj,
                                                  const int lvl,
                                                  CSRMatrixType& ilu )
{
    if ( lvl < 0 )
        return false;
    if ( lvl == 0 )
        return ILULevel0Symbolic( size, ai, aj, ilu );

    ilu.rows = size;
    ilu.cols = size;
    ilu.ResizeAI( size + 1 );
    ilu.ResizeDiagonal( size );
    const auto base = ai[0];
    typename CSRMatrixType::ROWTYPE nnz_cap = 3 * ( ai[size] - base ); // initial guess
    ilu.ResizeAJ( nnz_cap );
    ilu.ResizeAV( nnz_cap );
    _levels.assign( nnz_cap, 0 );
    using ROWT = typename CSRMatrixType::ROWTYPE;
    const ROWT MARKER_ABSENT = std::numeric_limits<ROWT>::max();
    _marker.assign( size, MARKER_ABSENT );
    auto* ilu_ai = ilu.AI();
    auto* ilu_aj = ilu.AJ();
    auto* ilu_diag = ilu.Diagonal();
    ilu_ai[0] = base;

    // Algorithm overview (BFS style ILU(k) symbolic factorization):
    // 1. Start from original sparsity pattern of row i (A_i,:); push columns < i into queue.
    // 2. Pop k from queue (k < i). If level(i,k) < lvl expand row k (its already-built ILU row)
    //    considering only entries j with k < j <= i (lower part + diagonal) because fill in U
    //    influences current row. For each candidate j compute new level = level(i,k)+level(k,j)+1.
    // 3. Insert/update (i,j) using a marker array to get O(1) membership test; if level improved
    //    and j < i we re-enqueue j to allow deeper / cheaper paths.
    // 4. After expansion ensure diagonal exists, then sort columns and append to global arrays.
    // Complexity: each discovered fill is processed proportional to the number of times its level
    // improves (bounded by lvl); membership operations are O(1). Avoids repeated full-row merges.
    // Data structures:
    //   cols/levels : local unsorted storage of pattern and their levels.
    //   _marker     : maps col -> position in cols (or MARKER_ABSENT if absent).
    //   q           : deque for BFS/level relaxation.
    // Potential further micro-optimizations: replace std::deque with custom ring buffer, use
    // integer vector as queue with two indices, and bucket queue by current fill level.

    // reuse member containers to avoid reallocations
    _cl.clear();
    if ( _cl.capacity() < 64 )
        _cl.reserve( 64 );
    _q.clear();

    for ( typename CSRMatrixType::COLTYPE i = 0; i < size; i++ )
    {
        _cl.clear();
        _q.clear();
        // load original sparsity row i
        for ( auto p = ai[i] - base; p < ai[i + 1] - base; ++p )
        {
            auto c = aj[p] - base;
            _marker[c] = static_cast<ROWT>( _cl.size() );
            _cl.push_back( { c, 0 } );
            if ( c < i )
                _q.push_back( c ); // candidate pivot in elimination
        }
        // BFS style expansion limited by level
        while ( !_q.empty() )
        {
            auto k = _q.front();
            _q.pop_front();
            if ( k >= i )
                continue; // only consider columns < i
            auto k_pos = _marker[k];
            int lvl_ik = _cl[k_pos].level;
            if ( lvl_ik >= lvl )
                continue; // no need to expand further

            // iterate over (k, j) with j > k (upper part of row k) starting at diagonal+1
            auto row_start = ( ilu_diag[k] - base ) + 1;
            auto row_end = ilu_ai[k + 1] - base;
            for ( auto rp = row_start; rp < row_end; ++rp )
            {
                auto j = ilu_aj[rp] - base;
                // j is guaranteed > k since we started after diagonal
                int new_level = lvl_ik + _levels[rp] + 1; // candidate level(i,j)
                if ( new_level > lvl )
                    continue;
                auto pos = _marker[j];
                if ( pos == MARKER_ABSENT )
                {
                    // add new fill
                    pos = static_cast<ROWT>( _cl.size() );
                    _cl.push_back( { j, new_level } );
                    _marker[j] = pos;
                    // only columns < i can produce further fill into row i (lower part)
                    if ( j < i )
                        _q.push_back( j );
                }
                else
                {
                    if ( new_level < _cl[pos].level )
                    {
                        _cl[pos].level = new_level;
                        if ( j < i )
                            _q.push_back( j ); // improved path -> re-expand
                    }
                }
            }
        }
        // ensure diagonal exists
        if ( _marker[i] == MARKER_ABSENT )
        {
            _marker[i] = static_cast<ROWT>( _cl.size() );
            _cl.push_back( { i, 0 } );
        }
        // sort by column index using ranges
        std::ranges::sort( _cl, {}, &ColLevel::col );
        auto row_nnz = static_cast<typename CSRMatrixType::ROWTYPE>( _cl.size() );
        auto needed = ( ilu_ai[i] - base ) + row_nnz;
        if ( needed > nnz_cap )
        {
            nnz_cap = std::max<decltype( nnz_cap )>( nnz_cap * 2, needed );
            ilu_aj = ilu.ResizeAJ( nnz_cap );
            ilu.ResizeAV( nnz_cap );
            _levels.resize( nnz_cap );
        }
        auto write_pos = ilu_ai[i] - base;
        for ( const auto& e : _cl )
        {
            ilu_aj[write_pos] = e.col + base;
            _levels[write_pos] = e.level;
            if ( e.col == i )
                ilu_diag[i] = write_pos + base; // store diag position
            ++write_pos;
        }
        ilu_ai[i + 1] = write_pos + base;
        // clear marker entries for this row
        for ( const auto& e : _cl )
            _marker[e.col] = MARKER_ABSENT;
    }
    ilu.ResizeAV( ilu_ai[size] - base );
    return true;
}

template <ResizableDiagonal CSRMatrixType, bool keepdiag>
bool ILULevelSymbolicParallelU<CSRMatrixType, keepdiag>::operator()(const COLTYPE size, ROWTYPE const* ai,
                                                         COLTYPE const* aj, const int lvl, CSRMatrixType& U)
{
    U.rows = size;
    U.cols = size;
    _Ui.resize(size);
    const auto base = ai[0];
    U.ResizeAI(size + 1);
    auto* u_ai = U.AI();
    u_ai[0] = base;
    const int reserve_size = 32;
    const COLTYPE not_visited = std::numeric_limits<COLTYPE>::max();
    _Ui.resize(size);
    const int chunk_size = 32;

#pragma omp parallel num_threads(_nthreads)
    {
        const int tid = omp_get_thread_num();
        auto& visited_thread = _visited[tid];
        visited_thread.resize(size);
        std::fill(visited_thread.begin(), visited_thread.end(), not_visited);

        auto& Q_thread = _Q[tid];
        auto& Q_next_thread = _Q_next[tid];
        Q_thread.reserve(reserve_size);
        Q_next_thread.reserve(reserve_size);

#pragma omp for schedule(dynamic, chunk_size)
        for (COLTYPE i = 0; i < size; i++)
        {
            Q_thread.clear();
            Q_next_thread.clear();
            Q_thread.push_back(i);
            int level = 0;
            visited_thread[i] = i;
            _Ui[i].clear();
            if constexpr (keepdiag)
            {
                _Ui[i].push_back(i + base);
            }
            while (level <= lvl && Q_thread.size())
            {
                for (const auto row : Q_thread)
                {
                    const ROWTYPE row_start = ai[row] - base;
                    const ROWTYPE row_end = ai[row + 1] - base;
                    for (auto adj_idx = row_start; adj_idx < row_end; ++adj_idx)
                    {
                        const COLTYPE col = aj[adj_idx] - base;
                        if (visited_thread[col] == i)
                            continue;
                        visited_thread[col] = i;
                        if (col < i)
                        {
                            Q_next_thread.push_back(col);
                        }
                        else if (col > i)
                        {
                            _Ui[i].push_back(col + base);
                        }
                    }
                }
                level++;
                std::swap(Q_thread, Q_next_thread);
            }
            if (lvl > 0)
                std::sort(_Ui[i].begin() + keepdiag, _Ui[i].end());
            u_ai[i + 1] = static_cast<ROWTYPE>(_Ui[i].size());
        }
    }
    utils::ParallelPrefixSumInplace(_nthreads, u_ai, u_ai + size + 1);
    const ROWTYPE nnz = u_ai[size] - base;
    U.ResizeAJ(nnz);
    U.ResizeAV(nnz);
    auto *u_aj = U.AJ();
    
    // Parallel copy _Ui to u_aj using prefix-based load balancing
#pragma omp parallel num_threads(_nthreads)
    {
        const int tid = omp_get_thread_num();
    auto [copy_start, copy_end] = utils::LoadPrefixBalancedPartitionPos(u_ai, u_ai + size, tid, _nthreads);
        
        for (COLTYPE i = copy_start; i < copy_end; i++)
        {
            const ROWTYPE row_start = u_ai[i] - base;
            const size_t num_bytes = _Ui[i].size() * sizeof(COLTYPE);
            std::memcpy(u_aj + row_start, _Ui[i].data(), num_bytes);
        }
    }
    return true;
}

template <ResizableDiagonal CSRMatrixType, bool keepdiag>
bool ILULevelSymbolicParallelL<CSRMatrixType, keepdiag>::operator()(const COLTYPE size,
                                                                    ROWTYPE const* ai, COLTYPE const* aj,
                                                                    const int lvl, CSRMatrixType& L)
{
    L.rows = size;
    L.cols = size;
    _Li.resize(size);
    const auto base = ai[0];
    L.ResizeAI(size + 1);
    auto* l_ai = L.AI();
    l_ai[0] = base;
    const int reserve_size = 64;
    const int chunk_size = 32;
    const COLTYPE not_visited = std::numeric_limits<COLTYPE>::max();
    const COLTYPE invalid_peak = std::numeric_limits<COLTYPE>::max();

#pragma omp parallel num_threads(_nthreads)
    {
        const int tid = omp_get_thread_num();
        auto& visited_thread = _visited[tid];
        visited_thread.resize(size);

        auto& added_thread = _added[tid];
        added_thread.resize(size);

        auto& Q_thread = _Q[tid];
        auto& Q_next_thread = _Q_next[tid];
        Q_thread.reserve(reserve_size);
        Q_next_thread.reserve(reserve_size);

        for (auto& node : visited_thread)
        {
            node.index = not_visited;
            node.peak = invalid_peak;
        }
        std::fill(added_thread.begin(), added_thread.end(), not_visited);

#pragma omp for schedule(dynamic, chunk_size)
        for (COLTYPE i = 0; i < size; i++)
        {
            Q_thread.clear();
            Q_next_thread.clear();
            _Li[i].clear();

            // Initialize Q with columns from row i where col < i
            for (ROWTYPE row_idx = ai[i] - base; row_idx < ai[i + 1] - base; ++row_idx)
            {
                const COLTYPE col = aj[row_idx] - base;
                if (col >= i)
                    break;
                Q_thread.push_back({col, col});
                _Li[i].push_back(col + base);
                auto& visited_col = visited_thread[col];
                visited_col.index = i;
                visited_col.peak = col;
                added_thread[col] = i;
            }

            int level = 0;
            while (level < lvl && !Q_thread.empty())
            {
                for (const auto& node : Q_thread)
                {
                    const COLTYPE idx = node.index;
                    const COLTYPE peak = node.peak;

                    // Expand to neighbors: iterate over row idx where col < i
                    for (ROWTYPE adj_idx = ai[idx] - base; adj_idx < ai[idx + 1] - base; ++adj_idx)
                    {
                        const COLTYPE k = aj[adj_idx] - base;
                        if (k >= i)
                            break;

                        auto& visited_k = visited_thread[k];
                        if (visited_k.index != i)
                        {
                            visited_k.index = i;
                            visited_k.peak = invalid_peak;
                        }

                        // Skip if we've already found a path with smaller or equal peak
                        if (visited_k.peak <= peak)
                            continue;

                        // Update peak for this node
                        visited_k.peak = peak;
                        const bool is_new = (added_thread[k] != i);

                        // Add to result if this is a new node and k > peak (k is the max in path)
                        if (is_new && k > peak)
                        {
                            added_thread[k] = i;
                            _Li[i].push_back(k + base);
                        }

                        // Enqueue for next level with updated peak
                        const COLTYPE new_peak = std::max(peak, k);
                        Q_next_thread.push_back({k, new_peak});
                    }
                }

                level++;
                std::swap(Q_thread, Q_next_thread);
                Q_next_thread.clear();
            }
            if (lvl > 0)
                std::sort(_Li[i].begin(), _Li[i].end());
            if constexpr (keepdiag)
            {
                _Li[i].push_back(i + base);
            }

            l_ai[i + 1] = static_cast<ROWTYPE>(_Li[i].size());
        }
    }

    utils::ParallelPrefixSumInplace(_nthreads, l_ai, l_ai + size + 1);
    const ROWTYPE nnz = l_ai[size] - base;
    L.ResizeAJ(nnz);
    L.ResizeAV(nnz);
    auto* l_aj = L.AJ();

    // Parallel copy _Li to l_aj using prefix-based load balancing
#pragma omp parallel num_threads(_nthreads)
    {
        const int tid = omp_get_thread_num();
        auto [copy_start, copy_end] = utils::LoadPrefixBalancedPartitionPos(l_ai, l_ai + size, tid, _nthreads);

        for (COLTYPE i = copy_start; i < copy_end; i++)
        {
            const ROWTYPE row_start = l_ai[i] - base;
            const size_t num_bytes = _Li[i].size() * sizeof(COLTYPE);
            std::memcpy(l_aj + row_start, _Li[i].data(), num_bytes);
        }
    }
    return true;
}

// Policy classes for sequential vs parallel elimination
struct SequentialPolicy
{
    SequentialPolicy(std::size_t /*size*/) {}
    
    template<typename COLT>
    void wait_for_row(COLT /*k*/) const {}
    
    template<typename COLT>
    void mark_row_ready(COLT /*i*/) {}
};

struct ParallelPolicy
{
    std::atomic<std::int32_t>* ready;
    
    ParallelPolicy(std::size_t size) : ready(nullptr) {}
    
    void set_ready_array(std::atomic<std::int32_t>* ready_arr)
    {
        ready = ready_arr;
    }
    
    template<typename COLT>
    void wait_for_row(COLT k) const
    {
        while (ready[k].load(std::memory_order_acquire) == 0)
        {
            // Spin-wait until row k is ready
        }
    }
    
    template<typename COLT>
    void mark_row_ready(COLT i)
    {
        ready[i].store(1, std::memory_order_release);
    }
};

// Helper struct for ILU row elimination (factorization without initialization)
template <typename ROWT, typename COLT, typename VALT, typename Policy = SequentialPolicy>
struct ILURowEliminator
{
    static constexpr ROWT MARKER_ABSENT = std::numeric_limits<ROWT>::max();
    
    // Workspace for marker array
    std::vector<ROWT> marker;
    Policy policy;
    
    ILURowEliminator(COLT size) : marker(size, MARKER_ABSENT), policy(size) {}
    
    // For parallel version: set the ready array
    void set_ready_array(std::atomic<std::int32_t>* ready_arr)
    {
        if constexpr (std::is_same_v<Policy, ParallelPolicy>)
        {
            policy.set_ready_array(ready_arr);
        }
    }
    
    // Perform elimination on row i using previously factorized rows
    // Assumes row i has been initialized with values from A
    // Returns false if singular pivot encountered
    bool eliminate_row(
        COLT i,
        ROWT base,
        ROWT row_start,
        ROWT row_end,
        // ILU matrix being built
        ROWT const* ilu_ai, COLT const* ilu_aj, VALT* ilu_av,
        ROWT const* ilu_diag)
    {
        // ---- Elimination: process L part entries (pivot columns k < i) ----
        for ( ROWT k_pos = row_start; k_pos < row_end; ++k_pos )
        {
            COLT k = ilu_aj[k_pos] - base;
            if ( k >= i )
                break; // reached diagonal / upper part
            if ( ilu_av[k_pos] == VALT( 0 ) )
                continue; // nothing to eliminate
            
            // Wait for row k to be ready (no-op in sequential, spin-wait in parallel)
            policy.wait_for_row(k);
            
            const VALT akk = ilu_av[ilu_diag[k] - base];
            if ( akk == VALT( 0 ) )
                return false; // singular pivot
            const VALT aik = ( ilu_av[k_pos] /= akk );

            // Iterate over U portion of row k: columns > k
            const ROWT k_u_begin = ( ilu_diag[k] - base ) + 1;
            const ROWT k_u_end = ilu_ai[k + 1] - base;
            for ( ROWT j_pos = k_u_begin; j_pos < k_u_end; ++j_pos )
            {
                COLT j = ilu_aj[j_pos] - base;
                ROWT pos_i = marker[j];
                if ( pos_i == MARKER_ABSENT )
                    continue; // fill not in numeric pattern (due to level dropping)
                // a_ij -= a_ik * a_kj (FMA form)
                ilu_av[pos_i] = std::fma( -aik, ilu_av[j_pos], ilu_av[pos_i] );
            }
        }

        // Clear marker entries touched in row i (make them ABSENT for next row)
        for ( ROWT pos = row_start; pos < row_end; ++pos )
        {
            COLT col = ilu_aj[pos] - base;
            marker[col] = MARKER_ABSENT;
        }
        
        // Mark this row as ready (no-op in sequential)
        policy.mark_row_ready(i);
        
        return true;
    }
};

template <ResizableDiagonal CSRMatrixType>
bool ILULevelNumeric( const typename CSRMatrixType::COLTYPE size,
                      typename CSRMatrixType::ROWTYPE const* ai,
                      typename CSRMatrixType::COLTYPE const* aj,
                      typename CSRMatrixType::VALTYPE const* av,
                      const int lvl,
                      CSRMatrixType& ilu )
{
    const auto base = ai[0];
    auto const* ilu_ai = ilu.AI();
    auto const* ilu_aj = ilu.AJ();
    auto* ilu_av = ilu.AV();
    auto const* ilu_diag = ilu.Diagonal();

    using ROWT = typename CSRMatrixType::ROWTYPE;
    using COLT = typename CSRMatrixType::COLTYPE;
    using VALT = typename CSRMatrixType::VALTYPE;

    // Create row eliminator with workspace
    ILURowEliminator<ROWT, COLT, VALT> eliminator(size);

    for ( COLT i = 0; i < size; i++ )
    {
        // ---- Initialize row i entries ----
        const ROWT row_start = ilu_ai[i] - base;
        const ROWT row_end = ilu_ai[i + 1] - base;
        ROWT a_pos = ai[i] - base;
        const ROWT a_row_end = ai[i + 1] - base;
        
        // Build marker for current row and copy values from A
        for ( ROWT pos = row_start; pos < row_end; ++pos )
        {
            COLT col = ilu_aj[pos] - base;
            eliminator.marker[col] = pos; // record position
            if ( a_pos == a_row_end || aj[a_pos] != ilu_aj[pos] )
            {
                ilu_av[pos] = VALT( 0 );
            }
            else
            {
                ilu_av[pos] = av[a_pos++ - base];
            }
        }

        // Perform elimination
        if (!eliminator.eliminate_row(i, base, row_start, row_end, ilu_ai, ilu_aj, ilu_av, ilu_diag))
            return false;
    }
    return true;
}

template <ResizableDiagonal CSRMatrixType>
bool ICCLevelSymbolicParallel<CSRMatrixType>::operator()( const COLTYPE size,
                                                          ROWTYPE const* ai,
                                                          COLTYPE const* aj,
                                                          const int lvl,
                                                          CSRMatrixType& L )
{
    _Li.resize( size );
    const auto base = ai[0];
    L.ResizeAI( size + 1 );
    L.rows = size;
    L.cols = size;
    L.AI()[0] = base;
    std::atomic<COLTYPE> counter( 0 );
    const int chunk_size = 32;
#pragma omp parallel num_threads( _num_threads )
    {
        const int tid = omp_get_thread_num();
        _visited[tid].resize( size );
        std::fill( _visited[tid].begin(), _visited[tid].end(), 0 );
        _Li_path_max[tid].resize( size );

#pragma omp for schedule( dynamic, chunk_size )
        for ( COLTYPE i = 0; i < size; i++ )
        {
            counter++;
#pragma omp critical
            {
                utils::printProgress( counter * 1. / size );
            }
            _Q[tid].clear();
            _Q_next[tid].clear();
            _Li[i].clear();
            for ( auto j_idx = ai[i] - base; j_idx < ai[i + 1] - base; j_idx++ )
            {
                const auto j = aj[j_idx] - base;
                if ( j > i )
                {
                    break;
                }
                _Q[tid][j] = j;
            }
            const COLTYPE visited_token = i + 1;
            int level = 0;
            while ( level <= lvl && !_Q[tid].empty() )
            {
                // #pragma omp critical
                //         {
                //           std::cout << "thread " << tid << " total threads " <<
                //           nthreads
                //                     << " processing row " << i << " level " << level
                //                     << " Q size " << _Q[tid].size() << std::endl;
                //         }
                for ( const auto& [k, path_max] : _Q[tid] )
                {
                    // skip if the the destination has been visited with a
                    // smaller path_max
                    if ( _visited[tid][k] == visited_token && _Li_path_max[tid][k] <= path_max )
                    {
                        continue;
                    }
                    if ( _visited[tid][k] != visited_token )
                    {
                        _visited[tid][k] = visited_token;
                    }
                    if ( path_max == k )
                        _Li[i].push_back( k );
                    _Li_path_max[tid][k] = path_max;
                    if ( level == lvl )
                    {
                        continue;
                    }

                    // iterative k->j paths
                    for ( auto j_idx = ai[k] - base; j_idx < ai[k + 1] - base; j_idx++ )
                    {
                        auto j = aj[j_idx] - base;
                        if ( j >= i )
                        {
                            break;
                        }
                        const COLTYPE kj_path_max = std::max( j, path_max );
                        auto it = _Q_next[tid].find( j );
                        if ( it != _Q_next[tid].end() )
                        {
                            it->second = std::min( it->second, kj_path_max );
                        }
                        else
                        {
                            _Q_next[tid][j] = kj_path_max;
                        }
                    }
                }
                _Q[tid].swap( _Q_next[tid] );
                _Q_next[tid].clear();
                level++;
            }
            std::sort( _Li[i].begin(), _Li[i].end() );
            L.AI()[i + 1] = _Li[i].size();
        }
#pragma omp single
        {
            for ( COLTYPE i = 0; i < size; i++ )
            {
                L.AI()[i + 1] += L.AI()[i];
            }
            L.ResizeAJ( L.AI()[size] - base );
            L.ResizeAV( L.AI()[size] - base );
        }

        auto [start, end] = utils::LoadPrefixBalancedPartitionPos(
            L.AI(), L.AI() + size, tid, _num_threads );
        for ( auto i = start; i < end; i++ )
        {
            ROWTYPE pos = L.AI()[i] - base;
            for ( const auto& s : _Li[i] )
            {
                L.AJ()[pos++] = s + base;
            }
        }
    }

    return true;
}

template <ResizableDiagonal CSRMatrixType>
bool ICCLevelNumericFixedPoint<CSRMatrixType>::operator()( const COLTYPE size,
                                                           ROWTYPE const* ai,
                                                           COLTYPE const* aj,
                                                           VALTYPE const* av,
                                                           CSRMatrixType& L )
{
    _av.resize( L.NNZ() );
    _ai.resize( L.NNZ() );
    _L_av_init.resize( L.NNZ() );
    // std::fill(_L_av_init.begin(), _L_av_init.end(), 0);
    _L_av_next.resize( L.NNZ() );
    std::fill( _L_av_next.begin(), _L_av_next.end(), 0 );
    const auto base = ai[0];
    assert( base == L.AI()[0] );

#pragma omp parallel num_threads( _num_threads )
    {
        const int tid = omp_get_thread_num();

        // copy av to _av according to L's sparsity pattern
        // and initialize _L_av_init
        auto [start, end] = utils::LoadPrefixBalancedPartitionPos(
            L.AI(), L.AI() + size, tid, _num_threads );
        for ( auto i = start; i < end; i++ )
        {
            ROWTYPE i_idx = ai[i] - base;
            VALTYPE sum_square = 0;
            ROWTYPE ilu_i_idx;
            for ( ilu_i_idx = L.AI()[i] - base; ilu_i_idx < L.AI()[i + 1] - base; ilu_i_idx++ )
            {
                _ai[ilu_i_idx] = i + base;
                if ( i_idx == ai[i + 1] - base || aj[i_idx] != L.AJ()[ilu_i_idx] )
                {
                    _av[ilu_i_idx] = 0; // initialize to zero
                    _L_av_init[ilu_i_idx] = 0;
                }
                else
                {
                    _av[ilu_i_idx] = av[i_idx]; // copy the value
                    if ( i == aj[i_idx] - base )
                        _av[ilu_i_idx] *= 2;
                    _L_av_init[ilu_i_idx] = _av[ilu_i_idx];
                    sum_square += _av[ilu_i_idx] * _av[ilu_i_idx];
                    i_idx++;
                }
            }
            assert( L.AJ()[ilu_i_idx - 1] == i + base );
            assert( sum_square > 0 );
            const VALTYPE weight = std::sqrt( _av[ilu_i_idx - 1] / sum_square );

            for ( ROWTYPE ilu_i_idx = L.AI()[i] - base;
                  ilu_i_idx < L.AI()[i + 1] - base; ilu_i_idx++ )
            {
                _L_av_init[ilu_i_idx] *= weight;
            }
        }
    }
    bool success = true;
    for ( int sweep = 0; sweep < _sweeps; sweep++ )
    {
#pragma omp parallel for num_threads( _num_threads )
        for ( ROWTYPE idx = 0; idx < L.NNZ(); idx++ )
        {
            if ( !success )
            {
                continue;
            }
            VALTYPE s = _av[idx];
            COLTYPE i = _ai[idx] - base;
            COLTYPE j = L.AJ()[idx] - base;
            ROWTYPE j_idx, i_idx;
            for ( j_idx = L.AI()[j] - base, i_idx = L.AI()[i] - base;
                  j_idx < L.AI()[j + 1] - base - 1 && i_idx < L.AI()[i + 1] - base; )
            {
                if ( L.AJ()[j_idx] == L.AJ()[i_idx] )
                {
                    s -= _L_av_init[j_idx] * _L_av_init[i_idx];
                    j_idx++;
                    i_idx++;
                }
                else if ( L.AJ()[j_idx] < L.AJ()[i_idx] )
                {
                    j_idx++;
                }
                else
                {
                    i_idx++;
                }
            }
            assert( L.AJ()[j_idx] == j + base );
            if ( i != j )
            {
                _L_av_next[idx] = s / _L_av_init[j_idx];
                if ( std::isnan( _L_av_next[idx] ) || std::isinf( _L_av_next[idx] ) )
                {
                    success = false;
                }
            }
            else
            {
                if ( s <= 0 )
                {
                    s = 1e-8;
                    // success = false;
                    // #pragma omp critical
                    //           {
                    //             std::cout << "Non-positive pivot encountered: " << s
                    //                       << ", i = " << i << ", j = " << j <<
                    //                       std::endl;
                    //           }
                }
                _L_av_next[idx] = std::sqrt( s );
            }
        }
        std::swap( _L_av_init, _L_av_next );
    }
    std::copy( _L_av_init.begin(), _L_av_init.end(), L.AV() );
    if ( !success )
        std::cout << "ICC did not converge!" << std::endl;
    return success;
}

template <ResizableDiagonal CSRMatrixType>
bool ILUTNumeric( const typename CSRMatrixType::COLTYPE size,
                  typename CSRMatrixType::ROWTYPE const* ai,
                  typename CSRMatrixType::COLTYPE const* aj,
                  typename CSRMatrixType::VALTYPE const* av,
                  const typename CSRMatrixType::VALTYPE tau,
                  CSRMatrixType& ilu )
{
    const auto base = ai[0];
    ilu.rows = size;
    ilu.cols = size;
    ilu.ResizeAI( size + 1 );
    ilu.ResizeDiagonal( size );
    auto* ilu_ai = ilu.AI();
    auto* ilu_diag = ilu.Diagonal();
    ilu_ai[0] = base;

    using ROWT = typename CSRMatrixType::ROWTYPE;
    using COLT = typename CSRMatrixType::COLTYPE;
    using VALT = typename CSRMatrixType::VALTYPE;

    using MarkerT = COLT;
    const MarkerT MARKER_ABSENT = std::numeric_limits<MarkerT>::max();
    std::vector<MarkerT> marker( size, MARKER_ABSENT );

    const ROWT nnz_a = ai[size] - base;
    ROWT nnz_cap = std::max<ROWT>( ROWT( 1 ), nnz_a );
    ilu.ResizeAJ( nnz_cap );
    ilu.ResizeAV( nnz_cap );
    auto* ilu_aj = ilu.AJ();
    auto* ilu_av = ilu.AV();

    if ( size == COLT( 0 ) )
        return true;

    const size_t reserve_hint = static_cast<size_t>( std::max<ROWT>(
        ROWT( 8 ), ( nnz_a / static_cast<ROWT>( size ) ) + ROWT( 4 ) ) );

    struct Entry
    {
        COLT col;
        VALT val;
    };

    std::vector<COLT> lower_cols;
    std::vector<Entry> row_entries;
    using LowerColsDiffT = typename std::vector<COLT>::difference_type;
    lower_cols.reserve( reserve_hint );
    row_entries.reserve( reserve_hint );

    for ( COLT i = 0; i < size; ++i )
    {
        row_entries.clear();
        lower_cols.clear();
        // L2 norm (Euclidean) of the ORIGINAL (unfactored) row i from A
        VALT original_row_l2_sq = VALT( 0 );

        // gather original row pattern and track max
        for ( ROWT pos = ai[i] - base; pos < ai[i + 1] - base; ++pos )
        {
            const COLT col = aj[pos] - base;
            marker[col] = static_cast<MarkerT>( row_entries.size() );
            const VALT val = av[pos];
            row_entries.push_back( { col, val } );
            original_row_l2_sq = std::fma( val, val, original_row_l2_sq );
            if ( col < i )
                lower_cols.push_back( col );
        }

        const VALT drop_tol = std::sqrt( original_row_l2_sq ) * tau;

        if ( marker[i] == MARKER_ABSENT )
        {
            marker[i] = static_cast<MarkerT>( row_entries.size() );
            row_entries.push_back( { i, VALT( 0 ) } );
        }

        // lower_cols is already sorted because the input CSR rows are sorted
        // and we insert new entries using upper_bound below.
        size_t lower_idx = 0;

        while ( lower_idx < lower_cols.size() )
        {
            const COLT k = lower_cols[lower_idx++];
            const MarkerT pos_k = marker[k];
            VALT& aik = row_entries[pos_k].val;
            if ( aik == VALT( 0 ) )
                continue;

            const ROWT diag_offset = ilu_diag[k] - base;
            const VALT akk = ilu_av[diag_offset];
            if ( akk == VALT( 0 ) ){
                std::cerr << "Error: zero diagonal in ILU preconditioner.\n";
                return false;
            }
            aik /= akk;
            if ( std::abs( aik ) < drop_tol )
            {
                aik = VALT( 0 );
                continue;
            }
            // dropping now based only on ORIGINAL row L2 norm; no need to track max during elimination

            const ROWT k_u_begin = diag_offset + 1;
            const ROWT k_u_end = ilu_ai[k + 1] - base;
            for ( ROWT j_pos = k_u_begin; j_pos < k_u_end; ++j_pos )
            {
                const COLT j = ilu_aj[j_pos] - base;
                const VALT u_kj = ilu_av[j_pos];
                MarkerT row_pos = marker[j];
                if ( row_pos == MARKER_ABSENT )
                {
                    row_pos = static_cast<MarkerT>( row_entries.size() );
                    marker[j] = row_pos;
                    const VALT new_val = -aik * u_kj;
                    row_entries.push_back( { j, new_val } );
                    // no row_max update (using original L2)
#if defined( ILUT_MANUAL_LOWER_INSERT )
                    if ( j < i )
                    {
                        const auto prev_size = lower_cols.size();
                        lower_cols.resize( prev_size + 1 );
                        auto insert_pos = static_cast<LowerColsDiffT>( prev_size );
                        while ( insert_pos >
                                static_cast<LowerColsDiffT>( lower_idx ) &&
                                lower_cols[insert_pos - 1] > j )
                        {
                            lower_cols[insert_pos] = lower_cols[insert_pos - 1];
                            --insert_pos;
                        }
                        lower_cols[insert_pos] = j;
                    }
#else
                    if ( j < i )
                    {
                        auto it = std::upper_bound(
                            lower_cols.begin() + static_cast<LowerColsDiffT>( lower_idx ),
                            lower_cols.end(), j );
                        lower_cols.insert( it, j );
                    }
#endif
                }
                else
                {
                    row_entries[row_pos].val =
                        std::fma( -aik, u_kj, row_entries[row_pos].val );
                    // no row_max update (using original L2)
                }
            }
        }

        bool diag_present = false;
        VALT diag_value = VALT( 0 );
        size_t compact_pos = 0;
        for ( size_t idx = 0; idx < row_entries.size(); ++idx )
        {
            const Entry entry = row_entries[idx];
            const COLT col = entry.col;
            const VALT val = entry.val;
            marker[col] = MARKER_ABSENT;

            if ( col == i )
            {
                diag_present = true;
                diag_value = val;
            }
            else if ( std::abs( val ) <= drop_tol ){
                continue;
            }

            row_entries[compact_pos++] = entry;
        }
        row_entries.resize( compact_pos );

        if ( !diag_present ){
            std::cerr << "Error: missing diagonal in ILU preconditioner.\n";
            return false;
        }

        std::sort( row_entries.begin(), row_entries.end(),
                   []( const Entry& a, const Entry& b ) { return a.col < b.col; } );

        const ROWT write_pos = ilu_ai[i] - base;
        const ROWT row_nnz = static_cast<ROWT>( row_entries.size() );
        const ROWT needed = write_pos + row_nnz;
        if ( needed > nnz_cap )
        {
            nnz_cap = std::max<ROWT>( needed, nnz_cap * 2 );
            ilu_aj = ilu.ResizeAJ( nnz_cap );
            ilu_av = ilu.ResizeAV( nnz_cap );
        }

        ROWT out_pos = write_pos;
        for ( const auto& entry : row_entries )
        {
            ilu_aj[out_pos] = entry.col + base;
            ilu_av[out_pos] = entry.val;
            if ( entry.col == i )
                ilu_diag[i] = out_pos + base;
            ++out_pos;
        }

        ilu_ai[i + 1] = out_pos + base;
    }
    return true;
}

template void ICCLevel0SymSymbolic<CSRMatrix<int, int, double>>(
    const int rows, int const* ai, int const* aj, CSRMatrix<int, int, double>& icc );

template void ICCLevelSymbolic0<CSRMatrix<int, int, double>>(
    const int rows,
    int const* ai,
    int const* aj,
    int const* diag_pos,
    const int lvl,
    CSRMatrix<int, int, double>& icc );

template void ICCLevelSymbolic1<CSRMatrix<int, int, double>>(
    const int rows,
    int const* ai,
    int const* aj,
    int const* diag_pos,
    const int lvl,
    CSRMatrix<int, int, double>& icc );

template void ICCLevelSymbolic2<CSRMatrix<int, int, double>>(
    const int rows,
    int const* ai,
    int const* aj,
    int const* diag_pos,
    const int lvl,
    CSRMatrix<int, int, double>& icc );

template void ICCLevelSymbolic3<CSRMatrix<int, int, double>>(
    const int rows,
    int const* ai,
    int const* aj,
    int const* diag_pos,
    const int lvl,
    CSRMatrix<int, int, double>& icc );

template bool ICCLevelNumeric<int, int, double>( const int rows,
                                                 int const* ai,
                                                 int const* aj,
                                                 double const* av,
                                                 int const* diag_pos,
                                                 const int lvl,
                                                 const double omega,
                                                 int const* icc_ai,
                                                 int const* icc_aj,
                                                 double* icc_av );

template class ICCLevelSymbolicParallel<matrix_utils::CSRMatrix<int, int, double>>;
template class ICCLevelNumericFixedPoint<matrix_utils::CSRMatrix<int, int, double>>;
template bool ILULevel0Symbolic<matrix_utils::CSRMatrix<int, int, double>>(
    const int size,
    int const* ai,
    int const* aj,
    matrix_utils::CSRMatrix<int, int, double>& ilu );
template class ILULevelSymbolic<matrix_utils::CSRMatrix<int, int, double>>;
template class ILULevelSymbolicParallelU<matrix_utils::CSRMatrix<int, int, double>, false>;
template class ILULevelSymbolicParallelU<matrix_utils::CSRMatrix<int, int, double>, true>;
template class ILULevelSymbolicParallelL<matrix_utils::CSRMatrix<int, int, double>, false>;
template class ILULevelSymbolicParallelL<matrix_utils::CSRMatrix<int, int, double>, true>;
template bool ILULevelNumeric<matrix_utils::CSRMatrix<int, int, double>>(
    const int size,
    int const* ai,
    int const* aj,
    double const* av,
    const int lvl,
    matrix_utils::CSRMatrix<int, int, double>& ilu );
template bool ILUTNumeric<matrix_utils::CSRMatrix<int, int, double>>(
    const int size,
    int const* ai,
    int const* aj,
    double const* av,
    const double tau,
    matrix_utils::CSRMatrix<int, int, double>& ilu );
} // namespace matrix_utils
