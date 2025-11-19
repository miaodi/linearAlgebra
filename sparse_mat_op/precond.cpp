#include "precond.hpp"
#include "matrix_utils.hpp"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <deque>
#include <iomanip>
#include <numeric>
#include <ranges>

namespace matrix_utils
{
template <ResizableDiagonalType CSRMatrixType>
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

template void ICCLevel0SymSymbolic<int, int, CSRMatrix<int, int, double>>(
    const int rows, int const* ai, int const* aj, CSRMatrix<int, int, double>& icc );

template void ICCLevelSymbolic0<int, int, CSRMatrix<int, int, double>>(
    const int rows,
    int const* ai,
    int const* aj,
    int const* diag_pos,
    const int lvl,
    CSRMatrix<int, int, double>& icc );

template void ICCLevelSymbolic1<int, int, CSRMatrix<int, int, double>>(
    const int rows,
    int const* ai,
    int const* aj,
    int const* diag_pos,
    const int lvl,
    CSRMatrix<int, int, double>& icc );

template void ICCLevelSymbolic2<int, int, CSRMatrix<int, int, double>>(
    const int rows,
    int const* ai,
    int const* aj,
    int const* diag_pos,
    const int lvl,
    CSRMatrix<int, int, double>& icc );

template void ICCLevelSymbolic3<int, int, CSRMatrix<int, int, double>>(
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

template <ResizableDiagonalType CSRMatrixType>
bool ICCLevelSymbolicSerial<CSRMatrixType>::operator()( const COLTYPE size,
                                                        ROWTYPE const* ai,
                                                        COLTYPE const* aj,
                                                        const int lvl,
                                                        CSRMatrixType& L )
{
    _Q.resize( 64 );
    const auto base = ai[0];
    L.ResizeAI( size + 1 );
    ROWTYPE nnz = ai[size] - base;
    std::cout << "nnz: " << nnz << std::endl;
    L.ResizeAJ( nnz );
    L.rows = size;
    L.cols = size;
    L.AI()[0] = base;
    for ( COLTYPE i = 0; i < size; i++ )
    {
        _S.clear();
        _Q.push( std::make_pair( i, 0 ) );
        _S[i] = 0;
        int level = 0;
        // _level.clear();
        // _level[i] = 0;
        while ( level <= lvl )
        {
            for ( auto lvl_size = _Q.size(); lvl_size > 0; lvl_size-- )
            {
                auto [k, path_max] = _Q.shift();
                // if (level > _level[k] && path_max != _S[k]) {
                //   continue;
                // }
                for ( auto j_idx = ai[k] - base; j_idx < ai[k + 1] - base; j_idx++ )
                {
                    auto j = aj[j_idx] - base;
                    if ( j >= i )
                    {
                        break;
                    }
                    COLTYPE j_path_max = std::max( j, path_max );
                    if ( !_S.contains( j ) || _S[j] > j_path_max )
                    {
                        _S[j] = j_path_max;
                        if ( level < lvl )
                        {
                            if ( _Q.isFull() )
                            {
                                std::cout << "resize _Q" << std::endl;
                                _Q.resizePreserve( _Q.size() * 2 );
                            }
                            _Q.push( std::make_pair( j, j_path_max ) );
                            // _level[j] = level;
                        }
                    }
                }
            }
            level++;
        }
        ROWTYPE pos = L.AI()[i] - base;
        if ( nnz < pos + _S.size() )
        {
            if ( 2 * ( i - base ) >= size )
                nnz *= 2;
            else
                nnz = nnz * std::ceil( size * 1. / ( i - base ) );
            L.ResizeAJ( nnz );
        }
        for ( const auto& s : _S )
        {
            if ( s.second <= s.first )
            {
                L.AJ()[pos++] = s.first + base;
            }
        }
        L.AI()[i + 1] = pos + base;
    }
    L.ResizeAV( L.AI()[size] - base );
    return true;
}

template <ResizableDiagonalType CSRMatrixType>
bool ICCLevelSymbolicSerial2<CSRMatrixType>::operator()( const COLTYPE size,
                                                         ROWTYPE const* ai,
                                                         COLTYPE const* aj,
                                                         const int lvl,
                                                         CSRMatrixType& L )
{
    const auto base = ai[0];
    L.ResizeAI( size + 1 );
    ROWTYPE nnz = ai[size] - base;
    _Q_temp.clear();
    L.ResizeAJ( nnz );
    L.rows = size;
    L.cols = size;
    L.AI()[0] = base;
    for ( COLTYPE i = 0; i < size; i++ )
    {
        _Q.clear();
        _Q[i] = 0;
        _S.clear();
        _S[i] = 0;
        int level = 0;
        while ( level <= lvl )
        {
            for ( const auto& [k, path_max] : _Q )
            {
                for ( auto j_idx = ai[k] - base; j_idx < ai[k + 1] - base; j_idx++ )
                {
                    auto j = aj[j_idx] - base;
                    if ( j >= i )
                    {
                        break;
                    }
                    COLTYPE j_path_max = std::max( j, path_max );
                    if ( !_S.contains( j ) || _S[j] > j_path_max )
                    {
                        _S[j] = j_path_max;
                        if ( level < lvl )
                        {
                            _Q_temp[j] = j_path_max;
                        }
                    }
                }
            }
            _Q.clear();
            _Q.swap( _Q_temp );
            level++;
        }
        ROWTYPE pos = L.AI()[i] - base;
        if ( nnz < pos + _S.size() )
        {
            if ( 2 * ( i - base ) >= size )
                nnz *= 2;
            else
                nnz = nnz * std::ceil( size * 1. / ( i - base ) );
            L.ResizeAJ( nnz );
        }
        for ( const auto& s : _S )
        {
            if ( s.second <= s.first )
            {
                L.AJ()[pos++] = s.first + base;
            }
        }
        L.AI()[i + 1] = pos + base;
    }
    L.ResizeAV( L.AI()[size] - base );
    return true;
}

template <ResizableDiagonalType CSRMatrixType>
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
    const ROWT MARKER_ABSENT = std::numeric_limits<ROWT>::max();
    // Reusable marker: position of column j in current row i (or ABSENT)
    std::vector<ROWT> marker( size, MARKER_ABSENT );

    for ( COLT i = 0; i < size; i++ )
    {
        // ---- Initialize row i entries ----
        const ROWT row_start = ilu_ai[i] - base;
        const ROWT row_end = ilu_ai[i + 1] - base;
        ROWT a_pos = ai[i] - base;
        const ROWT a_row_end = ai[i + 1] - base;
        // build marker for current row
        for ( ROWT pos = row_start; pos < row_end; ++pos )
        {
            COLT col = ilu_aj[pos] - base;
            marker[col] = pos; // record position
            if ( a_pos == a_row_end || aj[a_pos] != ilu_aj[pos] )
            {
                ilu_av[pos] = VALT( 0 );
            }
            else
            {
                ilu_av[pos] = av[a_pos++ - base];
            }
        }

        // ---- Elimination: process L part entries (pivot columns k < i) ----
        for ( ROWT k_pos = row_start; k_pos < row_end; ++k_pos )
        {
            COLT k = ilu_aj[k_pos] - base;
            if ( k >= i )
                break; // reached diagonal / upper part
            if ( ilu_av[k_pos] == VALT( 0 ) )
                continue; // nothing to eliminate
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
    }
    return true;
}

template class ICCLevelSymbolicSerial<matrix_utils::CSRMatrix<int, int, double>>;
template class ICCLevelSymbolicSerial2<matrix_utils::CSRMatrix<int, int, double>>;

template <ResizableDiagonalType CSRMatrixType>
bool ICCLevelSymbolicSerial3<CSRMatrixType>::operator()( const COLTYPE size,
                                                         ROWTYPE const* ai,
                                                         COLTYPE const* aj,
                                                         const int lvl,
                                                         CSRMatrixType& L )
{
    _visited.resize( size );
    std::fill( _visited.begin(), _visited.end(), 0 );
    _Li_path_max.resize( size );
    const auto base = ai[0];
    L.ResizeAI( size + 1 );
    ROWTYPE nnz = ai[size] - base;
    L.ResizeAJ( nnz );
    L.rows = size;
    L.cols = size;
    L.AI()[0] = base;
    COLTYPE visited_token = 0;
    for ( COLTYPE i = 0; i < size; i++ )
    {
        _Q.clear();
        _Q_next.clear();
        for ( auto j_idx = ai[i] - base; j_idx < ai[i + 1] - base; j_idx++ )
        {
            const auto j = aj[j_idx] - base;
            if ( j > i )
            {
                break;
            }
            _Q[j] = j;
        }

        visited_token++;
        _Li.clear();

        int level = 0;
        while ( level <= lvl )
        {
            for ( const auto& [k, path_max] : _Q )
            {
                // skip if the the destination has been visited with a smaller path_max
                if ( _visited[k] == visited_token && _Li_path_max[k] <= path_max )
                {
                    continue;
                }
                if ( _visited[k] != visited_token )
                {
                    _visited[k] = visited_token;
                }
                if ( path_max == k )
                    _Li.push_back( k );

                _Li_path_max[k] = path_max;
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
                    auto it = _Q_next.find( j );
                    if ( it != _Q_next.end() )
                    {
                        it->second = std::min( it->second, kj_path_max );
                    }
                    else
                    {
                        _Q_next[j] = kj_path_max;
                    }
                }
            }

            _Q.swap( _Q_next );
            _Q_next.clear();
            level++;
        }

        ROWTYPE pos = L.AI()[i] - base;
        if ( nnz < pos + _Li.size() )
        {
            if ( 2 * i >= size )
                nnz *= 2;
            else
                nnz = nnz * std::ceil( size * 1. / i );
            L.ResizeAJ( nnz );
        }

        std::sort( _Li.begin(), _Li.end() );
        for ( const auto& s : _Li )
        {
            L.AJ()[pos++] = s + base;
        }
        L.AI()[i + 1] = pos + base;
    }
    L.ResizeAV( L.AI()[size] - base );
    return true;
}

template <ResizableDiagonalType CSRMatrixType>
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
#pragma omp parallel num_threads( _num_threads )
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        _visited[tid].resize( size );
        std::fill( _visited[tid].begin(), _visited[tid].end(), 0 );
        _Li_path_max[tid].resize( size );

#pragma omp for schedule( dynamic, 100 )
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

template <ResizableDiagonalType CSRMatrixType>
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

template <ResizableDiagonalType CSRMatrixType>
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
        VALT row_max = VALT( 0 );

        // gather original row pattern and track max
        for ( ROWT pos = ai[i] - base; pos < ai[i + 1] - base; ++pos )
        {
            const COLT col = aj[pos] - base;
            marker[col] = static_cast<MarkerT>( row_entries.size() );
            const VALT val = av[pos];
            row_entries.push_back( { col, val } );
            row_max = std::max( row_max, std::abs( val ) );
            if ( col < i )
                lower_cols.push_back( col );
        }

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
            row_max = std::max( row_max, std::abs( aik ) );

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
                    row_max = std::max( row_max, std::abs( new_val ) );
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
                    row_max = std::max( row_max, std::abs( row_entries[row_pos].val ) );
                }
            }
        }

        const VALT drop_tol = row_max * tau;

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

template class ICCLevelSymbolicSerial3<matrix_utils::CSRMatrix<int, int, double>>;
template class ICCLevelSymbolicParallel<matrix_utils::CSRMatrix<int, int, double>>;
template class ICCLevelNumericFixedPoint<matrix_utils::CSRMatrix<int, int, double>>;
template class ILULevelSymbolic<matrix_utils::CSRMatrix<int, int, double>>;
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
