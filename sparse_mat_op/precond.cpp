#include "precond.hpp"
#include "matrix_utils.hpp"
#include <algorithm>
#include <deque>
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
    typename CSRMatrixType::ROWTYPE i_idx, ilu_i_idx, k_idx, j_idx2, j_idx;
    typename CSRMatrixType::COLTYPE j, k;

    auto const* ilu_ai = ilu.AI();
    auto const* ilu_aj = ilu.AJ();
    auto* ilu_av = ilu.AV();
    auto const* ilu_diag = ilu.Diagonal();
    typename CSRMatrixType::VALTYPE akk, aik;

    for ( typename CSRMatrixType::COLTYPE i = 0; i < size; i++ )
    {
        // initialize the current row's nonzeros
        i_idx = ai[i] - base;
        for ( ilu_i_idx = ilu_ai[i] - base; ilu_i_idx < ilu_ai[i + 1] - base; ilu_i_idx++ )
        {
            if ( i_idx == ai[i + 1] - base || aj[i_idx] != ilu_aj[ilu_i_idx] )
            {
                ilu_av[ilu_i_idx] = 0; // initialize to zero
            }
            else
            {
                ilu_av[ilu_i_idx] = av[i_idx++]; // copy the value
            }
        }
        k_idx = ilu_ai[i] - base;
        while ( true )
        {
            k = ilu_aj[k_idx] - base;
            if ( k >= i )
            {
                break;
            }
            akk = ilu_av[ilu_diag[k] - base];
            if ( akk == 0 )
            {
                // akk = ilu_av[ilu_diag[k] - base] = 1e-16;
                return false;
            }
            ilu_av[k_idx] /= akk; // a_{ik} = a_{ik} / a_{kk}
            aik = ilu_av[k_idx];

            j_idx2 = ++k_idx; // j_idx2 is for ith row, start after current k element
            for ( j_idx = ilu_diag[k] - base + 1; j_idx < ilu_ai[k + 1] - base; )
            {
                if ( ilu_aj[j_idx] == ilu_aj[j_idx2] )
                {
                    ilu_av[j_idx2++] -= aik * ilu_av[j_idx++];
                }
                else if ( ilu_aj[j_idx] < ilu_aj[j_idx2] )
                {
                    j_idx++;
                }
                else
                {
                    j_idx2++;
                }
            }
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

template class ICCLevelSymbolicSerial3<matrix_utils::CSRMatrix<int, int, double>>;
template class ILULevelSymbolic<matrix_utils::CSRMatrix<int, int, double>>;
template bool ILULevelNumeric<matrix_utils::CSRMatrix<int, int, double>>(
    const int size,
    int const* ai,
    int const* aj,
    double const* av,
    const int lvl,
    matrix_utils::CSRMatrix<int, int, double>& ilu );
} // namespace matrix_utils