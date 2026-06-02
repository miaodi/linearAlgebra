#include "lu_symbolic.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace factorization
{

template <matrix_utils::AppendableCSR CSRMatrixType>
bool SymbolicLUEdags<CSRMatrixType>::apply( const COLTYPE nnodes, ROWTYPE const* ai, COLTYPE const* aj, CSRMatrixType& lu )
{
    if ( !initialize( nnodes, ai, aj ) )
    {
        return false;
    }

    lu.rows = nnodes;
    lu.cols = nnodes;
    lu.aj.clear();
    lu.av.clear();
    lu.ResizeAI( static_cast<std::size_t>( nnodes ) + 1 );

    for ( COLTYPE row = 0; row < nnodes; ++row )
    {
        lu.AI()[row] = static_cast<ROWTYPE>( lu.aj.size() ) + _base;

        appendLowerRowPattern( row, ai, aj, lu );
        const std::size_t lower_begin = static_cast<std::size_t>( lu.AI()[row] - _base );
        const std::size_t lower_end = lu.aj.size();
        reduceLowerEdagRow( row, lower_begin == lower_end ? nullptr : lu.aj.data() + lower_begin,
                            lower_begin == lower_end ? nullptr : lu.aj.data() + lower_end );

        appendUpperRowPattern( row, ai, aj, lu );
        const std::size_t upper_end = lu.aj.size();
        lu.AI()[row + 1] = static_cast<ROWTYPE>( upper_end ) + _base;
        extendUpperEdagColumn( row );
    }

    packUpperEdag();
    lu.av.resize( lu.aj.size() );
    return true;
}

template <matrix_utils::AppendableCSR CSRMatrixType>
bool SymbolicLUEdags<CSRMatrixType>::initialize( const COLTYPE nnodes, ROWTYPE const* ai, COLTYPE const* aj )
{
    if ( nnodes <= 0 || ai == nullptr || aj == nullptr )
    {
        return false;
    }

    _nnodes = nnodes;
    _base = ai[0];
    _diag.resize( nnodes );

    using VALTYPE = typename CSRMatrixType::VALTYPE;
    const bool has_diagonal = matrix_utils::Diagonal( nnodes, ai, aj, static_cast<VALTYPE const*>( nullptr ),
                                                      _diag.data(), static_cast<VALTYPE*>( nullptr ) );
    if ( !has_diagonal )
    {
        return false;
    }

    _reachVisited.assign( static_cast<std::size_t>( nnodes ), COLTYPE{} );
    _reduceVisited.assign( static_cast<std::size_t>( nnodes ), COLTYPE{} );
    _unionVisited.assign( static_cast<std::size_t>( nnodes ), COLTYPE{} );
    _uCursor.assign( static_cast<std::size_t>( nnodes ), _base );
    _uEdagRows.assign( static_cast<std::size_t>( nnodes ), {} );
    _uColumnRows.assign( static_cast<std::size_t>( nnodes ), {} );
    _reachEpoch = COLTYPE{};
    _reduceEpoch = COLTYPE{};
    _unionEpoch = COLTYPE{};
    _stack.clear();

    _lEdag.rows = nnodes;
    _lEdag.cols = nnodes;
    _lEdag.ai.assign( static_cast<std::size_t>( nnodes ) + 1, _base );
    _lEdag.aj.clear();

    _uEdag.rows = nnodes;
    _uEdag.cols = nnodes;
    _uEdag.ai.assign( static_cast<std::size_t>( nnodes ) + 1, _base );
    _uEdag.aj.clear();

    return true;
}

template <matrix_utils::AppendableCSR CSRMatrixType>
bool SymbolicLUEdags<CSRMatrixType>::setUpperEdag( const GraphType& upper_edag )
{
    if ( !validGraph( upper_edag ) )
    {
        return false;
    }

    _uEdag = upper_edag;
    return true;
}

template <matrix_utils::AppendableCSR CSRMatrixType>
void SymbolicLUEdags<CSRMatrixType>::appendLowerRowPattern( const COLTYPE row,
                                                            ROWTYPE const* ai,
                                                            COLTYPE const* aj,
                                                            CSRMatrixType& lu )
{
    assert( row >= 0 && row < _nnodes );
    assert( ai != nullptr && aj != nullptr );
    assert( _diag.size() == static_cast<std::size_t>( _nnodes ) );
    assert( _reachVisited.size() == static_cast<std::size_t>( _nnodes ) );
    assert( _uEdagRows.size() == static_cast<std::size_t>( _nnodes ) );
    assert( lu.rows == _nnodes && lu.cols == _nnodes );

    _stack.clear();
    const std::size_t row_start = lu.aj.size();

    const COLTYPE label = nextEpoch( _reachVisited, _reachEpoch );
    auto visit = [&]( const COLTYPE node )
    {
        if ( _reachVisited[node] == label )
        {
            return;
        }

        _reachVisited[node] = label;
        lu.aj.push_back( node + _base );
        _stack.push_back( node );
    };

    for ( ROWTYPE p = ai[row] - _base; p < _diag[row] - _base; ++p )
    {
        visit( aj[p] - _base );
    }

    while ( !_stack.empty() )
    {
        const COLTYPE node = _stack.back();
        _stack.pop_back();

        for ( const COLTYPE next : _uEdagRows[node] )
        {
            if ( next < row )
            {
                visit( next );
            }
        }
    }

    const std::size_t row_end = lu.aj.size();
    std::sort( lu.aj.begin() + row_start, lu.aj.begin() + row_end );
}

template <matrix_utils::AppendableCSR CSRMatrixType>
void SymbolicLUEdags<CSRMatrixType>::reduceLowerEdagRow( const COLTYPE row,
                                                         COLTYPE const* candidates_begin,
                                                         COLTYPE const* candidates_end )
{
    assert( row >= 0 && row < _nnodes );

    reduceEdagRow( _lEdag, row, candidates_begin, candidates_end, ReduceOrder::Reverse, false );
}

template <matrix_utils::AppendableCSR CSRMatrixType>
void SymbolicLUEdags<CSRMatrixType>::extendUpperEdagColumn( const COLTYPE col )
{
    assert( col >= 0 && col < _nnodes );
    assert( _uColumnRows.size() == static_cast<std::size_t>( _nnodes ) );
    assert( _uEdagRows.size() == static_cast<std::size_t>( _nnodes ) );

    const auto& sources = _uColumnRows[col];
    for ( auto source_it = sources.rbegin(); source_it != sources.rend(); ++source_it )
    {
        const COLTYPE source = *source_it;
        assert( source >= 0 && source < col );

        if ( !upperEdagReaches( source, col ) )
        {
            _uEdagRows[source].push_back( col );
        }
    }
}

template <matrix_utils::AppendableCSR CSRMatrixType>
bool SymbolicLUEdags<CSRMatrixType>::upperEdagReaches( const COLTYPE source, const COLTYPE target )
{
    assert( source >= 0 && source < target && target < _nnodes );
    assert( _reduceVisited.size() == static_cast<std::size_t>( _nnodes ) );
    assert( _uEdagRows.size() == static_cast<std::size_t>( _nnodes ) );

    const COLTYPE label = nextEpoch( _reduceVisited, _reduceEpoch );
    _stack.clear();

    auto pushIfNew = [&]( const COLTYPE node ) -> bool
    {
        if ( node == target )
        {
            return true;
        }

        if ( node < target && _reduceVisited[node] != label )
        {
            _reduceVisited[node] = label;
            _stack.push_back( node );
        }
        return false;
    };

    for ( const COLTYPE next : _uEdagRows[source] )
    {
        if ( pushIfNew( next ) )
        {
            return true;
        }
    }

    while ( !_stack.empty() )
    {
        const COLTYPE node = _stack.back();
        _stack.pop_back();

        for ( const COLTYPE next : _uEdagRows[node] )
        {
            if ( pushIfNew( next ) )
            {
                return true;
            }
        }
    }

    return false;
}

template <matrix_utils::AppendableCSR CSRMatrixType>
void SymbolicLUEdags<CSRMatrixType>::packUpperEdag()
{
    assert( _uEdagRows.size() == static_cast<std::size_t>( _nnodes ) );

    _uEdag.ai.assign( static_cast<std::size_t>( _nnodes ) + 1, _base );
    _uEdag.aj.clear();
    for ( COLTYPE row = 0; row < _nnodes; ++row )
    {
        for ( const COLTYPE next : _uEdagRows[row] )
        {
            _uEdag.aj.push_back( next + _base );
        }
        _uEdag.ai[row + 1] = static_cast<ROWTYPE>( _uEdag.aj.size() ) + _base;
    }
}

template <matrix_utils::AppendableCSR CSRMatrixType>
void SymbolicLUEdags<CSRMatrixType>::reduceEdagRow( GraphType& edag,
                                                    const COLTYPE row,
                                                    COLTYPE const* candidates_begin,
                                                    COLTYPE const* candidates_end,
                                                    const ReduceOrder order,
                                                    const bool skip_self )
{
    assert( row >= 0 && row < _nnodes );
    assert( _reduceVisited.size() == static_cast<std::size_t>( _nnodes ) );
    assert( validGraph( edag ) );
    assert( ( candidates_begin == nullptr ) == ( candidates_end == nullptr ) );

    const std::size_t edge_begin = edag.aj.size();
    const COLTYPE label = nextEpoch( _reduceVisited, _reduceEpoch );

    auto markReachable = [&]( const COLTYPE source )
    {
        _reduceVisited[source] = label;
        _stack.clear();
        _stack.push_back( source );

        while ( !_stack.empty() )
        {
            const COLTYPE node = _stack.back();
            _stack.pop_back();

            for ( ROWTYPE p = edag.ai[node] - _base; p < edag.ai[node + 1] - _base; ++p )
            {
                const COLTYPE next = edag.aj[p] - _base;
                if ( _reduceVisited[next] != label )
                {
                    _reduceVisited[next] = label;
                    _stack.push_back( next );
                }
            }
        }
    };

    auto processCandidate = [&]( const COLTYPE candidate_with_base )
    {
        const COLTYPE candidate = candidate_with_base - _base;
        assert( candidate >= 0 && candidate < _nnodes );

        if ( skip_self && candidate == row )
        {
            return;
        }

        if ( _reduceVisited[candidate] == label )
        {
            return;
        }

        edag.aj.push_back( candidate + _base );
        markReachable( candidate );
    };

    if ( order == ReduceOrder::Reverse )
    {
        for ( auto candidate = candidates_end; candidate != candidates_begin; )
        {
            --candidate;
            processCandidate( *candidate );
        }
    }
    else
    {
        for ( auto candidate = candidates_begin; candidate != candidates_end; ++candidate )
        {
            processCandidate( *candidate );
        }
    }

    const std::size_t edge_end = edag.aj.size();
    edag.ai[row] = static_cast<ROWTYPE>( edge_begin ) + _base;
    edag.ai[row + 1] = static_cast<ROWTYPE>( edge_end ) + _base;
}

template <matrix_utils::AppendableCSR CSRMatrixType>
void SymbolicLUEdags<CSRMatrixType>::appendUpperRowPattern( const COLTYPE row,
                                                            ROWTYPE const* ai,
                                                            COLTYPE const* aj,
                                                            CSRMatrixType& lu )
{
    assert( row >= 0 && row < _nnodes );
    assert( ai != nullptr && aj != nullptr );
    assert( _unionVisited.size() == static_cast<std::size_t>( _nnodes ) );
    assert( _uCursor.size() == static_cast<std::size_t>( _nnodes ) );
    assert( _diag.size() == static_cast<std::size_t>( _nnodes ) );
    assert( validGraph( _lEdag ) );
    assert( lu.rows == _nnodes && lu.cols == _nnodes );
    assert( lu.AI() != nullptr );

    const std::size_t upper_begin = lu.aj.size();
    _uCursor[row] = static_cast<ROWTYPE>( upper_begin ) + _base;
    const COLTYPE label = nextEpoch( _unionVisited, _unionEpoch );

    auto appendIfNew = [&]( const COLTYPE node )
    {
        if ( _unionVisited[node] == label )
        {
            return;
        }

        _unionVisited[node] = label;
        lu.aj.push_back( node + _base );
        if ( node > row )
        {
            _uColumnRows[node].push_back( row );
        }
    };

    for ( ROWTYPE p = _diag[row] - _base; p < ai[row + 1] - _base; ++p )
    {
        appendIfNew( aj[p] - _base );
    }

    assert( _lEdag.ai[row] == _lEdag.ai[row + 1] || !lu.aj.empty() );

    for ( ROWTYPE edge = _lEdag.ai[row] - _base; edge < _lEdag.ai[row + 1] - _base; ++edge )
    {
        const COLTYPE dependency = _lEdag.aj[edge] - _base;
        ROWTYPE cursor = _uCursor[dependency] - _base;
        const ROWTYPE end = lu.AI()[dependency + 1] - _base;

        while ( cursor < end && lu.aj[cursor] - _base <= row )
        {
            ++cursor;
        }

        _uCursor[dependency] = cursor + _base;
        for ( ROWTYPE p = cursor; p < end; ++p )
        {
            appendIfNew( lu.aj[p] - _base );
        }
    }

    const std::size_t upper_end = lu.aj.size();
    std::sort( lu.aj.begin() + upper_begin, lu.aj.begin() + upper_end );
}

template <matrix_utils::AppendableCSR CSRMatrixType>
typename SymbolicLUEdags<CSRMatrixType>::COLTYPE SymbolicLUEdags<CSRMatrixType>::nextEpoch( std::vector<COLTYPE>& visited,
                                                                                            COLTYPE& epoch )
{
    if ( epoch == std::numeric_limits<COLTYPE>::max() )
    {
        std::fill( visited.begin(), visited.end(), COLTYPE{} );
        epoch = COLTYPE{ 1 };
    }
    else
    {
        ++epoch;
    }

    return epoch;
}

template <matrix_utils::AppendableCSR CSRMatrixType>
bool SymbolicLUEdags<CSRMatrixType>::validGraph( const GraphType& graph ) const
{
    return graph.rows == _nnodes && graph.cols == _nnodes &&
           graph.ai.size() >= static_cast<std::size_t>( _nnodes ) + 1 &&
           ( graph.ai.empty() || graph.ai[0] == _base );
}

template class SymbolicLUEdags<matrix_utils::CSRMatrixVec<std::int32_t, std::int32_t, double>>;
template class SymbolicLUEdags<matrix_utils::CSRMatrixVec<std::int64_t, std::int64_t, double>>;

} // namespace factorization
