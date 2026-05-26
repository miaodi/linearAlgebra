#include "MinimumDegree.hpp"
#include <algorithm>
#include <assert.h>
#include <limits> // Required for std::numeric_limits
#include "permutation.hpp"
namespace reordering
{

template <typename COLTYPE, MinimumDegree MD>
template <typename ROWTYPE>
void QuotientGraph<COLTYPE, MD>::initialize( const COLTYPE nnodes, ROWTYPE const* ai, COLTYPE const* aj, COLTYPE*& perm )
{
    using ObjectType = typename decltype( _pool )::value_type;
    const COLTYPE size = std::max( (COLTYPE)100, (COLTYPE)( nnodes / 1000. ) );
    _pool.setObjectPrep(
        [size]( ObjectType* obj )
        {
            obj->reserve( size );
            obj->clear();
        } );
    using CBType = typename decltype( _cb_pool )::value_type;
    _cb_pool.setObjectPrep(
        [size]( CBType* obj )
        {
            if ( obj->size() < size )
                obj->resize( size );
            obj->clear();
        } );

    _nodes.resize( nnodes );
    _union_find.reset( nnodes ); // reset union-find structure
    _degree_to_principle.clear();
    const ROWTYPE base = ai[0];
    for ( COLTYPE i = 0; i < nnodes; i++ )
    {
        COLTYPE row_size = ai[i + 1] - ai[i];

        if ( row_size == 0 )
        {
            *perm++ = i + base;
            continue;
        }

        _nodes[i].adjacent_variables.reserve( row_size );
        for ( ROWTYPE j_idx = ai[i] - base; j_idx < ai[i + 1] - base; j_idx++ )
        {
            COLTYPE j = aj[j_idx] - base; // internally use 0-based indexing
            if ( j == i )
                continue;
            _nodes[i].adjacent_variables.push_back( j );
        }
        _nodes[i].degree = _nodes[i].adjacent_variables.size();
        _nodes[i].simple_variables.push_back( i );

        auto it = _degree_to_principle.try_emplace( _nodes[i].degree, std::move( _cb_pool.acquire() ) );
        it.first->second->push_back( i );
        // _degree_to_principle[_nodes[i].degree].insert(i);
    }
}

template <typename COLTYPE, MinimumDegree MD>
template <typename ROWTYPE>
void QuotientGraph<COLTYPE, MD>::operator()( const COLTYPE nnodes,
                                             ROWTYPE const* ai,
                                             COLTYPE const* aj,
                                             COLTYPE* perm,
                                             COLTYPE* iperm )
{
    const ROWTYPE base = ai[0];
    COLTYPE idx = 0;
    bool found;
    typename decltype( _degree_to_principle )::iterator it;
    initialize( nnodes, ai, aj, perm );
    while ( !_degree_to_principle.empty() )
    {
        // std::cout << _degree_to_principle.size() << " " << _cb_pool.size() << " "
        //           << _pool.size() << std::endl;
        it = _degree_to_principle.begin();
        found = false;
        while ( !it->second->empty() )
        {
            idx = it->second->first();
            it->second->pop_front();
            if ( _union_find.Find( idx ) == idx && _nodes[idx].degree == it->first )
            {
                found = true;
                break; // found a valid principle
            }
        }
        if ( !found )
        {
            _degree_to_principle.erase( it );
        }
        else
        {
            // std::cout << " Eliminating principle node " << idx << " with degree "
            //           << _nodes[idx].degree << std::endl;
            eliminatePrincipleNode( idx, perm );
        }
    }
    matrix_utils::invPerm( nnodes, base, perm - nnodes, iperm );
}

template <typename COLTYPE, MinimumDegree MD>
bool QuotientGraph<COLTYPE, MD>::isDistinguishable( const COLTYPE i, const COLTYPE j ) const
{
    // diff in quotient graph
    if ( _nodes[i].adjacent_variables.size() != _nodes[j].adjacent_variables.size() )
        return true;
    if ( _nodes[i].adjacent_elements.size() != _nodes[j].adjacent_elements.size() )
        return true;
    if ( _nodes[i].adjacent_variables.size() == 0 && _nodes[i].adjacent_elements.size() == 0 )
        return true; // both nodes are empty, distinguishable
    // std::cout << "hello!\n";
    // check if they have the same adjacent variables
    size_t k = 0, l = 0;
    while ( k < _nodes[i].adjacent_variables.size() && l < _nodes[j].adjacent_variables.size() )
    {
        if ( _nodes[i].adjacent_variables[k] != _nodes[j].adjacent_variables[l] )
        {
            if ( _nodes[i].adjacent_variables[k] == j )
                k++;
            else if ( _nodes[j].adjacent_variables[l] == i )
                l++;
            else
                return true; // different neighbours
        }
        else
        {
            k++;
            l++;
        }
    }
    if ( k < _nodes[i].adjacent_variables.size() )
    {
        if ( _nodes[i].adjacent_variables[k] == j )
            k++;
        else
            return true; // different neighbours
    }
    if ( l < _nodes[j].adjacent_variables.size() )
    {
        if ( _nodes[j].adjacent_variables[l] == i )
            l++;
        else
            return true; // different neighbours
    }
    if ( k != _nodes[i].adjacent_variables.size() || l != _nodes[j].adjacent_variables.size() )
        return true; // different neighbours

    // check if they have the same adjacent elements
    if ( _nodes[i].adjacent_elements == _nodes[j].adjacent_elements )
        return false; // same adjacent elements, distinguishable
    return true;      // different adjacent elements
}

template <typename COLTYPE, MinimumDegree MD>
void QuotientGraph<COLTYPE, MD>::getFillins( const COLTYPE p, std::vector<COLTYPE>& Lp )
{
    // TODO: check if really needed
    principleVector( _nodes[p].adjacent_variables );

    __vectors.clear();
    __vectors.push_back( &_nodes[p].adjacent_variables );
    auto& adj_elements = _nodes[p].adjacent_elements;
    COLTYPE pos = 0;
    for ( size_t i = 0; i < adj_elements.size(); ++i )
    {
        COLTYPE j = adj_elements[i];
        if ( _nodes[j].degree != ELEMENT )
        {
            continue;
        }
        principleVector( _nodes[j].adjacent_variables );
        __vectors.push_back( &_nodes[j].adjacent_variables );
        adj_elements[pos++] = j; // keep only element nodes
    }
    adj_elements.resize( pos ); // remove non-element nodes
    assert( Lp.empty() );
    mergeKVectors( __vectors, Lp, std::optional<COLTYPE>( p ) );
    // vectorSubtract(Lp, _nodes[p].simple_variables);
}

template <typename COLTYPE, MinimumDegree MD>
void QuotientGraph<COLTYPE, MD>::merge( const COLTYPE i, const COLTYPE j )
{
    // std::cout << "merging " << i << " degree: " << _nodes[i].degree
    //           << " with " << j << " degree: " << _nodes[j].degree << std::endl;
    assert( _union_find.Find( i ) != _union_find.Find( j ) ); // cannot merge the same element
    std::vector<COLTYPE> temp;
    temp.reserve( _nodes[i].simple_variables.size() + _nodes[j].simple_variables.size() );
    auto it = std::set_union( _nodes[i].simple_variables.begin(), _nodes[i].simple_variables.end(),
                              _nodes[j].simple_variables.begin(), _nodes[j].simple_variables.end(),
                              std::back_inserter( temp ) );
    assert( std::is_sorted( temp.begin(), temp.end() ) );
    assert( std::adjacent_find( temp.begin(), temp.end() ) == temp.end() );
    std::swap( _nodes[i].simple_variables, temp );
    _nodes[i].degree -= 1;
    clearNode( j );
    assert( i == _union_find.Unite( i, j ) );
}

template <typename COLTYPE, MinimumDegree MD>
void QuotientGraph<COLTYPE, MD>::eliminatePrincipleNode( const COLTYPE p, COLTYPE*& perm )
{
    // std::cout << "eliminate " << p << std::endl;
    assert( p == _union_find.Find( p ) );
    auto Lp_ptr = massElimination( p );
    for ( auto j : _nodes[p].simple_variables )
    {
        *perm++ = j;
    }
    supervariableMerge( *Lp_ptr );
    toElementNode( p );
}

template <typename COLTYPE, MinimumDegree MD>
auto QuotientGraph<COLTYPE, MD>::massElimination( const COLTYPE p )
{
    auto Lp = _pool.acquire();
    getFillins( p, *Lp ); // getFillins will replace simple_variables with
                          //  principle variables
    _nodes[p].adjacent_variables = *Lp;
    for ( auto i : _nodes[p].adjacent_elements )
    {
        removeElementNode( i );
    }
    for ( auto i : *Lp )
    {
        // remove redundant variables
        vectorSubtract( _nodes[i].adjacent_variables, *Lp, std::optional<COLTYPE>( p ) );

        // element absorption
        vectorSubtract( _nodes[i].adjacent_elements,
                        _nodes[p].adjacent_elements ); // \ \epsilon_p
        _nodes[i].adjacent_elements.insert(
            std::upper_bound( _nodes[i].adjacent_elements.begin(), _nodes[i].adjacent_elements.end(), p ),
            p ); // \cup p

        // update degree and reinsert to principle map
        _nodes[i].degree = getDegree( i );
        auto it = _degree_to_principle.try_emplace( _nodes[i].degree, std::move( _cb_pool.acquire() ) );
        it.first->second->push_back( i ); // update node i's degree
    }
    return std::move( Lp );
}

template <typename COLTYPE, MinimumDegree MD>
void QuotientGraph<COLTYPE, MD>::supervariableMerge( const std::vector<COLTYPE>& fillins )
{
    _hash_table.clear();
    for ( size_t i = 0; i < fillins.size(); ++i )
    {
        auto it = _hash_table.try_emplace( hash( fillins[i] ), std::move( _pool.acquire() ) );
        it.first->second->push_back( fillins[i] ); // insert fillin to hash table
    }

    for ( auto& it : _hash_table )
    {
        auto& vec = *( it.second );
        for ( size_t i = 0; i < vec.size(); ++i )
        {
            const COLTYPE node = vec[i];
            if ( node == INVALID )
            {
                // skip merged
                continue;
            }
            auto degree = _nodes[node].degree;
            for ( size_t j = i + 1; j < vec.size(); ++j )
            {
                principleVector( _nodes[j].adjacent_variables );
                if ( isDistinguishable( node, vec[j] ) )
                {
                    continue; // distinguishable
                }
                merge( node, vec[j] );
                vec[j] = INVALID; // mark as merged
            }

            // if node degree is changed, reinsert to principle map
            if ( _nodes[node].degree != degree )
            {
                // std::cout << "Node: " << node << " update degree " << _nodes[node].degree << " original degree " << degree << std::endl;
                auto it2 = _degree_to_principle.try_emplace( _nodes[node].degree,
                                                             std::move( _cb_pool.acquire() ) );
                it2.first->second->push_back( node );
            }
        }
    }
}

template <typename COLTYPE, MinimumDegree MD>
COLTYPE QuotientGraph<COLTYPE, MD>::getExternalDegree( const COLTYPE p )
{
    principleVector( _nodes[p].adjacent_variables );
    COLTYPE degree = vectorSubtractSize( _nodes[p].adjacent_variables, _nodes[p].simple_variables );

    auto& adj_elements = _nodes[p].adjacent_elements;
    __vectors.clear();
    COLTYPE pos = 0;
    for ( size_t i = 0; i < adj_elements.size(); ++i )
    {
        COLTYPE j = adj_elements[i];
        if ( _nodes[j].degree != ELEMENT )
        {
            continue;
        }
        principleVector( _nodes[j].adjacent_variables );
        __vectors.push_back( &_nodes[j].adjacent_variables );
        adj_elements[pos++] = j; // keep only element nodes
    }
    adj_elements.resize( pos ); // remove non-element nodes
    auto temp = _pool.acquire();
    mergeKVectors( __vectors, *temp, std::optional<COLTYPE>( p ) );
    degree += vectorSubtractSize( *temp, _nodes[p].simple_variables );
    return degree;
}

template <typename COLTYPE, MinimumDegree MD>
COLTYPE QuotientGraph<COLTYPE, MD>::getExactDegree( const COLTYPE i )
{
    auto temp1 = _pool.acquire();
    getFillins( i, *temp1 );
    return static_cast<COLTYPE>( temp1->size() );
}

template <typename COLTYPE, MinimumDegree MD>
COLTYPE QuotientGraph<COLTYPE, MD>::getApproximateDegree( const COLTYPE i )
{
    auto temp1 = _pool.acquire();
    getFillins( i, *temp1 );
    return static_cast<COLTYPE>( temp1->size() );
}

template <typename COLTYPE, MinimumDegree MD>
COLTYPE QuotientGraph<COLTYPE, MD>::hash( const COLTYPE i ) const
{
    COLTYPE hash_value = 0;
    for ( auto j : _nodes[i].adjacent_variables )
    {
        assert( j == _union_find.Find( j ) );
        hash_value += j;
        hash_value %= static_cast<COLTYPE>( _nodes.size() );
    }
    for ( auto j : _nodes[i].adjacent_elements )
    {
        assert( j == _union_find.Find( j ) );
        hash_value += j;
        hash_value %= static_cast<COLTYPE>( _nodes.size() );
    }
    return hash_value;
}

template <typename COLTYPE, MinimumDegree MD>
void QuotientGraph<COLTYPE, MD>::principleVector( std::vector<COLTYPE>& vec )
{
    bool modified = false;
    bool has_invalid = false;
    bool needs_sort = false;
    size_t j = 0;
    for ( size_t i = 0; i < vec.size(); ++i )
    {
        auto principleNode = _union_find.Find( vec[i] );
        if ( principleNode != vec[i] )
        {
            modified = true;
        }
        if ( _nodes[principleNode].degree != INVALID )
        {
            if ( !needs_sort && modified )
                needs_sort = true;
            vec[j++] = principleNode; // keep only principle nodes
        }
        else
        {
            has_invalid = true; // principle node is INVALID, so we need to remove it
        }
    }
    if ( modified || has_invalid )
    {
        vec.resize( j ); // remove non-principle nodes
        if ( needs_sort )
            std::sort( vec.begin(), vec.end() );
        vec.erase( std::unique( vec.begin(), vec.end() ), vec.end() ); // remove duplicates
    }
}

template <typename COLTYPE, MinimumDegree MD>
void QuotientGraph<COLTYPE, MD>::removeElementNode( const COLTYPE i )
{
    assert( i == _union_find.Find( i ) );
    assert( _nodes[i].degree == ELEMENT );
    assert( _nodes[i].adjacent_elements.empty() );
    assert( _nodes[i].simple_variables.empty() );
    _nodes[i].adjacent_variables.clear();
    _nodes[i].adjacent_variables.shrink_to_fit();
    _nodes[i].degree = INVALID;
}

template <typename COLTYPE, MinimumDegree MD>
void QuotientGraph<COLTYPE, MD>::toElementNode( const COLTYPE i )
{
    assert( i == _union_find.Find( i ) );
    _nodes[i].adjacent_elements.clear();
    _nodes[i].adjacent_elements.shrink_to_fit();
    _nodes[i].simple_variables.clear();
    _nodes[i].simple_variables.shrink_to_fit();
    _nodes[i].degree = ELEMENT;
}

template <typename COLTYPE, MinimumDegree MD>
void QuotientGraph<COLTYPE, MD>::clearNode( const COLTYPE i )
{
    assert( i == _union_find.Find( i ) );
    _nodes[i].adjacent_variables.clear();
    _nodes[i].adjacent_variables.shrink_to_fit();
    _nodes[i].adjacent_elements.clear();
    _nodes[i].adjacent_elements.shrink_to_fit();
    _nodes[i].simple_variables.clear();
    _nodes[i].simple_variables.shrink_to_fit();
    _nodes[i].degree = INVALID;
}

// template <typename COLTYPE, MinimumDegree MD>
// void QuotientGraph<COLTYPE, MD>::updateWeight(const std::vector<COLTYPE> &Lp)
// {
//   for (auto i : Lp) {
//     assert(i == _union_find.Find(i));
//     for (auto j : _nodes[i].adjacent_elements) {
//       assert(j == _union_find.Find(j));
//       _nodes[j].weight =
//           static_cast<COLTYPE>(_nodes[j].adjacent_variables.size());
//     }
//   }
// }

template class QuotientGraph<int>;
template void QuotientGraph<int>::operator()( const int nnodes, int const* ai, int const* aj, int* perm, int* iperm );

template void QuotientGraph<int>::initialize( const int nnodes, int const* ai, int const* aj, int*& perm );

} // namespace reordering