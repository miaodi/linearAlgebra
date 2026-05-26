#pragma once
#include "../config.h"
#include "ObjectPool.hpp"
#include "UnionFind.h"
#include "circularbuffer.hpp"
#include <map>
#include <numeric>
#include <optional>
#include <queue>
#include <set>
#include <utility>
#include <vector>
#include <unordered_map>

// An implementation of the minimum degree reordering algorithm
// (10.1137/S0895479894278952)

// used https://github.com/rvp-group/srrg2_solver.git as a reference

namespace reordering
{

enum class MinimumDegree
{
    External,   // external minimum degree
    Exact,      // exact minimum degree
    Approximate // approximate minimum degree
};

template <typename COLTYPE, MinimumDegree MD = MinimumDegree::External>
class QuotientGraph
{
    struct Node
    {
        COLTYPE degree; // node degree

        std::conditional_t<MD == MinimumDegree::Approximate,
                           COLTYPE,
                           std::monostate>
            weight; // weight of the node, only used in approximate minimum degree
        std::vector<COLTYPE> adjacent_variables; // indices of the adjacent variables
        std::vector<COLTYPE> adjacent_elements;  // indices of the adjacent elements
        std::vector<COLTYPE> simple_variables;   // indices of the simple variables
    };
    static const COLTYPE INVALID{ std::numeric_limits<COLTYPE>::max() };
    static const COLTYPE ELEMENT{ std::numeric_limits<COLTYPE>::max() - 1 };
    static const COLTYPE ELIMINATED{ std::numeric_limits<COLTYPE>::max() };

public:
    QuotientGraph() = default;

    // perm: the permutation vector such that perm[i] = j means P(i, j) = 1. Aperm
    // = P * A ->Aperm(i, *) = A(j, *) iper: the inverse permutation vector such
    // that iperm[j] = i means perm[i] = j
    // Note: Assuming that provided matrix is symmetric.
    template <typename ROWTYPE>
    void operator()( const COLTYPE nnodes, ROWTYPE const* ai, COLTYPE const* aj, COLTYPE* perm, COLTYPE* iperm );

protected:
    // check if two nodes are distinguishable
    bool isDistinguishable( const COLTYPE i, const COLTYPE j ) const;

    // Adj_G(i) output from Lp
    void getFillins( const COLTYPE p, std::vector<COLTYPE>& Lp );

    // merge two nodes in the quotient graph i = {i, j}
    void merge( const COLTYPE i, const COLTYPE j );

    void eliminatePrincipleNode( const COLTYPE p, COLTYPE*& perm );

    auto massElimination( const COLTYPE i );

    void supervariableMerge( const std::vector<COLTYPE>& fillins );

    void principleVector( std::vector<COLTYPE>& vec );

    template <typename ROWTYPE>
    void initialize( const COLTYPE nnodes, ROWTYPE const* ai, COLTYPE const* aj, COLTYPE*& perm );

    COLTYPE getDegree( const COLTYPE i )
    {
        if constexpr ( MD == MinimumDegree::External )
        {
            return getExternalDegree( i );
        }
        else if constexpr ( MD == MinimumDegree::Exact )
        {
            return getExactDegree( i );
        }
        else if constexpr ( MD == MinimumDegree::Approximate )
        {
            return getApproximateDegree( i );
        }
        else
        {
            return INVALID;
        }
    }

    COLTYPE getExternalDegree( const COLTYPE i );

    COLTYPE getExactDegree( const COLTYPE i );

    COLTYPE getApproximateDegree( const COLTYPE i );

    COLTYPE hash( const COLTYPE i ) const;

    void removeElementNode( const COLTYPE i );

    void toElementNode( const COLTYPE i );

    void clearNode( const COLTYPE i );

    // // Algorithm 2, for computing |L_e/L_p|, only used for approximate minimum
    // degree
    //   void updateWeight(const std::vector<COLTYPE> &Lp);

public:
    std::vector<Node> _nodes;
    mutable reordering::UnionFind<COLTYPE, false> _union_find; // union-find structure
    std::map<COLTYPE,
             typename utils::ObjectPool<utils::CircularBuffer<COLTYPE>>::ptr_type>
        _degree_to_principle; // map variable to element

    utils::ObjectPool<std::vector<COLTYPE>> _pool; // object pool of vectors of size nnodes
    utils::ObjectPool<utils::CircularBuffer<COLTYPE>> _cb_pool; // object pool of circular buffers of size nnodes
    std::vector<std::vector<COLTYPE>*> __vectors;

    std::unordered_map<COLTYPE, typename utils::ObjectPool<std::vector<COLTYPE>>::ptr_type> _hash_table;
};

template <typename COLTYPE>
void mergeKVectors( const std::vector<std::vector<COLTYPE>*>& input_vectors,
                    std::vector<COLTYPE>& output_vector,
                    std::optional<COLTYPE> disregard_value = std::nullopt )
{
    using Entry = std::tuple<COLTYPE, COLTYPE,
                             COLTYPE>; // (value, list_index, element_index)

    // Min-heap
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> minHeap;

    // Initialize heap with the first element of each list
    for ( int i = 0; i < input_vectors.size(); ++i )
    {
        if ( !( *input_vectors[i] ).empty() )
        {
            minHeap.emplace( ( *input_vectors[i] )[0], i, 0 );
        }
    }
    // Clear the output vector
    output_vector.clear();
    // Merge the vectors
    while ( !minHeap.empty() )
    {
        auto [val, row, col] = minHeap.top();
        minHeap.pop();

        if ( output_vector.size() > 0 && output_vector.back() == val )
        {
            continue; // skip duplicates
        }
        if ( disregard_value.has_value() && val == disregard_value.value() )
        {
            continue; // skip the disregarded value
        }
        output_vector.push_back( val );

        if ( col + 1 < ( *input_vectors[row] ).size() )
        {
            minHeap.emplace( ( *input_vectors[row] )[col + 1], row, col + 1 );
        }
    }
}

template <typename COLTYPE>
void vectorSubtract( std::vector<COLTYPE>& op1,
                     const std::vector<COLTYPE>& op2,
                     std::optional<COLTYPE> disregard_value = std::nullopt )
{
    auto it1 = op1.begin();
    auto it1_run = op1.begin();
    auto it2 = op2.begin();

    while ( it1_run != op1.end() && it2 != op2.end() )
    {
        if ( disregard_value.has_value() && *it1_run == disregard_value.value() )
        {
            it1_run++;
            continue; // skip the disregarded value in op1
        }
        if ( disregard_value.has_value() && *it2 == disregard_value.value() )
        {
            it2++;
            continue; // skip the disregarded value in op2
        }
        if ( *it1_run < *it2 )
        {
            *it1 = *it1_run;
            it1++;
            it1_run++;
        }
        else if ( *it1_run > *it2 )
        {
            it2++;
        }
        else
        {
            it1_run++;
        }
    }
    while ( it1_run != op1.end() )
    {
        if ( disregard_value.has_value() && *it1_run == disregard_value.value() )
        {
            it1_run++;
            continue; // skip the disregarded value in op1
        }
        *it1 = *it1_run;
        it1++;
        it1_run++;
    }
    op1.resize( it1 - op1.begin() );
}

template <typename COLTYPE>
COLTYPE vectorSubtractSize( const std::vector<COLTYPE>& op1, const std::vector<COLTYPE>& op2 )
{
    auto it1 = op1.begin();
    auto it2 = op2.begin();
    COLTYPE size = 0;

    while ( it1 != op1.end() && it2 != op2.end() )
    {
        if ( *it1 < *it2 )
        {
            size++;
            it1++;
        }
        else if ( *it1 > *it2 )
        {
            it2++;
        }
        else
        {
            it1++;
            it2++;
        }
    }
    size += std::distance( it1, op1.end() );
    return size;
}
} // namespace reordering