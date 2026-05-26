#include "cholesky_multifrontal.hpp"

#include "matrix_utils.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

namespace factorization
{

template <matrix_utils::ResizableCSR CSRMatrixType>
bool MultifrontalCholesky<CSRMatrixType>::apply( const COLTYPE nnodes,
                                                 const ROWTYPE* ai_begin,
                                                 const ROWTYPE* ai_end,
                                                 const COLTYPE* aj,
                                                 const VALTYPE* av,
                                                 const graph::EliminationTree<COLTYPE>& etree,
                                                 CSRMatrixType& L )
{
    if ( ai_begin == nullptr || ai_end == nullptr || aj == nullptr || av == nullptr )
    {
        return false;
    }
    if ( L.rows != nnodes || L.cols != nnodes )
    {
        return false;
    }
    if ( L.AI() == nullptr || L.AJ() == nullptr )
    {
        return false;
    }

    const auto base = static_cast<COLTYPE>( L.Base() );
    if ( etree.nnodes() != nnodes || etree.base() != base )
    {
        return false;
    }

    if ( !prepareNumericValues( L ) )
    {
        return false;
    }
    if ( !buildChildToParentMaps( nnodes, base, etree, L ) )
    {
        return false;
    }

    const auto* topological_order = etree.topologicalOrder();
    for ( COLTYPE pos = 0; pos < nnodes; pos++ )
    {
        const auto node = topological_order[pos] - base;
        if ( !processNode( node, nnodes, ai_begin, ai_end, aj, av, etree, L ) )
        {
            return false;
        }
    }
    return true;
}

template <matrix_utils::ResizableCSR CSRMatrixType>
bool MultifrontalCholesky<CSRMatrixType>::prepareNumericValues( CSRMatrixType& L )
{
    if ( L.AI() == nullptr || L.AJ() == nullptr )
    {
        return false;
    }
    const COLTYPE nnodes = L.rows;
    const auto base = L.Base();
    const auto nnz = L.AI()[nnodes] - base;
    if ( nnz < 0 )
    {
        return false;
    }

    L.ResizeAV( static_cast<std::size_t>( nnz ) );
    return true;
}

template <matrix_utils::ResizableCSR CSRMatrixType>
bool MultifrontalCholesky<CSRMatrixType>::buildChildToParentMaps( const COLTYPE nnodes,
                                                                  const COLTYPE base,
                                                                  const graph::EliminationTree<COLTYPE>& etree,
                                                                  const CSRMatrixType& L )
{
    _nodes.clear();
    _nodes.resize( nnodes );
    const auto* lp = L.AI();
    const auto* li = L.AJ();
    const auto* parent = etree.parent();

    for ( COLTYPE node = 0; node < nnodes; node++ )
    {
        const auto node_start = lp[node] - base;
        const auto node_end = lp[node + 1] - base;
        assert( node_start < node_end );
        assert( li[node_start] == node + base );

        const auto parent_node = parent[node] - base;
        if ( parent_node == node )
        {
            continue;
        }
        assert( parent_node >= 0 );
        assert( parent_node < nnodes );

        const auto parent_start = lp[parent_node] - base;
        const auto parent_end = lp[parent_node + 1] - base;
        assert( parent_start < parent_end );
        assert( li[parent_start] == parent[node] );

        auto& map = _nodes[node].map_to_parent;
        const auto update_size = node_end - node_start - 1;
        map.resize( static_cast<std::size_t>( update_size ) );
        auto parent_pos = parent_start;
        for ( ROWTYPE i = 0; i < update_size; i++ )
        {
            const auto label = li[node_start + 1 + i];
            while ( parent_pos < parent_end && li[parent_pos] < label )
            {
                parent_pos++;
            }
            if ( parent_pos == parent_end || li[parent_pos] != label )
            {
                return false;
            }
            map[static_cast<std::size_t>( i )] = static_cast<COLTYPE>( parent_pos - parent_start );
        }
    }
    return true;
}

template <matrix_utils::ResizableCSR CSRMatrixType>
bool MultifrontalCholesky<CSRMatrixType>::processNode( const COLTYPE node,
                                                       const COLTYPE,
                                                       const ROWTYPE* ai_begin,
                                                       const ROWTYPE* ai_end,
                                                       const COLTYPE* aj,
                                                       const VALTYPE* av,
                                                       const graph::EliminationTree<COLTYPE>& etree,
                                                       CSRMatrixType& L )
{
    const auto front_size = static_cast<COLTYPE>( L.AI()[node + 1] - L.AI()[node] );
    if ( front_size <= 0 )
    {
        return false;
    }

    _worker.ensureSize( front_size );
    _worker.front().setZero();

    // Start with the original matrix entries that belong to this frontal column,
    // then accumulate Schur-complement updates from already-factorized children.
    if ( !initializeFront( node, ai_begin, ai_end, aj, av, L, front_size ) )
    {
        return false;
    }
    if ( !assembleChildren( node, etree ) )
    {
        return false;
    }

    // Factor the front, write the numeric column into L, and retain the update
    // matrix only when this node has a parent to assemble into later.
    if ( !factorFront( node, etree, L, front_size ) )
    {
        return false;
    }
    return true;
}

template <matrix_utils::ResizableCSR CSRMatrixType>
bool MultifrontalCholesky<CSRMatrixType>::initializeFront( const COLTYPE node,
                                                           const ROWTYPE* ai_begin,
                                                           const ROWTYPE* ai_end,
                                                           const COLTYPE* aj,
                                                           const VALTYPE* av,
                                                           const CSRMatrixType& L,
                                                           const COLTYPE front_size )
{
    const auto base = static_cast<COLTYPE>( L.Base() );
    const auto* vars = L.AJ() + L.AI()[node] - base;
    const auto* vars_end = vars + front_size;
    auto F = _worker.front();

    if ( ai_begin[node] > ai_end[node] )
    {
        return false;
    }

    for ( auto pos = ai_begin[node] - base; pos < ai_end[node] - base; pos++ )
    {
        const auto col = aj[pos];
        if ( col < node + base )
        {
            continue;
        }

        COLTYPE local = 0;
        [[maybe_unused]] const bool found = findLocalIndex( vars, vars_end, col, local );
        assert( found );
        F( local, 0 ) = av[pos];
        F( 0, local ) = av[pos];
    }
    return true;
}

template <matrix_utils::ResizableCSR CSRMatrixType>
bool MultifrontalCholesky<CSRMatrixType>::assembleChildren( const COLTYPE node,
                                                            const graph::EliminationTree<COLTYPE>& etree )
{
    const auto* child_offsets = etree.childOffsets();
    const auto* children = etree.children();
    auto F = _worker.front();
    for ( auto child_pos = child_offsets[node]; child_pos < child_offsets[node + 1]; child_pos++ )
    {
        const auto child = children[child_pos];
        if ( child < 0 || static_cast<std::size_t>( child ) >= _nodes.size() )
        {
            return false;
        }

        const auto& V = _nodes[child].V;
        const auto& map = _nodes[child].map_to_parent;
        const auto update_size = static_cast<COLTYPE>( V.rows() );
        if ( V.cols() != V.rows() || map.size() != static_cast<std::size_t>( update_size ) )
        {
            return false;
        }

        for ( COLTYPE i = 0; i < update_size; i++ )
        {
            const auto parent_i = map[static_cast<std::size_t>( i )];
            for ( COLTYPE j = 0; j < update_size; j++ )
            {
                const auto parent_j = map[static_cast<std::size_t>( j )];
                F( parent_i, parent_j ) += V( i, j );
            }
        }
    }
    return true;
}

template <matrix_utils::ResizableCSR CSRMatrixType>
bool MultifrontalCholesky<CSRMatrixType>::factorFront( const COLTYPE node,
                                                       const graph::EliminationTree<COLTYPE>& etree,
                                                       CSRMatrixType& L,
                                                       const COLTYPE front_size )
{
    const auto base = static_cast<COLTYPE>( L.Base() );
    const auto l_col_start = L.AI()[node] - base;
    const auto update_size = front_size - 1;
    auto F = _worker.front();
    const auto pivot = F( 0, 0 );
    if ( !( pivot > VALTYPE{} ) )
    {
        return false;
    }

    const auto lkk = static_cast<VALTYPE>( std::sqrt( pivot ) );
    L.AV()[l_col_start] = lkk;

    auto& V = _nodes[node].V;
    const auto parent_node = etree.parent()[node] - base;
    if ( update_size == 0 )
    {
        V.resize( 0, 0 );
        return true;
    }

    Eigen::Map<Eigen::Matrix<VALTYPE, Eigen::Dynamic, 1>> ltail( L.AV() + l_col_start + 1, update_size );
    ltail.noalias() = F.block( 1, 0, update_size, 1 ) / lkk;

    if ( parent_node == node )
    {
        V.resize( 0, 0 );
        return true;
    }

    V = F.block( 1, 1, update_size, update_size );
    V.noalias() -= ltail * ltail.transpose();
    return true;
}

template <matrix_utils::ResizableCSR CSRMatrixType>
bool MultifrontalCholesky<CSRMatrixType>::findLocalIndex( const COLTYPE* begin,
                                                          const COLTYPE* end,
                                                          const COLTYPE label,
                                                          COLTYPE& local_index )
{
    const auto it = std::lower_bound( begin, end, label );
    if ( it == end || *it != label )
    {
        return false;
    }
    local_index = static_cast<COLTYPE>( it - begin );
    return true;
}

template class MultifrontalCholesky<::matrix_utils::CSRMatrix<std::int32_t, std::int32_t, double>>;
template class MultifrontalCholesky<::matrix_utils::CSRMatrix<std::int64_t, std::int64_t, double>>;

} // namespace factorization
