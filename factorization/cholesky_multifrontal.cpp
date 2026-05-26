#include "cholesky_multifrontal.hpp"

#include "matrix_utils.hpp"

#include <Eigen/Cholesky>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

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
#ifndef NDEBUG
    assert( ai_begin != nullptr );
    assert( ai_end != nullptr );
    assert( aj != nullptr );
    assert( av != nullptr );
    assert( L.rows == nnodes );
    assert( L.cols == nnodes );
    assert( L.AI() != nullptr );
    assert( L.AJ() != nullptr );
#endif

    const auto base = static_cast<COLTYPE>( L.Base() );
    if ( etree.nnodes() != nnodes || etree.base() != base )
    {
        return false;
    }

    prepareNumericValues( L );
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
void MultifrontalCholesky<CSRMatrixType>::prepareNumericValues( CSRMatrixType& L )
{
    const COLTYPE nnodes = L.rows;
    const auto base = L.Base();
    const auto nnz = L.AI()[nnodes] - base;

    L.ResizeAV( static_cast<std::size_t>( nnz ) );
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

    _worker.ensureSize( front_size );
    _worker.front().setZero();

    // Start with the original matrix entries that belong to this frontal column,
    // then accumulate Schur-complement updates from already-factorized children.
    initializeFront( node, ai_begin, ai_end, aj, av, L );
    assembleChildren( node, etree );

    // Factor the front, write the numeric column into L, and retain the update
    // matrix only when this node has a parent to assemble into later.
    if ( !factorFront( node, etree, L, front_size ) )
    {
        return false;
    }
    return true;
}

template <matrix_utils::ResizableCSR CSRMatrixType>
void MultifrontalCholesky<CSRMatrixType>::initializeFront( const COLTYPE node,
                                                           const ROWTYPE* ai_begin,
                                                           const ROWTYPE* ai_end,
                                                           const COLTYPE* aj,
                                                           const VALTYPE* av,
                                                           const CSRMatrixType& L )
{
    const auto base = static_cast<COLTYPE>( L.Base() );
    const auto* vars = L.AJ() + L.AI()[node] - base;
    auto F = _worker.front();

    COLTYPE local = 0;
    for ( auto pos = ai_begin[node] - base; pos < ai_end[node] - base; pos++ )
    {
        const auto label = aj[pos];
        if ( label < node + base )
        {
            continue;
        }

        while ( vars[local] < label )
        {
            local++;
        }
        F( local, 0 ) = av[pos];
        F( 0, local ) = av[pos];
    }
}

template <matrix_utils::ResizableCSR CSRMatrixType>
void MultifrontalCholesky<CSRMatrixType>::assembleChildren( const COLTYPE node,
                                                            const graph::EliminationTree<COLTYPE>& etree )
{
    const auto* child_offsets = etree.childOffsets();
    const auto* children = etree.children();
    auto F = _worker.front();
    for ( auto child_pos = child_offsets[node]; child_pos < child_offsets[node + 1]; child_pos++ )
    {
        const auto child = children[child_pos];
        const auto& V = _nodes[child].V;
        const auto& map = _nodes[child].map_to_parent;
        const auto update_size = static_cast<COLTYPE>( V.rows() );

        for ( COLTYPE j = 0; j < update_size; j++ )
        {
            const auto parent_j = map[static_cast<std::size_t>( j )];
            for ( COLTYPE i = 0; i < update_size; i++ )
            {
                const auto parent_i = map[static_cast<std::size_t>( i )];
                F( parent_i, parent_j ) += V( i, j );
            }
        }
    }
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
bool MultifrontalCholeskySuperNodal<CSRMatrixType>::apply( const COLTYPE nnodes,
                                                           const ROWTYPE* ai_begin,
                                                           const ROWTYPE* ai_end,
                                                           const COLTYPE* aj,
                                                           const VALTYPE* av,
                                                           const graph::EliminationTree<COLTYPE>& etree,
                                                           CSRMatrixType& L )
{
#ifndef NDEBUG
    assert( ai_begin != nullptr );
    assert( ai_end != nullptr );
    assert( aj != nullptr );
    assert( av != nullptr );
    assert( L.rows == nnodes );
    assert( L.cols == nnodes );
    assert( L.AI() != nullptr );
    assert( L.AJ() != nullptr );
#endif

    if ( !analyzeSupernodes( nnodes, etree, L ) )
    {
        return false;
    }
    prepareNumericValues( L );

    const auto base = static_cast<COLTYPE>( _assembly_tree.base() );
    const auto* topological_order = _assembly_tree.topologicalOrder();
    const auto nsupernodes = static_cast<COLTYPE>( _nodes.size() );
    for ( COLTYPE pos = 0; pos < nsupernodes; pos++ )
    {
        const auto supernode = topological_order[pos] - base;
        if ( !processSupernode( supernode, ai_begin, ai_end, aj, av, L ) )
        {
            return false;
        }
    }
    return true;
}

template <matrix_utils::ResizableCSR CSRMatrixType>
bool MultifrontalCholeskySuperNodal<CSRMatrixType>::analyzeSupernodes( const COLTYPE nnodes,
                                                                       const graph::EliminationTree<COLTYPE>& etree,
                                                                       const CSRMatrixType& L )
{
    const auto base = static_cast<COLTYPE>( L.Base() );
    if ( etree.nnodes() != nnodes || etree.base() != base )
    {
        return false;
    }
    buildSupernodes( nnodes, base, etree, L );
    if ( !convertEliminationTreeToAssemblyTree( etree ) )
    {
        return false;
    }
    buildFrontalIndexListsAndChildMaps( base, L );
    return true;
}

template <matrix_utils::ResizableCSR CSRMatrixType>
void MultifrontalCholeskySuperNodal<CSRMatrixType>::prepareNumericValues( CSRMatrixType& L )
{
    const COLTYPE nnodes = L.rows;
    const auto base = L.Base();
    const auto nnz = L.AI()[nnodes] - base;

    L.ResizeAV( static_cast<std::size_t>( nnz ) );
}

template <matrix_utils::ResizableCSR CSRMatrixType>
void MultifrontalCholeskySuperNodal<CSRMatrixType>::buildSupernodes( const COLTYPE nnodes,
                                                                     const COLTYPE base,
                                                                     const graph::EliminationTree<COLTYPE>& etree,
                                                                     const CSRMatrixType& L )
{
    const auto* lp = L.AI();
    const auto* parent = etree.parent();

    _supernode_prefix.clear();
    _column_to_supernode.assign( static_cast<std::size_t>( nnodes ), COLTYPE{} );
    _supernode_prefix.push_back( 0 );

    COLTYPE supernode = 0;
    COLTYPE start = 0;
    while ( start < nnodes )
    {
        const auto start_count = lp[start + 1] - lp[start];
        COLTYPE end = start + 1;
        while ( end < nnodes )
        {
            const auto previous = end - 1;
            const auto previous_parent = parent[previous] - base;
            const auto current_count = lp[end + 1] - lp[end];
            if ( previous_parent != end || start_count != current_count + end - start )
            {
                break;
            }
            end++;
        }

        for ( COLTYPE col = start; col < end; col++ )
        {
            _column_to_supernode[static_cast<std::size_t>( col )] = supernode;
        }
        _supernode_prefix.push_back( end );
        supernode++;
        start = end;
    }

    _nodes.clear();
    _nodes.resize( static_cast<std::size_t>( supernode ) );
}

template <matrix_utils::ResizableCSR CSRMatrixType>
bool MultifrontalCholeskySuperNodal<CSRMatrixType>::convertEliminationTreeToAssemblyTree(
    const graph::EliminationTree<COLTYPE>& etree )
{
    if ( _supernode_prefix.empty() )
    {
        return false;
    }

    const auto base = static_cast<COLTYPE>( etree.base() );
    const auto* parent = etree.parent();
    const auto nsupernodes = static_cast<COLTYPE>( _nodes.size() );
    _assembly_parent.resize( static_cast<std::size_t>( nsupernodes ) );

    for ( COLTYPE supernode = 0; supernode < nsupernodes; supernode++ )
    {
        const auto last_column = _supernode_prefix[supernode + 1] - 1;
        const auto parent_column = parent[last_column] - base;
        if ( parent_column < 0 || parent_column >= etree.nnodes() )
        {
            return false;
        }

        const auto parent_supernode = _column_to_supernode[static_cast<std::size_t>( parent_column )];
        if ( parent_supernode == supernode )
        {
            _assembly_parent[static_cast<std::size_t>( supernode )] = supernode + base;
        }
        else
        {
            _assembly_parent[static_cast<std::size_t>( supernode )] = parent_supernode + base;
        }
    }

    return _assembly_tree.analyze( nsupernodes, base, _assembly_parent.data() );
}

template <matrix_utils::ResizableCSR CSRMatrixType>
void MultifrontalCholeskySuperNodal<CSRMatrixType>::buildFrontalIndexListsAndChildMaps( const COLTYPE base,
                                                                                        const CSRMatrixType& L )
{
    const auto* lp = L.AI();
    const auto* li = L.AJ();
    const auto nsupernodes = static_cast<COLTYPE>( _nodes.size() );

    const auto* assembly_parent = _assembly_tree.parent();
    for ( COLTYPE supernode = 0; supernode < nsupernodes; supernode++ )
    {
        const auto parent_supernode = assembly_parent[supernode] - base;
        auto& map = _nodes[static_cast<std::size_t>( supernode )].map_to_parent;
        map.clear();
        if ( parent_supernode == supernode )
        {
            continue;
        }
        assert( parent_supernode >= 0 );
        assert( parent_supernode < nsupernodes );

        const auto supernode_size = _supernode_prefix[supernode + 1] - _supernode_prefix[supernode];
        const auto first_column = _supernode_prefix[supernode];
        const auto front_begin = lp[first_column] - base;
        const auto front_end = lp[first_column + 1] - base;
        const auto update_size = static_cast<COLTYPE>( front_end - front_begin ) - supernode_size;

        const auto parent_first_column = _supernode_prefix[parent_supernode];
        const auto parent_front_begin = lp[parent_first_column] - base;
        const auto parent_front_end = lp[parent_first_column + 1] - base;
        map.resize( static_cast<std::size_t>( update_size ) );

        auto parent_pos = parent_front_begin;
        for ( COLTYPE i = 0; i < update_size; i++ )
        {
            const auto label = li[front_begin + supernode_size + i];
            while ( parent_pos < parent_front_end && li[parent_pos] < label )
            {
                parent_pos++;
            }
            assert( parent_pos != parent_front_end );
            assert( li[parent_pos] == label );
            map[static_cast<std::size_t>( i )] = static_cast<COLTYPE>( parent_pos - parent_front_begin );
        }
    }
}

template <matrix_utils::ResizableCSR CSRMatrixType>
bool MultifrontalCholeskySuperNodal<CSRMatrixType>::processSupernode( const COLTYPE supernode,
                                                                      const ROWTYPE* ai_begin,
                                                                      const ROWTYPE* ai_end,
                                                                      const COLTYPE* aj,
                                                                      const VALTYPE* av,
                                                                      CSRMatrixType& L )
{
    const auto base = static_cast<COLTYPE>( L.Base() );
    const auto first_column = _supernode_prefix[supernode];
    const auto front_size = static_cast<COLTYPE>( L.AI()[first_column + 1] - L.AI()[first_column] );

    _worker.ensureSize( front_size );
    _worker.front().setZero();

    initializeFront( supernode, ai_begin, ai_end, aj, av, L );
    assembleChildren( supernode );
    if ( !factorFront( supernode, L, front_size ) )
    {
        return false;
    }
    return true;
}

template <matrix_utils::ResizableCSR CSRMatrixType>
void MultifrontalCholeskySuperNodal<CSRMatrixType>::initializeFront( const COLTYPE supernode,
                                                                     const ROWTYPE* ai_begin,
                                                                     const ROWTYPE* ai_end,
                                                                     const COLTYPE* aj,
                                                                     const VALTYPE* av,
                                                                     const CSRMatrixType& L )
{
    const auto base = static_cast<COLTYPE>( L.Base() );
    const auto first_column = _supernode_prefix[supernode];
    const auto supernode_size = _supernode_prefix[supernode + 1] - _supernode_prefix[supernode];
    const auto vars_begin = L.AJ() + L.AI()[first_column] - base;
    auto F = _worker.front();

    for ( COLTYPE local_col = 0; local_col < supernode_size; local_col++ )
    {
        const auto column = first_column + local_col;
        const auto column_label = column + base;

        COLTYPE local_row = local_col;
        for ( auto pos = ai_begin[column] - base; pos < ai_end[column] - base; pos++ )
        {
            const auto label = aj[pos];
            if ( label < column_label )
            {
                continue;
            }

            while ( vars_begin[local_row] < label )
            {
                local_row++;
            }

            F( local_row, local_col ) = av[pos];
            F( local_col, local_row ) = av[pos];
        }
    }
}

template <matrix_utils::ResizableCSR CSRMatrixType>
void MultifrontalCholeskySuperNodal<CSRMatrixType>::assembleChildren( const COLTYPE supernode )
{
    const auto* child_offsets = _assembly_tree.childOffsets();
    const auto* children = _assembly_tree.children();
    auto F = _worker.front();
    for ( auto child_pos = child_offsets[supernode]; child_pos < child_offsets[supernode + 1]; child_pos++ )
    {
        const auto child = children[child_pos];
        const auto& V = _nodes[static_cast<std::size_t>( child )].V;
        const auto& map = _nodes[static_cast<std::size_t>( child )].map_to_parent;
        const auto update_size = static_cast<COLTYPE>( V.rows() );

        for ( COLTYPE j = 0; j < update_size; j++ )
        {
            const auto parent_j = map[static_cast<std::size_t>( j )];
            for ( COLTYPE i = 0; i < update_size; i++ )
            {
                const auto parent_i = map[static_cast<std::size_t>( i )];
                F( parent_i, parent_j ) += V( i, j );
            }
        }
    }
}

template <matrix_utils::ResizableCSR CSRMatrixType>
bool MultifrontalCholeskySuperNodal<CSRMatrixType>::factorFront( const COLTYPE supernode,
                                                                 CSRMatrixType& L,
                                                                 const COLTYPE front_size )
{
    const auto base = static_cast<COLTYPE>( L.Base() );
    const auto first_column = _supernode_prefix[supernode];
    const auto supernode_size = _supernode_prefix[supernode + 1] - _supernode_prefix[supernode];
    const auto update_size = front_size - supernode_size;
    const auto supernode_size_eigen = static_cast<Eigen::Index>( supernode_size );
    const auto update_size_eigen = static_cast<Eigen::Index>( update_size );
    auto F = _worker.front();

    DenseMatrix L11 = F.block( 0, 0, supernode_size_eigen, supernode_size_eigen );
    if ( supernode_size == 1 )
    {
        const auto pivot = L11( 0, 0 );
        if ( !( pivot > VALTYPE{} ) )
        {
            return false;
        }
        L11( 0, 0 ) = static_cast<VALTYPE>( std::sqrt( pivot ) );
    }
    else
    {
        Eigen::LLT<DenseMatrix> llt( L11 );
        if ( llt.info() != Eigen::Success )
        {
            return false;
        }
        L11 = llt.matrixL();
    }

    RowMajorDenseMatrix L12( supernode_size_eigen, update_size_eigen );
    if ( update_size > 0 )
    {
        L12 = F.block( 0, supernode_size_eigen, supernode_size_eigen, update_size_eigen );
        L11.template triangularView<Eigen::Lower>().solveInPlace( L12 );
    }

    const auto* front_labels = L.AJ() + L.AI()[first_column] - base;
    for ( COLTYPE local_col = 0; local_col < supernode_size; local_col++ )
    {
        const auto column = first_column + local_col;
        const auto l_col_start = L.AI()[column] - base;
#ifndef NDEBUG
        const auto l_col_end = L.AI()[column + 1] - base;
        const auto column_size = static_cast<COLTYPE>( l_col_end - l_col_start );
        assert( column_size == front_size - local_col );

        for ( COLTYPE local_row = local_col; local_row < front_size; local_row++ )
        {
            const auto l_pos = l_col_start + local_row - local_col;
            assert( L.AJ()[l_pos] == front_labels[local_row] );
        }
#endif

        const auto l11_tail_size = supernode_size - local_col;
        std::memcpy( L.AV() + l_col_start, &L11( local_col, local_col ),
                     static_cast<std::size_t>( l11_tail_size ) * sizeof( VALTYPE ) );
        if ( update_size > 0 )
        {
            std::memcpy( L.AV() + l_col_start + l11_tail_size, &L12( local_col, 0 ),
                         static_cast<std::size_t>( update_size ) * sizeof( VALTYPE ) );
        }
    }

    auto& V = _nodes[static_cast<std::size_t>( supernode )].V;
    const auto parent_supernode = _assembly_tree.parent()[supernode] - base;
    if ( update_size == 0 || parent_supernode == supernode )
    {
        V.resize( 0, 0 );
        return true;
    }

    V = F.block( supernode_size_eigen, supernode_size_eigen, update_size_eigen, update_size_eigen );
    V.noalias() -= L12.transpose() * L12;
    return true;
}

template class MultifrontalCholesky<::matrix_utils::CSRMatrix<std::int32_t, std::int32_t, double>>;
template class MultifrontalCholesky<::matrix_utils::CSRMatrix<std::int64_t, std::int64_t, double>>;
template class MultifrontalCholeskySuperNodal<::matrix_utils::CSRMatrix<std::int32_t, std::int32_t, double>>;
template class MultifrontalCholeskySuperNodal<::matrix_utils::CSRMatrix<std::int64_t, std::int64_t, double>>;

} // namespace factorization
