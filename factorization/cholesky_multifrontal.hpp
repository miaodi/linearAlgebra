#pragma once

#include "sparse_mat_traits.hpp"
#include "tree.hpp"

#include <Eigen/Dense>

#include <cstddef>
#include <memory>
#include <vector>

namespace factorization
{

/// @brief Sequential basic multifrontal Cholesky factorization.
///
/// This implements the one-column frontal method from Algorithm 5.8 in
/// @cite scott2023algorithms. `L` supplies the CSC symbolic pattern on input;
/// this class updates only its numeric values. The symbolic factor pattern in
/// `L` must be sorted within each front/column.
template <matrix_utils::ResizableCSR CSRMatrixType>
class MultifrontalCholesky
{
public:
    using ROWTYPE = typename CSRMatrixType::ROWTYPE;
    using COLTYPE = typename CSRMatrixType::COLTYPE;
    using VALTYPE = typename CSRMatrixType::VALTYPE;
    using DenseMatrix = Eigen::Matrix<VALTYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    /// @brief Factor A using caller-provided elimination-tree analysis.
    ///
    /// `etree` must correspond to the same ordering and base as `L`. `L.AI()` and
    /// `L.AJ()` must already contain the symbolic CSC pattern of the Cholesky
    /// factor, with each front/column sorted. This function only resizes and
    /// writes `L.AV()`.
    ///
    /// `ai_begin[k]` and `ai_end[k]` are base-indexed positions in `aj`/`av` for
    /// the row segment used to initialize front k. For a full sorted symmetric CSR
    /// matrix, pass diagonal positions as `ai_begin` and `A.AI() + 1` as `ai_end`.
    /// For an upper-triangular CSR matrix, pass `A.AI()` as `ai_begin` and
    /// `A.AI() + 1` as `ai_end`.
    bool apply( const COLTYPE nnodes,
                const ROWTYPE* ai_begin,
                const ROWTYPE* ai_end,
                const COLTYPE* aj,
                const VALTYPE* av,
                const graph::EliminationTree<COLTYPE>& etree,
                CSRMatrixType& L );

private:
    using DenseMap = Eigen::Map<DenseMatrix>;

    struct FrontalNode
    {
        DenseMatrix V;
        std::vector<COLTYPE> map_to_parent;
    };

    struct FrontalWorker
    {
        std::unique_ptr<VALTYPE[]> F;
        COLTYPE active_size = 0;
        COLTYPE capacity = 0;

        void ensureSize( const COLTYPE size )
        {
            if ( size > capacity )
            {
                const auto n = static_cast<std::size_t>( size );
                F = std::make_unique<VALTYPE[]>( n * n );
                capacity = size;
            }
            active_size = size;
        }

        DenseMap front()
        {
            const auto n = static_cast<Eigen::Index>( active_size );
            // Map the active front as a contiguous prefix, not with capacity stride.
            return DenseMap( F.get(), n, n );
        }
    };

    bool prepareNumericValues( CSRMatrixType& L );
    bool buildChildToParentMaps( const COLTYPE nnodes,
                                 const COLTYPE base,
                                 const graph::EliminationTree<COLTYPE>& etree,
                                 const CSRMatrixType& L );
    bool processNode( const COLTYPE node,
                      const COLTYPE nnodes,
                      const ROWTYPE* ai_begin,
                      const ROWTYPE* ai_end,
                      const COLTYPE* aj,
                      const VALTYPE* av,
                      const graph::EliminationTree<COLTYPE>& etree,
                      CSRMatrixType& L );
    bool initializeFront( const COLTYPE node,
                          const ROWTYPE* ai_begin,
                          const ROWTYPE* ai_end,
                          const COLTYPE* aj,
                          const VALTYPE* av,
                          const CSRMatrixType& L,
                          const COLTYPE front_size );
    bool assembleChildren( const COLTYPE node, const graph::EliminationTree<COLTYPE>& etree );
    bool factorFront( const COLTYPE node, const graph::EliminationTree<COLTYPE>& etree, CSRMatrixType& L, const COLTYPE front_size );
    static bool findLocalIndex( const COLTYPE* begin, const COLTYPE* end, const COLTYPE label, COLTYPE& local_index );

    FrontalWorker _worker;
    std::vector<FrontalNode> _nodes;
};

} // namespace factorization
