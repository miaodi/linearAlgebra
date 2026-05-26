#pragma once

#include "Transformation.hpp"
#include <memory>
#include <vector>

namespace solver
{

// Transformation sequence that applies multiple transformations in order
template <SwappableResizableCSR MatType, typename VecType = std::vector<typename MatType::VALTYPE>>
    requires VectorForMatrix<VecType, MatType>
class TransformSeq : public TransformationBase<MatType, VecType>
{
public:
    using VALTYPE = typename MatType::VALTYPE;
    using ROWTYPE = typename MatType::ROWTYPE;
    using COLTYPE = typename MatType::COLTYPE;
    using TransformPtr = std::shared_ptr<TransformationBase<MatType, VecType>>;

    explicit TransformSeq() = default;

    /**
     * Add a transformation to the sequence
     * Transformations are applied in the order they are added
     */
    void addTransformation( TransformPtr transform ) { _transforms.push_back( transform ); }

    /**
     * Clear all transformations
     */
    void clear() { _transforms.clear(); }

    /**
     * Get number of transformations in the sequence
     */
    size_t size() const { return _transforms.size(); }

    /**
     * Apply all transformations to operator: A_out = T_n * ... * T_2 * T_1 * A_in
     */
    void applyToOperator( MatType& in, MatType& out, int nthreads = 1 ) const override
    {
        for ( const auto& transform : _transforms )
        {
            transform->applyToOperator( in, out, nthreads );
            std::swap( in, out );
        }
        std::swap( in, out );
    }

    /**
     * Apply all transformations to RHS: b_out = T_n * ... * T_2 * T_1 * b_in
     */
    void applyToRHS( VecType& in, VecType& out, int nthreads = 1 ) const override
    {
        for ( const auto& transform : _transforms )
        {
            transform->applyToRHS( in, out, nthreads );
            std::swap( in, out );
        }
        std::swap( in, out );
    }

    /**
     * Apply all transformations to X: x_out = T_1 * T_2 * ... * T_n * x_in
     * used to convert solution of transformed system back to original system
     */
    void applyToX( VecType& in, VecType& out, int nthreads = 1 ) const override
    {
        for ( auto it = _transforms.rbegin(); it != _transforms.rend(); ++it )
        {
            ( *it )->applyToX( in, out, nthreads );
            std::swap( in, out );
        }
        std::swap( in, out );
    }

    /**
     * Apply inverse transformations to X: x_out = T_n^{-1} * ... * T_2^{-1} * T_1^{-1} * x_in
     * used to convert initial guess to the transformed system back to the original system
     */
    void applyInverseToX( VecType& in, VecType& out, int nthreads = 1 ) const override
    {
        // Apply in reverse order
        for ( const auto& transform : _transforms )
        {
            transform->applyInverseToX( in, out, nthreads );
            std::swap( in, out );
        }
        std::swap( in, out );
    }

private:
    std::vector<TransformPtr> _transforms;
};

} // namespace solver
