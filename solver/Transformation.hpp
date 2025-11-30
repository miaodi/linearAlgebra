#pragma once

#include <memory>
#include <vector>
#include <concepts>
#include <span>
#include "sparse_mat_traits.hpp"
#include "scaling.hpp"
#include "permutation.hpp"

namespace solver {

// Use VectorLike concept and SwappableResizableCSR from matrix_utils
using matrix_utils::VectorLike;
using matrix_utils::SwappableResizableCSR;

// Helper concept for valid vector type for a given matrix type
template <typename VecType, typename MatType>
concept VectorForMatrix = SwappableResizableCSR<MatType> && 
    VectorLike<VecType, typename MatType::VALTYPE>;

// Concept for transformation types
template <typename T, typename MatType, typename VecType>
concept Transformation = VectorForMatrix<VecType, MatType> &&
    requires(T t, VecType& vec, MatType& mat, int nthreads) {
    { t.applyToOperator(mat, mat, nthreads) } -> std::same_as<void>;
    { t.applyToRHS(vec, vec, nthreads) } -> std::same_as<void>;
    { t.applyToX(vec, vec, nthreads) } -> std::same_as<void>;
    { t.applyInverseToX(vec, vec, nthreads) } -> std::same_as<void>;
};

// Base interface for transformations
template <SwappableResizableCSR MatType, typename VecType = std::vector<typename MatType::VALTYPE>>
    requires VectorForMatrix<VecType, MatType>
class TransformationBase {
public:
    using VALTYPE = typename MatType::VALTYPE;
    using ROWTYPE = typename MatType::ROWTYPE;
    using COLTYPE = typename MatType::COLTYPE;
    
    virtual ~TransformationBase() = default;
    
    // Apply transformation to operator (matrix): out = Tr * A * Tc^{-1} (for row/col transforms)
    virtual void applyToOperator(MatType& in, MatType& out, int nthreads = 1) const = 0;

    // Apply transformation to RHS: out = T * in
    virtual void applyToRHS(VecType& in, VecType& out, int nthreads = 1) const = 0;
    
    // Apply transformation to solution: out = T * in
    virtual void applyToX(VecType& in, VecType& out, int nthreads = 1) const = 0;
    
    // Apply inverse transformation to solution: out = T^{-1} * in
    virtual void applyInverseToX(VecType& in, VecType& out, int nthreads = 1) const = 0;
};

// ============================================================================
// Permutation Transformations
// ============================================================================

// Row permutation transformation: P_r
template <SwappableResizableCSR MatType, typename VecType = std::vector<typename MatType::VALTYPE>>
    requires VectorForMatrix<VecType, MatType>
class RowPermutation : public TransformationBase<MatType, VecType> {
public:
    using VALTYPE = typename MatType::VALTYPE;
    using ROWTYPE = typename MatType::ROWTYPE;
    using COLTYPE = typename MatType::COLTYPE;
    
    explicit RowPermutation(const COLTYPE* perm, size_t size, int base = 0) 
        : _perm(perm, size), _base(base) {}
    
    explicit RowPermutation(std::span<const COLTYPE> perm, int base = 0)
        : RowPermutation(perm.data(), perm.size(), base) {}
    
    void applyToOperator(MatType& in, MatType& out, int nthreads = 1) const override {
        // For row permutation: out = P * A (permute rows of matrix)
        // This is matrix-specific and should be handled by LinearSolverSystem
        matrix_utils::permuteMat(in.rows, in.cols, _perm.data(), static_cast<const COLTYPE*>(nullptr),
                                 in.AI(), in.AJ(), in.AV(), out.AI(), out.AJ(), out.AV(), nthreads);
    }
    
    void applyToRHS(VecType& in, VecType& out, int nthreads = 1) const override {
        // out = P * in
        matrix_utils::permVec(_perm.size(), _base, in.data(), _perm.data(), out.data(), nthreads);
    }
    
    void applyToX(VecType& in, VecType& out, int nthreads = 1) const override {
        // For row permutation applied to x: x is not affected
        std::swap(in, out);
    }
    
    void applyInverseToX(VecType& in, VecType& out, int nthreads = 1) const override {
        // For row permutation: x is not affected by inverse either
        std::swap(in, out);
    }
    
private:
    std::span<const COLTYPE> _perm;
    int _base;
};

// Column permutation transformation: Q_c
template <SwappableResizableCSR MatType, typename VecType = std::vector<typename MatType::VALTYPE>>
    requires VectorForMatrix<VecType, MatType>
class ColumnPermutation : public TransformationBase<MatType, VecType> {
public:
    using VALTYPE = typename MatType::VALTYPE;
    using ROWTYPE = typename MatType::ROWTYPE;
    using COLTYPE = typename MatType::COLTYPE;
    
    explicit ColumnPermutation(const COLTYPE* perm, size_t size, int base = 0)
        : _perm(perm, size), _base(base) {}
    
    explicit ColumnPermutation(std::span<const COLTYPE> perm, int base = 0)
        : ColumnPermutation(perm.data(), perm.size(), base) {}
    
    void applyToOperator(MatType& in, MatType& out, int nthreads = 1) const override {
        // For column permutation: out = A * Q (permute columns of matrix)
        // This is matrix-specific and should be handled by LinearSolverSystem
        matrix_utils::permuteMat(in.rows, in.cols, static_cast<const COLTYPE*>(nullptr), _perm.data(),
                                 in.AI(), in.AJ(), in.AV(), out.AI(), out.AJ(), out.AV(), nthreads);
    }
    
    void applyToRHS(VecType& in, VecType& out, int nthreads = 1) const override {
        // Column permutation doesn't affect RHS
        std::swap(in, out);
    }

    void applyToX(VecType& in, VecType& out, int nthreads = 1) const override
    {
        // out = Q * in (apply column permutation to solution)k
        matrix_utils::invPermVec(_perm.size(), _base, in.data(), _perm.data(), out.data(), nthreads);
    }

    void applyInverseToX(VecType& in, VecType& out, int nthreads = 1) const override
    {
        // out = Q^{-1} * in
        matrix_utils::permVec(_perm.size(), _base, in.data(), _perm.data(), out.data(), nthreads);
    }

private:
    std::span<const COLTYPE> _perm;
    int _base;
};

// Row and column permutation transformation: P_r * A * Q_c^T
template <SwappableResizableCSR MatType, typename VecType = std::vector<typename MatType::VALTYPE>>
    requires VectorForMatrix<VecType, MatType>
class RowColPermutation : public TransformationBase<MatType, VecType> {
public:
    using VALTYPE = typename MatType::VALTYPE;
    using ROWTYPE = typename MatType::ROWTYPE;
    using COLTYPE = typename MatType::COLTYPE;
    
    explicit RowColPermutation(const COLTYPE* row_perm, const COLTYPE* col_perm, size_t size, int base = 0)
        : _row_perm(row_perm, size), _col_perm(col_perm, size), _base(base) {}
    
    explicit RowColPermutation(std::span<const COLTYPE> row_perm, std::span<const COLTYPE> col_perm, int base = 0)
        : RowColPermutation(row_perm.data(), col_perm.data(), row_perm.size(), base) {}
    
    void applyToOperator(MatType& in, MatType& out, int nthreads = 1) const override {
        // For row and column permutation: out = P_r * A * Q_c^T
        matrix_utils::permuteMat(in.rows, in.cols, _row_perm.data(), _col_perm.data(),
                                 in.AI(), in.AJ(), in.AV(), out.AI(), out.AJ(), out.AV(), nthreads);
    }
    
    void applyToRHS(VecType& in, VecType& out, int nthreads = 1) const override {
        // out = P_r * in (apply row permutation to RHS)
        matrix_utils::permVec(_row_perm.size(), _base, in.data(), _row_perm.data(), out.data(), nthreads);
    }
    
    void applyToX(VecType& in, VecType& out, int nthreads = 1) const override {
        // out = Q_c * in (apply column permutation to solution)
        matrix_utils::invPermVec(_col_perm.size(), _base, in.data(), _col_perm.data(), out.data(), nthreads);
    }
    
    void applyInverseToX(VecType& in, VecType& out, int nthreads = 1) const override {
        // out = Q_c^{-1} * in (apply inverse column permutation)
        matrix_utils::permVec(_col_perm.size(), _base, in.data(), _col_perm.data(), out.data(), nthreads);
    }
    
private:
    std::span<const COLTYPE> _row_perm;
    std::span<const COLTYPE> _col_perm;
    int _base;
};

// ============================================================================
// Scaling Transformations
// ============================================================================

// Row scaling transformation: S_r (scales rows of matrix, scales RHS)
template <SwappableResizableCSR MatType, typename VecType = std::vector<typename MatType::VALTYPE>>
    requires VectorForMatrix<VecType, MatType>
class RowScaling : public TransformationBase<MatType, VecType> {
public:
    using VALTYPE = typename MatType::VALTYPE;
    using ROWTYPE = typename MatType::ROWTYPE;
    using COLTYPE = typename MatType::COLTYPE;
    
    explicit RowScaling(const VALTYPE* scales, size_t size, int base = 0)
        : _scales(scales, size) {}
    
    explicit RowScaling(std::span<const VALTYPE> scales, int base = 0)
        : RowScaling(scales.data(), scales.size(), base) {}

    void applyToOperator(MatType& in, MatType& out, int nthreads = 1) const override
    {
        // For row scaling: out = S_r * A (scale rows of matrix)
        // This is matrix-specific and should be handled by LinearSolverSystem
        std::swap(in, out);
        matrix_utils::ScaleMat(out.rows, out.AI(), out.AJ(), out.AV(), _scales.data(),
                               static_cast<VALTYPE*>(nullptr), nthreads);
    }

    void applyToRHS(VecType& in, VecType& out, int nthreads = 1) const override {
        // out = S_r * in (scale RHS by row scales)
        std::swap(in, out);
        matrix_utils::ScaleVec(_scales.size(), out.data(), _scales.data(), nthreads);
    }
    
    void applyToX(VecType& in, VecType& out, int nthreads = 1) const override {
        // Row scaling doesn't affect solution x
        std::swap(in, out);
    }
    
    void applyInverseToX(VecType& in, VecType& out, int nthreads = 1) const override {
        // Row scaling doesn't affect solution x
        std::swap(in, out);
    }
    
private:
    std::span<const VALTYPE> _scales;
};

// Column scaling transformation: S_c (scales columns of matrix, scales solution)
template <SwappableResizableCSR MatType, typename VecType = std::vector<typename MatType::VALTYPE>>
    requires VectorForMatrix<VecType, MatType>
class ColumnScaling : public TransformationBase<MatType, VecType> {
public:
    using VALTYPE = typename MatType::VALTYPE;
    using ROWTYPE = typename MatType::ROWTYPE;
    using COLTYPE = typename MatType::COLTYPE;
    
    explicit ColumnScaling(const VALTYPE* scales, size_t size, int base = 0)
        : _scales(scales, size) {}
    
    explicit ColumnScaling(std::span<const VALTYPE> scales, int base = 0)
        : ColumnScaling(scales.data(), scales.size(), base) {}
    
    void applyToOperator(MatType& in, MatType& out, int nthreads = 1) const override {
        // For column scaling: out = A * S_c (scale columns of matrix)
        // This is matrix-specific and should be handled by LinearSolverSystem
        std::swap(in, out);
        matrix_utils::ScaleMat(out.rows, out.AI(), out.AJ(), out.AV(),
                               static_cast<VALTYPE*>(nullptr), _scales.data(), nthreads);
    }
    
    void applyToRHS(VecType& in, VecType& out, int nthreads = 1) const override {
        // Column scaling doesn't affect RHS
        std::swap(in, out);
    }
    
    void applyToX(VecType& in, VecType& out, int nthreads = 1) const override {
        // out = S_c * in (scale solution by column scales)
        std::swap(in, out);
        matrix_utils::ScaleVec(_scales.size(), out.data(), _scales.data(), nthreads);
    }
    
    void applyInverseToX(VecType& in, VecType& out, int nthreads = 1) const override {
        // out = S_c^{-1} * in
        std::swap(in, out);
        matrix_utils::InvScaleVec(_scales.size(), out.data(), _scales.data(), nthreads);
    }
    
private:
    std::span<const VALTYPE> _scales;
};

// Row and column scaling transformation: S_r * A * S_c (scales both rows and columns)
template <SwappableResizableCSR MatType, typename VecType = std::vector<typename MatType::VALTYPE>>
    requires VectorForMatrix<VecType, MatType>
class RowColScaling : public TransformationBase<MatType, VecType> {
public:
    using VALTYPE = typename MatType::VALTYPE;
    using ROWTYPE = typename MatType::ROWTYPE;
    using COLTYPE = typename MatType::COLTYPE;
    
    explicit RowColScaling(const VALTYPE* row_scales, const VALTYPE* col_scales, 
                           size_t size, int base = 0)
        : _row_scales(row_scales, size), _col_scales(col_scales, size) {}
    
    explicit RowColScaling(std::span<const VALTYPE> row_scales, std::span<const VALTYPE> col_scales,
                           int base = 0)
        : RowColScaling(row_scales.data(), col_scales.data(), row_scales.size(), base) {}
    
    void applyToOperator(MatType& in, MatType& out, int nthreads = 1) const override {
        // out = S_r * A * S_c (scale both rows and columns)
        std::swap(in, out);
        matrix_utils::ScaleMat(out.rows, out.AI(), out.AJ(), out.AV(),
                               _row_scales.data(), _col_scales.data(), nthreads);
    }
    
    void applyToRHS(VecType& in, VecType& out, int nthreads = 1) const override {
        // out = S_r * in (scale RHS by row scales)
        std::swap(in, out);
        matrix_utils::ScaleVec(_row_scales.size(), out.data(), _row_scales.data(), nthreads);
    }
    
    void applyToX(VecType& in, VecType& out, int nthreads = 1) const override {
        // out = S_c * in (scale solution by column scales)
        std::swap(in, out);
        matrix_utils::ScaleVec(_col_scales.size(), out.data(), _col_scales.data(), nthreads);
    }
    
    void applyInverseToX(VecType& in, VecType& out, int nthreads = 1) const override {
        // out = S_c^{-1} * in (apply inverse column scaling)
        std::swap(in, out);
        matrix_utils::InvScaleVec(_col_scales.size(), out.data(), _col_scales.data(), nthreads);
    }
    
private:
    std::span<const VALTYPE> _row_scales;
    std::span<const VALTYPE> _col_scales;
};

// ============================================================================
// Identity Transformation
// ============================================================================

// Identity transformation (no-op)
template <SwappableResizableCSR MatType, typename VecType = std::vector<typename MatType::VALTYPE>>
    requires VectorForMatrix<VecType, MatType>
class IdentityTransformation : public TransformationBase<MatType, VecType> {
public:
    using VALTYPE = typename MatType::VALTYPE;
    using ROWTYPE = typename MatType::ROWTYPE;
    using COLTYPE = typename MatType::COLTYPE;
    
    explicit IdentityTransformation() = default;
    
    void applyToOperator(MatType& in, MatType& out, int nthreads = 1) const override {
        std::swap(in, out);
    }
    
    void applyToRHS(VecType& in, VecType& out, int nthreads = 1) const override {
        std::swap(in, out);
    }
    
    void applyToX(VecType& in, VecType& out, int nthreads = 1) const override {
        std::swap(in, out);
    }
    
    void applyInverseToX(VecType& in, VecType& out, int nthreads = 1) const override {
        std::swap(in, out);
    }
};

} // namespace solver