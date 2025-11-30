#pragma once

#include <memory>
#include <vector>
#include <concepts>
#include <span>
#include "../sparse_mat_op/sparse_mat_traits.hpp"

namespace solver {

// Use VectorLike concept and SwappableResizableCSR from matrix_utils
using matrix_utils::VectorLike;
using matrix_utils::SwappableResizableCSR;

// Concept for transformation types
template <typename T, typename VALTYPE, typename VecType, typename MatType>
concept Transformation = VectorLike<VecType, VALTYPE> && SwappableResizableCSR<MatType> &&
    requires(T t, VecType& vec, MatType& mat) {
    { t.applyToOperator(mat, mat) } -> std::same_as<void>;
    { t.applyToRHS(vec, vec) } -> std::same_as<void>;
    { t.applyToX(vec, vec) } -> std::same_as<void>;
    { t.applyInverseToX(vec, vec) } -> std::same_as<void>;
};

// Base interface for transformations
template <typename VALTYPE, typename VecType = std::vector<VALTYPE>>
    requires VectorLike<VecType, VALTYPE>
class TransformationBase {
public:
    virtual ~TransformationBase() = default;
    
    // Apply transformation to operator (matrix): out = T * A * T^{-1} (for row/col transforms)
    // MatType must satisfy SwappableResizableCSR concept
    // Note: This is a template method that acts like a virtual function - derived classes should override it
    template<SwappableResizableCSR Mat>
    void applyToOperator(Mat& in, Mat& out) const {
        // Empty default - derived classes should provide their own implementation
        // Most transformations will just swap for identity or implement specific logic
    }
    
    // Apply transformation to RHS: out = T * in
    virtual void applyToRHS(VecType& in, VecType& out) const = 0;
    
    // Apply transformation to solution: out = T * in
    virtual void applyToX(VecType& in, VecType& out) const = 0;
    
    // Apply inverse transformation to solution: out = T^{-1} * in
    virtual void applyInverseToX(VecType& in, VecType& out) const = 0;
    
    // Get dimension
    virtual size_t dimension() const = 0;
    
    // Get base (0 or 1)
    virtual int base() const = 0;
};

// Row permutation transformation: P_r
template <typename VALTYPE, typename VecType = std::vector<VALTYPE>, typename INDEXTYPE = int>
    requires VectorLike<VecType, VALTYPE>
class RowPermutation : public TransformationBase<VALTYPE, VecType> {
public:
    explicit RowPermutation(const INDEXTYPE* perm, size_t size, int base = 0) 
        : _perm(perm, size), _base(base) {}
    
    explicit RowPermutation(std::span<const INDEXTYPE> perm, int base = 0)
        : RowPermutation(perm.data(), perm.size(), base) {}
    
    template<SwappableResizableCSR Mat>
    void applyToOperator(Mat& in, Mat& out) const {
        // For row permutation: out = P * A (permute rows of matrix)
        // This is matrix-specific and should be handled by LinearSolverSystem
        std::swap(in, out);
    }
    
    void applyToRHS(VecType& in, VecType& out) const override {
        // out = P * in
        for (size_t i = 0; i < _perm.size(); ++i) {
            out[i] = in[_perm[i] - _base];
        }
        std::swap(in, out);
    }
    
    void applyToX(VecType& in, VecType& out) const override {
        // For row permutation applied to x: x is not affected
        std::swap(in, out);
    }
    
    void applyInverseToX(VecType& in, VecType& out) const override {
        // For row permutation: x is not affected by inverse either
        std::swap(in, out);
    }
    
    size_t dimension() const override { return _perm.size(); }
    int base() const override { return _base; }
    
    std::span<const INDEXTYPE> permutation() const { return _perm; }
    std::span<const INDEXTYPE> inversePermutation() const { 
        ensureInverseComputed();
        return _inv_perm; 
    }
    
private:
    void ensureInverseComputed() const {
        if (_inv_perm_data.empty()) {
            _inv_perm_data.resize(_perm.size());
            _inv_perm = std::span<INDEXTYPE>(_inv_perm_data);
            // Compute inverse permutation
            for (size_t i = 0; i < _perm.size(); ++i) {
                _inv_perm[_perm[i] - _base] = static_cast<INDEXTYPE>(i + _base);
            }
        }
    }
    
    std::span<const INDEXTYPE> _perm;
    mutable std::vector<INDEXTYPE> _inv_perm_data;
    mutable std::span<INDEXTYPE> _inv_perm;
    int _base;
};

// Column permutation transformation: Q_c
template <typename VALTYPE, typename VecType = std::vector<VALTYPE>, typename INDEXTYPE = int>
    requires VectorLike<VecType, VALTYPE>
class ColumnPermutation : public TransformationBase<VALTYPE, VecType> {
public:
    explicit ColumnPermutation(const INDEXTYPE* perm, size_t size, int base = 0)
        : _perm(perm, size), _base(base) {}
    
    explicit ColumnPermutation(std::span<const INDEXTYPE> perm, int base = 0)
        : ColumnPermutation(perm.data(), perm.size(), base) {}
    
    template<SwappableResizableCSR Mat>
    void applyToOperator(Mat& in, Mat& out) const {
        // For column permutation: out = A * Q (permute columns of matrix)
        // This is matrix-specific and should be handled by LinearSolverSystem
        std::swap(in, out);
    }
    
    void applyToRHS(VecType& in, VecType& out) const override {
        // Column permutation doesn't affect RHS
        std::swap(in, out);
    }
    
    void applyToX(VecType& in, VecType& out) const override {
        // out = Q * in (apply column permutation to solution)
        ensureInverseComputed();
        for (size_t i = 0; i < _inv_perm.size(); ++i) {
            out[i] = in[_inv_perm[i] - _base];
        }
        std::swap(in, out);
    }
    
    void applyInverseToX(VecType& in, VecType& out) const override {
        // out = Q^{-1} * in
        for (size_t i = 0; i < _perm.size(); ++i) {
            out[i] = in[_perm[i] - _base];
        }
        std::swap(in, out);
    }
    
    size_t dimension() const override { return _perm.size(); }
    int base() const override { return _base; }
    
    std::span<const INDEXTYPE> permutation() const { return _perm; }
    std::span<const INDEXTYPE> inversePermutation() const { 
        ensureInverseComputed();
        return _inv_perm; 
    }
    
private:
    void ensureInverseComputed() const {
        if (_inv_perm_data.empty()) {
            _inv_perm_data.resize(_perm.size());
            _inv_perm = std::span<INDEXTYPE>(_inv_perm_data);
            // Compute inverse permutation
            for (size_t i = 0; i < _perm.size(); ++i) {
                _inv_perm[_perm[i] - _base] = static_cast<INDEXTYPE>(i + _base);
            }
        }
    }
    
    std::span<const INDEXTYPE> _perm;
    mutable std::vector<INDEXTYPE> _inv_perm_data;
    mutable std::span<INDEXTYPE> _inv_perm;
    int _base;
};

// Diagonal scaling transformation: S_r or S_c
template <typename VALTYPE, typename VecType = std::vector<VALTYPE>>
    requires VectorLike<VecType, VALTYPE>
class DiagonalScaling : public TransformationBase<VALTYPE, VecType> {
public:
    explicit DiagonalScaling(const VALTYPE* scales, size_t size, int base = 0)
        : _scales(scales, size), _base(base) {}
    
    explicit DiagonalScaling(std::span<const VALTYPE> scales, int base = 0)
        : DiagonalScaling(scales.data(), scales.size(), base) {}
    
    template<SwappableResizableCSR Mat>
    void applyToOperator(Mat& in, Mat& out) const {
        // For diagonal scaling: handled by LinearSolverSystem for matrix operations
        std::swap(in, out);
    }
    
    void applyToRHS(VecType& in, VecType& out) const override {
        // out = S * in (scale RHS)
        for (size_t i = 0; i < _scales.size(); ++i) {
            out[i] = _scales[i] * in[i];
        }
        std::swap(in, out);
    }
    
    void applyToX(VecType& in, VecType& out) const override {
        // out = S * in (scale solution)
        for (size_t i = 0; i < _scales.size(); ++i) {
            out[i] = _scales[i] * in[i];
        }
        std::swap(in, out);
    }
    
    void applyInverseToX(VecType& in, VecType& out) const override {
        // out = S^{-1} * in
        ensureInverseComputed();
        for (size_t i = 0; i < _inv_scales.size(); ++i) {
            out[i] = _inv_scales[i] * in[i];
        }
        std::swap(in, out);
    }
    
    size_t dimension() const override { return _scales.size(); }
    int base() const override { return _base; }
    
    std::span<const VALTYPE> scales() const { return _scales; }
    std::span<const VALTYPE> inverseScales() const { 
        ensureInverseComputed();
        return _inv_scales; 
    }
    
private:
    void ensureInverseComputed() const {
        if (_inv_scales_data.empty()) {
            _inv_scales_data.resize(_scales.size());
            _inv_scales = std::span<VALTYPE>(_inv_scales_data);
            // Compute inverse scales
            for (size_t i = 0; i < _scales.size(); ++i) {
                _inv_scales[i] = static_cast<VALTYPE>(1) / _scales[i];
            }
        }
    }
    
    std::span<const VALTYPE> _scales;
    mutable std::vector<VALTYPE> _inv_scales_data;
    mutable std::span<VALTYPE> _inv_scales;
    int _base;
};

// Row scaling transformation: S_r (scales rows of matrix, scales RHS)
template <typename VALTYPE, typename VecType = std::vector<VALTYPE>>
    requires VectorLike<VecType, VALTYPE>
class RowScaling : public TransformationBase<VALTYPE, VecType> {
public:
    explicit RowScaling(const VALTYPE* scales, size_t size, int base = 0)
        : _scales(scales, size), _base(base) {}
    
    explicit RowScaling(std::span<const VALTYPE> scales, int base = 0)
        : RowScaling(scales.data(), scales.size(), base) {}
    
    template<SwappableResizableCSR Mat>
    void applyToOperator(Mat& in, Mat& out) const {
        // For row scaling: out = S_r * A (scale rows of matrix)
        // This is matrix-specific and should be handled by LinearSolverSystem
        std::swap(in, out);
    }
    
    void applyToRHS(VecType& in, VecType& out) const override {
        // out = S_r * in (scale RHS by row scales)
        for (size_t i = 0; i < _scales.size(); ++i) {
            out[i] = _scales[i] * in[i];
        }
        std::swap(in, out);
    }
    
    void applyToX(VecType& in, VecType& out) const override {
        // Row scaling doesn't affect solution x
        std::swap(in, out);
    }
    
    void applyInverseToX(VecType& in, VecType& out) const override {
        // Row scaling doesn't affect solution x
        std::swap(in, out);
    }
    
    size_t dimension() const override { return _scales.size(); }
    int base() const override { return _base; }
    
    std::span<const VALTYPE> scales() const { return _scales; }
    std::span<const VALTYPE> inverseScales() const { 
        ensureInverseComputed();
        return _inv_scales; 
    }
    
private:
    void ensureInverseComputed() const {
        if (_inv_scales_data.empty()) {
            _inv_scales_data.resize(_scales.size());
            _inv_scales = std::span<VALTYPE>(_inv_scales_data);
            // Compute inverse scales
            for (size_t i = 0; i < _scales.size(); ++i) {
                _inv_scales[i] = static_cast<VALTYPE>(1) / _scales[i];
            }
        }
    }
    
    std::span<const VALTYPE> _scales;
    mutable std::vector<VALTYPE> _inv_scales_data;
    mutable std::span<VALTYPE> _inv_scales;
    int _base;
};

// Column scaling transformation: S_c (scales columns of matrix, scales solution)
template <typename VALTYPE, typename VecType = std::vector<VALTYPE>>
    requires VectorLike<VecType, VALTYPE>
class ColumnScaling : public TransformationBase<VALTYPE, VecType> {
public:
    explicit ColumnScaling(const VALTYPE* scales, size_t size, int base = 0)
        : _scales(scales, size), _base(base) {}
    
    explicit ColumnScaling(std::span<const VALTYPE> scales, int base = 0)
        : ColumnScaling(scales.data(), scales.size(), base) {}
    
    template<SwappableResizableCSR Mat>
    void applyToOperator(Mat& in, Mat& out) const {
        // For column scaling: out = A * S_c (scale columns of matrix)
        // This is matrix-specific and should be handled by LinearSolverSystem
        std::swap(in, out);
    }
    
    void applyToRHS(VecType& in, VecType& out) const override {
        // Column scaling doesn't affect RHS
        std::swap(in, out);
    }
    
    void applyToX(VecType& in, VecType& out) const override {
        // out = S_c * in (scale solution by column scales)
        for (size_t i = 0; i < _scales.size(); ++i) {
            out[i] = _scales[i] * in[i];
        }
        std::swap(in, out);
    }
    
    void applyInverseToX(VecType& in, VecType& out) const override {
        // out = S_c^{-1} * in
        ensureInverseComputed();
        for (size_t i = 0; i < _inv_scales.size(); ++i) {
            out[i] = _inv_scales[i] * in[i];
        }
        std::swap(in, out);
    }
    
    size_t dimension() const override { return _scales.size(); }
    int base() const override { return _base; }
    
    std::span<const VALTYPE> scales() const { return _scales; }
    std::span<const VALTYPE> inverseScales() const { 
        ensureInverseComputed();
        return _inv_scales; 
    }
    
private:
    void ensureInverseComputed() const {
        if (_inv_scales_data.empty()) {
            _inv_scales_data.resize(_scales.size());
            _inv_scales = std::span<VALTYPE>(_inv_scales_data);
            // Compute inverse scales
            for (size_t i = 0; i < _scales.size(); ++i) {
                _inv_scales[i] = static_cast<VALTYPE>(1) / _scales[i];
            }
        }
    }
    
    std::span<const VALTYPE> _scales;
    mutable std::vector<VALTYPE> _inv_scales_data;
    mutable std::span<VALTYPE> _inv_scales;
    int _base;
};

// Identity transformation (no-op)
template <typename VALTYPE, typename VecType = std::vector<VALTYPE>>
    requires VectorLike<VecType, VALTYPE>
class IdentityTransformation : public TransformationBase<VALTYPE, VecType> {
public:
    explicit IdentityTransformation(size_t dim, int base = 0) : _dim(dim), _base(base) {}
    
    template<SwappableResizableCSR Mat>
    void applyToOperator(Mat& in, Mat& out) const {
        std::swap(in, out);
    }
    
    void applyToRHS(VecType& in, VecType& out) const override {
        std::swap(in, out);
    }
    
    void applyToX(VecType& in, VecType& out) const override {
        std::swap(in, out);
    }
    
    void applyInverseToX(VecType& in, VecType& out) const override {
        std::swap(in, out);
    }
    
    size_t dimension() const override { return _dim; }
    int base() const override { return _base; }
    
private:
    size_t _dim;
    int _base;
};

}