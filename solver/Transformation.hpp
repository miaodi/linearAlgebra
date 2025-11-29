#pragma once

#include <memory>
#include <vector>
#include <concepts>
#include <span>
#include "sparse_mat_traits.hpp"

namespace solver {

// Use VectorLike concept from matrix_utils
using matrix_utils::VectorLike;// Concept for transformation types
template <typename T, typename VALTYPE, typename VecType>
concept Transformation = VectorLike<VecType, VALTYPE> && requires(T t, VecType& vec) {
    { t.applyToOperator(vec, vec) } -> std::same_as<void>;
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
    virtual void applyToOperator(VecType& in, VecType& out) const = 0;
    
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
    
    void applyToOperator(VecType& in, VecType& out) const override {
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
    
    void applyToOperator(VecType& in, VecType& out) const override {
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
    
    void applyToOperator(VecType& in, VecType& out) const override {
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

// Identity transformation (no-op)
template <typename VALTYPE, typename VecType = std::vector<VALTYPE>>
    requires VectorLike<VecType, VALTYPE>
class IdentityTransformation : public TransformationBase<VALTYPE, VecType> {
public:
    explicit IdentityTransformation(size_t dim, int base = 0) : _dim(dim), _base(base) {}
    
    void applyToOperator(VecType& in, VecType& out) const override {
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