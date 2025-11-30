#pragma once

#include "Transformation.hpp"   
#include "matrix_utils.hpp"
#include <memory>
#include <vector>
#include <stdexcept>

namespace solver
{

/**
 * Linear solver system with matrix preconditioning transformations
 * 
 * Handles the transformation chain:
 * \bar{A} = P_r^3 S_r^2 P_r^1 A S_c^1 S_c^2 Q_c^3
 * \bar{b} = P_r^3 S_r^2 P_r^1 b
 * \bar{x} = (S_c^1 S_c^2 Q_c^3)^{-1} x_init
 * 
 * After solving \bar{A} \bar{x} = \bar{b}, recovers x = S_c^1 S_c^2 Q_c^3 \bar{x}
 */
template <typename CSRMatrixType>
class LinearSolverSystem
{
public:
    using ROWTYPE = typename CSRMatrixType::ROWTYPE;
    using COLTYPE = typename CSRMatrixType::COLTYPE;
    using VALTYPE = typename CSRMatrixType::VALTYPE;
    using TransformPtr = std::shared_ptr<TransformationBase<VALTYPE>>;
    
    LinearSolverSystem() = default;
    
    /**
     * Set the original matrix A
     */
    void setMatrix(const CSRMatrixType& A) {
        _original_matrix = A;
        _rows = A.rows;
        _cols = A.rows; // Assuming square matrix
        _transformed = false;
    }
    
    /**
     * Add a row transformation (applied left to right: newest first)
     * E.g., addRowTransformation(P_r^1), then addRowTransformation(S_r^2), then addRowTransformation(P_r^3)
     * Results in: P_r^3 S_r^2 P_r^1
     */
    void addRowTransformation(TransformPtr transform) {
        if (transform->dimension() != _rows) {
            throw std::runtime_error("Row transformation dimension mismatch");
        }
        _row_transforms.push_back(transform);
        _transformed = false;
    }
    
    /**
     * Add a column transformation (applied right to left: newest last)
     * E.g., addColumnTransformation(S_c^1), then addColumnTransformation(S_c^2), then addColumnTransformation(Q_c^3)
     * Results in: S_c^1 S_c^2 Q_c^3
     */
    void addColumnTransformation(TransformPtr transform) {
        if (transform->dimension() != _cols) {
            throw std::runtime_error("Column transformation dimension mismatch");
        }
        _col_transforms.push_back(transform);
        _transformed = false;
    }
    
    /**
     * Clear all transformations
     */
    void clearTransformations() {
        _row_transforms.clear();
        _col_transforms.clear();
        _transformed = false;
    }
    
    /**
     * Apply all transformations to get \bar{A}
     * This performs: \bar{A} = P_r^3 S_r^2 P_r^1 A S_c^1 S_c^2 Q_c^3
     */
    CSRMatrixType getTransformedMatrix() {
        if (_transformed) {
            return _transformed_matrix;
        }
        
        CSRMatrixType result = _original_matrix;
        
        // Apply column transformations: A * S_c^1 * S_c^2 * Q_c^3
        for (const auto& transform : _col_transforms) {
            result = applyColumnTransform(result, transform);
        }
        
        // Apply row transformations: P_r^3 * S_r^2 * P_r^1 * (previous result)
        for (auto it = _row_transforms.rbegin(); it != _row_transforms.rend(); ++it) {
            result = applyRowTransform(result, *it);
        }
        
        _transformed_matrix = result;
        _transformed = true;
        return result;
    }
    
    /**
     * Transform RHS: \bar{b} = P_r^3 S_r^2 P_r^1 b
     */
    std::vector<VALTYPE> transformRHS(const std::vector<VALTYPE>& b) const {
        std::vector<VALTYPE> result = b;
        std::vector<VALTYPE> temp(b.size());
        
        // Apply row transformations in order: P_r^1, S_r^2, P_r^3
        for (const auto& transform : _row_transforms) {
            transform->apply(result.data(), temp.data());
            result = std::move(temp);
            temp.resize(result.size());
        }
        
        return result;
    }
    
    /**
     * Transform initial solution: \bar{x} = (S_c^1 S_c^2 Q_c^3)^{-1} x_init
     */
    std::vector<VALTYPE> transformSolution(const std::vector<VALTYPE>& x_init) const {
        std::vector<VALTYPE> result = x_init;
        std::vector<VALTYPE> temp(x_init.size());
        
        // Apply inverse of column transformations in reverse order: Q_c^3^{-1}, S_c^2^{-1}, S_c^1^{-1}
        for (auto it = _col_transforms.rbegin(); it != _col_transforms.rend(); ++it) {
            (*it)->applyInverseToX(result.data(), temp.data());
            result = std::move(temp);
            temp.resize(result.size());
        }
        
        return result;
    }
    
    /**
     * Recover original solution: x = S_c^1 S_c^2 Q_c^3 \bar{x}
     */
    std::vector<VALTYPE> recoverSolution(const std::vector<VALTYPE>& x_bar) const {
        std::vector<VALTYPE> result = x_bar;
        std::vector<VALTYPE> temp(x_bar.size());
        
        // Apply column transformations in order: S_c^1, S_c^2, Q_c^3
        for (const auto& transform : _col_transforms) {
            transform->applyToX(result.data(), temp.data());
            result = std::move(temp);
            temp.resize(result.size());
        }
        
        return result;
    }
    
    /**
     * Get the original matrix
     */
    const CSRMatrixType& getOriginalMatrix() const { return _original_matrix; }
    
    /**
     * Check if matrix has been transformed
     */
    bool isTransformed() const { return _transformed; }
    
    /**
     * Get number of row transformations
     */
    size_t numRowTransformations() const { return _row_transforms.size(); }
    
    /**
     * Get number of column transformations
     */
    size_t numColumnTransformations() const { return _col_transforms.size(); }

private:
    CSRMatrixType _original_matrix;
    CSRMatrixType _transformed_matrix;
    std::vector<TransformPtr> _row_transforms;
    std::vector<TransformPtr> _col_transforms;
    size_t _rows = 0;
    size_t _cols = 0;
    bool _transformed = false;
    
    /**
     * Apply column transformation to matrix: A_new = A * T
     */
    CSRMatrixType applyColumnTransform(const CSRMatrixType& A, TransformPtr transform) {
        // For column transformation, we need to scale/permute columns
        CSRMatrixType result = A;
        
        // Handle different transformation types
        if (auto* perm = dynamic_cast<ColumnPermutation<VALTYPE>*>(transform.get())) {
            result = applyColumnPermutation(A, perm->permutation());
        } else if (auto* scale = dynamic_cast<DiagonalScaling<VALTYPE>*>(transform.get())) {
            result = applyColumnScaling(A, scale->scales());
        }
        
        return result;
    }
    
    /**
     * Apply row transformation to matrix: A_new = T * A
     */
    CSRMatrixType applyRowTransform(const CSRMatrixType& A, TransformPtr transform) {
        // For row transformation, we need to scale/permute rows
        CSRMatrixType result = A;
        
        // Handle different transformation types
        if (auto* perm = dynamic_cast<RowPermutation<VALTYPE>*>(transform.get())) {
            result = applyRowPermutation(A, perm->permutation());
        } else if (auto* scale = dynamic_cast<DiagonalScaling<VALTYPE>*>(transform.get())) {
            result = applyRowScaling(A, scale->scales());
        }
        
        return result;
    }
    
    /**
     * Apply row permutation: permute rows according to perm
     */
    CSRMatrixType applyRowPermutation(const CSRMatrixType& A, std::span<const int> perm) {
        CSRMatrixType result;
        result.rows = A.rows;
        result.ai.resize(A.rows + 1);
        result.ai[0] = A.Base();
        
        // Permute rows
        for (size_t i = 0; i < A.rows; ++i) {
            size_t src_row = perm[i];
            ROWTYPE nnz_in_row = A.ai[src_row + 1] - A.ai[src_row];
            result.ai[i + 1] = result.ai[i] + nnz_in_row;
        }
        
        result.aj.reserve(A.NNZ());
        result.av.reserve(A.NNZ());
        
        for (size_t i = 0; i < A.rows; ++i) {
            size_t src_row = perm[i];
            for (ROWTYPE j = A.ai[src_row]; j < A.ai[src_row + 1]; ++j) {
                result.aj.push_back(A.aj[j - A.Base()]);
                result.av.push_back(A.av[j - A.Base()]);
            }
        }
        
        return result;
    }
    
    /**
     * Apply row scaling: multiply each row by its scale factor
     */
    CSRMatrixType applyRowScaling(const CSRMatrixType& A, std::span<const VALTYPE> scales) {
        CSRMatrixType result = A;
        
        for (size_t i = 0; i < A.rows; ++i) {
            VALTYPE scale = scales[i];
            for (ROWTYPE j = A.ai[i]; j < A.ai[i + 1]; ++j) {
                result.av[j - A.Base()] *= scale;
            }
        }
        
        return result;
    }
    
    /**
     * Apply column permutation: permute columns according to perm
     */
    CSRMatrixType applyColumnPermutation(const CSRMatrixType& A, std::span<const int> perm) {
        CSRMatrixType result = A;
        
        // For each nonzero, update its column index according to permutation
        for (size_t i = 0; i < A.NNZ(); ++i) {
            COLTYPE old_col = A.aj[i] - A.Base();
            result.aj[i] = perm[old_col] + A.Base();
        }
        
        return result;
    }
    
    /**
     * Apply column scaling: multiply each column by its scale factor
     */
    CSRMatrixType applyColumnScaling(const CSRMatrixType& A, std::span<const VALTYPE> scales) {
        CSRMatrixType result = A;
        
        for (size_t i = 0; i < A.NNZ(); ++i) {
            COLTYPE col = A.aj[i] - A.Base();
            result.av[i] *= scales[col];
        }
        
        return result;
    }
};

} // namespace solver
