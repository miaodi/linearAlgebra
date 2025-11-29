#pragma once

#include "LinearSolverSystem.hpp"
#include "Transformation.hpp"
#include "matrix_utils.hpp"
#include <vector>

/**
 * Example usage of LinearSolverSystem with transformations
 * 
 * This demonstrates how to:
 * 1. Create a linear system with matrix A, RHS b, and initial solution x_init
 * 2. Apply row and column transformations (permutations and scalings)
 * 3. Get the transformed system \bar{A}\bar{x} = \bar{b}
 * 4. Solve the transformed system
 * 5. Recover the original solution x
 */

namespace solver {

template <typename CSRMatrixType>
class LinearSolverExample {
public:
    using VALTYPE = typename CSRMatrixType::VALTYPE;
    
    static void demonstrateUsage() {
        // 1. Setup original system: A x = b
        CSRMatrixType A = createExampleMatrix();
        std::vector<VALTYPE> b = createExampleRHS(A.rows);
        std::vector<VALTYPE> x_init(A.rows, 0.0); // Initial guess
        
        // 2. Create solver system
        LinearSolverSystem<CSRMatrixType> system;
        system.setMatrix(A);
        
        // 3. Add transformations
        // Example: Row scaling S_r^1 (e.g., equilibration)
        auto row_scaling = createRowScaling(A);
        system.addRowTransformation(row_scaling);
        
        // Example: Row permutation P_r^2 (e.g., for stability)
        auto row_perm = createRowPermutation(A.rows);
        system.addRowTransformation(row_perm);
        
        // Example: Column scaling S_c^1 (e.g., equilibration)
        auto col_scaling = createColumnScaling(A);
        system.addColumnTransformation(col_scaling);
        
        // Example: Column permutation Q_c^2 (e.g., for fill reduction)
        auto col_perm = createColumnPermutation(A.rows);
        system.addColumnTransformation(col_perm);
        
        // 4. Get transformed system
        CSRMatrixType A_bar = system.getTransformedMatrix();
        std::vector<VALTYPE> b_bar = system.transformRHS(b);
        std::vector<VALTYPE> x_bar = system.transformSolution(x_init);
        
        // 5. Solve transformed system: A_bar * x_bar = b_bar
        // (Use your favorite solver here)
        std::vector<VALTYPE> x_bar_solution = solveSystem(A_bar, b_bar, x_bar);
        
        // 6. Recover original solution: x = S_c^1 S_c^2 Q_c^3 * x_bar_solution
        std::vector<VALTYPE> x_solution = system.recoverSolution(x_bar_solution);
        
        // x_solution now contains the solution to the original system A x = b
    }
    
private:
    static CSRMatrixType createExampleMatrix() {
        // Create a simple example matrix
        CSRMatrixType A;
        A.rows = 4;
        A.ai = {0, 2, 4, 6, 8};
        A.aj = {0, 1, 0, 1, 2, 3, 2, 3};
        A.av = {4.0, -1.0, -1.0, 4.0, 4.0, -1.0, -1.0, 4.0};
        return A;
    }
    
    static std::vector<VALTYPE> createExampleRHS(size_t n) {
        return std::vector<VALTYPE>(n, 1.0);
    }
    
    static std::shared_ptr<TransformationBase<VALTYPE>> createRowScaling(const CSRMatrixType& A) {
        // Example: Scale each row by 1/max(|row|)
        std::vector<VALTYPE> scales(A.rows);
        for (size_t i = 0; i < A.rows; ++i) {
            VALTYPE max_val = 0;
            for (auto j = A.ai[i]; j < A.ai[i + 1]; ++j) {
                max_val = std::max(max_val, std::abs(A.av[j - A.Base()]));
            }
            scales[i] = (max_val > 0) ? (1.0 / max_val) : 1.0;
        }
        return std::make_shared<DiagonalScaling<VALTYPE>>(scales);
    }
    
    static std::shared_ptr<TransformationBase<VALTYPE>> createColumnScaling(const CSRMatrixType& A) {
        // Example: Scale each column by 1/max(|column|)
        std::vector<VALTYPE> scales(A.rows, 0);
        
        // Find max in each column
        for (size_t i = 0; i < A.rows; ++i) {
            for (auto j = A.ai[i]; j < A.ai[i + 1]; ++j) {
                size_t col = A.aj[j - A.Base()] - A.Base();
                scales[col] = std::max(scales[col], std::abs(A.av[j - A.Base()]));
            }
        }
        
        // Compute scaling factors
        for (auto& s : scales) {
            s = (s > 0) ? (1.0 / s) : 1.0;
        }
        
        return std::make_shared<DiagonalScaling<VALTYPE>>(scales);
    }
    
    static std::shared_ptr<TransformationBase<VALTYPE>> createRowPermutation(size_t n) {
        // Example: Identity permutation (could be replaced with AMD, RCM, etc.)
        std::vector<int> perm(n);
        for (size_t i = 0; i < n; ++i) {
            perm[i] = static_cast<int>(i);
        }
        return std::make_shared<RowPermutation<VALTYPE>>(perm);
    }
    
    static std::shared_ptr<TransformationBase<VALTYPE>> createColumnPermutation(size_t n) {
        // Example: Identity permutation (could be replaced with AMD, RCM, etc.)
        std::vector<int> perm(n);
        for (size_t i = 0; i < n; ++i) {
            perm[i] = static_cast<int>(i);
        }
        return std::make_shared<ColumnPermutation<VALTYPE>>(perm);
    }
    
    static std::vector<VALTYPE> solveSystem(const CSRMatrixType& A, 
                                           const std::vector<VALTYPE>& b,
                                           const std::vector<VALTYPE>& x_init) {
        // Placeholder for actual solver (e.g., GMRES, BiCGSTAB, etc.)
        return x_init;
    }
};

} // namespace solver
