#include "ruiz_scale.hpp"
#include <iostream>
#include <numeric>

namespace scaling
{

// Implementation of Ruiz scaling algorithm
// Reference: D. Ruiz, "A scaling algorithm to equilibrate both rows and columns norms in matrices"
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, RuizScalingNormType NORM>
bool RuizScaleSerial(const COLTYPE rows, const COLTYPE cols, ROWTYPE const* ai, COLTYPE const* aj,
                     VALTYPE* av, VALTYPE* dr, VALTYPE* dc, const int max_iters,
                     const VALTYPE tol)
{
    const auto base = ai[0];
    constexpr VALTYPE zero = static_cast<VALTYPE>(0);
    constexpr VALTYPE one = static_cast<VALTYPE>(1);
    constexpr VALTYPE epsilon = static_cast<VALTYPE>(1e-20);
    
    // Initialize scaling factors to 1
    std::memset(dr, static_cast<VALTYPE>(1), sizeof(VALTYPE) * rows);
    std::memset(dc, static_cast<VALTYPE>(1), sizeof(VALTYPE) * cols);
    std::vector<VALTYPE> row_norms(rows);
    std::vector<VALTYPE> col_norms(cols);

    VALTYPE avg_row_norm = zero;

    for (int iter = 0; iter < max_iters; ++iter)
    {
        // Step 1: Compute row and column norms
        std::fill(row_norms.begin(), row_norms.end(), zero);
        std::fill(col_norms.begin(), col_norms.end(), zero);
        
        for (COLTYPE i = 0; i < rows; ++i)
        {
            for (ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; ++j)
            {
                const COLTYPE col = aj[j] - base;
                const VALTYPE val = av[j];
                if constexpr (NORM == RuizScalingNormType::MaxNorm)
                {
                    row_norms[i] = std::max(row_norms[i], std::abs(val));
                }
                else if constexpr (NORM == RuizScalingNormType::L2Norm)
                {
                    row_norms[i] = std::fma(val, val, row_norms[i]);
                }

                if constexpr (NORM == RuizScalingNormType::MaxNorm)
                {
                    col_norms[col] = std::max(col_norms[col], std::abs(val));
                }
                else if constexpr (NORM == RuizScalingNormType::L2Norm)
                {
                    col_norms[col] = std::fma(val, val, col_norms[col]);
                }
            }
            // Finalize L2 norm with sqrt
            if constexpr (NORM == RuizScalingNormType::L2Norm)
            {
                row_norms[i] = std::sqrt(row_norms[i]);
            }
            avg_row_norm += row_norms[i];
            
            // Step 2: Compute scaling factor as 1/sqrt(norm)
            // Using sqrt gives better conditioning than direct inversion
            if (row_norms[i] > zero)
            {
                row_norms[i] = one / std::sqrt(row_norms[i]);
            }
            else
            {
                row_norms[i] = one;
            }
        }
        
        VALTYPE avg_col_norm = zero;
        // Finalize column norms and compute scaling factors
        for (COLTYPE j = 0; j < cols; ++j)
        {
            if constexpr (NORM == RuizScalingNormType::L2Norm)
            {
                col_norms[j] = std::sqrt(col_norms[j]);
            }
            avg_col_norm += col_norms[j];
            
            if (col_norms[j] > zero)
            {
                col_norms[j] = one / std::sqrt(col_norms[j]);
            }
            else
            {
                col_norms[j] = one;
            }
        }

        avg_row_norm /= static_cast<VALTYPE>(rows);
        avg_col_norm /= static_cast<VALTYPE>(cols);

        // Step 3: Apply scaling to matrix and accumulate into dr, dc
        VALTYPE max_dev = zero;
        for (COLTYPE i = 0; i < rows; ++i)
        {
            dr[i] *= row_norms[i];
        }
        for (COLTYPE j = 0; j < cols; ++j)
        {
            dc[j] *= col_norms[j];
        }
        
        // Scale matrix entries and track maximum relative change
        for (COLTYPE i = 0; i < rows; ++i)
        {
            for (ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; ++j)
            {
                const COLTYPE col = aj[j] - base;
                const VALTYPE old_val = av[j];
                const VALTYPE new_val = old_val * row_norms[i] * col_norms[col];
                av[j] = new_val;
                const VALTYPE dev = std::abs(new_val - old_val) / (std::abs(old_val) + epsilon);
                max_dev = std::max(max_dev, dev);
            }
        }
        
        std::cout << "Iteration " << iter << ": avg_row_norm = " << avg_row_norm 
                  << ", avg_col_norm = " << avg_col_norm 
                  << ", max_dev = " << max_dev << std::endl;
        
        // Check convergence: if maximum relative change is below tolerance, stop
        if (max_dev < tol)
        {
            std::cout << "Converged after " << (iter + 1) << " iterations" << std::endl;
            return true;
        }
    }
    std::cout << "Did not converge after " << max_iters << " iterations" << std::endl;
    return false;
}

// Explicit template instantiations for common types
#define INSTANTIATE_RUIZ_SCALE(ROWTYPE, COLTYPE, VALTYPE) \
    template bool RuizScaleSerial<ROWTYPE, COLTYPE, VALTYPE, RuizScalingNormType::MaxNorm>( \
        const COLTYPE, const COLTYPE, ROWTYPE const*, COLTYPE const*, \
        VALTYPE*, VALTYPE*, VALTYPE*, const int, const VALTYPE); \
    template bool RuizScaleSerial<ROWTYPE, COLTYPE, VALTYPE, RuizScalingNormType::L2Norm>( \
        const COLTYPE, const COLTYPE, ROWTYPE const*, COLTYPE const*, \
        VALTYPE*, VALTYPE*, VALTYPE*, const int, const VALTYPE);

// int32, float
INSTANTIATE_RUIZ_SCALE(int32_t, int32_t, float)
// int32, double
INSTANTIATE_RUIZ_SCALE(int32_t, int32_t, double)
// int64, float
INSTANTIATE_RUIZ_SCALE(int64_t, int64_t, float)
// int64, double
INSTANTIATE_RUIZ_SCALE(int64_t, int64_t, double)

#undef INSTANTIATE_RUIZ_SCALE

} // namespace scaling
