#pragma once

#include <immintrin.h>
#include <type_traits>

namespace matrix_utils
{
enum class BetaMode
{
    Zero,
    One,
    Generic
};
enum class RowDotKernel
{
    Scalar,
    Simd
};

/// @brief Apply beta scaling to result: y = ax + beta * x_old
/// @details Optimized for common beta values (0 and 1) via template specialization
/// @tparam Mode BetaMode::Zero (beta=0), One (beta=1), or Generic (any beta)
/// @param ax The computed value (alpha * A * x)
/// @param beta The scaling factor for x_old
/// @param x_old The previous value of x
/// @return The final result: ax + beta * x_old
template <BetaMode Mode, typename VALTYPE>
inline VALTYPE apply_beta(const VALTYPE ax, const VALTYPE beta, const VALTYPE x_old)
{
    if constexpr (Mode == BetaMode::Zero)
    {
        return ax;
    }
    else if constexpr (Mode == BetaMode::One)
    {
        return ax + x_old;
    }
    else
    {
        return ax + (beta == static_cast<VALTYPE>(0) ? static_cast<VALTYPE>(0) : beta * x_old);
    }
}

/// @brief Kernel-selectable dot product over CSR value/index slices [start, end)
/// @details Computes the dot product of a sparse matrix row segment with a dense vector.
///          Supports both scalar and SIMD (AVX2) implementations.
///
/// @note SIMD Implementation Constraints:
///       - AVX2 gather instructions require 32-bit indices
///       - If COLTYPE is int64_t or any non-32-bit type, SIMD falls back to scalar
///       - This prevents potential overflow when downcasting 64-bit indices to 32-bit
///
/// @tparam Kernel RowDotKernel::Scalar or RowDotKernel::Simd
/// @tparam Base Matrix indexing base (0 or 1)
/// @tparam ROWTYPE Integer type for row pointers
/// @tparam COLTYPE Integer type for column indices (SIMD only works with 32-bit types)
/// @tparam VALTYPE Value type (double or float)
/// @param start Starting index in the CSR arrays
/// @param end Ending index (exclusive) in the CSR arrays
/// @param aj Column indices array
/// @param av Values array
/// @param b Dense vector to multiply with
/// @return Dot product result
template <RowDotKernel Kernel, int Base = 0, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
inline VALTYPE DotRangeSIMD(const ROWTYPE start, const ROWTYPE end, COLTYPE const* __restrict aj,
                            VALTYPE const* __restrict av, VALTYPE const* __restrict b)
{
#if defined(AVX2_SUPPORTED) || defined(__AVX2__)
    constexpr bool is_double = std::is_same_v<VALTYPE, double>;
    constexpr bool is_float = std::is_same_v<VALTYPE, float>;
#endif

    if constexpr (Kernel == RowDotKernel::Simd)
    {
#if defined(AVX2_SUPPORTED) || defined(__AVX2__)
        // AVX2 gather instructions require 32-bit indices
        // Fall back to scalar if COLTYPE is not 32-bit (e.g., int64_t)
        constexpr bool can_use_gather = (sizeof(COLTYPE) == sizeof(int32_t));

        if constexpr (is_double && can_use_gather)
        {
            const ROWTYPE simd_end = start + ((end - start) & (~ROWTYPE(3)));
            // Use union to access SIMD register as array without explicit store
            // vacc.vec for SIMD operations, vacc.arr for scalar extraction
            union {
                __m256d vec;
                double arr[4];
            } vacc;
            vacc.vec = _mm256_setzero_pd();
// #pragma unroll(32)
            for (ROWTYPE idx = start; idx < simd_end; idx += 4)
            {
                __m128i j_idx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(aj + idx));
                if constexpr (Base != 0)
                {
                    j_idx = _mm_sub_epi32(j_idx, _mm_set1_epi32(Base));
                }
                __m256d vb = _mm256_i32gather_pd(b, j_idx, 8);
                __m256d va = _mm256_loadu_pd(av + idx);
                vacc.vec = _mm256_fmadd_pd(va, vb, vacc.vec);
            }
            // Extract and sum 4 doubles directly from union
            VALTYPE sum = vacc.arr[0] + vacc.arr[1] + vacc.arr[2] + vacc.arr[3];
            if constexpr (Base != 0)
            {
                for (ROWTYPE idx = simd_end; idx < end; ++idx)
                {
                    sum += av[idx] * b[aj[idx] - Base];
                }
            }
            else
            {
                for (ROWTYPE idx = simd_end; idx < end; ++idx)
                {
                    sum += av[idx] * b[aj[idx]];
                }
            }
            return sum;
        }
        else if constexpr (is_float && can_use_gather)
        {
            const ROWTYPE simd_end = start + ((end - start) & (~ROWTYPE(7)));
            // Use union to access SIMD register as array without explicit store
            // vacc.vec for SIMD operations, vacc.arr for scalar extraction
            union {
                __m256 vec;
                float arr[8];
            } vacc;
            vacc.vec = _mm256_setzero_ps();
            for (ROWTYPE idx = start; idx < simd_end; idx += 8)
            {
                __m256i j_idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(aj + idx));
                if constexpr (Base != 0)
                {
                    j_idx = _mm256_sub_epi32(j_idx, _mm256_set1_epi32(Base));
                }
                __m256 vb = _mm256_i32gather_ps(b, j_idx, 4);
                __m256 va = _mm256_loadu_ps(reinterpret_cast<const float*>(av + idx));
                vacc.vec = _mm256_fmadd_ps(va, vb, vacc.vec);
            }
            // Extract and sum 8 floats directly from union
            VALTYPE sum = static_cast<VALTYPE>(vacc.arr[0] + vacc.arr[1] + vacc.arr[2] + vacc.arr[3] +
                                               vacc.arr[4] + vacc.arr[5] + vacc.arr[6] + vacc.arr[7]);
            if constexpr (Base != 0)
            {
                for (ROWTYPE idx = simd_end; idx < end; ++idx)
                {
                    sum += av[idx] * b[aj[idx] - Base];
                }
            }
            else
            {
                for (ROWTYPE idx = simd_end; idx < end; ++idx)
                {
                    sum += av[idx] * b[aj[idx]];
                }
            }
            return sum;
        }
        // If SIMD requested but can't use gather (e.g., COLTYPE is int64_t), fall through to scalar
#endif
    }
    // Scalar fallback (also used when COLTYPE is not 32-bit)
    VALTYPE sum = 0;
    if constexpr (Base != 0)
    {
// #pragma omp simd
#pragma unroll(8)
        for (ROWTYPE idx = start; idx < end; ++idx)
        {
            sum += av[idx] * b[aj[idx] - Base];
        }
    }
    else
    {
// #pragma omp simd
#pragma unroll(8)
        for (ROWTYPE idx = start; idx < end; ++idx)
        {
            sum += av[idx] * b[aj[idx]];
        }
    }
    return sum;
}

/// @brief Runtime dispatcher for DotRangeSIMD based on base value
/// @details Dispatches to the appropriate compile-time Base template based on runtime base value
/// @param base Runtime base value (0 or non-zero)
template <RowDotKernel Kernel, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
inline VALTYPE DotRangeSIMD_dispatch(const ROWTYPE start, const ROWTYPE end, const int base,
                                     COLTYPE const* __restrict aj, VALTYPE const* __restrict av,
                                     VALTYPE const* __restrict b)
{
    if (base == 0)
    {
        return DotRangeSIMD<Kernel, 0>(start, end, aj, av, b);
    }
    else
    {
        return DotRangeSIMD<Kernel, 1>(start, end, aj, av, b);
    }
}

} // namespace matrix_utils
