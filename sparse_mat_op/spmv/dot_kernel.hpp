#pragma once

#if defined( __x86_64__ ) || defined( __i386__ ) || defined( _M_X64 ) || defined( _M_IX86 )
#define MATRIX_UTILS_X86_INTRINSICS_AVAILABLE 1
#include <immintrin.h>
#endif
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
inline VALTYPE apply_beta( const VALTYPE ax, const VALTYPE beta, const VALTYPE x_old )
{
    if constexpr ( Mode == BetaMode::Zero )
    {
        return ax;
    }
    else if constexpr ( Mode == BetaMode::One )
    {
        return ax + x_old;
    }
    else
    {
        return ax + ( beta == static_cast<VALTYPE>( 0 ) ? static_cast<VALTYPE>( 0 ) : beta * x_old );
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
inline VALTYPE DotRangeSIMD( const ROWTYPE start,
                             const ROWTYPE end,
                             COLTYPE const* __restrict aj,
                             VALTYPE const* __restrict av,
                             VALTYPE const* __restrict b )
{
#if defined( MATRIX_UTILS_X86_INTRINSICS_AVAILABLE ) && ( defined( AVX2_SUPPORTED ) || defined( __AVX2__ ) )
    constexpr bool is_double = std::is_same_v<VALTYPE, double>;
    constexpr bool is_float = std::is_same_v<VALTYPE, float>;
#endif

    if constexpr ( Kernel == RowDotKernel::Simd )
    {
#if defined( MATRIX_UTILS_X86_INTRINSICS_AVAILABLE ) && ( defined( AVX2_SUPPORTED ) || defined( __AVX2__ ) )
        // AVX2 gather instructions require 32-bit indices
        // Fall back to scalar if COLTYPE is not 32-bit (e.g., int64_t)
        constexpr bool can_use_gather = ( sizeof( COLTYPE ) == sizeof( int32_t ) );

        if constexpr ( is_double && can_use_gather )
        {
            const ROWTYPE row_nnz = end - start;
            const ROWTYPE simd_end = start + ( ( end - start ) & ( ~ROWTYPE( 3 ) ) );
            ROWTYPE idx = start;

            auto gather_fmadd = [&]( const ROWTYPE idx, __m256d acc ) -> __m256d
            {
                __m128i j_idx = _mm_loadu_si128( reinterpret_cast<const __m128i*>( aj + idx ) );
                if constexpr ( Base != 0 )
                {
                    j_idx = _mm_sub_epi32( j_idx, _mm_set1_epi32( Base ) );
                }
                __m256d vb = _mm256_i32gather_pd( b, j_idx, 8 );
                __m256d va = _mm256_loadu_pd( av + idx );
                return _mm256_fmadd_pd( va, vb, acc );
            };

            __m256d vacc = _mm256_setzero_pd();
            if ( row_nnz >= 16 )
            {
                const ROWTYPE unrolled_end = start + ( row_nnz & ( ~ROWTYPE( 15 ) ) );
                __m256d vacc0 = _mm256_setzero_pd();
                __m256d vacc1 = _mm256_setzero_pd();
                __m256d vacc2 = _mm256_setzero_pd();
                __m256d vacc3 = _mm256_setzero_pd();

                for ( ; idx < unrolled_end; idx += 16 )
                {
                    vacc0 = gather_fmadd( idx, vacc0 );
                    vacc1 = gather_fmadd( idx + 4, vacc1 );
                    vacc2 = gather_fmadd( idx + 8, vacc2 );
                    vacc3 = gather_fmadd( idx + 12, vacc3 );
                }
                vacc = _mm256_add_pd( _mm256_add_pd( vacc0, vacc1 ), _mm256_add_pd( vacc2, vacc3 ) );
            }
            for ( ; idx < simd_end; idx += 4 )
            {
                vacc = gather_fmadd( idx, vacc );
            }

            double lanes[4];
            _mm256_storeu_pd( lanes, vacc );
            VALTYPE sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
            if constexpr ( Base != 0 )
            {
                for ( ROWTYPE idx = simd_end; idx < end; ++idx )
                {
                    sum += av[idx] * b[aj[idx] - Base];
                }
            }
            else
            {
                for ( ROWTYPE idx = simd_end; idx < end; ++idx )
                {
                    sum += av[idx] * b[aj[idx]];
                }
            }
            return sum;
        }
        else if constexpr ( is_float && can_use_gather )
        {
            const ROWTYPE row_nnz = end - start;
            const ROWTYPE simd_end = start + ( ( end - start ) & ( ~ROWTYPE( 7 ) ) );
            ROWTYPE idx = start;

            auto gather_fmadd = [&]( const ROWTYPE idx, __m256 acc ) -> __m256
            {
                __m256i j_idx = _mm256_loadu_si256( reinterpret_cast<const __m256i*>( aj + idx ) );
                if constexpr ( Base != 0 )
                {
                    j_idx = _mm256_sub_epi32( j_idx, _mm256_set1_epi32( Base ) );
                }
                __m256 vb = _mm256_i32gather_ps( b, j_idx, 4 );
                __m256 va = _mm256_loadu_ps( reinterpret_cast<const float*>( av + idx ) );
                return _mm256_fmadd_ps( va, vb, acc );
            };

            __m256 vacc = _mm256_setzero_ps();
            if ( row_nnz >= 32 )
            {
                const ROWTYPE unrolled_end = start + ( row_nnz & ( ~ROWTYPE( 31 ) ) );
                __m256 vacc0 = _mm256_setzero_ps();
                __m256 vacc1 = _mm256_setzero_ps();
                __m256 vacc2 = _mm256_setzero_ps();
                __m256 vacc3 = _mm256_setzero_ps();

                for ( ; idx < unrolled_end; idx += 32 )
                {
                    vacc0 = gather_fmadd( idx, vacc0 );
                    vacc1 = gather_fmadd( idx + 8, vacc1 );
                    vacc2 = gather_fmadd( idx + 16, vacc2 );
                    vacc3 = gather_fmadd( idx + 24, vacc3 );
                }
                vacc = _mm256_add_ps( _mm256_add_ps( vacc0, vacc1 ), _mm256_add_ps( vacc2, vacc3 ) );
            }
            for ( ; idx < simd_end; idx += 8 )
            {
                vacc = gather_fmadd( idx, vacc );
            }

            float lanes[8];
            _mm256_storeu_ps( lanes, vacc );
            VALTYPE sum = static_cast<VALTYPE>( lanes[0] + lanes[1] + lanes[2] + lanes[3] +
                                                lanes[4] + lanes[5] + lanes[6] + lanes[7] );
            if constexpr ( Base != 0 )
            {
                for ( ROWTYPE idx = simd_end; idx < end; ++idx )
                {
                    sum += av[idx] * b[aj[idx] - Base];
                }
            }
            else
            {
                for ( ROWTYPE idx = simd_end; idx < end; ++idx )
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
    if constexpr ( Base != 0 )
    {
// #pragma omp simd
#pragma unroll( 8 )
        for ( ROWTYPE idx = start; idx < end; ++idx )
        {
            sum += av[idx] * b[aj[idx] - Base];
        }
    }
    else
    {
// #pragma omp simd
#pragma unroll( 8 )
        for ( ROWTYPE idx = start; idx < end; ++idx )
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
inline VALTYPE DotRangeSIMD_dispatch( const ROWTYPE start,
                                      const ROWTYPE end,
                                      const int base,
                                      COLTYPE const* __restrict aj,
                                      VALTYPE const* __restrict av,
                                      VALTYPE const* __restrict b )
{
    if ( base == 0 )
    {
        return DotRangeSIMD<Kernel, 0>( start, end, aj, av, b );
    }
    else
    {
        return DotRangeSIMD<Kernel, 1>( start, end, aj, av, b );
    }
}

} // namespace matrix_utils
