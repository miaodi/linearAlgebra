#pragma once

#include <immintrin.h>
#include <type_traits>

namespace matrix_utils {
enum class BetaMode { Zero, One, Generic };
enum class RowDotKernel { Scalar, Simd };

template <BetaMode Mode, typename VALTYPE>
inline VALTYPE apply_beta(const VALTYPE ax, const VALTYPE beta,
                          const VALTYPE x_old) {
  if constexpr (Mode == BetaMode::Zero) {
    return ax;
  } else if constexpr (Mode == BetaMode::One) {
    return ax + x_old;
  } else {
    return ax + (beta == static_cast<VALTYPE>(0) ? static_cast<VALTYPE>(0)
                                                 : beta * x_old);
  }
}

// Kernel-selectable dot product over CSR value/index slices [start, end)
template <RowDotKernel Kernel, int Base = 0, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
inline VALTYPE DotRangeSIMD(const ROWTYPE start, const ROWTYPE end,
                            COLTYPE const *__restrict aj,
                            VALTYPE const *__restrict av,
                            VALTYPE const *__restrict b) {
#if defined(AVX2_SUPPORTED) || defined(__AVX2__)
  constexpr bool is_double = std::is_same_v<VALTYPE, double>;
  constexpr bool is_float = std::is_same_v<VALTYPE, float>;
#endif

  if constexpr (Kernel == RowDotKernel::Simd) {
#if defined(AVX2_SUPPORTED) || defined(__AVX2__)
    if constexpr (is_double) {
      const ROWTYPE simd_end = start + ((end - start) & (~ROWTYPE(3)));
      __m256d vacc = _mm256_setzero_pd();
#pragma unroll(4)
      for (ROWTYPE idx = start; idx < simd_end; idx += 4) {
        if constexpr (std::is_same_v<COLTYPE, int>) {
          __m128i j_idx =
              _mm_loadu_si128(reinterpret_cast<const __m128i *>(aj + idx));
          if constexpr (Base != 0) {
            j_idx = _mm_sub_epi32(j_idx, _mm_set1_epi32(Base));
          }
          __m256d vb = _mm256_i32gather_pd(b, j_idx, 8);
          __m256d va = _mm256_loadu_pd(av + idx);
          vacc = _mm256_fmadd_pd(va, vb, vacc);
        } else {
          int indices[4];
          if constexpr (Base != 0) {
            indices[0] = static_cast<int>(aj[idx] - Base);
            indices[1] = static_cast<int>(aj[idx + 1] - Base);
            indices[2] = static_cast<int>(aj[idx + 2] - Base);
            indices[3] = static_cast<int>(aj[idx + 3] - Base);
          } else {
            indices[0] = static_cast<int>(aj[idx]);
            indices[1] = static_cast<int>(aj[idx + 1]);
            indices[2] = static_cast<int>(aj[idx + 2]);
            indices[3] = static_cast<int>(aj[idx + 3]);
          }
          __m128i j_idx =
              _mm_loadu_si128(reinterpret_cast<const __m128i *>(indices));
          __m256d vb = _mm256_i32gather_pd(b, j_idx, 8);
          __m256d va = _mm256_loadu_pd(av + idx);
          vacc = _mm256_fmadd_pd(va, vb, vacc);
        }
      }
      alignas(32) double tmp[4];
      _mm256_store_pd(tmp, vacc);
      VALTYPE sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
      if constexpr (Base != 0) {
        for (ROWTYPE idx = simd_end; idx < end; ++idx) {
          sum += av[idx] * b[aj[idx] - Base];
        }
      } else {
        for (ROWTYPE idx = simd_end; idx < end; ++idx) {
          sum += av[idx] * b[aj[idx]];
        }
      }
      return sum;
    } else if constexpr (is_float) {
      const ROWTYPE simd_end = start + ((end - start) & (~ROWTYPE(7)));
      __m256 vacc = _mm256_setzero_ps();
      for (ROWTYPE idx = start; idx < simd_end; idx += 8) {
        __m256i j_idx;
        if constexpr (std::is_same_v<COLTYPE, int>) {
          j_idx = _mm256_loadu_si256(
              reinterpret_cast<const __m256i *>(aj + idx));
          if constexpr (Base != 0) {
            j_idx = _mm256_sub_epi32(j_idx, _mm256_set1_epi32(Base));
          }
        } else {
          alignas(32) int indices[8];
          if constexpr (Base != 0) {
            indices[0] = static_cast<int>(aj[idx] - Base);
            indices[1] = static_cast<int>(aj[idx + 1] - Base);
            indices[2] = static_cast<int>(aj[idx + 2] - Base);
            indices[3] = static_cast<int>(aj[idx + 3] - Base);
            indices[4] = static_cast<int>(aj[idx + 4] - Base);
            indices[5] = static_cast<int>(aj[idx + 5] - Base);
            indices[6] = static_cast<int>(aj[idx + 6] - Base);
            indices[7] = static_cast<int>(aj[idx + 7] - Base);
          } else {
            indices[0] = static_cast<int>(aj[idx]);
            indices[1] = static_cast<int>(aj[idx + 1]);
            indices[2] = static_cast<int>(aj[idx + 2]);
            indices[3] = static_cast<int>(aj[idx + 3]);
            indices[4] = static_cast<int>(aj[idx + 4]);
            indices[5] = static_cast<int>(aj[idx + 5]);
            indices[6] = static_cast<int>(aj[idx + 6]);
            indices[7] = static_cast<int>(aj[idx + 7]);
          }
          j_idx = _mm256_load_si256(reinterpret_cast<const __m256i *>(indices));
        }
        __m256 vb = _mm256_i32gather_ps(b, j_idx, 4);
        __m256 va = _mm256_loadu_ps(reinterpret_cast<const float *>(av + idx));
        vacc = _mm256_fmadd_ps(va, vb, vacc);
      }
      alignas(32) float tmp[8];
      _mm256_store_ps(tmp, vacc);
      VALTYPE sum = static_cast<VALTYPE>(tmp[0] + tmp[1] + tmp[2] + tmp[3] +
                                         tmp[4] + tmp[5] + tmp[6] + tmp[7]);
      if constexpr (Base != 0) {
        for (ROWTYPE idx = simd_end; idx < end; ++idx) {
          sum += av[idx] * b[aj[idx] - Base];
        }
      } else {
        for (ROWTYPE idx = simd_end; idx < end; ++idx) {
          sum += av[idx] * b[aj[idx]];
        }
      }
      return sum;
    }
#endif
  }
  VALTYPE sum = 0;
#pragma unroll(8)
  if constexpr (Base != 0) {
    for (ROWTYPE idx = start; idx < end; ++idx) {
      sum += av[idx] * b[aj[idx] - Base];
    }
  } else {
    for (ROWTYPE idx = start; idx < end; ++idx) {
      sum += av[idx] * b[aj[idx]];
    }
  }
  return sum;
}
/// @brief Runtime dispatcher for DotRangeSIMD based on base value
/// @details Dispatches to the appropriate compile-time Base template based on runtime base value
/// @param base Runtime base value (0 or non-zero)
template <RowDotKernel Kernel, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
inline VALTYPE DotRangeSIMD_dispatch(const ROWTYPE start, const ROWTYPE end,
                                     const int base, COLTYPE const *__restrict aj,
                                     VALTYPE const *__restrict av,
                                     VALTYPE const *__restrict b) {
  if (base == 0) {
    return DotRangeSIMD<Kernel, 0>(start, end, aj, av, b);
  } else {
    return DotRangeSIMD<Kernel, 1>(start, end, aj, av, b);
  }
}

} // namespace matrix_utils
