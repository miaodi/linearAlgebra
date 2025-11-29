#pragma once

#include "BitVector.hpp"
#include "matrix_utils.hpp"
#include <concepts>
#include <cstring>
#include <immintrin.h>
#include <omp.h>

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
template <RowDotKernel Kernel, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
inline VALTYPE DotRangeSIMD(const ROWTYPE start, const ROWTYPE end,
                            const int base, COLTYPE const *__restrict aj,
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
      for (ROWTYPE idx = start; idx < simd_end; idx += 4) {
        if constexpr (std::is_same_v<COLTYPE, int>) {
          __m128i j_idx =
              _mm_loadu_si128(reinterpret_cast<const __m128i *>(aj + idx));
          j_idx = _mm_sub_epi32(j_idx, _mm_set1_epi32(base));
          __m256d vb = _mm256_i32gather_pd(b, j_idx, 8);
          __m256d va = _mm256_loadu_pd(av + idx);
          vacc = _mm256_fmadd_pd(va, vb, vacc);
        } else {
          int indices[4];
          indices[0] = static_cast<int>(aj[idx] - base);
          indices[1] = static_cast<int>(aj[idx + 1] - base);
          indices[2] = static_cast<int>(aj[idx + 2] - base);
          indices[3] = static_cast<int>(aj[idx + 3] - base);
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
      for (ROWTYPE idx = simd_end; idx < end; ++idx) {
        sum += av[idx] * b[aj[idx] - base];
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
          j_idx = _mm256_sub_epi32(j_idx, _mm256_set1_epi32(base));
        } else {
          alignas(32) int indices[8];
          indices[0] = static_cast<int>(aj[idx] - base);
          indices[1] = static_cast<int>(aj[idx + 1] - base);
          indices[2] = static_cast<int>(aj[idx + 2] - base);
          indices[3] = static_cast<int>(aj[idx + 3] - base);
          indices[4] = static_cast<int>(aj[idx + 4] - base);
          indices[5] = static_cast<int>(aj[idx + 5] - base);
          indices[6] = static_cast<int>(aj[idx + 6] - base);
          indices[7] = static_cast<int>(aj[idx + 7] - base);
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
      for (ROWTYPE idx = simd_end; idx < end; ++idx) {
        sum += av[idx] * b[aj[idx] - base];
      }
      return sum;
    }
#endif
  }
  VALTYPE sum = 0;
  for (ROWTYPE idx = start; idx < end; ++idx) {
    sum += av[idx] * b[aj[idx] - base];
  }
  return sum;
}

// compute x = alpha * A * b + beta * x

struct SerialSPMV {
  SerialSPMV() = default;

  template <BetaMode Mode, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
  void compute(const COLTYPE size, const int base,
               ROWTYPE const * __restrict ai,
               COLTYPE const * __restrict aj,
               VALTYPE const * __restrict av,
               VALTYPE const * __restrict const b,
               VALTYPE * __restrict const x, const VALTYPE alpha,
               const VALTYPE beta) const {
    for (COLTYPE i = 0; i < size; i++) {
      VALTYPE val = 0;
#pragma unroll
      for (ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; j++) {
        val += av[j] * b[aj[j] - base];
      }
      const VALTYPE ax = alpha * val;
      x[i] = apply_beta<Mode>(ax, beta, x[i]);
    }
  }

  template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
  void operator()(const COLTYPE size, const int base,
                  ROWTYPE const * __restrict ai,
                  COLTYPE const * __restrict aj,
                  VALTYPE const * __restrict av,
                  VALTYPE const * __restrict const b,
                  VALTYPE * __restrict const x, const VALTYPE alpha,
                  const VALTYPE beta) const {
    if (beta == static_cast<VALTYPE>(0)) {
      compute<BetaMode::Zero>(size, base, ai, aj, av, b, x, alpha, beta);
    } else if (beta == static_cast<VALTYPE>(1)) {
      compute<BetaMode::One>(size, base, ai, aj, av, b, x, alpha, beta);
    } else {
      compute<BetaMode::Generic>(size, base, ai, aj, av, b, x, alpha, beta);
    }
  }
};

template <typename ROWTYPE = int, typename COLTYPE = int, typename VALTYPE = double,
          RowDotKernel Kernel = RowDotKernel::Scalar>
struct ParallelSPMV {
  ParallelSPMV(const int num_threads = omp_get_max_threads())
      : _nthreads{num_threads} {}

  void setNumThreads(const int num_threads) { _nthreads = num_threads; }

  template <BetaMode Mode>
  void compute(const COLTYPE size, const int base,
               ROWTYPE const *__restrict ai, COLTYPE const *__restrict aj,
               VALTYPE const *__restrict av,
               VALTYPE const *__restrict const b,
               VALTYPE *__restrict const x, const VALTYPE alpha,
               const VALTYPE beta) const {
#pragma omp parallel num_threads(_nthreads)
    {
      const int tid = omp_get_thread_num();
      const int nthreads = omp_get_num_threads();

      auto [start, end] =
          utils::LoadPrefixBalancedPartitionPos(ai, ai + size, tid, nthreads);

      for (COLTYPE i = start; i < end; i++) {
        VALTYPE val = DotRangeSIMD<Kernel>(ai[i] - base, ai[i + 1] - base, base,
                                           aj, av, b);
        const VALTYPE ax = alpha * val;
        x[i] = apply_beta<Mode>(ax, beta, x[i]);
      }
    }
  }

  void operator()(const COLTYPE size, const int base,
                  ROWTYPE const *__restrict ai,
                  COLTYPE const *__restrict aj,
                  VALTYPE const *__restrict av,
                  VALTYPE const *__restrict const b,
                  VALTYPE *__restrict const x, const VALTYPE alpha,
                  const VALTYPE beta) const {
    if (beta == static_cast<VALTYPE>(0)) {
      compute<BetaMode::Zero>(size, base, ai, aj, av, b, x, alpha, beta);
    } else if (beta == static_cast<VALTYPE>(1)) {
      compute<BetaMode::One>(size, base, ai, aj, av, b, x, alpha, beta);
    } else {
      compute<BetaMode::Generic>(size, base, ai, aj, av, b, x, alpha, beta);
    }
  }

private:
  int _nthreads;
};

template <typename ROWTYPE = int, typename COLTYPE = int, typename VALTYPE = double,
          RowDotKernel Kernel = RowDotKernel::Scalar>
class ALBUSSPMV
{
public:
    ALBUSSPMV(const int num_threads = omp_get_num_threads()) : _nthreads{num_threads} {}

    void setNumThreads(const int num_threads) { _nthreads = num_threads; }

    void preprocess(const COLTYPE size, const int base, ROWTYPE const* __restrict ai,
                    COLTYPE const* __restrict aj, VALTYPE const* __restrict av)
    {
        const ROWTYPE nnz = ai[size] - base;

        if (nnz <= 0)
        {
            _nthreads = 1;
        }
        else
        {
            if (_nthreads <= 0)
                _nthreads = 1;
            if (_nthreads > nnz)
                _nthreads = static_cast<int>(nnz);
        }

        _threadBlockSizePrefix.assign(_nthreads + 1, ROWTYPE(0));

        const ROWTYPE work_per_thread = (nnz > 0) ? (nnz / _nthreads) : 0;
        const int resid = (nnz > 0) ? (nnz % _nthreads) : 0;
        for (int i = 0; i < _nthreads; i++)
        {
            _threadBlockSizePrefix[i + 1] =
                i >= resid ? ((i + 1) * work_per_thread + resid) : ((i + 1) * (work_per_thread + 1));
        }

        _threadStartRow.resize(_nthreads + 1);
        for (size_t i = 0; i <= _nthreads; i++)
        {
            const ROWTYPE target = _threadBlockSizePrefix[i] + base;
            auto it = std::upper_bound(ai, ai + size + 1, target);
            ROWTYPE row = static_cast<ROWTYPE>(std::distance(ai, it)) - 1;
            if (row < 0)
                row = 0;
            if (row > static_cast<ROWTYPE>(size))
                row = size;
            _threadStartRow[i] = static_cast<COLTYPE>(row);
        }

        _threadBoundaryValue.assign(2 * _nthreads, static_cast<VALTYPE>(0));
    }

    void operator()(const COLTYPE size, ROWTYPE const* __restrict ai, COLTYPE const* __restrict aj,
                    VALTYPE const* __restrict av, const VALTYPE* __restrict const b, VALTYPE* __restrict const x,
                    const VALTYPE alpha, const VALTYPE beta) const
    {
        const int base = static_cast<int>(ai ? ai[0] : 0);
        if (beta == static_cast<VALTYPE>(0))
        {
            compute<BetaMode::Zero>(size, base, ai, aj, av, b, x, alpha, beta);
        }
        else if (beta == static_cast<VALTYPE>(1))
        {
            compute<BetaMode::One>(size, base, ai, aj, av, b, x, alpha, beta);
        }
        else
        {
            compute<BetaMode::Generic>(size, base, ai, aj, av, b, x, alpha, beta);
        }
    }

private:
    template <BetaMode Mode>
    void compute(const COLTYPE size, const int base, ROWTYPE const* __restrict ai,
                 COLTYPE const* __restrict aj, VALTYPE const* __restrict av, const VALTYPE* __restrict const b,
                 VALTYPE* __restrict const x, const VALTYPE alpha, const VALTYPE beta) const
    {
        const ROWTYPE nnz = ai[size] - base;
        if (nnz == 0)
        {
            if constexpr (Mode == BetaMode::Zero)
            {
                std::fill(x, x + size, VALTYPE(0));
            }
            else if constexpr (Mode == BetaMode::One)
            {
                // leave x untouched
            }
            else
            {
                if (beta == VALTYPE(0))
                {
                    std::fill(x, x + size, VALTYPE(0));
                }
                else if (beta != VALTYPE(1))
                {
#pragma omp simd
                    for (COLTYPE i = 0; i < size; ++i)
                        x[i] *= beta;
                }
            }
            return;
        }

        std::memset(_threadBoundaryValue.data(), 0, sizeof(VALTYPE) * 2 * _nthreads);
#pragma omp parallel num_threads(_nthreads)
        {
            const int tid = omp_get_thread_num();
            const int tidBegin = tid << 1;
            const int tidEnd = tidBegin | 1;
            const COLTYPE startRow = _threadStartRow[tid];
            const COLTYPE endRow = _threadStartRow[tid + 1];
            const ROWTYPE nzStart = _threadBlockSizePrefix[tid];
            const ROWTYPE nzEnd = _threadBlockSizePrefix[tid + 1];
            VALTYPE val;
            if (startRow < endRow)
            {
                COLTYPE row = startRow;
                val = DotRangeSIMD<Kernel>(nzStart, ai[startRow + 1] - base, base, aj, av, b);
                _threadBoundaryValue[tidBegin] = alpha * val;

#pragma unroll(32)
                for (row = startRow + 1; row < endRow; row++)
                {
                    val = DotRangeSIMD<Kernel>(ai[row] - base, ai[row + 1] - base, base, aj, av, b);
                    const VALTYPE ax = alpha * val;
                    x[row] = apply_beta<Mode>(ax, beta, x[row]);
                }

                val = DotRangeSIMD<Kernel>(ai[endRow] - base, nzEnd, base, aj, av, b);
                _threadBoundaryValue[tidEnd] = alpha * val;
            }
            else
            {
                val = DotRangeSIMD<Kernel>(nzStart, nzEnd, base, aj, av, b);
                _threadBoundaryValue[tidBegin] = alpha * val;
            }
        }

        COLTYPE idx = std::numeric_limits<COLTYPE>::max();
#pragma unroll(32)
        for (int tid = 0; tid < _nthreads; tid++)
        {
            if (_threadStartRow[tid] != idx)
            {
                idx = _threadStartRow[tid];
                x[idx] = apply_beta<Mode>(_threadBoundaryValue[2 * tid], beta, x[idx]);
            }
            else
            {
                x[idx] += _threadBoundaryValue[2 * tid];
            }
            if (tid == _nthreads - 1)
                break;
            if (_threadStartRow[tid + 1] != idx)
            {
                idx = _threadStartRow[tid + 1];
                x[idx] = apply_beta<Mode>(_threadBoundaryValue[2 * tid + 1], beta, x[idx]);
            }
            else
            {
                x[idx] += _threadBoundaryValue[2 * tid + 1];
            }
        }
    }

    int _nthreads;
    mutable std::vector<ROWTYPE> _threadBlockSizePrefix;
    mutable std::vector<COLTYPE> _threadStartRow;
    mutable std::vector<VALTYPE> _threadBoundaryValue;
};
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, typename T>
constexpr bool spmv_has_preprocess = requires(const COLTYPE size,
                                              const int base, ROWTYPE const * __restrict ai,
                                              COLTYPE const * __restrict aj,
                                              VALTYPE const * __restrict av, T &t) {
  t.preprocess(size, base, ai, aj, av);
};

/** 
 * Sparse Matrix-Vector Multiplication (SPMV) operator
 * After preprocess(), the operator() can be called to perform SPMV
 * Usage:
 *  matrix_utils::SPMV<CSRMatrixType, SPMVType> spmv;
 *  spmv.setMatrix(&csr_matrix);
 *  spmv.preprocess();
 *  spmv(b, x, alpha, beta); // x = alpha * A * b + beta * x
 */
template <typename CSRMatrixType, typename SPMVType> struct SPMV {
  using ROWTYPE = typename CSRMatrixType::ROWTYPE;
  using COLTYPE = typename CSRMatrixType::COLTYPE;
  using VALTYPE = typename CSRMatrixType::VALTYPE;

  SPMV() : _matrix{nullptr} {}

  void setMatrix(CSRMatrixType const *matrix) { _matrix = matrix; }

  void preprocess() {
    if constexpr (spmv_has_preprocess<ROWTYPE, COLTYPE, VALTYPE, SPMVType>) {
      _spmv.preprocess(_matrix->rows, _matrix->Base(), _matrix->AI(),
                       _matrix->AJ(), _matrix->AV());
    }
  }
  COLTYPE size() const {
    if (_matrix) {
      return _matrix->rows;
    } else {
      return 0;
    }
  }
  
  void operator()(const VALTYPE * __restrict const b,
                  VALTYPE * __restrict const x, const VALTYPE alpha = 1.,
                  const VALTYPE beta = 0.) const {
    if constexpr (requires { _spmv(_matrix->rows, _matrix->Base(), _matrix->AI(),
                                   _matrix->AJ(), _matrix->AV(), b, x, alpha,
                                   beta); }) {
      _spmv(_matrix->rows, _matrix->Base(), _matrix->AI(), _matrix->AJ(),
            _matrix->AV(), b, x, alpha, beta);
    } else if constexpr (requires { _spmv(_matrix->rows, _matrix->AI(),
                                          _matrix->AJ(), _matrix->AV(), b, x,
                                          alpha, beta); }) {
      _spmv(_matrix->rows, _matrix->AI(), _matrix->AJ(), _matrix->AV(), b, x,
            alpha, beta);
    } else {
      static_assert(sizeof(SPMVType) == 0,
                    "SPMVType operator() signature not supported");
    }
  }

  CSRMatrixType const *_matrix;
  SPMVType _spmv;
};

} // namespace matrix_utils
