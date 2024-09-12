#pragma once

#include "BitVector.hpp"
#include "matrix_utils.hpp"
#include <concepts>
#include <omp.h>

namespace matrix_utils {

// compute y = alpha * A * x + beta * y

struct SerialSPMV {
  SerialSPMV() = default;

  template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
  void operator()(const COLTYPE size, const int base, ROWTYPE const *ai,
                  COLTYPE const *aj, VALTYPE const *av, VALTYPE const *const b,
                  VALTYPE *const x, const VALTYPE alpha,
                  const VALTYPE beta) const {
    for (COLTYPE i = 0; i < size; i++) {
      VALTYPE val = beta == 0 ? 0 : beta * x[i];
#pragma unroll
      for (ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; j++) {
        val += alpha * av[j] * b[aj[j] - base];
      }
      x[i] = val;
    }
  }
};

struct ParallelSPMV {
  ParallelSPMV() {}

  template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
  void operator()(const COLTYPE size, const int base, ROWTYPE const *ai,
                  COLTYPE const *aj, VALTYPE const *av, VALTYPE const *const b,
                  VALTYPE *const x, const VALTYPE alpha,
                  const VALTYPE beta) const {
#pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      const int nthreads = omp_get_num_threads();

      auto [start, end] =
          utils::LoadPrefixBalancedPartitionPos(ai, ai + size, tid, nthreads);

      for (COLTYPE i = start; i < end; i++) {
        VALTYPE val = beta == 0 ? 0 : beta * x[i];
#pragma unroll
        for (ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; j++) {
          val += alpha * av[j] * b[aj[j] - base];
        }
        x[i] = val;
      }
    }
  }
};

template <typename ROWTYPE = int, typename COLTYPE = int,
          typename VALTYPE = double>
class SegSumSPMV {
public:
  SegSumSPMV(const int num_threads = omp_get_num_threads())
      : _nthreads{num_threads} {}

  void setNumThreads(const int num_threads) { _nthreads = num_threads; }

  void preprocess(const COLTYPE size, const int base, ROWTYPE const *ai,
                  COLTYPE const *aj, VALTYPE const *av) {
    _threadBv.resize(_nthreads + 1);
    _threadProduct.resize(_nthreads + 1);
    _threadProduct[0] = 0;

    const ROWTYPE nnz = ai[size] - base;
    _bv.resize(nnz);
    _product.resize(nnz);
  }

  void operator()(const COLTYPE size, const int base, ROWTYPE const *ai,
                  COLTYPE const *aj, VALTYPE const *av, const VALTYPE *const b,
                  VALTYPE *const x, const VALTYPE alpha,
                  const VALTYPE beta) const {
    const ROWTYPE nnz = ai[size] - base;
    _bv.clearAll();
    _threadBv.clearAll();
#pragma omp parallel num_threads(_nthreads)
    {
      const int tid = omp_get_thread_num();
      const int nthreads = omp_get_num_threads();

      auto [start_row, end_row] =
          utils::LoadBalancedPartitionPos(size, tid, nthreads);

      for (COLTYPE i = start_row; i < end_row; i++) {
        _bv.set(ai[i] - base);
      }

#pragma omp barrier
      auto [start, end] = utils::LoadBalancedPartitionPos(nnz, tid, nthreads);
#pragma unroll
      for (ROWTYPE i = start; i < end; i++) {
        _product[i] = alpha * av[i] * b[aj[i] - base];
        if (i > start && !_bv.get(i)) {
          _product[i] += _product[i - 1];
          if (_bv.get(i - 1)) {
            _bv.set(i);
          }
        }
      }
      _threadProduct[tid + 1] = 0;
      if (start < end) {
        _threadProduct[tid + 1] = _product[end - 1];
        if (_bv.get(end - 1))
          _threadBv.set(tid + 1);
      }

#pragma omp barrier
#pragma omp single
      {
        for (int i = 1; i <= nthreads; i++) {
          if (!_threadBv.get(i)) {
            _threadProduct[i] += _threadProduct[i - 1];
            if (_threadBv.get(i - 1)) {
              _threadBv.set(i);
            }
          }
          // std::cout << std::setprecision(16) << _threadProduct[i] << " ";
        }
        // std::cout << std::endl;
      }

      for (ROWTYPE i = start; i < end; i++) {
        if (!_bv.get(i)) {
          _product[i] += _threadProduct[tid];
          // if (_threadBv.get(tid + 1)) {
          //   _bv.set(i);
          // }
        } else {
          break;
        }
      }

      //       // debug
      // #pragma omp barrier
      // #pragma omp single
      //       {
      //         std::ifstream f("test.bin", std::ios::binary);
      //         if (!f.good()) {
      //           std::ofstream ofs("test.bin", std::ios::binary);
      //           ofs.write((char *)&_product[0], _product.size() *
      //           sizeof(double)); ofs.close();
      //         } else {
      //           std::ifstream ofs("test.bin", std::ios::binary);
      //           std::vector<double> ref(_nnz);
      //           ofs.read(reinterpret_cast<char *>(&ref[0]), _nnz *
      //           sizeof(double)); ofs.close();

      //           for (int i = 0; i < _nnz; i++) {
      //             if (ref[i] != _product[i]) {
      //               std::cout << i << " " << ref[i] << " " << _product[i]
      //                         << std::endl;
      //             }
      //           }
      //         }
      //       }

#pragma omp barrier
      for (COLTYPE i = start_row; i < end_row; i++) {
        if (ai[i + 1] > ai[i]) {
          const COLTYPE jIDX = ai[i + 1] - base - 1;
          x[i] = _product[jIDX] + beta * x[i];
        } else {
          x[i] = beta * x[i];
        }
      }
    }
  }

private:
  int _nthreads;
  mutable utils::BitVector<ROWTYPE> _bv;
  mutable std::vector<VALTYPE> _product;
  mutable utils::BitVector<int> _threadBv;
  mutable std::vector<VALTYPE> _threadProduct;
};

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, typename T>
constexpr bool spmv_has_preprocess =
    requires(const COLTYPE size, const int base, ROWTYPE const *ai,
             COLTYPE const *aj, VALTYPE const *av,
             T &t) { t.preprocess(size, base, ai, aj, av); };

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

  void operator()(const VALTYPE *const b, VALTYPE *const x,
                  const VALTYPE alpha = 1., const VALTYPE beta = 0.) const {
    _spmv(_matrix->rows, _matrix->Base(), _matrix->AI(), _matrix->AJ(),
          _matrix->AV(), b, x, alpha, beta);
  }

  CSRMatrixType const *_matrix;
  SPMVType _spmv;
};

} // namespace matrix_utils