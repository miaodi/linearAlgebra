#pragma once

#include "BitVector.hpp"
#include "matrix_utils.hpp"
#include <omp.h>

namespace matrix_utils {

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void SPMV(const COLTYPE size, const int base, ROWTYPE const *ai,
          COLTYPE const *aj, VALTYPE const *av, VALTYPE const *const b,
          VALTYPE *const x) {
  for (COLTYPE i = 0; i < size; i++) {
    VALTYPE val = 0;
    for (ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; j++) {
      val += av[j] * b[aj[j] - base];
    }
    x[i] = val;
  }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void ParallelSPMV(const COLTYPE size, const int base, ROWTYPE const *ai,
                  COLTYPE const *aj, VALTYPE const *av, VALTYPE const *const b,
                  VALTYPE *const x) {

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();

    auto [start, end] =
        utils::LoadPrefixBalancedPartitionPos(ai, ai + size, tid, nthreads);

#pragma omp for
    for (COLTYPE i = start; i < end; i++) {
      VALTYPE val = 0;
      for (ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; j++) {
        val += av[j] * b[aj[j] - base];
      }
      x[i] = val;
    }
  }
}
template <typename ROWTYPE = int, typename COLTYPE = int,
          typename VALTYPE = double>
class SegSumSPMV {
public:
  SegSumSPMV(const int num_threads = omp_get_num_threads())
      : _nthreads{num_threads} {
    _threadBv.resize(_nthreads + 1);
    _threadProduct.resize(_nthreads + 1);
    _threadProduct[0] = 0;
  }

  void set_mat(const COLTYPE size, const int base, ROWTYPE const *ai,
               COLTYPE const *aj, VALTYPE const *av) {
    _size = size;
    _base = base;
    _ai = ai;
    _aj = aj;
    _av = av;
    _nnz = ai[size] - base;
    _bv.resize(_nnz);
    _product.resize(_nnz);
  }

  void operator()(const VALTYPE *const b, VALTYPE *const x) const {
    _bv.clearAll();
    _threadBv.clearAll();
#pragma omp parallel num_threads(_nthreads)
    {
      const int tid = omp_get_thread_num();
      const int nthreads = omp_get_num_threads();

      auto [start_row, end_row] =
          utils::LoadBalancedPartitionPos(_size, tid, nthreads);

      for (COLTYPE i = start_row; i < end_row; i++) {
        _bv.set(_ai[i] - _base);
      }

#pragma omp barrier
      auto [start, end] = utils::LoadBalancedPartitionPos(_nnz, tid, nthreads);

      for (ROWTYPE i = start; i < end; i++) {
        _product[i] = _av[i] * b[_aj[i] - _base];
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
        if (_ai[i + 1] > _ai[i]) {
          const COLTYPE jIDX = _ai[i + 1] - _base - 1;
          x[i] = _product[jIDX];
        } else {
          x[i] = 0;
        }
      }
    }
  }

private:
  int _nthreads;
  COLTYPE _size;
  ROWTYPE _nnz;
  int _base;
  ROWTYPE const *_ai;
  COLTYPE const *_aj;
  VALTYPE const *_av;

  mutable utils::BitVector<ROWTYPE> _bv;
  mutable std::vector<VALTYPE> _product;
  mutable utils::BitVector<int> _threadBv;
  mutable std::vector<VALTYPE> _threadProduct;
};

} // namespace matrix_utils