#include "BitVector.hpp"
#include "utils.h"
// #include <cereal/archives/binary.hpp>
// #include <cereal/types/vector.hpp>
#include <chrono>
#include <execution>
#include <fstream>
#include <memory>
#include <numeric>
#include <omp.h>
#include <span>
#include <thread>
#include <tuple>
#include <type_traits>

#include "matrix_utils.hpp"

namespace matrix_utils {
template <FBSubstitutionType FBST, TriangularSolve TS, typename ROWTYPE,
          typename COLTYPE, typename VALTYPE>
void OptimizedTriangularSolve<FBST, TS, ROWTYPE, COLTYPE, VALTYPE>::analysis(
    const COLTYPE rows, const int base, ROWTYPE const *ai, COLTYPE const *aj,
    VALTYPE const *av, VALTYPE const *diag) {
  _diag = diag;
  _size = rows;
  _vec.resize(_size);
  const auto nnz = ai[rows] - base;
  _reorderedMat.ai.resize(rows + 1);
  _reorderedMat.aj.resize(nnz);
  _reorderedMat.av.resize(nnz);
  _reorderedMat.base = base;
  _reorderedMat.rows = rows;
  matrix_utils::TopologicalSort2<TS>(rows, base, ai, aj, _iperm, _levelPrefix);
  _levels = _levelPrefix.size() - 1;
  _threadlevels.resize(_nthreads);
  _threadiperm.resize(rows);

#pragma omp parallel num_threads(_nthreads)
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    // #pragma omp single
    //       std::cout << "nthreads: " << nthreads << std::endl;

    _threadlevels[tid].resize(_levels + 1);
    _threadlevels[tid][0] = 0;

    for (COLTYPE l = 0; l < _levels; l++) {
      // TODO: a better load balancing is needed
      auto [start, end] = utils::LoadBalancedPartitionPos(
          _levelPrefix[l + 1] - _levelPrefix[l], tid, nthreads);
      const COLTYPE size = end - start;
      // #pragma omp critical
      //         std::cout << "tid: " << tid << " , size: " << size <<
      //         std::endl;
      _threadlevels[tid][l + 1] = _threadlevels[tid][l] + size;
    }

#pragma omp barrier
#pragma omp single
    {
      COLTYPE size = 0;
      for (int tid = 1; tid < nthreads; tid++) {
        size += _threadlevels[tid - 1][_levels];
        _threadlevels[tid][0] = size;
      }
    }

    for (COLTYPE l = 0; l < _levels; l++) {
      _threadlevels[tid][l + 1] += _threadlevels[tid][0];
    }
    // up to this point, _threadlevels becomes the prefix of size of each
    // super task

#pragma omp barrier
    COLTYPE cur = _threadlevels[tid][0];

    for (COLTYPE l = 0; l < _levels; l++) {
      auto [start, end] = utils::LoadBalancedPartitionPos(
          _levelPrefix[l + 1] - _levelPrefix[l], tid, nthreads);
      for (auto i = start; i != end; i++) {
        _threadiperm[cur++] = _iperm[i + _levelPrefix[l]];
      }
    }
  }

  utils::inversePermute(_threadperm, _threadiperm, base);

  // matrix_utils::permute(rows, base, ai, aj, av, _threadiperm.data(),
  //                       _threadperm.data(), _reorderedMat.ai.data(),
  //                       _reorderedMat.aj.data(), _reorderedMat.av.data());

  matrix_utils::permuteRow(rows, base, ai, aj, av, _threadiperm.data(),
                           _reorderedMat.ai.data(), _reorderedMat.aj.data(),
                           _reorderedMat.av.data());

  if constexpr (FBST == FBSubstitutionType::NoBarrierSuperNode) {
    build_task_graph();
    // for (auto i = 0; i < _taskInvAdjGraph.rows; i++) {
    //   std::cout << "taks " << i << ": ";
    //   for (auto j = _taskInvAdjGraph.ai[i]; j < _taskInvAdjGraph.ai[i + 1];
    //        j++) {
    //     std::cout << _taskInvAdjGraph.aj[j] << " ";
    //   }
    //   std::cout << std::endl;
    // }
  }

  if constexpr (FBST == FBSubstitutionType::NoBarrier)
    _bv.resize(_size);
  else if constexpr (FBST == FBSubstitutionType::NoBarrierSuperNode)
    _bv.resize(_tasks);
}

template <FBSubstitutionType FBST, TriangularSolve TS, typename ROWTYPE,
          typename COLTYPE, typename VALTYPE>
void OptimizedTriangularSolve<FBST, TS, ROWTYPE, COLTYPE, VALTYPE>::operator()(
    const VALTYPE *const b, VALTYPE *const x) const {
  if constexpr (FBST == FBSubstitutionType::Barrier)
    BarrierOp(b, x);
  else if constexpr (FBST == FBSubstitutionType::NoBarrier)
    NoBarrierOp(b, x);
  else if constexpr (FBST == FBSubstitutionType::NoBarrierSuperNode)
    NoBarrierSuperNodeOp(b, x);
}

template <FBSubstitutionType FBST, TriangularSolve TS, typename ROWTYPE,
          typename COLTYPE, typename VALTYPE>
void OptimizedTriangularSolve<FBST, TS, ROWTYPE, COLTYPE, VALTYPE>::BarrierOp(
    const VALTYPE *const b, VALTYPE *const x) const {
#pragma omp parallel num_threads(_nthreads)
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    for (COLTYPE l = 0; l < _levels; l++) {
      const COLTYPE start = _threadlevels[tid][l];
      const COLTYPE end = _threadlevels[tid][l + 1];
      for (COLTYPE i = start; i < end; i++) {
        const COLTYPE idx = _threadiperm[i] - _reorderedMat.base;
        VALTYPE val = 0;
        for (auto j = _reorderedMat.ai[i] - _reorderedMat.base;
             j < _reorderedMat.ai[i + 1] - _reorderedMat.base; j++) {
          const COLTYPE j_idx = _reorderedMat.aj[j] - _reorderedMat.base;
          val += _reorderedMat.av[j] * x[j_idx];
        }
        x[idx] = _diag ? (b[idx] - val) / _diag[idx] : (b[idx] - val);
      }
#pragma omp barrier
    }
  }
  // std::copy(_vec.begin(), _vec.end(), x);
  // matrix_utils::permuteVec(_size, _reorderedMat.base, _vec.data(),
  //                          _threadperm.data(), x);
}

template <FBSubstitutionType FBST, TriangularSolve TS, typename ROWTYPE,
          typename COLTYPE, typename VALTYPE>
void OptimizedTriangularSolve<FBST, TS, ROWTYPE, COLTYPE, VALTYPE>::NoBarrierOp(
    const VALTYPE *const b, VALTYPE *const x) const {
  _bv.clearAll();
#pragma omp parallel num_threads(_nthreads)
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    for (COLTYPE l = 0; l < _levels; l++) {
      const COLTYPE start = _threadlevels[tid][l];
      const COLTYPE end = _threadlevels[tid][l + 1];
      for (COLTYPE i = start; i < end; i++) {
        const COLTYPE idx = _threadiperm[i] - _reorderedMat.base;
        VALTYPE val = 0;
        for (auto j = _reorderedMat.ai[i] - _reorderedMat.base;
             j < _reorderedMat.ai[i + 1] - _reorderedMat.base; j++) {
          const COLTYPE j_idx = _reorderedMat.aj[j] - _reorderedMat.base;
          while (!_bv.get(j_idx)) {
            std::this_thread::yield();
          }
          val += _reorderedMat.av[j] * x[j_idx];
        }
        x[idx] = _diag ? (b[idx] - val) / _diag[idx] : (b[idx] - val);
        _bv.set(idx);
      }
    }
  }
}

template <FBSubstitutionType FBST, TriangularSolve TS, typename ROWTYPE,
          typename COLTYPE, typename VALTYPE>
void OptimizedTriangularSolve<FBST, TS, ROWTYPE, COLTYPE, VALTYPE>::
    NoBarrierSuperNodeOp(const VALTYPE *const b, VALTYPE *const x) const {
  _bv.clearAll();
#pragma omp parallel num_threads(_nthreads)
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    for (COLTYPE task = _threadTaskPrefix[tid];
         task < _threadTaskPrefix[tid + 1]; task++) {

      for (COLTYPE i = _taskInvAdjGraph2.ai[task];
           i < _taskInvAdjGraph2.ai[task + 1]; i++) {
        const COLTYPE j_idx = _taskInvAdjGraph2.aj[i];
        while (!_bv.get(j_idx)) {
          std::this_thread::yield();
          // std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
      }

      for (COLTYPE i = _taskBoundaryPrefix[task];
           i < _taskBoundaryPrefix[task + 1]; i++) {
        const COLTYPE idx = _threadiperm[i] - _reorderedMat.base;
        VALTYPE val = 0;
        for (auto j = _reorderedMat.ai[i] - _reorderedMat.base;
             j < _reorderedMat.ai[i + 1] - _reorderedMat.base; j++) {
          const COLTYPE j_idx = _reorderedMat.aj[j] - _reorderedMat.base;
          val += _reorderedMat.av[j] * x[j_idx];
        }
        x[idx] = _diag ? (b[idx] - val) / _diag[idx] : (b[idx] - val);
      }
      _bv.set(task);
    }
  }
}

} // namespace matrix_utils