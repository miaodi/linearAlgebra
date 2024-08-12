#pragma once

#include "BitVector.hpp"
#include "utils.h"
#include <chrono>
#include <execution>
#include <memory>
#include <numeric>
#include <omp.h>
#include <span>
#include <thread>
#include <tuple>
#include <type_traits>

#include "matrix_utils.hpp"

namespace matrix_utils {

/// @brief Forward-substitution algorithm for low triangular csr matrix L. Note
/// that the diagonal term is assumed to be 1
/// @tparam SIZE
/// @tparam R
/// @tparam C
/// @tparam V
/// @tparam VALTYPE
/// @param size
/// @param base
/// @param ai
/// @param aj
/// @param av
/// @param rhs
/// @param x
template <typename SIZE, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void ForwardSubstitution(const SIZE size, const int base, ROWTYPE const *ai,
                         COLTYPE const *aj, VALTYPE const *av,
                         VALTYPE const *const b, VALTYPE *const x) {
  ROWTYPE j;
  for (SIZE i = 0; i < size; i++) {
    x[i] = b[i];
    for (j = ai[i] - base; j < ai[i + 1] - base; j++) {
      x[i] -= av[j] * x[aj[j] - base];
    }
  }
}

/// @brief Forward-substitution algorithm for low triangular csr matrix L
/// obtained by the transpose of a strict upper triangular csr matrix. Note that
/// the diagonal term is assumed to be 1
/// @tparam SIZE
/// @tparam R
/// @tparam C
/// @tparam V
/// @tparam VALTYPE
/// @param size
/// @param base
/// @param ai
/// @param aj
/// @param av
/// @param rhs
/// @param x
template <typename SIZE, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void ForwardSubstitutionT(const SIZE size, const int base, ROWTYPE const *ai,
                          COLTYPE const *aj, VALTYPE const *av,
                          VALTYPE const *const b, VALTYPE *const x) {
  ROWTYPE j;
  std::copy(b, b + size, x);
  for (SIZE i = 0; i < size; i++) {
    for (j = ai[i] - base; j < ai[i + 1] - base; j++) {
      x[aj[j] - base] -= av[j] * x[i];
    }
  }
}

template <typename SIZE, typename ROWTYPE, typename COLTYPE, typename VALTYPE,
          typename VEC>
void LevelScheduleForwardSubstitution(const VEC &iperm, const VEC &prefix,
                                      const SIZE rows, const int base,
                                      ROWTYPE const *ai, COLTYPE const *aj,
                                      VALTYPE const *av, VALTYPE const *const b,
                                      VALTYPE *const x) {
#pragma omp parallel
  {
    for (int l = 0; l < prefix.size() - 1; l++) {
#pragma omp for
      for (SIZE i = prefix[l]; i < prefix[l + 1]; i++) {
        const SIZE idx = iperm[i] - base;
        x[idx] = b[idx];
        for (auto j = ai[idx] - base; j < ai[idx + 1] - base; j++) {
          x[idx] -= av[j] * x[aj[j] - base];
        }
      }
#pragma omp barrier
    }
  }
}

template <bool WithBarrier = true, typename SIZE = int, typename ROWTYPE = int,
          typename COLTYPE = int, typename VALTYPE = double>
class OptimizedForwardSubstitution {
public:
  OptimizedForwardSubstitution() : _nthreads{omp_get_max_threads()} {}

  void analysis(const SIZE rows, const int base, ROWTYPE const *ai,
                COLTYPE const *aj, VALTYPE const *av) {
    _size = rows;
    if constexpr (!WithBarrier)
      _bv.resize(rows);
    const auto nnz = ai[rows] - base;
    _reorderedMat.ai.resize(rows + 1);
    _reorderedMat.aj.resize(nnz);
    _reorderedMat.av.resize(nnz);
    _reorderedMat.base = base;
    _reorderedMat.rows = rows;
    _nthreads = omp_get_max_threads();
    matrix_utils::TopologicalSort2<matrix_utils::TriangularSolve::L>(
        rows, base, ai, aj, _iperm, _levelPrefix);
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
        auto [start, end] = utils::LoadBalancedPartition(
            _iperm.data() + _levelPrefix[l],
            _iperm.data() + _levelPrefix[l + 1], tid, nthreads);
        const COLTYPE size = std::distance(start, end);
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
        auto [start, end] = utils::LoadBalancedPartition(
            _iperm.data() + _levelPrefix[l],
            _iperm.data() + _levelPrefix[l + 1], tid, nthreads);
        for (auto it = start; it != end; it++) {
          _threadiperm[cur++] = *it;
        }
      }
    }
    matrix_utils::permuteRow(rows, base, ai, aj, av, _threadiperm.data(),
                             _reorderedMat.ai.data(), _reorderedMat.aj.data(),
                             _reorderedMat.av.data());
  }

  void operator()(const VALTYPE *const b, VALTYPE *const x) const {
    if constexpr (!WithBarrier) {
      _bv.clearAll();
    }
#pragma omp parallel num_threads(_nthreads)
    {
      const int tid = omp_get_thread_num();
      const int nthreads = omp_get_num_threads();
      for (COLTYPE l = 0; l < _levels; l++) {
        const COLTYPE start = _threadlevels[tid][l];
        const COLTYPE end = _threadlevels[tid][l + 1];
        // std::cout << "tid: " << tid << " , start: " << start
        //           << " , end: " << end << std::endl;
        for (COLTYPE i = start; i < end; i++) {
          const SIZE idx = _threadiperm[i] - _reorderedMat.base;
          // std::cout << _reorderedMat.ai[i] << " " << _reorderedMat.ai[i +
          // 1]
          //           << std::endl;
          x[idx] = b[idx];
          for (auto j = _reorderedMat.ai[i] - _reorderedMat.base;
               j < _reorderedMat.ai[i + 1] - _reorderedMat.base; j++) {
            const COLTYPE j_idx = _reorderedMat.aj[j] - _reorderedMat.base;

            if constexpr (!WithBarrier) {
              while (!_bv.get(j_idx)) {
                std::this_thread::yield();
                // sleep(0);
                // std::this_thread::sleep_for(std::chrono::nanoseconds(1));
                // std::cout << "j_idx: " << j_idx << std::endl;
                // continue;
              }
            }
            x[idx] -= _reorderedMat.av[j] * x[j_idx];
          }
          if constexpr (!WithBarrier) {
            _bv.set(idx);
          }
        }
        if constexpr (WithBarrier) {
#pragma omp barrier
        }
      }
    }
  }

  void build_task_graph() {
    _threadTaskPrefix.resize(_nthreads + 1);
    _threadPrefixSum.resize(_nthreads + 1);
    _threadPrefixSum[0] = 0;
    _threadPrefixSum2.resize(_nthreads + 1);
    std::fill(_threadPrefixSum2.begin(), _threadPrefixSum2.end(), 0);
    _reorderedRowIdToTaskId.resize(_size);
    std::cout << "levels: " << _levels << std::endl;

    // const SIZE num_tasks = _nthreads * _levels;

    // _permedToTask.resize(num_tasks);

#pragma omp parallel num_threads(_nthreads)
    {
      const int tid = omp_get_thread_num();
      const int nthreads = omp_get_num_threads();

      COLTYPE cnt = 0;
      for (COLTYPE l = 0; l < _levels; l++) {
        if (_threadlevels[tid][l + 1] > _threadlevels[tid][l])
          ++cnt;
      }
      _threadTaskPrefix[tid + 1] = cnt;

#pragma omp barrier
#pragma omp single
      {
        _threadTaskPrefix[0] = 0;
        std::inclusive_scan(_threadTaskPrefix.begin(), _threadTaskPrefix.end(),
                            _threadTaskPrefix.begin());
        _tasks = _threadTaskPrefix[_nthreads];

        std::cout << "tasks: " << _tasks << std::endl;
        _taskBoundaryPrefix.resize(_tasks + 1);
        _nnzPerTask.resize(_tasks);

        _taskInvAdjGraph.rows = _tasks;
        _taskInvAdjGraph.cols = _tasks;
        _taskInvAdjGraph.base = 0;
        _taskInvAdjGraph.ai.resize(_tasks + 1);
        _taskInvAdjGraph.ai[0] = 0; // zero based
        _taskInvAdjGraph.aj.resize(_reorderedMat.nnz());

        _taskInvAdjGraphTemp.rows = _tasks;
        _taskInvAdjGraphTemp.cols = _tasks;
        _taskInvAdjGraphTemp.base = 0;
        _taskInvAdjGraphTemp.ai.resize(_tasks + 1);
        _taskInvAdjGraphTemp.ai[0] = 0; // zero based
        _taskInvAdjGraphTemp.aj.resize(_reorderedMat.nnz());
      }

      COLTYPE taskOffset = _threadTaskPrefix[tid];
      for (COLTYPE l = 0; l < _levels; l++) {
        if (_threadlevels[tid][l + 1] > _threadlevels[tid][l])
          _taskBoundaryPrefix[taskOffset++] =
              _threadlevels[tid][l + 1] - _threadlevels[tid][l];
      }

#pragma omp barrier
#pragma omp single
      {
        _taskBoundaryPrefix[0] = 0;
        std::inclusive_scan(_taskBoundaryPrefix.begin(),
                            _taskBoundaryPrefix.end(),
                            _taskBoundaryPrefix.begin());
      }

      // split  tasks to each  thread
      auto [start, end] =
          utils::LoadBalancedPartitionPos(_tasks, tid, nthreads);
#pragma omp critical
      {
        std::cout << "tid: " << tid << " start:  " << start << "  end: " << end
                  << std::endl;
      }
      _threadPrefixSum[tid + 1] = 0;
      for (COLTYPE task = start; task < end; task++) {
        COLTYPE invAdjSizePerTask = 0;
        for (COLTYPE i = _taskBoundaryPrefix[task];
             i < _taskBoundaryPrefix[task + 1]; i++) {
          invAdjSizePerTask += _reorderedMat.ai[i + 1] - _reorderedMat.ai[i];
          _reorderedRowIdToTaskId[i] = task;
        }
        _threadPrefixSum[tid + 1] += invAdjSizePerTask;
        _taskInvAdjGraph.ai[task + 1] = _threadPrefixSum[tid + 1];
        _nnzPerTask[task] = invAdjSizePerTask;
        // #pragma omp critical
        //         {
        //           std::cout << "tid: " << tid << " task: " << task
        //                     << " ai:  " << _taskInvAdjGraph.ai[task + 1] <<
        //                     std::endl;
        //         }
      }

#pragma omp barrier
#pragma omp single
      {
        std::inclusive_scan(_threadPrefixSum.begin(), _threadPrefixSum.end(),
                            _threadPrefixSum.begin());
      }
      for (COLTYPE task = start; task < end; task++) {
        _taskInvAdjGraph.ai[task + 1] += _threadPrefixSum[tid];
      }

#pragma omp barrier
      // #pragma omp barrier
      // #pragma omp single
      //       {
      //         for (auto i = 0; i <= _tasks; i++) {
      //           std::cout << _taskInvAdjGraph.ai[i] << std::endl;
      //         }
      //       }

      // rebalance the work load
      std::tie(start, end) = utils::LoadPrefixBalancedPartitionPos(
          _taskInvAdjGraph.ai.begin(), _taskInvAdjGraph.ai.begin() + _tasks,
          tid, nthreads);

      COLTYPE maxInvAdjSize = 0;
      for (auto task = start; task < end; task++) {
        maxInvAdjSize = std::max(maxInvAdjSize, _nnzPerTask[task]);
      }

#pragma omp critical
      {
        std::cout << "tid: " << tid << " startPos: " << start
                  << " endPos: " << end << std::endl;
      }

      auto startThread =
          std::distance(_threadTaskPrefix.begin(),
                        upper_bound(_threadTaskPrefix.begin(),
                                    _threadTaskPrefix.end(), start)) -
          1;
      auto endThread =
          std::distance(_threadTaskPrefix.begin(),
                        upper_bound(_threadTaskPrefix.begin(),
                                    _threadTaskPrefix.end(), end)) -
          1;
      endThread =
          std::min(endThread, static_cast<decltype(endThread)>(_nthreads) - 1);

      // building task inverse adjacency graph
      _taskInvAdj.resize(maxInvAdjSize);
      for (auto thread = startThread; thread <= endThread; thread++) {
        ROWTYPE threadCount = 0;
        const COLTYPE threadBegin = _threadTaskPrefix[thread];
        const COLTYPE threadEnd = _threadTaskPrefix[thread + 1];
        const COLTYPE startTask = std::max(start, threadBegin);
        const COLTYPE endTask = std::min(end, threadEnd);
#pragma omp critical
        std::cout << "tid: " << tid << " startTask: " << startTask
                  << " endTask: " << endTask << std::endl;
        for (auto task = startTask; task < endTask; task++) {
          maxInvAdjSize = 0;
          if (task != startTask)
            _taskInvAdj[maxInvAdjSize++] = task - 1;
          for (COLTYPE row = _taskBoundaryPrefix[task];
               row < _taskBoundaryPrefix[task + 1]; row++) {
            for (COLTYPE j = _reorderedMat.ai[row] - _reorderedMat.base;
                 j < _reorderedMat.ai[row + 1] - _reorderedMat.base; j++) {
              auto col = _reorderedRowIdToTaskId[_reorderedMat.aj[j] -
                                                 _reorderedMat.base];
              if (col < threadBegin || col >= threadEnd) {
                _taskInvAdj[maxInvAdjSize++] = col;
              }
            }
          }
          std::sort(_taskInvAdj.begin(), _taskInvAdj.begin() + maxInvAdjSize);
          maxInvAdjSize =
              std::distance(_taskInvAdj.begin(),
                            std::unique(_taskInvAdj.begin(),
                                        _taskInvAdj.begin() + maxInvAdjSize));
          // #pragma omp critical
          //           std::cout << "tid: " << tid << " maxInvAdjSize: " <<
          //           maxInvAdjSize
          //                     << std::endl;
          _taskInvAdjGraphTemp.ai[task + 1] = maxInvAdjSize;
          std::copy(_taskInvAdj.begin(), _taskInvAdj.begin() + maxInvAdjSize,
                    _taskInvAdjGraph.aj.begin() + _taskInvAdjGraph.ai[task]);
          threadCount += maxInvAdjSize;
        }
        __sync_fetch_and_add(&_threadPrefixSum2[thread + 1], threadCount);
        std::cout << threadCount << std::endl;
      }

      // #pragma omp barrier
      // #pragma omp single
      //       {
      //         for (auto i : _threadPrefixSum2) {
      //           std::cout << i << std::endl;
      //         }
      //       }
    }
  }

protected:
  int _nthreads;
  SIZE _size;
  std::vector<COLTYPE> _iperm;
  std::vector<COLTYPE> _levelPrefix;

  COLTYPE _levels;
  std::vector<std::vector<COLTYPE>>
      _threadlevels; // level prefix for each thread
  std::vector<COLTYPE> _threadiperm;
  CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE> _reorderedMat;

  // always zero based
  COLTYPE _tasks;
  CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE> _taskInvAdjGraph;
  CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE> _taskInvAdjGraphTemp;
  std::vector<COLTYPE> _threadTaskPrefix; // tasks on each thread
  std::vector<COLTYPE>
      _taskBoundaryPrefix; // num of rows in each task size task + 1
  std::vector<ROWTYPE> _threadPrefixSum;  //
  std::vector<ROWTYPE> _threadPrefixSum2; //
  std::vector<COLTYPE> _reorderedRowIdToTaskId;
  std::vector<COLTYPE> _taskInvAdj;
  std::vector<COLTYPE> _nnzPerTask;

  mutable utils::BitVector<COLTYPE> _bv;
};
} // namespace matrix_utils