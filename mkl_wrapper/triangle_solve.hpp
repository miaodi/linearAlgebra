#pragma once

#include "BitVector.hpp"
#include "utils.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
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
    _taskInvAdj.resize(_nthreads);
    _threadTaskPrefix.resize(_nthreads + 1);
    _threadPrefixSum.resize(_nthreads + 1);
    _threadPrefixSum[0] = 0;
    // _threadPrefixSum2.resize(_nthreads + 1);
    // std::fill(_threadPrefixSum2.begin(), _threadPrefixSum2.end(), 0);
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

        _taskInvAdjGraph.rows = _tasks;
        _taskInvAdjGraph.cols = _tasks;
        _taskInvAdjGraph.base = 0;
        _taskInvAdjGraph.ai.resize(_tasks + 1);
        _taskInvAdjGraph.ai[0] = 0; // zero based
        _taskInvAdjGraph.aj.resize(_reorderedMat.nnz());

        _taskAdjGraph.rows = _tasks;
        _taskAdjGraph.cols = _tasks;
        _taskAdjGraph.base = 0;
        _taskAdjGraph.ai.resize(_tasks + 1);
        _taskAdjGraph.ai[0] = 0; // zero based
        // _taskAdjGraph.aj.resize(_reorderedMat.nnz());
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
      _threadPrefixSum[tid + 1] = 0; // reset
      // #pragma omp barrier
      // #pragma omp single
      //       {
      //         for (auto i = 0; i <= _tasks; i++) {
      //           std::cout << _taskInvAdjGraph.ai[i] << std::endl;
      //         }
      //       }

      // rebalance the work load
      auto [start2, end2] = utils::LoadPrefixBalancedPartitionPos(
          _taskInvAdjGraph.ai.begin(), _taskInvAdjGraph.ai.begin() + _tasks,
          tid, nthreads);

      COLTYPE maxInvAdjSize = 0;
      for (auto task = start; task < end; task++) {
        maxInvAdjSize = std::max(maxInvAdjSize, _taskInvAdjGraph.ai[task + 1] -
                                                    _taskInvAdjGraph.ai[task]);
      }

#pragma omp critical
      {
        std::cout << "tid: " << tid << " startPos: " << start
                  << " endPos: " << end << std::endl;
      }

      auto startThread =
          std::distance(_threadTaskPrefix.begin(),
                        upper_bound(_threadTaskPrefix.begin(),
                                    _threadTaskPrefix.end(),
                                    static_cast<COLTYPE>(start2))) -
          1;
      auto endThread = std::distance(_threadTaskPrefix.begin(),
                                     upper_bound(_threadTaskPrefix.begin(),
                                                 _threadTaskPrefix.end(),
                                                 static_cast<COLTYPE>(end2))) -
                       1;
      endThread =
          std::min(endThread, static_cast<decltype(endThread)>(_nthreads) - 1);

      // building task inverse adjacency graph
      _taskInvAdj[tid].resize(maxInvAdjSize);
      for (auto thread = startThread; thread <= endThread; thread++) {
        ROWTYPE threadCount = 0;
        const COLTYPE threadBegin = _threadTaskPrefix[thread];
        const COLTYPE threadEnd = _threadTaskPrefix[thread + 1];
        const COLTYPE startTask =
            std::max(static_cast<COLTYPE>(start2), threadBegin);
        const COLTYPE endTask = std::min(static_cast<COLTYPE>(end2), threadEnd);
#pragma omp critical
        std::cout << "tid: " << tid << " startTask: " << startTask
                  << " endTask: " << endTask << std::endl;
        for (auto task = startTask; task < endTask; task++) {
          maxInvAdjSize = 0;
          if (task != startTask)
            _taskInvAdj[tid][maxInvAdjSize++] = task - 1;
          for (COLTYPE row = _taskBoundaryPrefix[task];
               row < _taskBoundaryPrefix[task + 1]; row++) {
            for (COLTYPE j = _reorderedMat.ai[row] - _reorderedMat.base;
                 j < _reorderedMat.ai[row + 1] - _reorderedMat.base; j++) {
              auto col = _reorderedRowIdToTaskId[_reorderedMat.aj[j] -
                                                 _reorderedMat.base];
              if (col < threadBegin || col >= threadEnd) {
                _taskInvAdj[tid][maxInvAdjSize++] = col;
              }
            }
          }
          std::sort(_taskInvAdj[tid].begin(),
                    _taskInvAdj[tid].begin() + maxInvAdjSize);
          maxInvAdjSize = std::distance(
              _taskInvAdj[tid].begin(),
              std::unique(_taskInvAdj[tid].begin(),
                          _taskInvAdj[tid].begin() + maxInvAdjSize));
          // // #pragma omp critical
          // //           std::cout << "tid: " << tid << " maxInvAdjSize: " <<
          // //           maxInvAdjSize
          // //                     << std::endl;
          _taskAdjGraph.ai[task + 1] = maxInvAdjSize;
          std::copy(_taskInvAdj[tid].begin(),
                    _taskInvAdj[tid].begin() + maxInvAdjSize,
                    _taskInvAdjGraph.aj.begin() + _taskInvAdjGraph.ai[task]);
          threadCount += maxInvAdjSize;
        }
        __atomic_add_fetch(&_threadPrefixSum[thread + 1], threadCount,
                           __ATOMIC_RELAXED);
        // #pragma omp critical
        //         std::cout << "tid: " << tid << " threadCount: " <<
        //         threadCount
        //                   << std::endl;
      }

#pragma omp barrier
#pragma omp single
      {
        std::inclusive_scan(_threadPrefixSum.begin(), _threadPrefixSum.end(),
                            _threadPrefixSum.begin());
        _taskAdjGraph.aj.resize(_threadPrefixSum[_nthreads]);
        std::cout << "_threadPrefixSum[_nthreads]: "
                  << _threadPrefixSum[_nthreads] << std::endl;
        _taskAdjGraph.ai[_tasks] = _threadPrefixSum[_nthreads];
      }

      _taskAdjGraph.ai[start] = _threadPrefixSum[tid];
      for (auto task = start; task < end - 1; task++) {
        _taskAdjGraph.ai[task + 1] += _taskAdjGraph.ai[task];
      }

#pragma omp barrier
      for (auto task = start2; task < end2; task++) {
        std::copy(_taskInvAdjGraph.aj.begin() + _taskInvAdjGraph.ai[task],
                  _taskInvAdjGraph.aj.begin() + _taskInvAdjGraph.ai[task] +
                      _taskAdjGraph.ai[task + 1] - _taskAdjGraph.ai[task],
                  _taskAdjGraph.aj.begin() + _taskAdjGraph.ai[task]);
      }
    }
    // std::ifstream f("test.bin");
    // if (!f.good()) {
    //   std::ofstream ofs("test.bin", std::ios::binary);
    //   std::stringstream ss;
    //   cereal::BinaryOutputArchive oarchive(ss);
    //   oarchive(_taskAdjGraph);
    //   ofs << ss.rdbuf();
    // } else {
    //   std::ifstream ofs("test.bin", std::ios::binary);
    //   std::stringstream ss;
    //   ss << ofs.rdbuf();
    //   ofs.close();
    //   CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE> temp;
    //   cereal::BinaryInputArchive iarchive(ss);
    //   iarchive(temp);
    //   for (auto i = 0; i < temp.aj.size(); i++) {
    //     if (temp.aj[i] != _taskAdjGraph.aj[i])
    //       std::cout << "fucked\n";
    //   }
    //   for (auto i = 0; i < temp.ai.size(); i++) {
    //     if (temp.ai[i] != _taskAdjGraph.ai[i])
    //       std::cout << "fucked\n";
    //   }
    // }

    std::swap(_taskAdjGraph, _taskInvAdjGraph);
    _taskAdjGraph.aj.resize(_taskInvAdjGraph.nnz());
    std::cout << "_taskInvAdjGraph: " << _taskInvAdjGraph.ai.size()
              << std::endl;
    std::cout << "_taskAdjGraph.aj: " << _taskAdjGraph.aj.size() << std::endl;
    matrix_utils::ParallelTranspose2(
        _taskInvAdjGraph.rows, _taskInvAdjGraph.cols, _taskInvAdjGraph.base,
        _taskInvAdjGraph.ai.data(), _taskInvAdjGraph.aj.data(),
        (VALTYPE const *)nullptr, _taskAdjGraph.ai.data(),
        _taskAdjGraph.aj.data(), (VALTYPE *)nullptr);

    for (auto i : _taskAdjGraph.ai) {
      std::cout << i << " ";
    }
    std::cout << std::endl;
    for (auto i : _taskInvAdjGraph.ai) {
      std::cout << i << " ";
    }
    std::cout << std::endl;

    _taskInvAdjGraph2.rows = _tasks;
    _taskInvAdjGraph2.cols = _tasks;
    _taskInvAdjGraph2.base = 0;
    _taskInvAdjGraph2.ai.resize(_tasks + 1);
    _taskInvAdjGraph2.ai[0] = 0; // zero based
    // _taskInvAdjGraph2.aj.resize(_taskInvAdjGraph.aj.size());

    std::cout << "_taskAdjGraph is valid: "
              << matrix_utils::ValidCSR(
                     _taskAdjGraph.rows, _taskAdjGraph.cols, _taskAdjGraph.base,
                     _taskAdjGraph.ai.data(), _taskAdjGraph.aj.data())
              << std::endl;

    std::cout << "_taskInvAdjGraph is valid: "
              << matrix_utils::ValidCSR(
                     _taskInvAdjGraph.rows, _taskInvAdjGraph.cols,
                     _taskInvAdjGraph.base, _taskInvAdjGraph.ai.data(),
                     _taskInvAdjGraph.aj.data())
              << std::endl;

    for (auto j = _taskInvAdjGraph.ai[38]; j < _taskInvAdjGraph.ai[38 + 1];
         j++) {
      std::cout << _taskInvAdjGraph.aj[j] << " ";
    }
    std::cout << std::endl;

    for (auto i = 0; i <= _taskInvAdjGraph.rows; i++) {
      // std::cout << _taskInvAdjGraph.ai[i] << std::endl;
      for (auto j = _taskInvAdjGraph.ai[i]; j < _taskInvAdjGraph.ai[i + 1];
           j++) {
        std::cout << _taskInvAdjGraph.aj[j] << " ";
      }
      std::cout << std::endl;
    }

    // #pragma omp parallel
    //     {
    //       const int tid = omp_get_thread_num();
    //       const int nthreads = omp_get_num_threads();

    //       // rebalance the work load
    //       auto [start3, end3] = utils::LoadPrefixBalancedPartitionPos(
    //           _taskAdjGraph.ai.begin(), _taskAdjGraph.ai.begin() + _tasks,
    //           tid, nthreads);

    //       auto startThread =
    //           std::distance(_threadTaskPrefix.begin(),
    //                         upper_bound(_threadTaskPrefix.begin(),
    //                                     _threadTaskPrefix.end(),
    //                                     static_cast<COLTYPE>(start3))) -
    //           1;
    //       auto endThread = std::distance(_threadTaskPrefix.begin(),
    //                                      upper_bound(_threadTaskPrefix.begin(),
    //                                                  _threadTaskPrefix.end(),
    //                                                  static_cast<COLTYPE>(end3)))
    //                                                  -
    //                        1;
    //       endThread =
    //           std::min(endThread, static_cast<decltype(endThread)>(_nthreads)
    //           - 1);

    //       ROWTYPE threadCount = 0;
    //       COLTYPE maxInvAdjSize = 0;
    //       for (auto thread = startThread; thread <= endThread; thread++) {
    //         threadCount = 0;
    //         const COLTYPE threadBegin = _threadTaskPrefix[thread];
    //         const COLTYPE threadEnd = _threadTaskPrefix[thread + 1];
    //         const COLTYPE startTask =
    //             std::max(static_cast<COLTYPE>(start3), threadBegin);
    //         const COLTYPE endTask = std::min(static_cast<COLTYPE>(end3),
    //         threadEnd);
    // #pragma omp critical
    //         std::cout << "tid: " << tid << " startTask: " << startTask
    //                   << " endTask: " << endTask << std::endl;
    //         for (auto task = startTask; task < endTask; task++) {
    //           maxInvAdjSize = 0;

    //           for (COLTYPE child = _taskAdjGraph.ai[task];
    //                child < _taskAdjGraph.ai[task + 1]; child++) {
    //             auto childPtr = _taskAdjGraph.aj.data() +
    //             _taskAdjGraph.ai[task]; auto childEndPtr =
    //                 _taskAdjGraph.aj.data() + _taskAdjGraph.ai[task + 1];
    //             auto parentPtr =
    //                 _taskInvAdjGraph.aj.data() + _taskInvAdjGraph.ai[child];
    //             auto parentEndPtr =
    //                 _taskInvAdjGraph.aj.data() + _taskInvAdjGraph.ai[child +
    //                 1];

    //             bool remove = false;
    //             if (childPtr < childEndPtr) {
    //               std::cout << std::distance(_taskInvAdjGraph.aj.data(),
    //                                          parentPtr) -
    //                                _taskInvAdjGraph.ai[child]
    //                         << std::endl;
    //               parentPtr = std::lower_bound(parentPtr, parentEndPtr,
    //               *childPtr);
    //             }
    //             if (parentPtr < parentEndPtr) {
    //               childPtr = std::lower_bound(childPtr, childEndPtr,
    //               *parentPtr);
    //             }

    //             while (parentPtr != parentEndPtr && childPtr != childEndPtr)
    //             {
    //               COLTYPE cmp = *parentPtr - *childPtr;
    //               if (0 == cmp) {
    //                 remove = true;
    //                 break;
    //               } else if (cmp < 0)
    //                 ++parentPtr;
    //               else
    //                 ++childPtr;
    //             }
    //             // if(!remove)
    //           }
    //         }
    //       }
    //     }
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
  CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE> _taskInvAdjGraph; // parents
  CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE> _taskAdjGraph;    // children
  CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE>
      _taskInvAdjGraph2; // parents after transisive edge removal
  std::vector<COLTYPE> _threadTaskPrefix; // tasks on each thread
  std::vector<COLTYPE>
      _taskBoundaryPrefix; // num of rows in each task size task + 1
  std::vector<ROWTYPE> _threadPrefixSum; //
  // std::vector<ROWTYPE> _threadPrefixSum2; //
  std::vector<COLTYPE> _reorderedRowIdToTaskId;
  std::vector<std::vector<COLTYPE>> _taskInvAdj; // thread local

  mutable utils::BitVector<COLTYPE> _bv;
}; // namespace matrix_utils
} // namespace matrix_utils