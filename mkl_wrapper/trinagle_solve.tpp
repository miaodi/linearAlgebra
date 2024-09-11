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
  _reorderedMat.ai[0] = base;
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

    // prepare cost for load balance of each level

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
        const COLTYPE idx = _threadiperm[i] - _reorderedMat.Base();
        VALTYPE val = 0;
#pragma unroll
        for (auto j = _reorderedMat.ai[i] - _reorderedMat.Base();
             j < _reorderedMat.ai[i + 1] - _reorderedMat.Base(); j++) {
          const COLTYPE j_idx = _reorderedMat.aj[j] - _reorderedMat.Base();
          val += _reorderedMat.av[j] * x[j_idx];
        }
        x[idx] = _diag ? (b[idx] - val) / _diag[idx] : (b[idx] - val);
      }
#pragma omp barrier
    }
  }
  // std::copy(_vec.begin(), _vec.end(), x);
  // matrix_utils::permuteVec(_size, _reorderedMat.Base(), _vec.data(),
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
        const COLTYPE idx = _threadiperm[i] - _reorderedMat.Base();
        VALTYPE val = 0;
        for (auto j = _reorderedMat.ai[i] - _reorderedMat.Base();
             j < _reorderedMat.ai[i + 1] - _reorderedMat.Base(); j++) {
          const COLTYPE j_idx = _reorderedMat.aj[j] - _reorderedMat.Base();
          while (!_bv.get(j_idx)) {
            // std::cout << "tid: " << tid << "yield\n";
            // std::this_thread::yield();
            _mm_pause();
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
          // std::this_thread::yield();
          _mm_pause();
          // std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
      }

      for (COLTYPE i = _taskBoundaryPrefix[task];
           i < _taskBoundaryPrefix[task + 1]; i++) {
        const COLTYPE idx = _threadiperm[i] - _reorderedMat.Base();
        VALTYPE val = 0;
#pragma unroll
        for (auto j = _reorderedMat.ai[i] - _reorderedMat.Base();
             j < _reorderedMat.ai[i + 1] - _reorderedMat.Base(); j++) {
          const COLTYPE j_idx = _reorderedMat.aj[j] - _reorderedMat.Base();
          val += _reorderedMat.av[j] * x[j_idx];
        }
        x[idx] = _diag ? (b[idx] - val) / _diag[idx] : (b[idx] - val);
      }
      _bv.set(task);
    }
  }
}

template <FBSubstitutionType FBST, TriangularSolve TS, typename ROWTYPE,
          typename COLTYPE, typename VALTYPE>
void OptimizedTriangularSolve<FBST, TS, ROWTYPE, COLTYPE,
                              VALTYPE>::build_task_graph() {
  _taskInvAdj.resize(_nthreads);
  _threadTaskPrefix.resize(_nthreads + 1);
  _threadPrefixSum.resize(_nthreads + 1);
  _threadPrefixSum[0] = 0;
  // _threadPrefixSum2.resize(_nthreads + 1);
  // std::fill(_threadPrefixSum2.begin(), _threadPrefixSum2.end(), 0);
  _reorderedRowIdToTaskId.resize(_size);
  // std::cout << "levels: " << _levels << std::endl;

#pragma omp parallel num_threads(_nthreads)
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();

    // count the number of tasks for each thread
    COLTYPE cnt = 0;
    for (COLTYPE l = 0; l < _levels; l++) {
      if (_threadlevels[tid][l + 1] > _threadlevels[tid][l])
        ++cnt;
    }
    _threadTaskPrefix[tid + 1] = cnt; // tasks in each thread

#pragma omp barrier
#pragma omp single
    {
      _threadTaskPrefix[0] = 0;
      std::inclusive_scan(_threadTaskPrefix.begin(), _threadTaskPrefix.end(),
                          _threadTaskPrefix.begin());
      _tasks = _threadTaskPrefix[_nthreads];

      // taskSizes.resize(_tasks);

      // std::cout << "tasks: " << _tasks << std::endl;
      _taskBoundaryPrefix.resize(_tasks + 1);

      _taskInvAdjGraph.rows = _tasks;
      _taskInvAdjGraph.cols = _tasks;
      _taskInvAdjGraph.ai.resize(_tasks + 1);
      _taskInvAdjGraph.ai[0] = 0; // zero based
      _taskInvAdjGraph.aj.resize(
          _reorderedMat.NNZ() +
          _tasks); // added _tasks for edges within super-tasks

      _taskAdjGraph.rows = _tasks;
      _taskAdjGraph.cols = _tasks;
      _taskAdjGraph.ai.resize(_tasks + 1);
      _taskAdjGraph.ai[0] = 0; // zero based
      // _taskAdjGraph.aj.resize(_reorderedMat.NNZ());
    }

    // build task boundary prefix (prefix of task sizes)
    COLTYPE taskOffset = _threadTaskPrefix[tid];
    for (COLTYPE l = 0; l < _levels; l++) {
      if (_threadlevels[tid][l + 1] > _threadlevels[tid][l]) {
        _taskBoundaryPrefix[++taskOffset] =
            _threadlevels[tid][l + 1] - _threadlevels[tid][l];
      }
    }

#pragma omp barrier
#pragma omp single
    {
      _taskBoundaryPrefix[0] = 0;
      std::inclusive_scan(_taskBoundaryPrefix.begin(),
                          _taskBoundaryPrefix.end(),
                          _taskBoundaryPrefix.begin());
    }

    // split tasks to each thread
    auto [start, end] = utils::LoadBalancedPartitionPos(_tasks, tid, nthreads);
    _threadPrefixSum[tid + 1] = 0;
    for (COLTYPE task = start; task < end; task++) {
      COLTYPE invAdjSizePerTask = 0;
      for (COLTYPE i = _taskBoundaryPrefix[task];
           i < _taskBoundaryPrefix[task + 1]; i++) {
        invAdjSizePerTask += _reorderedMat.ai[i + 1] - _reorderedMat.ai[i];
        _reorderedRowIdToTaskId[i] = task;
      }
      invAdjSizePerTask += 1; // added 1 for task -> task-1 dependency
                              // within each super-tasks
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
        _taskInvAdjGraph.ai.begin(), _taskInvAdjGraph.ai.begin() + _tasks, tid,
        nthreads);

    COLTYPE maxInvAdjSize = 0;
    for (auto task = start2; task < end2; task++) {
      maxInvAdjSize = std::max(maxInvAdjSize, _taskInvAdjGraph.ai[task + 1] -
                                                  _taskInvAdjGraph.ai[task]);
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

      for (auto task = startTask; task < endTask; task++) {
        maxInvAdjSize = 0;
        if (task != threadBegin)
          _taskInvAdj[tid][maxInvAdjSize++] = task - 1;
        for (COLTYPE row = _taskBoundaryPrefix[task];
             row < _taskBoundaryPrefix[task + 1]; row++) {
          for (COLTYPE i = _reorderedMat.ai[row] - _reorderedMat.Base();
               i < _reorderedMat.ai[row + 1] - _reorderedMat.Base(); i++) {
            COLTYPE j = _reorderedMat.aj[i] - _reorderedMat.Base();
            auto col =
                _reorderedRowIdToTaskId[_threadperm[j] - _reorderedMat.Base()];
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
      _taskAdjGraph.ai[_tasks] = _threadPrefixSum[_nthreads];
    }

    _taskAdjGraph.ai[_threadTaskPrefix[tid]] = _threadPrefixSum[tid];
    for (auto task = _threadTaskPrefix[tid];
         task < _threadTaskPrefix[tid + 1] - 1; task++) {
      _taskAdjGraph.ai[task + 1] += _taskAdjGraph.ai[task];
    }

#pragma omp barrier
    for (auto task = start2; task < end2; task++) {
      std::copy_n(_taskInvAdjGraph.aj.begin() + _taskInvAdjGraph.ai[task],
                  _taskAdjGraph.ai[task + 1] - _taskAdjGraph.ai[task],
                  _taskAdjGraph.aj.begin() + _taskAdjGraph.ai[task]);
    }
  }

  std::swap(_taskAdjGraph, _taskInvAdjGraph);

  // std::ifstream f("test.bin");
  // if (!f.good()) {
  //   std::ofstream ofs("test.bin", std::ios::binary);
  //   std::stringstream ss;
  //   cereal::BinaryOutputArchive oarchive(ss);
  //   oarchive(_taskInvAdjGraph);
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
  //     if (temp.aj[i] != _taskInvAdjGraph.aj[i])
  //       std::cout << "fucked\n";
  //   }
  //   for (auto i = 0; i < temp.ai.size(); i++) {
  //     if (temp.ai[i] != _taskInvAdjGraph.ai[i])
  //       std::cout << "fucked\n";
  //   }
  // }

  _taskAdjGraph.aj.resize(_taskInvAdjGraph.NNZ());
  matrix_utils::ParallelTranspose2(
      _taskInvAdjGraph.rows, _taskInvAdjGraph.cols, _taskInvAdjGraph.Base(),
      _taskInvAdjGraph.ai.data(), _taskInvAdjGraph.aj.data(),
      (VALTYPE const *)nullptr, _taskAdjGraph.ai.data(),
      _taskAdjGraph.aj.data(), (VALTYPE *)nullptr);
  // matrix_utils::SerialTranspose(
  //     _taskInvAdjGraph.rows, _taskInvAdjGraph.cols, _taskInvAdjGraph.Base(),
  //     _taskInvAdjGraph.ai.data(), _taskInvAdjGraph.aj.data(),
  //     (VALTYPE const *)nullptr, _taskAdjGraph.ai.data(),
  //     _taskAdjGraph.aj.data(), (VALTYPE *)nullptr);

  _taskInvAdjGraph2.rows = _tasks;
  _taskInvAdjGraph2.cols = _tasks;
  _taskInvAdjGraph2.ai.resize(_tasks + 1);
  _taskInvAdjGraph2.ai[0] = 0; // zero based
  // _taskInvAdjGraph2.aj.resize(_taskInvAdjGraph.aj.size());
  _transitiveEdgeRemoveAj.resize(_taskInvAdjGraph.aj.size());

#ifdef DEBUG
  std::cout << "_taskAdjGraph is valid: "
            << matrix_utils::ValidCSR(
                   _taskAdjGraph.rows, _taskAdjGraph.cols, _taskAdjGraph.Base(),
                   _taskAdjGraph.ai.data(), _taskAdjGraph.aj.data())
            << std::endl;

  std::cout << "_taskInvAdjGraph is valid: "
            << matrix_utils::ValidCSR(
                   _taskInvAdjGraph.rows, _taskInvAdjGraph.cols,
                   _taskInvAdjGraph.Base(), _taskInvAdjGraph.ai.data(),
                   _taskInvAdjGraph.aj.data())
            << std::endl;
#endif

  // for (ROWTYPE i = 0; i < _taskInvAdjGraph.rows; i++) {
  //   for (COLTYPE j = _taskInvAdjGraph.ai[i]; j < _taskInvAdjGraph.ai[i +
  //   1];
  //        j++) {
  //     std::cout << _taskInvAdjGraph.aj[j] << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // std::cout << std::endl;

#pragma omp parallel num_threads(_nthreads)
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();

    // rebalance the work load
    auto [start3, end3] = utils::LoadPrefixBalancedPartitionPos(
        _taskAdjGraph.ai.begin(), _taskAdjGraph.ai.begin() + _tasks, tid,
        nthreads);

    auto startThread =
        std::distance(_threadTaskPrefix.begin(),
                      upper_bound(_threadTaskPrefix.begin(),
                                  _threadTaskPrefix.end(),
                                  static_cast<COLTYPE>(start3))) -
        1;
    auto endThread = std::distance(_threadTaskPrefix.begin(),
                                   upper_bound(_threadTaskPrefix.begin(),
                                               _threadTaskPrefix.end(),
                                               static_cast<COLTYPE>(end3))) -
                     1;
    endThread =
        std::min(endThread, static_cast<decltype(endThread)>(_nthreads) - 1);

    ROWTYPE threadCount = 0;
    COLTYPE maxInvAdjSize = 0;
    COLTYPE parent;
    _threadPrefixSum[tid + 1] = 0;
    for (auto thread = startThread; thread <= endThread; thread++) {
      threadCount = 0;
      const COLTYPE threadBegin = _threadTaskPrefix[thread];
      const COLTYPE threadEnd = _threadTaskPrefix[thread + 1];
      const COLTYPE startTask =
          std::max(static_cast<COLTYPE>(start3), threadBegin);
      const COLTYPE endTask = std::min(static_cast<COLTYPE>(end3), threadEnd);
      for (auto task = startTask; task < endTask; task++) {
        maxInvAdjSize = 0;

        for (ROWTYPE parentID = _taskInvAdjGraph.ai[task];
             parentID < _taskInvAdjGraph.ai[task + 1]; parentID++) {
          parent = _taskInvAdjGraph.aj[parentID];
          auto parentPtr =
              _taskInvAdjGraph.aj.data() + _taskInvAdjGraph.ai[task];
          auto parentEndPtr =
              _taskInvAdjGraph.aj.data() + _taskInvAdjGraph.ai[task + 1];
          auto childPtr = _taskAdjGraph.aj.data() + _taskAdjGraph.ai[parent];
          auto childEndPtr =
              _taskAdjGraph.aj.data() + _taskAdjGraph.ai[parent + 1];

          bool remove = false;
          if (parentPtr < parentEndPtr) {
            childPtr = std::lower_bound(childPtr, childEndPtr, *parentPtr);
          }
          if (childPtr < childEndPtr) {
            parentPtr = std::lower_bound(parentPtr, parentEndPtr, *childPtr);
          }

          while (parentPtr != parentEndPtr && childPtr != childEndPtr) {
            COLTYPE cmp = *parentPtr - *childPtr;
            if (0 == cmp) {
              remove = true;
              break;
            } else if (cmp < 0)
              ++parentPtr;
            else
              ++childPtr;
          }
          if (!remove) {
            _taskInvAdj[tid][maxInvAdjSize++] = parent;
          }
        }
        _taskInvAdjGraph2.ai[task + 1] = maxInvAdjSize;
        std::copy(_taskInvAdj[tid].begin(),
                  _taskInvAdj[tid].begin() + maxInvAdjSize,
                  _transitiveEdgeRemoveAj.begin() + _taskInvAdjGraph.ai[task]);
        threadCount += maxInvAdjSize;
        // #pragma omp critical
        //           {
        //             std::cout << "tid: " << tid << " task " << task
        //                       << " : start point: " <<
        //                       _taskInvAdjGraph.ai[task]
        //                       << " | ";
        //             for (int i = 0; i < maxInvAdjSize; i++) {
        //               std::cout << _taskInvAdj[tid][i] << " ";
        //             }
        //             std::cout << std::endl;
        //           }
      }
      __atomic_add_fetch(&_threadPrefixSum[thread + 1], threadCount,
                         __ATOMIC_RELAXED);
    }
#pragma omp barrier
#pragma omp single
    {
      std::inclusive_scan(_threadPrefixSum.begin(), _threadPrefixSum.end(),
                          _threadPrefixSum.begin());
      _taskInvAdjGraph2.aj.resize(_threadPrefixSum[_nthreads]);
      _taskInvAdjGraph2.ai[_tasks] = _threadPrefixSum[_nthreads];
    }

    _taskInvAdjGraph2.ai[_threadTaskPrefix[tid]] = _threadPrefixSum[tid];
    for (auto task = _threadTaskPrefix[tid];
         task < _threadTaskPrefix[tid + 1] - 1; task++) {
      _taskInvAdjGraph2.ai[task + 1] += _taskInvAdjGraph2.ai[task];
    }

#pragma omp barrier

    for (auto task = start3; task < end3; task++) {
      std::copy(_transitiveEdgeRemoveAj.begin() + _taskInvAdjGraph.ai[task],
                _transitiveEdgeRemoveAj.begin() + _taskInvAdjGraph.ai[task] +
                    _taskInvAdjGraph2.ai[task + 1] - _taskInvAdjGraph2.ai[task],
                _taskInvAdjGraph2.aj.begin() + _taskInvAdjGraph2.ai[task]);
    }
  }

  // // sanity check
  // {
  //   std::cout << "_taskInvAdjGraph2 is valid: "
  //             << matrix_utils::ValidCSR(
  //                    _taskInvAdjGraph2.rows, _taskInvAdjGraph2.cols,
  //                    _taskInvAdjGraph2.Base(), _taskInvAdjGraph2.ai.data(),
  //                    _taskInvAdjGraph2.aj.data())
  //             << std::endl;

  //   std::ifstream f("test.bin");
  //   if (!f.good()) {
  //     std::ofstream ofs("test.bin", std::ios::binary);
  //     std::stringstream ss;
  //     cereal::BinaryOutputArchive oarchive(ss);
  //     oarchive(_taskInvAdjGraph2);
  //     ofs << ss.rdbuf();
  //   } else {
  //     std::ifstream ofs("test.bin", std::ios::binary);
  //     std::stringstream ss;
  //     ss << ofs.rdbuf();
  //     ofs.close();
  //     CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE> temp;
  //     cereal::BinaryInputArchive iarchive(ss);
  //     iarchive(temp);
  //     for (auto i = 0; i < temp.aj.size(); i++) {
  //       if (temp.aj[i] != _taskInvAdjGraph2.aj[i])
  //         std::cout << "fucked\n";
  //     }
  //     for (auto i = 0; i < temp.ai.size(); i++) {
  //       if (temp.ai[i] != _taskInvAdjGraph2.ai[i])
  //         std::cout << "fucked\n";
  //     }
  //     std::cout << _taskInvAdjGraph.NNZ() << " " <<
  //     _taskInvAdjGraph2.NNZ()
  //               << std::endl;
  //     std::cout << "finished check\n";
  //   }
  // }
}

} // namespace matrix_utils