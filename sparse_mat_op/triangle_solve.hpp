#pragma once

#include "BitVector.hpp"
#include "utils.h"
#include <chrono>
#include <execution>
#include <fstream>
#include <memory>
#include <numeric>
#include <omp.h>
#include <span>
#include <tuple>
#include <type_traits>

#include "matrix_utils.hpp"

namespace matrix_utils {

/// @brief Forward-substitution algorithm for low triangular csr matrix L. Note
/// that the diagonal term is assumed to be 1
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
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void ForwardSubstitution(const COLTYPE size, const int base, ROWTYPE const *ai,
                         COLTYPE const *aj, VALTYPE const *av,
                         VALTYPE const *const b, VALTYPE *const x) {
  for (COLTYPE i = 0; i < size; i++) {
    VALTYPE val = 0;
    for (ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; j++) {
      val += av[j] * x[aj[j] - base];
    }
    x[i] = b[i] - val;
  }
}

/// @brief Backword-substitution algorithm for low triangular csr matrix L. Note
/// that the diagonal term is assumed to be 1
/// @brief
/// @tparam ROWTYPE
/// @tparam COLTYPE
/// @tparam VALTYPE
/// @param size
/// @param base
/// @param ai   csr of the strict upper triangular matrix
/// @param aj
/// @param av
/// @param diag diagonal vector
/// @param b
/// @param x
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void BackwardSubstitution(const COLTYPE size, const int base, ROWTYPE const *ai,
                          COLTYPE const *aj, VALTYPE const *av,
                          VALTYPE const *diag, VALTYPE const *const b,
                          VALTYPE *const x) {
  for (COLTYPE i = size - 1; i >= 0; i--) {
    VALTYPE val = 0;
    for (ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; j++) {
      val += av[j] * x[aj[j] - base];
    }
    x[i] = (b[i] - val) / diag[i];
  }
}

/// @brief Forward-substitution algorithm for low triangular csr matrix L
/// obtained by the transpose of a strict upper triangular csr matrix. Note that
/// the diagonal term is assumed to be 1
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
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void ForwardSubstitutionT(const COLTYPE size, const int base, ROWTYPE const *ai,
                          COLTYPE const *aj, VALTYPE const *av,
                          VALTYPE const *const b, VALTYPE *const x) {
  ROWTYPE j;
  std::copy(b, b + size, x);
  for (COLTYPE i = 0; i < size; i++) {
    for (j = ai[i] - base; j < ai[i + 1] - base; j++) {
      x[aj[j] - base] -= av[j] * x[i];
    }
  }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void LevelScheduleForwardSubstitution(COLTYPE const *iperm,
                                      COLTYPE const *prefix, const COLTYPE lvls,
                                      const COLTYPE rows, const int base,
                                      ROWTYPE const *ai, COLTYPE const *aj,
                                      VALTYPE const *av, VALTYPE const *const b,
                                      VALTYPE *const x) {
#pragma omp parallel
  {
    for (int l = 0; l < lvls; l++) {
#pragma omp for
      for (COLTYPE i = prefix[l]; i < prefix[l + 1]; i++) {
        const COLTYPE idx = iperm[i] - base;
        VALTYPE val = 0;
        for (auto j = ai[idx] - base; j < ai[idx + 1] - base; j++) {
          val += av[j] * x[aj[j] - base];
        }
        x[idx] = b[idx] - val;
      }
#pragma omp barrier
    }
  }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void LevelScheduleBackwardSubstitution(
    COLTYPE const *iperm, COLTYPE const *prefix, const COLTYPE lvls,
    const COLTYPE rows, const int base, ROWTYPE const *ai, COLTYPE const *aj,
    VALTYPE const *av, VALTYPE const *diag, VALTYPE const *const b,
    VALTYPE *const x) {
#pragma omp parallel
  {
    for (int l = 0; l < lvls; l++) {
#pragma omp for
      for (COLTYPE i = prefix[l]; i < prefix[l + 1]; i++) {
        const COLTYPE idx = iperm[i] - base;
        VALTYPE val = 0;
        for (auto j = ai[idx] - base; j < ai[idx + 1] - base; j++) {
          val += av[j] * x[aj[j] - base];
        }
        x[idx] = (b[idx] - val) / diag[idx];
      }
#pragma omp barrier
    }
  }
}

enum class FBSubstitutionType { Barrier, NoBarrier, NoBarrierSuperNode };

template <FBSubstitutionType FBST = FBSubstitutionType::Barrier,
          TriangularMatrix TS = TriangularMatrix::L, typename ROWTYPE = int,
          typename COLTYPE = int, typename VALTYPE = double>
class OptimizedTriangularSolve {
public:
  OptimizedTriangularSolve(const int num_threads = omp_get_num_threads())
      : _nthreads{num_threads} {}

  void analysis(const COLTYPE rows, const int base, ROWTYPE const *ai,
                COLTYPE const *aj, VALTYPE const *av,
                VALTYPE const *diag = nullptr);

  void operator()(const VALTYPE *const b, VALTYPE *const x) const;

  void BarrierOp(const VALTYPE *const b, VALTYPE *const x) const;

  void NoBarrierOp(const VALTYPE *const b, VALTYPE *const x) const;

  void NoBarrierSuperNodeOp(const VALTYPE *const b, VALTYPE *const x) const;

  void build_task_graph();

  int get_num_threads() const { return _nthreads; }

protected:
  int _nthreads;
  COLTYPE _size;
  std::vector<COLTYPE> _iperm;
  std::vector<COLTYPE> _levelPrefix;
  mutable std::vector<double> _vec;

  COLTYPE _levels;
  std::vector<std::vector<COLTYPE>>
      _threadlevels; // level prefix for each thread, zero based
  std::vector<COLTYPE> _threadiperm;
  std::vector<COLTYPE> _threadperm;
  CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE> _reorderedMat;
  VALTYPE const *_diag{nullptr};

  // super node level scheduling data
  // always zero based
  COLTYPE _tasks;
  CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE> _taskAdjGraph;    // children
  CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE> _taskInvAdjGraph; // parents
  CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE>
      _taskInvAdjGraph2; // parents after transisive edge removal
  std::vector<COLTYPE> _threadTaskPrefix; // tasks on each thread
  std::vector<COLTYPE>
      _taskBoundaryPrefix; // num of rows in each task size task + 1
  std::vector<ROWTYPE> _threadPrefixSum; //
  // std::vector<ROWTYPE> _threadPrefixSum2; //
  std::vector<COLTYPE> _reorderedRowIdToTaskId;
  std::vector<std::vector<COLTYPE>> _taskInvAdj; // thread local
  std::vector<COLTYPE> _transitiveEdgeRemoveAj;

  mutable utils::BitVector<COLTYPE> _bv;

  // debugging
  // std::vector<COLTYPE> taskSizes;
};
} // namespace matrix_utils

#include "triangle_solve.tpp"