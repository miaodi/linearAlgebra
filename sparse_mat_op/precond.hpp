#pragma once

#include "circularbuffer.hpp"
#include "matrix_utils.hpp"
#include "sparse_mat_traits.hpp"
#include <deque>
#include <forward_list>
#include <map>
#include <omp.h>
#include <ranges>
#include <unordered_map>
#include <vector>

// for all preconditioner operators, assuming the diagonal entries are filled.
// In other words, a zero value should be provided for the diagonal entries if
// it is a void entry in A.
namespace matrix_utils {

// ICC helper declarations (definitions in precond.cpp)
template <ResizableCSRMatrixType CSRMatrixType>
void ICCLevel0SymSymbolic(const typename CSRMatrixType::COLTYPE size, 
                          typename CSRMatrixType::ROWTYPE const* ai, 
                          typename CSRMatrixType::COLTYPE const* aj, 
                          CSRMatrixType& icc);

template <typename CSRMatrixType>
void ICCLevelSymbolic0(const typename CSRMatrixType::COLTYPE size, 
                       typename CSRMatrixType::ROWTYPE const* ai, 
                       typename CSRMatrixType::COLTYPE const* aj, 
                       typename CSRMatrixType::COLTYPE const* diag_pos,
                       const int lvl, 
                       CSRMatrixType& icc);

template <ResizableCSRMatrixType CSRMatrixType>
void ICCLevelSymbolic1(const typename CSRMatrixType::COLTYPE size, 
                       typename CSRMatrixType::ROWTYPE const* ai, 
                       typename CSRMatrixType::COLTYPE const* aj, 
                       typename CSRMatrixType::COLTYPE const* diag_pos,
                       const int lvl, 
                       CSRMatrixType& icc);

template <ResizableCSRMatrixType CSRMatrixType>
void ICCLevelSymbolic2(const typename CSRMatrixType::COLTYPE size, 
                       typename CSRMatrixType::ROWTYPE const* ai, 
                       typename CSRMatrixType::COLTYPE const* aj, 
                       typename CSRMatrixType::COLTYPE const* diag_pos,
                       const int lvl, 
                       CSRMatrixType& icc);

template <ResizableCSRMatrixType CSRMatrixType>
void ICCLevelSymbolic3(const typename CSRMatrixType::COLTYPE size, 
                       typename CSRMatrixType::ROWTYPE const* ai, 
                       typename CSRMatrixType::COLTYPE const* aj, 
                       typename CSRMatrixType::COLTYPE const* diag_pos,
                       const int lvl, 
                       CSRMatrixType& icc);

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
bool ICCLevelNumeric(const COLTYPE size, 
                     ROWTYPE const* ai, 
                     COLTYPE const* aj, 
                     VALTYPE const* av,
                     COLTYPE const* diag_pos, 
                     const int lvl, 
                     const VALTYPE omega, 
                     ROWTYPE const* icc_ai,
                     COLTYPE const* icc_aj, 
                     VALTYPE* icc_av);

template <ResizableDiagonalType CSRMatrixType> class ICCLevelSymbolicParallel {
public:
  using COLTYPE = typename CSRMatrixType::COLTYPE;
  using ROWTYPE = typename CSRMatrixType::ROWTYPE;

  ICCLevelSymbolicParallel(const int num_threads)
      : _num_threads(num_threads), _Li_path_max(num_threads),
        _visited(num_threads), _Li(num_threads), _Q(num_threads),
        _Q_next(num_threads) {}

  bool operator()(const COLTYPE size, ROWTYPE const *ai, COLTYPE const *aj,
                  const int lvl, CSRMatrixType &L);

private:
  int _num_threads;
  std::vector<std::vector<COLTYPE>> _Li_path_max; //
  std::vector<std::vector<COLTYPE>> _visited;
  std::vector<std::vector<COLTYPE>> _Li;
  std::vector<std::unordered_map<COLTYPE, COLTYPE>> _Q;
  std::vector<std::unordered_map<COLTYPE, COLTYPE>> _Q_next;
};

template <ResizableDiagonalType CSRMatrixType> class ICCLevelNumericFixedPoint {
public:
  using COLTYPE = typename CSRMatrixType::COLTYPE;
  using ROWTYPE = typename CSRMatrixType::ROWTYPE;
  using VALTYPE = typename CSRMatrixType::VALTYPE;

  ICCLevelNumericFixedPoint(const int num_threads)
      : _num_threads(num_threads) {}

  bool operator()(const COLTYPE size, ROWTYPE const *ai, COLTYPE const *aj,
                  VALTYPE const *av, CSRMatrixType &L);

private:
  int _num_threads;         // number of threads to use in parallel region
  int _sweeps{100};      // number of sweeps to perform
  std::vector<VALTYPE> _av; // av in L's sparsity pattern
  std::vector<COLTYPE> _ai; // ai in COO format for L's sparsity pattern
  std::vector<VALTYPE> _L_av_init; // initial guess for L's av
  std::vector<VALTYPE> _L_av_next; // next iteration's L's av after a sweep
};

template <ResizableDiagonalType CSRMatrixType>
bool ILULevel0Symbolic(const typename CSRMatrixType::COLTYPE size,
                       typename CSRMatrixType::ROWTYPE const *ai,
                       typename CSRMatrixType::COLTYPE const *aj,
                       CSRMatrixType &ilu);

template <ResizableDiagonalType CSRMatrixType> class ILULevelSymbolic {
public:
  ILULevelSymbolic() = default;
  bool operator()(const typename CSRMatrixType::COLTYPE size,
                  typename CSRMatrixType::ROWTYPE const *ai,
                  typename CSRMatrixType::COLTYPE const *aj, const int lvl,
                  CSRMatrixType &ilu);

private:
  // Local (col, level) pair used during symbolic pattern construction of a row
  struct ColLevel {
    typename CSRMatrixType::COLTYPE col;
    int level;
  };
  std::vector<int> _levels; // level for each element
  // marker array for O(1) membership / position lookup in current row (MAX sentinel if absent)
  std::vector<typename CSRMatrixType::ROWTYPE> _marker;
  // Reusable storage to avoid per-row allocations
  std::vector<ColLevel> _cl;
  std::deque<typename CSRMatrixType::COLTYPE> _q; // queue of pivot candidates < i
};

template <ResizableDiagonalType CSRMatrixType>
class ILULevelSymbolicParallelU
{
public:
    using COLTYPE = typename CSRMatrixType::COLTYPE;
    using ROWTYPE = typename CSRMatrixType::ROWTYPE;
    ILULevelSymbolicParallelU(const int nthreads)
        : _nthreads(nthreads), _visited(nthreads), _Q(nthreads), _Q_next(nthreads), _Ui(nthreads)
    {
    }

    bool operator()(const typename CSRMatrixType::COLTYPE size, typename CSRMatrixType::ROWTYPE const* ai,
                    typename CSRMatrixType::COLTYPE const* aj, const int lvl, CSRMatrixType& U);

private:
    int _nthreads;
    std::vector<std::vector<COLTYPE>> _visited;
    std::vector<std::vector<COLTYPE>> _Q;
    std::vector<std::vector<COLTYPE>> _Q_next;
    std::vector<std::vector<COLTYPE>> _Ui;
};

template <ResizableDiagonalType CSRMatrixType>
bool ILULevelNumeric( const typename CSRMatrixType::COLTYPE size,
                      typename CSRMatrixType::ROWTYPE const* ai,
                      typename CSRMatrixType::COLTYPE const* aj,
                      typename CSRMatrixType::VALTYPE const* av,
                      const int lvl,
                      CSRMatrixType& ilu );

template <ResizableDiagonalType CSRMatrixType>
bool ILUTNumeric( const typename CSRMatrixType::COLTYPE size,
                  typename CSRMatrixType::ROWTYPE const* ai,
                  typename CSRMatrixType::COLTYPE const* aj,
                  typename CSRMatrixType::VALTYPE const* av,
                  const typename CSRMatrixType::VALTYPE tau,
                  CSRMatrixType& ilu );
  
                      

template <typename VT> class IdentityPrec {
public:
  using VALTYPE = VT;
  IdentityPrec(const std::size_t size) : _size(size) {}

  std::size_t size() const { return _size; }

  bool operator()(VALTYPE const *const b, VALTYPE *const x) const {
    for (std::size_t i = 0; i < _size; i++) {
      x[i] = b[i];
    }
    return true;
  }
  std::size_t _size;
};
} // namespace matrix_utils
