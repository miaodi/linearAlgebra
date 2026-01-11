#pragma once

#include "circularbuffer.hpp"
#include "matrix_utils.hpp"
#include "sparse_mat_traits.hpp"
#include <deque>
#include <forward_list>
#include <limits>
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
template <ResizableCSR CSRMatrixType>
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

template <ResizableCSR CSRMatrixType>
void ICCLevelSymbolic1(const typename CSRMatrixType::COLTYPE size, 
                       typename CSRMatrixType::ROWTYPE const* ai, 
                       typename CSRMatrixType::COLTYPE const* aj, 
                       typename CSRMatrixType::COLTYPE const* diag_pos,
                       const int lvl, 
                       CSRMatrixType& icc);

template <ResizableCSR CSRMatrixType>
void ICCLevelSymbolic2(const typename CSRMatrixType::COLTYPE size, 
                       typename CSRMatrixType::ROWTYPE const* ai, 
                       typename CSRMatrixType::COLTYPE const* aj, 
                       typename CSRMatrixType::COLTYPE const* diag_pos,
                       const int lvl, 
                       CSRMatrixType& icc);

template <ResizableCSR CSRMatrixType>
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

template <ResizableDiagonal CSRMatrixType> class ICCLevelSymbolicParallel {
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

template <ResizableDiagonal CSRMatrixType> class ICCLevelNumericFixedPoint {
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

template <ResizableDiagonal CSRMatrixType>
bool ILULevel0Symbolic(const typename CSRMatrixType::COLTYPE size,
                       typename CSRMatrixType::ROWTYPE const *ai,
                       typename CSRMatrixType::COLTYPE const *aj,
                       CSRMatrixType &ilu);

template <ResizableDiagonal CSRMatrixType> class ILULevelSymbolic {
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

// GS-Urow ILU(k) symbolic factorization with parallel U-row construction
// New sequential and scalable parallel algorithms for incomplete LU factor preconditioning
// Hysom 2001
template <ResizableDiagonal CSRMatrixType, bool keepdiag = false>
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

// GS-Lrow ILU(k) symbolic factorization with parallel L-row construction
// Similar to ILULevelSymbolicParallelU but constructs L rows instead of U rows
template <ResizableDiagonal CSRMatrixType, bool keepdiag = false>
class ILULevelSymbolicParallelL
{
public:
    using COLTYPE = typename CSRMatrixType::COLTYPE;
    using ROWTYPE = typename CSRMatrixType::ROWTYPE;
    
    struct NodeInfo {
        COLTYPE index;
        COLTYPE peak;
    };
    
    ILULevelSymbolicParallelL(const int nthreads)
        : _nthreads(nthreads), _visited(nthreads), _Q(nthreads), _Q_next(nthreads), _Li(nthreads)
    {
    }

    bool operator()(const typename CSRMatrixType::COLTYPE size, typename CSRMatrixType::ROWTYPE const* ai,
                    typename CSRMatrixType::COLTYPE const* aj, const int lvl, CSRMatrixType& L);

private:
    int _nthreads;
    std::vector<std::vector<NodeInfo>> _visited;
    std::vector<std::vector<NodeInfo>> _Q;
    std::vector<std::vector<NodeInfo>> _Q_next;
    std::vector<std::vector<COLTYPE>> _Li;
};

template <ResizableDiagonal CSRMatrixType>
bool ILULevelNumeric( const typename CSRMatrixType::COLTYPE size,
                      typename CSRMatrixType::ROWTYPE const* ai,
                      typename CSRMatrixType::COLTYPE const* aj,
                      typename CSRMatrixType::VALTYPE const* av,
                      const int lvl,
                      CSRMatrixType& ilu );

template <ResizableDiagonal CSRMatrixType>
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

// Jacobi (diagonal) preconditioner: x = D^{-1} b
// Requires diagonal entries present in the matrix; zeros are treated as 1.0 to avoid division by zero.
template <ResizableDiagonal CSRMatrixType> class JacobiPrec {
public:
  using ROWTYPE = typename CSRMatrixType::ROWTYPE;
  using COLTYPE = typename CSRMatrixType::COLTYPE;
  using VALTYPE = typename CSRMatrixType::VALTYPE;

  JacobiPrec(const CSRMatrixType &A, int nthreads = omp_get_max_threads())
      : _n(A.rows), _invD(A.rows), _nthreads(nthreads) {
    // Build inverse diagonal using utility function Diagonal
    // Ask Diagonal to compute the inverted diagonal directly (invert=true)
    const bool ok = matrix_utils::Diagonal(A.rows, A.AI(), A.AJ(), A.AV(), static_cast<ROWTYPE*>(nullptr), _invD.data(), true);
  }
  COLTYPE size() const { return _n; }

  bool operator()(VALTYPE const *const b, VALTYPE *const x) const {
    // Apply inverse diagonal
#pragma omp parallel for num_threads(_nthreads)
    for (COLTYPE i = 0; i < _n; ++i) {
      x[i] = _invD[i] * b[i];
    }
    return true;
  }

private:
  COLTYPE _n;
  std::vector<VALTYPE> _invD;
  int _nthreads;
};
} // namespace matrix_utils
