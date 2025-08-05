#pragma once
#include "sparse_mat_traits.hpp"
#include <vector>

namespace iterative_solver {
template <typename OpType> class GMRES {
public:
  enum class ErrorType : int {
    NO_ERROR = 0,
    MAX_ITER_REACHED = 1,
    INSUFFICIENT_MEMORY = 2,
    INVALID_MATRIX = 3
  };
  using ROWTYPE = typename matrix_utils::CSRMatrixRowType<OpType>::type;
  using COLTYPE = typename matrix_utils::CSRMatrixIndexType<OpType>::type;
  using VALTYPE = typename matrix_utils::CSRMatrixValueType<OpType>::type;
  GMRES() {}

  ErrorType operator()(OpType const *op, VALTYPE const *b, VALTYPE *x,
                       int max_iter = 100, VALTYPE tol = 1e-6,
                       int restart = 10) {
    _size = op->rows();
    _max_iter = max_iter;
    _tol = tol;
    _restart = restart;

    _H.resize(_restart * _restart);
    _Q.resize(_size * _restart);

    int iter = 0;
    int res = 0;
    ErrorType error = ErrorType::NO_ERROR;
    while (true) {
      if (res < _tol)
        break;
      if (iter >= _max_iter) {
        error = ErrorType::MAX_ITER_REACHED;
        break;
      }
      
    }
  }

private:
  COLTYPE _size;
  int _max_iter{100};
  VALTYPE _tol{1e-6};
  int _restart{10};
  std::vector<VALTYPE> _H;
  std::vector<VALTYPE> _Q;

public:
};
} // namespace iterative_solver