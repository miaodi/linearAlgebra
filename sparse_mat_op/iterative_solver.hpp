#pragma once
#include "sparse_mat_traits.hpp"
#include "vec_ops.hpp"
#include <vector>

namespace iterative_solver {
template <typename VALTYPE> class GMRES {
public:
  enum class ErrorType : int {
    NO_ERROR = 0,
    MAX_ITER_REACHED = 1,
    INSUFFICIENT_MEMORY = 2,
    INVALID_MATRIX = 3
  };
  GMRES() {}

  void setMaxIter(int max_iter) { _max_iter = max_iter; }
  void setAbsTol(VALTYPE abs_tol) { _abs_tol = abs_tol; }
  void setRelTol(VALTYPE rel_tol) { _rel_tol = rel_tol; }
  void setRestart(int restart) { _restart = restart; }

  template <matrix_utils::SpmvOpType Op, matrix_utils::PrecOpType PrecOp>
  ErrorType operator()(Op const *const op, PrecOp const *const prec,
                       VALTYPE const *b, VALTYPE *x) {

    static_assert(std::is_same_v<typename Op::VALTYPE, VALTYPE>,
                  "Op::VALTYPE must be the same as VALTYPE");
    static_assert(std::is_same_v<typename PrecOp::VALTYPE, VALTYPE>,
                  "PrecOp::VALTYPE must be the same as VALTYPE");

    const auto size = op->size();
    _H.resize(_restart * _restart);
    _Q.resize(size * (_restart + 1));
    _tmp.resize(size);
    _g.resize(_restart + 1);

    VALTYPE resid = 1;
    ErrorType error_code = ErrorType::NO_ERROR;

    vec_ops::copy_vec(size, b, _tmp.data());
    VALTYPE *v_j_ptr = _Q.data();
    // b-= Ax_0
    (*op)(x, _tmp.data(), (VALTYPE)(-1), (VALTYPE)(1));
    // resid = ||b - Ax_0||
    (*prec)(_tmp.data(), v_j_ptr);
    
    _g[0] = vec_ops::vec_l2_norm(size, v_j_ptr);
    for (int iter = 0; iter < _max_iter;) {
      v_j_ptr = _Q.data();
      if (_g[0] < _abs_tol) {
        return error_code;
      }
      vec_ops::scale_vec(size, (VALTYPE)(1) / _g[0], v_j_ptr);
      for (int j = 0; j < _restart; j++) {
        iter++;
        if (iter >= _max_iter) {
          error_code = ErrorType::MAX_ITER_REACHED;
          break;
        }

        // v_j_ptr = A * v_j_ptr
        (*op)(v_j_ptr, _tmp.data(), (VALTYPE)(1), (VALTYPE)(0));
        v_j_ptr += size;
        // Apply preconditioner
        (*prec)(_tmp.data(), v_j_ptr);

        VALTYPE *v_i_ptr = _Q.data();
        for (int i = 0; i <= j; i++) {
          // H[i][j] = v_i^T * v_j
          _H[i + j * _restart] = vec_ops::dot_product(size, v_j_ptr, v_i_ptr);
          // w -= H[i][j] * v_i
          vec_ops::axpy(size, -_H[i + j * _restart], v_i_ptr, v_j_ptr);

          v_i_ptr += size;
        }

        // H[j+1][j] = ||v_j_ptr||
        _H[j * _restart + j + 1] = vec_ops::vec_l2_norm(size, v_j_ptr);

        // // Update residual
        // beta = std::sqrt(beta * beta -
        //                  _H[i * _restart + i] * _H[i * _restart + i]);
      }
      v_j_ptr += size;
    }

    // while (true) {
    //   if (rel_err < _rel_tol)
    //     break;
    //   if (iter >= _max_iter) {
    //     error = ErrorType::MAX_ITER_REACHED;
    //     break;
    //   }
    // }
    std::copy(_tmp.begin(), _tmp.end(), x);
    return error_code;
  }

private:
  int _max_iter{100};
  VALTYPE _abs_tol{0.0};
  VALTYPE _rel_tol{1e-6};
  int _restart{10};
  std::vector<VALTYPE> _H;
  std::vector<VALTYPE> _Q;
  std::vector<VALTYPE> _tmp;
  std::vector<VALTYPE> _g;
};
} // namespace iterative_solver