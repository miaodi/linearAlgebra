#pragma once
#include "sparse_mat_traits.hpp"
#include "vec_ops.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace iterative_solver {

template <typename VALTYPE>
void givens_rotation(VALTYPE *const R, VALTYPE *const g, VALTYPE *const c,
                     VALTYPE *const s, const VALTYPE beta, VALTYPE &resid,
                     const int lda, const int j) {
  auto R_col_j = R + j * lda;
  // apply Givens rotation to R_col_j
  for (int i = 0; i < j; i++) {
    auto tmp = c[i] * R_col_j[i] - s[i] * R_col_j[i + 1];
    R_col_j[i + 1] = s[i] * R_col_j[i] + c[i] * R_col_j[i + 1];
    R_col_j[i] = tmp;
  }
  // compute Givens rotation for R_col_j
  auto div_r = VALTYPE(1) / std::hypot(R_col_j[j], beta);
  // auto div_r = VALTYPE(1) / std::sqrt(R_col_j[j] * R_col_j[j] + beta * beta);
  c[j] = div_r * R_col_j[j];
  s[j] = -div_r * beta;
  if (std::abs(s[j]) < 1e-16) {
    c[j] = 1;
    s[j] = 0;
  }

  R_col_j[j] = c[j] * R_col_j[j] - s[j] * beta;
  // apply Givens rotation to g
  g[j] = c[j] * resid;
  resid *= s[j];
}

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
    const auto restart1 = _restart + 1;
    _H.resize(_restart, _restart);
    _H.setZero();
    _Q.resize(size, restart1);
    _Q.setZero();

    _tmp.resize(size);
    _g.resize(_restart);
    _c.resize(_restart);
    _s.resize(_restart);
    Eigen::Map<Eigen::Matrix<VALTYPE, Eigen::Dynamic, 1>> x_vec(x, size);

    VALTYPE resid, init_resid, beta;
    ErrorType error_code = ErrorType::NO_ERROR;

    vec_ops::copy_vec(size, b, _tmp.data());
    // b-= Ax_0
    (*op)(x, _tmp.data(), (VALTYPE)(-1), (VALTYPE)(1));
    std::cout << _tmp << std::endl;

    // resid_0 = M^{-1}(b - Ax_0)
    (*prec)(_tmp.data(), _Q.col(0).data());
    std::cout << _Q << std::endl;
    int j, iter;
    for (iter = 0; iter < _max_iter;) {
      init_resid = resid = _Q.col(0).norm();
      if (init_resid < _abs_tol) {
        return error_code;
      }
      _Q.col(0) = _Q.col(0) / init_resid;

      for (j = 0; j < _restart; j++, iter++) {
        if (iter >= _max_iter) {
          error_code = ErrorType::MAX_ITER_REACHED;
          break;
        }
        // v_j_ptr = A * v_j_ptr
        (*op)(_Q.col(j).data(), _tmp.data(), (VALTYPE)(1), (VALTYPE)(0));
        // Apply preconditioner
        (*prec)(_tmp.data(), _Q.col(j + 1).data());

        for (int i = 0; i <= j; i++) {
          // H[i][j] = v_i^T * v_j
          _H(i, j) = _Q.col(i).dot(_Q.col(j + 1));
          // w -= H[i][j] * v_i
          _Q.col(j + 1) -= _H(i, j) * _Q.col(i);
        }

        // H[j+1][j] = ||v_j_ptr||
        beta = _Q.col(j + 1).norm();
        _Q.col(j + 1) = _Q.col(j + 1) / beta;
        givens_rotation(_H.data(), _g.data(), _c.data(), _s.data(), beta, resid,
                        _restart, j);
        std::cout << "iter: " << iter << " "
                  << "resid: " << std::abs(resid) << " "
                  << "relative resid: " << std::abs(resid) / init_resid << std::endl;
        if (std::abs(resid) < _abs_tol ||
            std::abs(resid) < _rel_tol * init_resid) {
          error_code = ErrorType::NO_ERROR;
          break;
        }
      }
      _H.block(0, 0, j, j)
          .template triangularView<Eigen::Upper>()
          .solveInPlace(_g.head(j));
      x_vec += _Q.leftCols(j) * _g.head(j);

      vec_ops::copy_vec(size, b, _tmp.data());
      // b-= Ax_i
      (*op)(x, _tmp.data(), (VALTYPE)(-1), (VALTYPE)(1));
      // resid_i = b - Ax_i
      (*prec)(_tmp.data(), _Q.col(0).data());
    }

    return error_code;
  }

private:
  int _max_iter{1000};
  VALTYPE _abs_tol{0.0};
  VALTYPE _rel_tol{1e-8};
  int _restart{100};
  Eigen::Matrix<VALTYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> _H;
  Eigen::Matrix<VALTYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> _Q;
  Eigen::Matrix<VALTYPE, Eigen::Dynamic, 1> _g;
  Eigen::Matrix<VALTYPE, Eigen::Dynamic, 1> _c;
  Eigen::Matrix<VALTYPE, Eigen::Dynamic, 1> _s;
  Eigen::Matrix<VALTYPE, Eigen::Dynamic, 1> _tmp;
};
} // namespace iterative_solver