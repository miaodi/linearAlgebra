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
                     const size_t lda, const size_t j) {
  auto R_col_j = R + j * lda;
  // apply Givens rotation to R_col_j
  for (size_t i = 0; i < j; i++) {
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

enum class State : int {
  CONVERGED = 0,
  RUNNING = 1,
  MAX_ITER_REACHED = 2,
  FAILED = 3
};
template <typename VALTYPE> class GMRES {
public:
  GMRES() {}

  void setMaxIter(size_t max_iter) { _max_iter = max_iter; }
  void setAbsTol(VALTYPE abs_tol) { _abs_tol = abs_tol; }
  void setRelTol(VALTYPE rel_tol) { _rel_tol = rel_tol; }
  void setRestart(size_t restart) { _restart = restart; }

  template <matrix_utils::SpmvOpType Op, matrix_utils::PrecOpType PrecOp>
  State operator()(Op const *const op, PrecOp const *const prec,
                   VALTYPE const *b, VALTYPE *x) {

    static_assert(std::is_same_v<typename Op::VALTYPE, VALTYPE>,
                  "Op::VALTYPE must be the same as VALTYPE");
    static_assert(std::is_same_v<typename PrecOp::VALTYPE, VALTYPE>,
                  "PrecOp::VALTYPE must be the same as VALTYPE");

    const auto size = op->size();
    _restart = std::min(static_cast<size_t>(size), _restart);
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
    State state_code = State::RUNNING;

    vec_ops::copy_vec(size, b, _tmp.data());

    // b-= Ax_0
    (*op)(x, _tmp.data(), (VALTYPE)(-1), (VALTYPE)(1));
    // resid_0 = M^{-1}(b - Ax_0)
    (*prec)(_tmp.data(), _Q.col(0).data());

    init_resid = resid = _Q.col(0).norm();
    if (init_resid < _abs_tol) {
      return State::CONVERGED;
    }

    size_t j, iter;
    for (iter = 0; iter < _max_iter;) {
      _Q.col(0) = _Q.col(0) / resid;

      for (j = 0; j < _restart; j++, iter++) {
        if (iter >= _max_iter) {
          state_code = State::MAX_ITER_REACHED;
          break;
        }
        // v_j_ptr = A * v_j_ptr
        (*op)(_Q.col(j).data(), _tmp.data(), (VALTYPE)(1), (VALTYPE)(0));
        // Apply preconditioner
        (*prec)(_tmp.data(), _Q.col(j + 1).data());

        for (size_t i = 0; i <= j; i++) {
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
                  << "relative resid: " << std::abs(resid) / init_resid
                  << std::endl;
        if (std::abs(resid) < _abs_tol ||
            std::abs(resid) < _rel_tol * init_resid) {
          state_code = State::CONVERGED;
          break;
        }
      }
      _H.block(0, 0, j, j)
          .template triangularView<Eigen::Upper>()
          .solveInPlace(_g.head(j));
      x_vec += _Q.leftCols(j) * _g.head(j);
      if (state_code != State::RUNNING) {
        break;
      }
      vec_ops::copy_vec(size, b, _tmp.data());
      // b-= Ax_i
      (*op)(x, _tmp.data(), (VALTYPE)(-1), (VALTYPE)(1));
      // resid_i = b - Ax_i
      (*prec)(_tmp.data(), _Q.col(0).data());
      resid = _Q.col(0).norm();
    }

    return state_code;
  }

private:
  size_t _max_iter{100};
  VALTYPE _abs_tol{0.0};
  VALTYPE _rel_tol{1e-8};
  size_t _restart{20};
  Eigen::Matrix<VALTYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> _H;
  Eigen::Matrix<VALTYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> _Q;
  Eigen::Matrix<VALTYPE, Eigen::Dynamic, 1> _g;
  Eigen::Matrix<VALTYPE, Eigen::Dynamic, 1> _c;
  Eigen::Matrix<VALTYPE, Eigen::Dynamic, 1> _s;
  Eigen::Matrix<VALTYPE, Eigen::Dynamic, 1> _tmp;
};
} // namespace iterative_solver