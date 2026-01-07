#pragma once
#include "sparse_mat_traits.hpp"
#include "vec_ops.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace iterative_solver
{

/**
 * @brief Enumeration of preconditioner types.
 * left: M^{-1} * A * x = M^{-1} * b
 * right: A * M^{-1} * y = b, x = M^{-1} * y
 * none: A * x = b
 */
enum class PreconditionerType
{
    NONE = 0,
    LEFT = 1,
    RIGHT = 2
};

// Apply Givens rotation to R and g, and compute new Givens rotation
/**
 * @brief Applies a Givens rotation to the Hessenberg matrix column and the
 * residual vector.
 *
 * @tparam VALTYPE Scalar type (e.g., float, double).
 * @param beta Scalar value representing the norm of the new vector.
 * @param lda Leading dimension (number of rows) of the Hessenberg matrix.
 * @param j Current iteration index (column being processed).
 * @param R Pointer to the Hessenberg matrix data (column-major).
 * @param g Pointer to the residual vector.
 * @param c Pointer to the cosine values for Givens rotations.
 * @param s Pointer to the sine values for Givens rotations.
 * @param resid Reference to the current residual value (will be updated).
 */
template <typename VALTYPE>
void givens_rotation(const VALTYPE beta, const size_t lda, const size_t j, VALTYPE* const R,
                     VALTYPE* const g, VALTYPE* const c, VALTYPE* const s, VALTYPE& resid)
{
    auto R_col_j = R + j * lda;
    // apply Givens rotation to R_col_j
    for (size_t i = 0; i < j; i++)
    {
        auto tmp = c[i] * R_col_j[i] - s[i] * R_col_j[i + 1];
        R_col_j[i + 1] = s[i] * R_col_j[i] + c[i] * R_col_j[i + 1];
        R_col_j[i] = tmp;
    }
    // compute Givens rotation for R_col_j
    auto div_r = VALTYPE(1) / std::hypot(R_col_j[j], beta);
    // auto div_r = VALTYPE(1) / std::sqrt(R_col_j[j] * R_col_j[j] + beta * beta);
    c[j] = div_r * R_col_j[j];
    s[j] = -div_r * beta;
    if (std::abs(s[j]) < 1e-16)
    {
        c[j] = 1;
        s[j] = 0;
    }

    R_col_j[j] = c[j] * R_col_j[j] - s[j] * beta;
    // apply Givens rotation to g
    g[j] = c[j] * resid;
    resid *= s[j];
}

enum class State : int
{
    CONVERGED = 0,
    RUNNING = 1,
    MAX_ITER_REACHED = 2,
    FAILED = 3
};

template <typename VALTYPE>
class GMRES
{
public:
    GMRES() {}

    void setMaxIter(size_t max_iter) { _max_iter = max_iter; }
    void setAbsTol(VALTYPE abs_tol) { _abs_tol = abs_tol; }
    void setRelTol(VALTYPE rel_tol) { _rel_tol = rel_tol; }
    void setRestart(size_t restart) { _restart = restart; }
    void setPreconditionerType(PreconditionerType prec_type) { _prec_type = prec_type; }

    template <matrix_utils::SpmvOp Op, matrix_utils::PrecOp PrecOp>
    State operator()(Op const* const op, PrecOp const* const prec, VALTYPE const* b, VALTYPE* x)
    {
        static_assert(std::is_same_v<typename Op::VALTYPE, VALTYPE>,
                      "Op::VALTYPE must be the same as VALTYPE");
        static_assert(std::is_same_v<typename PrecOp::VALTYPE, VALTYPE>,
                      "PrecOp::VALTYPE must be the same as VALTYPE");

        const size_t size = op->size();
        initialize_workspace(size);
        Eigen::Map<Eigen::Matrix<VALTYPE, Eigen::Dynamic, 1>> x_vec(x, size);
        VALTYPE init_resid = compute_residual(op, prec, b, x, size);
        if (init_resid < _abs_tol)
        {
            return State::CONVERGED;
        }

        VALTYPE resid = init_resid;
        for (size_t iter = 0; iter < _max_iter;)
        {
            size_t cycle_iterations;
            State restart_state =
                perform_restart_cycle(op, prec, init_resid, x_vec, iter, resid, cycle_iterations);

            // std::cout << _H << std::endl;
            if (restart_state == State::CONVERGED)
            {
                return State::CONVERGED;
            }

            if (restart_state == State::MAX_ITER_REACHED)
            {
                return State::MAX_ITER_REACHED;
            }

            // std::cout<<"after update: " << std::endl;
            // std::cout << x_vec << std::endl;

            if (restart_state != State::RUNNING)
            {
                break;
            }

            resid = compute_residual(op, prec, b, x, size);
        }

        return State::MAX_ITER_REACHED;
    }

private:
    void initialize_workspace(size_t size)
    {
        _restart = std::min(static_cast<size_t>(size), _restart);
        _H.resize(_restart, _restart);
        _H.setZero();
        _Q.resize(size, _restart + 1);
        _Q.setZero();
        _tmp.resize(size);
        _g.resize(_restart);
        _c.resize(_restart);
        _s.resize(_restart);
    }

    template <matrix_utils::SpmvOp Op, matrix_utils::PrecOp PrecOp>
    VALTYPE compute_residual(Op const* op, PrecOp const* prec, VALTYPE const* b, VALTYPE const* x, size_t size)
    {
        vec_ops::copy_vec(size, b, _tmp.data());
        // Compute residual r = b - Ax
        (*op)(x, _tmp.data(), (VALTYPE)(-1), (VALTYPE)(1));
        if (_prec_type == PreconditionerType::LEFT)
        {
            (*prec)(_tmp.data(), _Q.col(0).data());
        }
        else
        {
            vec_ops::copy_vec(size, _tmp.data(), _Q.col(0).data());
        }
        return _Q.col(0).norm();
    }

    template <matrix_utils::SpmvOp Op, matrix_utils::PrecOp PrecOp>
    void apply_operator_with_preconditioning(Op const* op, PrecOp const* prec, VALTYPE const* input, VALTYPE* output)
    {
        switch (_prec_type)
        {
        case PreconditionerType::RIGHT:
            (*prec)(input, _tmp.data());
            (*op)(_tmp.data(), output, (VALTYPE)(1), (VALTYPE)(0));
            break;
        case PreconditionerType::LEFT:
            (*op)(input, _tmp.data(), (VALTYPE)(1), (VALTYPE)(0));
            (*prec)(_tmp.data(), output);
            break;
        case PreconditionerType::NONE:
            (*op)(input, output, (VALTYPE)(1), (VALTYPE)(0));
            break;
        }
    }

    template <matrix_utils::SpmvOp Op, matrix_utils::PrecOp PrecOp>
    State perform_restart_cycle(Op const* op, PrecOp const* prec, VALTYPE init_resid,
                                Eigen::Map<Eigen::Matrix<VALTYPE, Eigen::Dynamic, 1>>& x_vec,
                                size_t& iter, VALTYPE& resid, size_t& cycle_iterations)
    {
        _Q.col(0) /= resid;

        size_t j;
        for (j = 0; j < _restart && iter < _max_iter; ++j, ++iter)
        {
            apply_operator_with_preconditioning(op, prec, _Q.col(j).data(), _Q.col(j + 1).data());

            // Modified Gram-Schmidt orthogonalization
            for (size_t i = 0; i <= j; ++i)
            {
                _H(i, j) = _Q.col(i).dot(_Q.col(j + 1));
                _Q.col(j + 1) -= _H(i, j) * _Q.col(i);
            }

            VALTYPE beta = _Q.col(j + 1).norm();

            // // Check for happy breakdown: if beta is very small, the Krylov subspace
            // // is complete and we have found the exact solution within the subspace
            // if ( beta < _breakdown_tol )
            // {
            //     // Happy breakdown occurred - solve with current subspace and return
            //     cycle_iterations = j + 1;
            //     solve_least_squares( j + 1 );
            //     update_solution( prec, cycle_iterations, x_vec );
            //     return State::CONVERGED;
            // }

            _Q.col(j + 1) /= beta;

            givens_rotation(beta, _restart, j, _H.data(), _g.data(), _c.data(), _s.data(), resid);

            print_iteration_info(iter, resid, init_resid);

            if (check_convergence(resid, init_resid))
            {
                cycle_iterations = j + 1;
                solve_least_squares(j + 1);
                update_solution(prec, cycle_iterations, x_vec);
                return State::CONVERGED;
            }
        }

        cycle_iterations = j;
        solve_least_squares(j);
        update_solution(prec, cycle_iterations, x_vec);
        return (iter >= _max_iter) ? State::MAX_ITER_REACHED : State::RUNNING;
    }

    void solve_least_squares(size_t j)
    {
        if (j > 0)
        {
            _H.block(0, 0, j, j).template triangularView<Eigen::Upper>().solveInPlace(_g.head(j));
        }
    }

    template <matrix_utils::PrecOp PrecOp>
    void update_solution(PrecOp const* prec, size_t j,
                         Eigen::Map<Eigen::Matrix<VALTYPE, Eigen::Dynamic, 1>>& x_vec)
    {
        if (j == 0)
            return;

        if (_prec_type == PreconditionerType::RIGHT)
        {
            Eigen::Matrix<VALTYPE, Eigen::Dynamic, 1> y = _Q.leftCols(j) * _g.head(j);
            (*prec)(y.data(), _tmp.data());
            x_vec += Eigen::Map<Eigen::Matrix<VALTYPE, Eigen::Dynamic, 1>>(_tmp.data(), x_vec.size());
        }
        else
        {
            x_vec += _Q.leftCols(j) * _g.head(j);
        }
    }

    void print_iteration_info(size_t iter, VALTYPE resid, VALTYPE init_resid) const
    {
        std::cout << "iter: " << std::setw(6) << iter << " "
                  << "resid: " << std::scientific << std::setprecision(14) << std::abs(resid) << " "
                  << "relative resid: " << std::scientific << std::setprecision(14)
                  << std::abs(resid) / init_resid << std::endl;
    }

    bool check_convergence(VALTYPE resid, VALTYPE init_resid) const
    {
        return std::abs(resid) < _abs_tol || std::abs(resid) < _rel_tol * init_resid;
    }

private:
    size_t _max_iter{100};
    VALTYPE _abs_tol{0.0};
    VALTYPE _rel_tol{1e-8};
    // VALTYPE _breakdown_tol{ 1e-14 };  // Tolerance for detecting happy breakdown
    size_t _restart{20};
    PreconditionerType _prec_type{PreconditionerType::LEFT};
    Eigen::Matrix<VALTYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> _H;
    Eigen::Matrix<VALTYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> _Q;
    Eigen::Matrix<VALTYPE, Eigen::Dynamic, 1> _g;
    Eigen::Matrix<VALTYPE, Eigen::Dynamic, 1> _c;
    Eigen::Matrix<VALTYPE, Eigen::Dynamic, 1> _s;
    Eigen::Matrix<VALTYPE, Eigen::Dynamic, 1> _tmp;
};

/**
 * @brief BiConjugate Gradient Stabilized (BiCGSTAB) solver.
 *
 * BiCGSTAB is an iterative method for solving non-symmetric linear systems Ax = b.
 * It combines ideas from BiCG and GMRES to provide smooth convergence without restarts.
 *
 * Algorithm 7.7 from "Iterative Methods for Sparse Linear Systems" by Yousef Saad
 *
 * @tparam VALTYPE Scalar type (e.g., float, double).
 */
template <typename VALTYPE>
class BICGSTAB
{
public:
    BICGSTAB() {}

    void setMaxIter(size_t max_iter) { _max_iter = max_iter; }
    void setAbsTol(VALTYPE abs_tol) { _abs_tol = abs_tol; }
    void setRelTol(VALTYPE rel_tol) { _rel_tol = rel_tol; }
    void setPreconditionerType(PreconditionerType prec_type) { _prec_type = prec_type; }

    template <matrix_utils::SpmvOp Op, matrix_utils::PrecOp PrecOp>
    State operator()(Op const* const op, PrecOp const* const prec, VALTYPE const* b, VALTYPE* x)
    {
        static_assert(std::is_same_v<typename Op::VALTYPE, VALTYPE>,
                      "Op::VALTYPE must be the same as VALTYPE");
        static_assert(std::is_same_v<typename PrecOp::VALTYPE, VALTYPE>,
                      "PrecOp::VALTYPE must be the same as VALTYPE");

        const size_t size = op->size();
        initialize_workspace(size);

        // Compute initial residual r0 = b - Ax0 (with preconditioning if needed)
        VALTYPE init_resid = compute_residual(op, prec, b, x, size);

        if (init_resid < _abs_tol)
        {
            return State::CONVERGED;
        }

        // Choose arbitrary r_tilde (commonly r_tilde = r0)
        vec_ops::copy_vec(size, _r.data(), _r_tilde.data());
        // Initialize p0 = r0
        vec_ops::copy_vec(size, _r.data(), _p.data());

        std::fill(_x_hat.begin(), _x_hat.end(), static_cast<VALTYPE>(0));

        VALTYPE rho = vec_ops::dot_product(size, _r_tilde.data(), _r.data());
        // if (std::abs(rho) < _breakdown_tol)
        // {
        //     std::cout << "BiCGSTAB breakdown: initial rho = " << rho << std::endl;
        //     return State::FAILED;
        // }
        std::cout << "rho initial: " << rho << std::endl;

        VALTYPE alpha = 1.0;
        VALTYPE omega = 1.0;

        for (size_t iter = 0; iter < _max_iter; ++iter)
        {
            // step 1: Compute alpha
            // step 1.1: Compute A * p
            apply_operator_with_preconditioning(op, prec, _p.data(), _v.data());
            // step 1.2: alpha = rho / (r_tilde, v)
            const VALTYPE rtilde_v = vec_ops::dot_product(size, _r_tilde.data(), _v.data());
            if (std::abs(rtilde_v) < _breakdown_tol)
            {
                std::cout << "BiCGSTAB breakdown: r_tilde_v = " << rtilde_v << std::endl;
                return State::FAILED;
            }
            alpha = rho / rtilde_v;

            // step 2: Compute s = r - alpha * v
            vec_ops::copy_vec(size, _r.data(), _s.data());
            vec_ops::axpy(size, -alpha, _v.data(), _s.data());

            // step 3: Compute omega
            // step 3.1: Compute t = A * s
            apply_operator_with_preconditioning(op, prec, _s.data(), _t.data());
            // step 3.2: omega = (t, s) / (t, t)
            const VALTYPE t_s = vec_ops::dot_product(size, _t.data(), _s.data());
            const VALTYPE t_t = vec_ops::dot_product(size, _t.data(), _t.data());
            // if (std::abs(t_t) < _breakdown_tol)
            // {
            //     std::cout << "BiCGSTAB breakdown: t_t = " << t_t << std::endl;
            //     return State::FAILED;
            // }
            omega = t_s / t_t;

            // step 4: Update solution x_hat = x_hat + alpha * p + omega * s
            vec_ops::axpy(size, alpha, _p.data(), _x_hat.data());
            vec_ops::axpy(size, omega, _s.data(), _x_hat.data());

            // step 5: Update residual r = s - omega * t
            vec_ops::copy_vec(size, _s.data(), _r.data());
            vec_ops::axpy(size, -omega, _t.data(), _r.data());

            // step 6: Check convergence
            const VALTYPE resid = vec_ops::vec_l2_norm(size, _r.data());
            print_iteration_info(iter, resid, init_resid);
            if (check_convergence(resid, init_resid))
            {
                update_solution(prec, x, size);
                return State::CONVERGED;
            }
            // step 7: Compute beta
            const VALTYPE rho_new = vec_ops::dot_product(size, _r_tilde.data(), _r.data());
            if (std::abs(rho_new) < _breakdown_tol)
            {
                std::cout << "BiCGSTAB breakdown: rho_new = " << rho_new
                            << std::endl;
                return State::FAILED;
            }
            const VALTYPE beta = (rho_new / rho) * (alpha / omega);
            rho = rho_new;
            // step 8: Update search direction p = r + beta * (p - omega * v)
            vec_ops::xpay_pbw(size, _r.data(), beta, _p.data(), -omega, _v.data(), _p.data());
        }

        return State::MAX_ITER_REACHED;
    }

private:
    void initialize_workspace(size_t size)
    {
        _r.resize(size);
        _r_tilde.resize(size);
        _p.resize(size);
        _s.resize(size);
        _t.resize(size);
        _v.resize(size);
        _tmp.resize(size);
        _x_hat.resize(size);
    }

    template <matrix_utils::SpmvOp Op, matrix_utils::PrecOp PrecOp>
    VALTYPE compute_residual(Op const* op, PrecOp const* prec, VALTYPE const* b, VALTYPE const* x, size_t size)
    {
        vec_ops::copy_vec(size, b, _r.data());
        // Compute residual r = b - Ax
        (*op)(x, _r.data(), (VALTYPE)(-1), (VALTYPE)(1));
        if (_prec_type == PreconditionerType::LEFT)
        {
            vec_ops::copy_vec(size, _r.data(), _tmp.data());
            (*prec)(_tmp.data(), _r.data());
        }
        return vec_ops::vec_l2_norm(size, _r.data());
    }

    template <matrix_utils::SpmvOp Op, matrix_utils::PrecOp PrecOp>
    void apply_operator_with_preconditioning(Op const* op, PrecOp const* prec, VALTYPE const* input, VALTYPE* output)
    {
        if (_prec_type == PreconditionerType::RIGHT)
        {
            (*prec)(input, _tmp.data());
            (*op)(_tmp.data(), output, (VALTYPE)(1), (VALTYPE)(0));
        }
        else if (_prec_type == PreconditionerType::LEFT)
        {
            (*op)(input, _tmp.data(), (VALTYPE)(1), (VALTYPE)(0));
            (*prec)(_tmp.data(), output);
        }
        else
        {
            (*op)(input, output, (VALTYPE)(1), (VALTYPE)(0));
        }
    }

    template <matrix_utils::PrecOp PrecOp>
    void update_solution(PrecOp const* prec, VALTYPE* x, size_t size)
    {
        if (_prec_type == PreconditionerType::RIGHT)
        {
            (*prec)(_x_hat.data(), _tmp.data());
            vec_ops::axpy(size, 1.0, _tmp.data(), x);
        }
        else
        {
            vec_ops::axpy(size, 1.0, _x_hat.data(), x);
        }
    }

    void print_iteration_info(size_t iter, VALTYPE resid, VALTYPE init_resid) const
    {
        std::cout << "iter: " << std::setw(6) << iter << " "
                  << "resid: " << std::scientific << std::setprecision(14) << std::abs(resid) << " "
                  << "relative resid: " << std::scientific << std::setprecision(14)
                  << std::abs(resid) / init_resid << std::endl;
    }

    bool check_convergence(VALTYPE resid, VALTYPE init_resid) const
    {
        return std::abs(resid) < _abs_tol || std::abs(resid) < _rel_tol * init_resid;
    }

private:
    size_t _max_iter{100};
    VALTYPE _abs_tol{0.0};
    VALTYPE _rel_tol{1e-8};
    VALTYPE _breakdown_tol{1e-14}; // Tolerance for detecting breakdown
    PreconditionerType _prec_type{PreconditionerType::LEFT};

    // Working vectors
    std::vector<VALTYPE> _r;       // Current residual
    std::vector<VALTYPE> _r_tilde; // Shadow residual (fixed)
    std::vector<VALTYPE> _p;       // Search direction
    std::vector<VALTYPE> _s;       // Intermediate residual
    std::vector<VALTYPE> _t;       // A * (M^{-1} * s) or M^{-1} * A * s
    std::vector<VALTYPE> _v;       // A * (M^{-1} * p) or M^{-1} * A * p
    std::vector<VALTYPE> _x_hat;   // x = x + M_r^{-1} * x_hat
    std::vector<VALTYPE> _tmp;     // Temporary vector for preconditioning
};
} // namespace iterative_solver
