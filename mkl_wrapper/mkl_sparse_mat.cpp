#include "mkl_sparse_mat.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <mkl_rci.h>
#include <vector>

namespace mkl_wrapper {

mkl_sparse_mat::mkl_sparse_mat(const MKL_INT row, const MKL_INT col,
                               const MKL_INT nnz) {
  _nrow = row;
  _ncol = col;
  _nnz = nnz;

  _ai.reset(new MKL_INT[_nrow + 1]);
  _aj.reset(new MKL_INT[_nnz]);
  _av.reset(new double[_nnz]);

  for (MKL_INT i = 0; i < _nnz; i++) {
    _aj[i] = 0;
    _av[i] = 0.0;
  }

  for (MKL_INT i = 0; i < _nrow + 1; i++) {
    _ai[i] = 0;
  }
  sp_fill();
}

mkl_sparse_mat::mkl_sparse_mat(const mkl_sparse_mat &other) {

  _nrow = other._nrow;
  _ncol = other._ncol;
  _nnz = other._nnz;

  _ai.reset(new MKL_INT[_nrow + 1]);
  _aj.reset(new MKL_INT[_nnz]);
  _av.reset(new double[_nnz]);

  for (MKL_INT i = 0; i < _nnz; i++) {
    _aj[i] = other._aj[i];
    _av[i] = other._av[i];
  }

  for (MKL_INT i = 0; i < _nrow + 1; i++) {
    _ai[i] = other._ai[i];
  }

  sp_fill();
}

mkl_sparse_mat &mkl_sparse_mat::operator=(const mkl_sparse_mat &other) {
  mkl_sparse_mat tmp(other);
  this->swap(tmp);
  return *this;
}

mkl_sparse_mat::mkl_sparse_mat(mkl_sparse_mat &&src) {
  // just swap the array pointers...
  src.swap(*this);
}

// move assignment operator
mkl_sparse_mat &mkl_sparse_mat::operator=(mkl_sparse_mat &&rhs) {
  mkl_sparse_mat temp(std::move(rhs)); // moves the array
  temp.swap(*this);
  return *this;
}

void mkl_sparse_mat::swap(mkl_sparse_mat &other) {

  std::swap(_mkl_mat, other._mkl_mat);
  std::swap(_mkl_stat, other._mkl_stat);
  std::swap(_mkl_base, other._mkl_base);
  std::swap(_mkl_descr, other._mkl_descr);
  std::swap(_mkl_descr, other._mkl_descr);
  std::swap(_pd, other._pd);
  std::swap(_nrow, other._nrow);
  std::swap(_ncol, other._ncol);
  std::swap(_nnz, other._nnz);
  std::swap(_ai, other._ai);
  std::swap(_aj, other._aj);
  std::swap(_av, other._av);
}

mkl_sparse_mat::mkl_sparse_mat(sparse_matrix_t mkl_mat) {
  if (mkl_mat == nullptr) {
    std::cerr << "sparse_matrix_t is an empty pointer, failed to create "
                 "mkl_sparse_mat."
              << std::endl;
    return;
  }
  MKL_INT *rows_start;
  MKL_INT *rows_end;
  MKL_INT *col_index;
  double *values;

  _mkl_stat = mkl_sparse_order(mkl_mat); // ordering in CSR format
  if (_mkl_stat != SPARSE_STATUS_SUCCESS) {
    std::cout << "mkl reorder CSR failed, code: " << _mkl_stat << "\n";
  }
  _mkl_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
  _mkl_descr.diag = SPARSE_DIAG_NON_UNIT;
  _mkl_descr.mode = SPARSE_FILL_MODE_FULL;
  _mkl_stat =
      mkl_sparse_d_export_csr(mkl_mat, &_mkl_base, &_nrow, &_ncol, &rows_start,
                              &rows_end, &col_index, &values);

  if (_mkl_stat != SPARSE_STATUS_SUCCESS) {
    std::cout << "MKL EXPORT CSR FAILED, CODE: " << _mkl_stat << "\n";
  }

  _nnz = rows_start[_nrow];

  _ai.reset(new MKL_INT[_nrow + 1]);
  std::copy(rows_start, rows_start + _nrow + 1, _ai.get());
  _aj.reset(new MKL_INT[_nnz]);
  std::copy(col_index, col_index + _nnz, _aj.get());
  _av.reset(new double[_nnz]);
  std::copy(values, values + _nnz, _av.get());

  sp_fill();
}

mkl_sparse_mat::mkl_sparse_mat(const MKL_INT row, const MKL_INT col,
                               const std::shared_ptr<MKL_INT[]> &ai,
                               const std::shared_ptr<MKL_INT[]> &aj,
                               const std::shared_ptr<double[]> &av,
                               const sparse_index_base_t base) {
  _nrow = row;
  _ncol = col;

  _mkl_base = base;
  _nnz = _mkl_base == SPARSE_INDEX_BASE_ZERO ? ai[_nrow] : ai[_nrow] - 1;

  _ai = ai;
  _aj = aj;
  _av = av;

  sp_fill();
}

void mkl_sparse_mat::to_one_based() {
  if (_mkl_base == SPARSE_INDEX_BASE_ZERO) {
    mkl_sparse_destroy(_mkl_mat);
    {
#pragma omp parallel
#pragma omp for nowait
      for (MKL_INT i = 0; i < _nnz; i++) {
        _aj[i] += 1;
      }
#pragma omp parallel
#pragma omp for nowait
      for (MKL_INT i = 0; i < _nrow + 1; i++) {
        _ai[i] += 1;
      }
    }
    _mkl_base = SPARSE_INDEX_BASE_ONE;
    sp_fill();
  }
}

void mkl_sparse_mat::to_zero_based() {
  if (_mkl_base == SPARSE_INDEX_BASE_ONE) {
    mkl_sparse_destroy(_mkl_mat);
    {
#pragma omp parallel
#pragma omp for nowait
      for (MKL_INT i = 0; i < _nnz; i++) {
        _aj[i] -= 1;
      }
#pragma omp parallel
#pragma omp for nowait
      for (MKL_INT i = 0; i < _nrow + 1; i++) {
        _ai[i] -= 1;
      }
    }
    _mkl_base = SPARSE_INDEX_BASE_ZERO;
    sp_fill();
  }
}

void mkl_sparse_mat::sp_fill() {
  _mkl_stat =
      mkl_sparse_d_create_csr(&_mkl_mat, _mkl_base, _nrow, _ncol, _ai.get(),
                              _ai.get() + 1, _aj.get(), _av.get());
  if (_mkl_stat != SPARSE_STATUS_SUCCESS) {
    std::cerr << "Matrix is not created" << std::endl;
  }

  _mkl_stat = mkl_sparse_order(_mkl_mat); // ordering in CSR format
  _mkl_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
  _mkl_descr.diag = SPARSE_DIAG_NON_UNIT;
  _mkl_descr.mode = SPARSE_FILL_MODE_FULL;
}

mkl_sparse_mat::~mkl_sparse_mat() {
  if (_mkl_mat)
    mkl_sparse_destroy(_mkl_mat);
}

bool mkl_sparse_mat::mult_vec(double const *const b, double *const x) {
  _mkl_stat = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, _mkl_mat,
                              _mkl_descr, b, 0.0, x);
  return _mkl_stat == SPARSE_STATUS_SUCCESS;
}

void mkl_sparse_mat::print() const {
  std::cout << "ai: ";
  for (MKL_INT i = 0; i <= _nrow; i++) {
    std::cout << _ai[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "aj: ";
  for (MKL_INT i = 0; i < _nnz; i++) {
    std::cout << _aj[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "av: ";
  for (MKL_INT i = 0; i < _nnz; i++) {
    std::cout << _av[i] << " ";
  }
  std::cout << std::endl;
  std::cout << std::endl;
}

MKL_INT mkl_sparse_mat::max_nz() const {
  MKL_INT res = 0;
  for (MKL_INT i = 0; i < _nrow; i++) {
    res = std::max(res, _ai[i + 1] - _ai[i]);
  }
  return res;
}

mkl_sparse_mat mkl_sparse_sum(mkl_sparse_mat &A, mkl_sparse_mat &B, double c) {
  A.to_zero_based();
  B.to_zero_based();
  sparse_matrix_t result;
  auto status = mkl_sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE,
                                 A.mkl_handler(), c, B.mkl_handler(), &result);
  if (status != SPARSE_STATUS_SUCCESS) {

    std::cerr << "mkl sparse sum failed, code: " << status << std::endl;
    return mkl_sparse_mat();
  }
  auto res = mkl_sparse_mat(result);
  mkl_sparse_destroy(result);
  return res;
}

// opA(A)*B
mkl_sparse_mat mkl_sparse_mult(mkl_sparse_mat &A, mkl_sparse_mat &B,
                               const sparse_operation_t opA) {
  A.to_zero_based();
  B.to_zero_based();
  sparse_matrix_t result;
  auto status = mkl_sparse_spmm(opA, A.mkl_handler(), B.mkl_handler(), &result);
  if (status != SPARSE_STATUS_SUCCESS) {

    std::cerr << "mkl sparse multiply failed, code: " << status << std::endl;
    return mkl_sparse_mat();
  }
  auto res = mkl_sparse_mat(result);
  mkl_sparse_destroy(result);
  return res;
}

mkl_ilu0::mkl_ilu0(mkl_sparse_mat *A) : mkl_sparse_mat(), _A(A) {

  _nrow = _A->rows();
  _ncol = _A->cols();
  _nnz = _A->nnz();
  _interm_vec.reset(new double[_nrow]);
  _ai = _A->get_ai();
  _aj = _A->get_aj();
}

bool mkl_ilu0::factorize() {
  if (_A == nullptr || _A->rows() != _A->cols())
    return false;

  MKL_INT ipar[128] = {0};
  double dpar[128] = {0.0};

  // parameters affecting the pre-conditioner
  if (_check_zero_diag) {
    ipar[30] = 1;
    dpar[30] = _zero_tol;
    dpar[31] = _zero_rep;
  }

  _av.reset(new double[_nnz]);

  _A->to_one_based();

  MKL_INT ierr = 0;
  dcsrilu0(&_nrow, _A->get_av().get(), _ai.get(), _aj.get(), _av.get(), ipar,
           dpar, &ierr);

  _A->to_zero_based();

  if (ierr != 0)
    return false;

  sp_fill();
  return true;
}

bool mkl_ilu0::solve(double const *const b, double *const x) {

  sparse_operation_t transA = SPARSE_OPERATION_NON_TRANSPOSE;
  _mkl_descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  _mkl_descr.mode = SPARSE_FILL_MODE_LOWER;
  _mkl_descr.diag = SPARSE_DIAG_UNIT;
  mkl_sparse_d_trsv(transA, 1.0, _mkl_mat, _mkl_descr, b, _interm_vec.get());

  _mkl_descr.mode = SPARSE_FILL_MODE_UPPER;
  _mkl_descr.diag = SPARSE_DIAG_NON_UNIT;
  mkl_sparse_d_trsv(transA, 1.0, _mkl_mat, _mkl_descr, _interm_vec.get(), x);
  return true;
}

mkl_ilut::mkl_ilut(mkl_sparse_mat *A) : mkl_sparse_mat(), _A(A) {

  _nrow = _A->rows();
  _ncol = _A->cols();
  _interm_vec.reset(new double[_nrow]);
}

bool mkl_ilut::factorize() {
  if (_A == nullptr || _A->rows() != _A->cols())
    return false;

  MKL_INT ipar[128] = {0};
  double dpar[128] = {0.0};

  // parameters affecting the pre-conditioner
  if (_check_zero_diag) {
    ipar[30] = 1;
    dpar[30] = _zero_tol;
    // dpar[31] = _zero_rep;
  } else {
    // do this to avoid a warning from the preconditioner
    dpar[30] = _tau;
  }

  _ai.reset(new MKL_INT[_nrow + 1]);
  _nnz = (2 * _max_fill + 1) * _nrow;
  ;
  _aj.reset(new MKL_INT[_nnz]);
  _av.reset(new double[_nnz]);

  MKL_INT ierr = 0;
  _A->to_one_based();
  dcsrilut(&_nrow, _A->get_av().get(), _A->get_ai().get(), _A->get_aj().get(),
           _av.get(), _ai.get(), _aj.get(), &_tau, &_max_fill, ipar, dpar,
           &ierr);
  _mkl_base = SPARSE_INDEX_BASE_ONE;
  // _A->to_zero_based();
  // _A->print();
  if (ierr != 0)
    return false;

  sp_fill();
  // to_zero_based();
  return true;
}

bool mkl_ilut::solve(double const *const b, double *const x) {

  sparse_operation_t transA = SPARSE_OPERATION_NON_TRANSPOSE;
  _mkl_descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  _mkl_descr.mode = SPARSE_FILL_MODE_LOWER;
  _mkl_descr.diag = SPARSE_DIAG_UNIT;
  mkl_sparse_d_trsv(transA, 1.0, _mkl_mat, _mkl_descr, b, _interm_vec.get());

  _mkl_descr.mode = SPARSE_FILL_MODE_UPPER;
  _mkl_descr.diag = SPARSE_DIAG_NON_UNIT;
  mkl_sparse_d_trsv(transA, 1.0, _mkl_mat, _mkl_descr, _interm_vec.get(), x);
  return true;
}

mkl_sparse_mat_sym::mkl_sparse_mat_sym(mkl_sparse_mat *A) : mkl_sparse_mat() {

  // only take upper triangular

  _nrow = A->rows();
  _ncol = A->cols();

  _ai.reset(new MKL_INT[_nrow + 1]);
  auto ai = A->get_ai();
  auto aj = A->get_aj();
  auto av = A->get_av();
  _nnz = 0;
  _ai[0] = 0;
  for (MKL_INT i = 0; i < _nrow; i++) {
    MKL_INT begin = ai[i];
    MKL_INT end = ai[i + 1];
    auto res = std::find(aj.get() + begin, aj.get() + end, i);
    if (res == aj.get() + end) {
      std::cout << "Could not find diagonal!" << std::endl;
    } else {
      _nnz += aj.get() + end - res;
    }
    _ai[i + 1] = _nnz;
  }
  // for (MKL_INT i = 0; i < _nrow + 1; i++) {
  //   std::cout << _ai[i] << std::endl;
  // }

  _aj.reset(new MKL_INT[_nnz]);
  _av.reset(new double[_nnz]);

  MKL_INT ind = 0;
  for (MKL_INT i = 0; i < _nrow; i++) {
    MKL_INT begin = ai[i];
    MKL_INT end = ai[i + 1];
    auto res = std::find(aj.get() + begin, aj.get() + end, i);
    begin = res - aj.get();
    for (auto i = begin; i != end; i++) {
      _aj[ind] = aj[i];
      _av[ind++] = av[i];
    }
  }
  sp_fill();
}

void mkl_sparse_mat_sym::sp_fill() {
  _mkl_stat =
      mkl_sparse_d_create_csr(&_mkl_mat, _mkl_base, _nrow, _ncol, _ai.get(),
                              _ai.get() + 1, _aj.get(), _av.get());
  if (_mkl_stat != SPARSE_STATUS_SUCCESS) {
    std::cout << "Matrix is not created" << std::endl;
  }

  _mkl_stat = mkl_sparse_order(_mkl_mat); // ordering in CSR format
  _mkl_descr.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
  _mkl_descr.diag = SPARSE_DIAG_NON_UNIT;
  _mkl_descr.mode = SPARSE_FILL_MODE_UPPER;
}

mkl_sparse_mat_diag::mkl_sparse_mat_diag(const MKL_INT size, const double val)
    : mkl_sparse_mat() {

  _nrow = size;
  _ncol = size;
  _nnz = size;

  _ai.reset(new MKL_INT[_nrow + 1]);
  _aj.reset(new MKL_INT[_nnz]);
  _av.reset(new double[_nnz]);

  for (MKL_INT i = 0; i < _nnz; i++) {
    _aj[i] = i;
    _av[i] = val;
  }
  _ai[0] = 0;
  for (MKL_INT i = 1; i < _nrow + 1; i++) {
    _ai[i] = _ai[i - 1] + 1;
  }
  sp_fill();
}

mkl_ic0::mkl_ic0(mkl_sparse_mat *A) : mkl_sparse_mat_sym(A) {
  _interm_vec.reset(new double[_nrow]);
}

bool mkl_ic0::factorize() {
  MKL_INT *row = _aj.get();
  MKL_INT *col = _ai.get();
  double *val = _av.get();

  std::vector<double> tmp(_nrow, 0.0);

  // fill in the values
  for (MKL_INT k = 0; k < _nrow; ++k) {
    // get the values for column k
    double *ak = val + (col[k]);
    MKL_INT *rowk = row + (col[k]);
    MKL_INT Lk = col[k + 1] - col[k];

    // sanity check
    if (rowk[0] != k) {
      fprintf(stderr,
              "Fatal error in incomplete Cholesky preconditioner:\nMatrix "
              "format error at row %d.",
              k);
      return false;
    }

    // make sure the diagonal element is not zero
    if (ak[0] == 0.0) {
      fprintf(stderr,
              "Fatal error in incomplete Cholesky preconditioner:\nZero "
              "diagonal element at row %d.",
              k);
      return false;
    }

    // make sure the diagonal element is not negative either
    if (ak[0] < 0.0) {
      fprintf(stderr,
              "Fatal error in incomplete Cholesky preconditioner:\nNegative "
              "diagonal element at row %d (value = %lg).",
              k, ak[0]);
      return false;
    }

    // set the diagonal element
    double akk = std::sqrt(ak[0]);
    ak[0] = akk;
    tmp[rowk[0]] = akk;

    // divide column by akk
    for (MKL_INT j = 1; j < Lk; ++j) {
      ak[j] /= akk;
      tmp[rowk[j]] = ak[j];
    }

    // loop over all other columns
    for (MKL_INT _j = 1; _j < Lk; ++_j) {
      MKL_INT j = rowk[_j];
      double tjk = tmp[j];
      if (tjk != 0.0) {
        double *aj = val + col[j];
        MKL_INT Lj = col[j + 1] - col[j];
        MKL_INT *rowj = row + col[j];

        for (MKL_INT i = 0; i < Lj; i++)
          aj[i] -= tmp[rowj[i]] * tjk;
      }
    }

    // reset temp buffer
    for (MKL_INT j = 0; j < Lk; ++j)
      tmp[rowk[j]] = 0.0;
  }

  return true;
}

bool mkl_ic0::solve(double const *const b, double *const x) {
  sparse_operation_t transA = SPARSE_OPERATION_TRANSPOSE;
  _mkl_descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  _mkl_descr.mode = SPARSE_FILL_MODE_UPPER;
  _mkl_descr.diag = SPARSE_DIAG_NON_UNIT;
  mkl_sparse_d_trsv(transA, 1.0, _mkl_mat, _mkl_descr, b, _interm_vec.get());

  transA = SPARSE_OPERATION_NON_TRANSPOSE;
  mkl_sparse_d_trsv(transA, 1.0, _mkl_mat, _mkl_descr, _interm_vec.get(), x);
  return true;
}
} // namespace mkl_wrapper