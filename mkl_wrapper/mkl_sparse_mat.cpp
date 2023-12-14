#include "mkl_sparse_mat.h"
#include <iostream>
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

mkl_sparse_mat::mkl_sparse_mat(const MKL_INT row, const MKL_INT col,
                               const std::shared_ptr<MKL_INT[]> &ai,
                               const std::shared_ptr<MKL_INT[]> &aj,
                               const std::shared_ptr<double[]> &av) {
  _nrow = row;
  _ncol = col;

  _nnz = ai[_nrow];

  _ai = ai;
  _aj = aj;
  _av = av;

  sp_fill();
}

void mkl_sparse_mat::sp_fill() {

  _mkl_stat =
      mkl_sparse_d_create_csr(&_mkl_mat, SPARSE_INDEX_BASE_ZERO, _nrow, _ncol,
                              _ai.get(), _ai.get() + 1, _aj.get(), _av.get());
  if (_mkl_stat != SPARSE_STATUS_SUCCESS) {
    std::cout << "Matrix is not created" << std::endl;
  }

  _mkl_stat = mkl_sparse_order(_mkl_mat); // ordering in CSR format
  _mkl_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
  _mkl_descr.diag = SPARSE_DIAG_NON_UNIT;
}

mkl_sparse_mat::~mkl_sparse_mat() {}

void mkl_sparse_mat::mult_vec(double const *const b, double *const x) {
  _mkl_stat = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, _mkl_mat,
                              _mkl_descr, b, 0.0, x);
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

  MKL_INT ierr = 0;
  {
    // I hate one-based
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
  // for (int i = 0; i < _nrow; i++) {
  //   for (int j = _ai[i] - 1; j < _ai[i + 1] - 1; j++) {
  //     std::cout << i + 1 << " " << _aj[j] << " " << _A->get_av()[j]
  //               << std::endl;
  //   }
  // }
  dcsrilu0(&_nrow, _A->get_av().get(), _ai.get(), _aj.get(), _av.get(), ipar,
           dpar, &ierr);
  {
    // I like zero-based
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
} // namespace mkl_wrapper