#include "mkl_sparse_mat.h"
#include <iostream>
namespace mkl_wrapper {

mkl_sparse_mat::mkl_sparse_mat(const MKL_INT row, const MKL_INT col,
                               const MKL_INT nnz) {
  _nrow = row;
  _ncol = col;
  _nnz = nnz;

  _ai = new MKL_INT[_nrow + 1];
  _aj = new MKL_INT[_nnz];
  _av = new double[_nnz];

  for (MKL_INT i = 0; i < _nnz; i++) {
    _aj[i] = 0;
    _av[i] = 0.0;
  }

  for (MKL_INT i = 0; i < _nrow + 1; i++) {
    _ai[i] = 0;
  }
  _owned = true;
  sp_fill();
}

mkl_sparse_mat::mkl_sparse_mat(const MKL_INT row, const MKL_INT col,
                               MKL_INT *ai, MKL_INT *aj, double *av) {
  _nrow = row;
  _ncol = col;

  _nnz = ai[_nrow];

  _ai = ai;
  _aj = aj;
  _av = av;

  _owned = false;
  sp_fill();
}

void mkl_sparse_mat::sp_fill() {

  _mkl_stat = mkl_sparse_d_create_csr(&_mkl_mat, SPARSE_INDEX_BASE_ZERO, _nrow,
                                      _ncol, _ai, _ai + 1, _aj, _av);
  if (_mkl_stat != SPARSE_STATUS_SUCCESS) {
    std::cout << "Matrix is not created" << std::endl;
  }

  _mkl_stat = mkl_sparse_order(_mkl_mat); // ordering in CSR format
  _mkl_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
  _mkl_descr.diag = SPARSE_DIAG_NON_UNIT;
}

mkl_sparse_mat::~mkl_sparse_mat() {
  if (_owned)
    mkl_sparse_destroy(_mkl_mat);
}

void mkl_sparse_mat::mult_vec(double const *const b, double *const x) {
  _mkl_stat = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, _mkl_mat,
                              _mkl_descr, b, 0.0, x);
}
} // namespace mkl_wrapper