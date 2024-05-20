#include "incomplete_cholesky.h"
#include <cmath>
#include <cstdio>
#include <execution>
#include <fstream>
#include <iostream>
#include <mkl_sparse_handle.h>
#include <mkl_spblas.h>

namespace mkl_wrapper {
incomplete_cholesky_k::incomplete_cholesky_k(const mkl_sparse_mat &A,
                                             const int level)
    : incomplete_fact(), _level{level} {
  std::ofstream myfile;
  myfile.open("sym_mat.svg");
  print_svg(myfile);
  myfile.close();
  matrix_descr lt_descr;
  lt_descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  lt_descr.mode = SPARSE_FILL_MODE_UPPER;
  sparse_matrix_t lt_mat;
  _mkl_stat = mkl_sparse_copy(A.mkl_handler(), lt_descr, &lt_mat);
  std::cout << "mkl_sparse_copy: " << _mkl_stat << std::endl;
  for (int l = 0; l < _level; l++) {
    lt_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    lt_descr.mode = SPARSE_FILL_MODE_UPPER;
    sparse_matrix_t result;
    _mkl_stat = mkl_sparse_sp2m(SPARSE_OPERATION_TRANSPOSE, lt_descr, lt_mat,
                                SPARSE_OPERATION_NON_TRANSPOSE, lt_descr,
                                lt_mat, SPARSE_STAGE_NNZ_COUNT, &result);
    std::cout << "mkl_sparse_sp2m: " << _mkl_stat << std::endl;
    _mkl_stat =
        mkl_sparse_sp2m(SPARSE_OPERATION_TRANSPOSE, lt_descr, lt_mat,
                        SPARSE_OPERATION_NON_TRANSPOSE, lt_descr, lt_mat,
                        SPARSE_STAGE_FINALIZE_MULT_NO_VAL, &result);
    std::cout << "mkl_sparse_sp2m: " << _mkl_stat << std::endl;
    // _mkl_stat =
    //     mkl_sparse_sp2m(SPARSE_OPERATION_TRANSPOSE, lt_descr, lt_mat,
    //                     SPARSE_OPERATION_NON_TRANSPOSE, lt_descr, lt_mat,
    //                     SPARSE_STAGE_FULL_MULT, &result);
    // std::cout << "mkl_sparse_sp2m: " << _mkl_stat << std::endl;
    mkl_sparse_destroy(lt_mat);
    lt_mat = result;
  }

  MKL_INT *rows_start{nullptr};
  MKL_INT *rows_end;
  MKL_INT *col_index{nullptr};
  double *values{nullptr};
  _mkl_stat =
      mkl_sparse_d_export_csr(lt_mat, &_mkl_base, &_nrow, &_ncol, &rows_start,
                              &rows_end, &col_index, &values);

  if (_mkl_stat != SPARSE_STATUS_SUCCESS) {
    std::cout << "MKL EXPORT CSR FAILED, CODE: " << _mkl_stat << "\n";
  }
  if (rows_start) {
    _nnz = rows_start[_nrow] - _mkl_base;

    _aj.reset(new MKL_INT[_nnz]);
    _av.reset(new double[_nnz]);
    _ai.reset(new MKL_INT[_nrow + 1]);
    std::copy(std::execution::seq, rows_start, rows_start + _nrow + 1,
              _ai.get());
  }

  if (col_index) {
    std::copy(std::execution::seq, col_index, col_index + _nnz, _aj.get());
    std::cout << std::endl;
  }

  if (values) {
    std::copy(std::execution::seq, values, values + _nnz, _av.get());
  }

  sp_fill();
  _interm_vec.reset(new double[_nrow]);
}

// bool incomplete_cholesky_k::factorize() {
//   MKL_INT *row = _aj.get();
//   MKL_INT *col = _ai.get();
//   double *val = _av.get();

//   std::vector<double> tmp(_nrow, 0.0);

//   // fill in the values
//   for (MKL_INT k = 0; k < _nrow; ++k) {
//     // get the values for column k
//     double *ak = val + (col[k] - _mkl_base);
//     MKL_INT *rowk = row + (col[k] - _mkl_base);
//     MKL_INT Lk = col[k + 1] - col[k];

//     // sanity check
//     if (rowk[0] - _mkl_base != k) {
//       fprintf(stderr,
//               "Fatal error in incomplete Cholesky preconditioner:\nMatrix "
//               "format error at row %d.",
//               k);
//       return false;
//     }

//     // make sure the diagonal element is not zero
//     if (ak[0] == 0.0) {
//       fprintf(stderr,
//               "Fatal error in incomplete Cholesky preconditioner:\nZero "
//               "diagonal element at row %d.",
//               k);
//       return false;
//     }

//     // make sure the diagonal element is not negative either
//     if (ak[0] < 0.0) {
//       fprintf(stderr,
//               "Fatal error in incomplete Cholesky preconditioner:\nNegative "
//               "diagonal element at row %d (value = %lg).",
//               k, ak[0]);
//       return false;
//     }

//     // set the diagonal element
//     double akk = std::sqrt(ak[0]);
//     ak[0] = akk;
//     tmp[rowk[0] - _mkl_base] = akk;

//     // divide column by akk
//     for (MKL_INT j = 1; j < Lk; ++j) {
//       ak[j] /= akk;
//       tmp[rowk[j] - _mkl_base] = ak[j];
//     }

//     // loop over all other columns
//     for (MKL_INT _j = 1; _j < Lk; ++_j) {
//       MKL_INT j = rowk[_j] - _mkl_base;
//       double tjk = tmp[j];
//       if (tjk != 0.0) {
//         double *aj = val + col[j] - _mkl_base;
//         MKL_INT Lj = col[j + 1] - col[j];
//         MKL_INT *rowj = row + col[j] - _mkl_base;

//         for (MKL_INT i = 0; i < Lj; i++)
//           aj[i] -= tmp[rowj[i] - _mkl_base] * tjk;
//       }
//     }

//     // reset temp buffer
//     for (MKL_INT j = 0; j < Lk; ++j)
//       tmp[rowk[j] - _mkl_base] = 0.0;
//   }

//   return true;
// }

// bool incomplete_cholesky_k::solve(double const *const b, double *const x) {
//   sparse_operation_t transA = SPARSE_OPERATION_TRANSPOSE;
//   _mkl_descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
//   _mkl_descr.mode = SPARSE_FILL_MODE_UPPER;
//   _mkl_descr.diag = SPARSE_DIAG_NON_UNIT;
//   mkl_sparse_d_trsv(transA, 1.0, _mkl_mat, _mkl_descr, b, _interm_vec.get());

//   transA = SPARSE_OPERATION_NON_TRANSPOSE;
//   mkl_sparse_d_trsv(transA, 1.0, _mkl_mat, _mkl_descr, _interm_vec.get(), x);
//   return true;
// }
} // namespace mkl_wrapper