#pragma once
#include <mkl.h>
namespace mkl_wrapper {

// Derived Class for storing Matrices in CSR Form with MKL Matrix Datatype
class mkl_sparse_mat {
public:
  mkl_sparse_mat(const MKL_INT row, const MKL_INT col, const MKL_INT nnz);
  mkl_sparse_mat(const MKL_INT row, const MKL_INT col, MKL_INT *ai, MKL_INT *aj,
                 double *av);
  ~mkl_sparse_mat();
  void sp_fill();

  sparse_matrix_t &mkl_handler() { return _mkl_mat; }
  MKL_INT rows() const { return _nrow; }
  MKL_INT cols() const { return _ncol; }
  void mult_vec(double const *const b, double *const x);

protected:
  sparse_matrix_t _mkl_mat;
  sparse_status_t _mkl_stat;
  matrix_descr _mkl_descr;

  MKL_INT _nrow; // Number of Rows
  MKL_INT _ncol; // Number of Columns
  MKL_INT _nnz;  // Number of Non-zeros

  MKL_INT *_ai{nullptr}; // Row Pointer
  MKL_INT *_aj{nullptr}; // Column Index
  double *_av{nullptr};  // Value Array

  bool _owned{false};
};
} // namespace mkl_wrapper