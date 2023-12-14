#pragma once
#include <memory>
#include <mkl.h>
namespace mkl_wrapper {

// Derived Class for storing Matrices in CSR Form with MKL Matrix Datatype
class mkl_sparse_mat {
public:
  mkl_sparse_mat() = default;
  mkl_sparse_mat(const MKL_INT row, const MKL_INT col, const MKL_INT nnz);
  mkl_sparse_mat(const MKL_INT row, const MKL_INT col,
                 const std::shared_ptr<MKL_INT[]> &ai,
                 const std::shared_ptr<MKL_INT[]> &aj,
                 const std::shared_ptr<double[]> &av);

  ~mkl_sparse_mat();
  void sp_fill();

  sparse_matrix_t &mkl_handler() { return _mkl_mat; }
  MKL_INT rows() const { return _nrow; }
  MKL_INT cols() const { return _ncol; }
  MKL_INT nnz() const { return _nnz; }
  void mult_vec(double const *const b, double *const x);

  std::shared_ptr<MKL_INT[]> get_ai() { return _ai; }
  std::shared_ptr<MKL_INT[]> get_aj() { return _aj; }
  std::shared_ptr<double[]> get_av() { return _av; }

  virtual bool solve(double const *const b, double *const x) { return false; }

protected:
  sparse_matrix_t _mkl_mat;
  sparse_status_t _mkl_stat;
  matrix_descr _mkl_descr;

  MKL_INT _nrow; // Number of Rows
  MKL_INT _ncol; // Number of Columns
  MKL_INT _nnz;  // Number of Non-zeros

  std::shared_ptr<MKL_INT[]> _ai{nullptr}; // Row Pointer
  std::shared_ptr<MKL_INT[]> _aj{nullptr}; // Column Index
  std::shared_ptr<double[]> _av{nullptr};  // Value Array
};

class mkl_ilu0 : public mkl_sparse_mat {
public:
  mkl_ilu0(mkl_sparse_mat *A);
  bool factorize();

  virtual bool solve(double const *const b, double *const x) override;

protected:
  mkl_sparse_mat *_A{nullptr};
  std::unique_ptr<double[]> _interm_vec{nullptr};

  bool _check_zero_diag{false}; // check for zero diagonals
  double _zero_tol{1e-16};      // threshold for zero diagonal check
  double _zero_rep{1e-10};      // replacement value for zero diagonal
};
} // namespace mkl_wrapper