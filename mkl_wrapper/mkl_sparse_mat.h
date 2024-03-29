#pragma once
#include <memory>
#include <mkl_cblas.h>
#include <mkl_spblas.h>
#include <vector>
namespace mkl_wrapper {

// Derived Class for storing Matrices in CSR Form with MKL Matrix Datatype
class mkl_sparse_mat {

public:
  mkl_sparse_mat() = default;

  // copy constructor
  mkl_sparse_mat(const mkl_sparse_mat &);
  // assignment operator
  mkl_sparse_mat &operator=(const mkl_sparse_mat &);

  // move constructor
  mkl_sparse_mat(mkl_sparse_mat &&src);
  // move assignment operator
  mkl_sparse_mat &operator=(mkl_sparse_mat &&rhs);

  mkl_sparse_mat(const MKL_INT row, const MKL_INT col, const MKL_INT nnz);

  mkl_sparse_mat(const MKL_INT row, const MKL_INT col,
                 const std::shared_ptr<MKL_INT[]> &ai,
                 const std::shared_ptr<MKL_INT[]> &aj,
                 const std::shared_ptr<double[]> &av,
                 const sparse_index_base_t base = SPARSE_INDEX_BASE_ZERO);

  // copy vector csr data to internal data member
  mkl_sparse_mat(const MKL_INT row, const MKL_INT col,
                 const std::vector<MKL_INT> &ai, const std::vector<MKL_INT> &aj,
                 const std::vector<double> &av,
                 const sparse_index_base_t base = SPARSE_INDEX_BASE_ZERO);

  // make a deep copy of mkl_mat so that mkl_sparse_mat will always keep the
  // ownership of csr data
  mkl_sparse_mat(sparse_matrix_t mkl_mat);

  ~mkl_sparse_mat();
  virtual void sp_fill();

  void optimize();

  void prune(const double tol = 1e-16);

  sparse_matrix_t &mkl_handler() { return _mkl_mat; }
  const sparse_matrix_t &mkl_handler() const { return _mkl_mat; }
  matrix_descr &mkl_descr() { return _mkl_descr; }
  const matrix_descr &mkl_descr() const { return _mkl_descr; }
  sparse_index_base_t mkl_base() const { return _mkl_base; }
  bool positive_definite() const { return _pd; }
  void set_positive_definite(bool pd) { _pd = pd; }
  MKL_INT rows() const { return _nrow; }
  MKL_INT cols() const { return _ncol; }
  MKL_INT nnz() const { return _nnz; }
  bool mult_vec(double const *const b, double *const x);
  bool transpose_mult_vec(double const *const b, double *const x);

  MKL_INT avg_nz() const { return _nnz / _nrow; }
  MKL_INT max_nz() const;

  std::shared_ptr<double[]> get_diag() const;
  std::shared_ptr<MKL_INT[]> get_ai() { return _ai; }
  std::shared_ptr<MKL_INT[]> get_aj() { return _aj; }
  std::shared_ptr<double[]> get_av() { return _av; }

  std::shared_ptr<const MKL_INT[]> get_ai() const { return _ai; }
  std::shared_ptr<const MKL_INT[]> get_aj() const { return _aj; }
  std::shared_ptr<const double[]> get_av() const { return _av; }

  virtual bool solve(double const *const b, double *const x) { return false; }

  void to_one_based();

  void to_zero_based();

  void swap(mkl_sparse_mat &other);

  void print() const;

  void check() const;

  void transpose();

  void clear();

protected:
  sparse_matrix_t _mkl_mat{nullptr};
  sparse_status_t _mkl_stat;
  sparse_index_base_t _mkl_base{SPARSE_INDEX_BASE_ZERO};
  matrix_descr _mkl_descr;

  bool _pd{false}; // positive definite

  MKL_INT _nrow; // Number of Rows
  MKL_INT _ncol; // Number of Columns
  MKL_INT _nnz;  // Number of Non-zeros

  std::shared_ptr<MKL_INT[]> _ai{nullptr}; // Row Pointer
  std::shared_ptr<MKL_INT[]> _aj{nullptr}; // Column Index
  std::shared_ptr<double[]> _av{nullptr};  // Value Array
};

// c*A+B
mkl_sparse_mat mkl_sparse_sum(const mkl_sparse_mat &A, const mkl_sparse_mat &B,
                              double c = 1.);

bool mkl_sparse_dense_prod(const mkl_sparse_mat &A, double const *B, double *C);

// opA(A)*B
mkl_sparse_mat
mkl_sparse_mult(const mkl_sparse_mat &A, const mkl_sparse_mat &B,
                const sparse_operation_t opA = SPARSE_OPERATION_NON_TRANSPOSE,
                const sparse_operation_t opB = SPARSE_OPERATION_NON_TRANSPOSE);

// PT*A*P
mkl_sparse_mat mkl_sparse_mult_ptap(mkl_sparse_mat &A, mkl_sparse_mat &P);

// P*A*PT
mkl_sparse_mat mkl_sparse_mult_papt(mkl_sparse_mat &A, mkl_sparse_mat &P);

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

class mkl_ilut : public mkl_sparse_mat {
public:
  mkl_ilut(mkl_sparse_mat *A);
  bool factorize();

  virtual bool solve(double const *const b, double *const x) override;
  void set_tau(const double tau) { _tau = tau; }
  void set_max_fill(const MKL_INT fill) { _max_fill = fill; }

protected:
  mkl_sparse_mat *_A{nullptr};
  std::unique_ptr<double[]> _interm_vec{nullptr};

  bool _check_zero_diag{false}; // check for zero diagonals
  double _zero_tol{1e-16};      // threshold for zero diagonal check
  // double _zero_rep{1e-10};      // replacement value for zero diagonal

  double _tau{1e-6};
  MKL_INT _max_fill{2};
};

// upper triangular
class mkl_sparse_mat_sym : public mkl_sparse_mat {
public:
  mkl_sparse_mat_sym() : mkl_sparse_mat() {}
  explicit mkl_sparse_mat_sym(const mkl_sparse_mat &A);

  mkl_sparse_mat_sym(const mkl_sparse_mat_sym &A);
  // user guarantee the data is upper triangular
  mkl_sparse_mat_sym(const MKL_INT row, const MKL_INT col,
                     const std::shared_ptr<MKL_INT[]> &ai,
                     const std::shared_ptr<MKL_INT[]> &aj,
                     const std::shared_ptr<double[]> &av,
                     const sparse_index_base_t base = SPARSE_INDEX_BASE_ZERO);
  // make a deep copy of mkl_mat so that mkl_sparse_mat will always keep the
  // ownership of csr data
  mkl_sparse_mat_sym(sparse_matrix_t mkl_mat);

  virtual void sp_fill();
};

class mkl_sparse_mat_diag : public mkl_sparse_mat {
public:
  mkl_sparse_mat_diag(const MKL_INT size, const double val);
};

// c*A+B
mkl_sparse_mat_sym mkl_sparse_sum(const mkl_sparse_mat_sym &A,
                                  const mkl_sparse_mat_sym &B, double c = 1.);

// PT*A*P
mkl_sparse_mat_sym mkl_sparse_mult_ptap(mkl_sparse_mat_sym &A,
                                        mkl_sparse_mat &P);

// P*A*PT
mkl_sparse_mat mkl_sparse_mult_papt(mkl_sparse_mat_sym &A, mkl_sparse_mat &P);

// Incomplete Cholesky ic0
class mkl_ic0 : public mkl_sparse_mat_sym {
public:
  mkl_ic0(const mkl_sparse_mat &A);

  bool factorize();

  virtual bool solve(double const *const b, double *const x) override;

protected:
  std::unique_ptr<double[]> _interm_vec{nullptr};
};

// col major
class dense_mat {
public:
  dense_mat() = default;
  dense_mat(MKL_INT m, MKL_INT n) : _m(m), _n(n) {
    _av.reset(new double[_m * _n]);
  }

  void resize(MKL_INT m, MKL_INT n) {
    _m = m;
    _n = n;
    _av.reset(new double[_m * _n]);
  }

  mkl_sparse_mat to_sparse_trans() const;

  MKL_INT rows() const { return _m; }
  MKL_INT cols() const { return _n; }

  std::shared_ptr<double[]> get_av() { return _av; }
  std::shared_ptr<const double[]> get_av() const { return _av; }
  double *col(int i) { return _av.get() + i * _m; }

protected:
  MKL_INT _m{0};
  MKL_INT _n{0};
  std::shared_ptr<double[]> _av{nullptr}; // Value Array
};

bool mkl_sparse_dense_mat_prod(
    const mkl_sparse_mat &A, const dense_mat &B, dense_mat &C,
    const sparse_operation_t opA = SPARSE_OPERATION_NON_TRANSPOSE);

bool dense_product(const dense_mat &A, const dense_mat &B, dense_mat &C,
                   const CBLAS_TRANSPOSE opA = CblasNoTrans,
                   const CBLAS_TRANSPOSE opB = CblasNoTrans);

// PT*A*P
dense_mat mkl_sparse_mult_ptap(const mkl_sparse_mat &A, const dense_mat &P);

// P*A*PT
dense_mat mkl_sparse_mult_papt(const mkl_sparse_mat &A, const dense_mat &P);
} // namespace mkl_wrapper