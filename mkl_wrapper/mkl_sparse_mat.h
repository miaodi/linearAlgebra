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

  sparse_matrix_t mkl_handler() { return _mkl_mat; }
  sparse_matrix_t mkl_handler() const { return _mkl_mat; }
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

  MKL_INT bandwidth() const;

  std::shared_ptr<double[]> get_diag() const;
  std::shared_ptr<MKL_INT[]> get_ai() { return _ai; }
  std::shared_ptr<MKL_INT[]> get_aj() { return _aj; }
  std::shared_ptr<double[]> get_av() { return _av; }

  std::shared_ptr<const MKL_INT[]> get_ai() const { return _ai; }
  std::shared_ptr<const MKL_INT[]> get_aj() const { return _aj; }
  std::shared_ptr<const double[]> get_av() const { return _av; }

  // get the graph representation without self-edge
  void get_adjacency_graph(std::vector<MKL_INT> &xadj,
                           std::vector<MKL_INT> &adjncy) const;

  virtual bool solve(double const *const b, double *const x) { return false; }

  void to_one_based();

  void to_zero_based();

  void swap(mkl_sparse_mat &other);

  void print() const;

  int check() const;

  void transpose();

  void clear();

  void print_svg(std::ostream &out) const;

  void print_gnuplot(std::ostream &out) const;

protected:
  MKL_INT _nrow; // Number of Rows
  MKL_INT _ncol; // Number of Columns
  MKL_INT _nnz;  // Number of Non-zeros

  std::shared_ptr<MKL_INT[]> _ai{nullptr}; // Row Pointer
  std::shared_ptr<MKL_INT[]> _aj{nullptr}; // Column Index
  std::shared_ptr<double[]> _av{nullptr};  // Value Array
  bool _pd{false};                         // positive definite

  sparse_matrix_t _mkl_mat{nullptr};
  sparse_index_base_t _mkl_base{SPARSE_INDEX_BASE_ZERO};
  matrix_descr _mkl_descr;

  sparse_status_t _mkl_stat;
};

// incomplete factorization preconditioner
class incomplete_fact : public mkl_sparse_mat {
public:
  incomplete_fact() : mkl_sparse_mat() {}

  virtual bool numeric_factorize(mkl_sparse_mat const *const A) {
    return false;
  }

  virtual bool symbolic_factorize(mkl_sparse_mat const *const A) {
    return true;
  }

protected:
  std::vector<double> _interm_vec;
};

std::shared_ptr<MKL_INT[]> permutedAI(const mkl_sparse_mat &A,
                                      MKL_INT const *const pinv);

std::tuple<std::shared_ptr<MKL_INT[]>, std::shared_ptr<MKL_INT[]>,
           std::shared_ptr<double[]>>
permuteRow(const mkl_sparse_mat &A, MKL_INT const *const pinv);

std::tuple<std::shared_ptr<MKL_INT[]>, std::shared_ptr<MKL_INT[]>,
           std::shared_ptr<double[]>>
permute(const mkl_sparse_mat &A, MKL_INT const *const pinv,
        MKL_INT const *const q);

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
mkl_sparse_mat_sym mkl_sparse_mult_papt(mkl_sparse_mat_sym &A,
                                        mkl_sparse_mat &P);

std::tuple<std::shared_ptr<MKL_INT[]>, std::shared_ptr<MKL_INT[]>,
           std::shared_ptr<double[]>>
symPermute(const mkl_sparse_mat &A, MKL_INT const *const pinv);

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

  bool orthogonalize();

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

// return a square random sparse matrix
mkl_sparse_mat random_sparse(const MKL_INT row, const MKL_INT nnzRow);
} // namespace mkl_wrapper