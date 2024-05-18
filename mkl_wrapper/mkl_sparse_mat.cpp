#include "mkl_sparse_mat.h"
#include "utils.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <execution>
#include <iostream>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>
#include <mkl_rci.h>
#include <mkl_sparse_handle.h>
#include <omp.h>
#include <vector>

#if !defined(MKL_ILP64)
#define IFORMAT "%i"
#else
#define IFORMAT "%lli"
#endif
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

  _mkl_stat =
      mkl_sparse_d_export_csr(mkl_mat, &_mkl_base, &_nrow, &_ncol, &rows_start,
                              &rows_end, &col_index, &values);

  if (_mkl_stat != SPARSE_STATUS_SUCCESS) {
    std::cout << "MKL EXPORT CSR FAILED, CODE: " << _mkl_stat << "\n";
  }

  _nnz = rows_start[_nrow] - _mkl_base;

  _ai.reset(new MKL_INT[_nrow + 1]);
  std::copy(std::execution::seq, rows_start, rows_start + _nrow + 1, _ai.get());
  _aj.reset(new MKL_INT[_nnz]);
  std::copy(std::execution::seq, col_index, col_index + _nnz, _aj.get());
  _av.reset(new double[_nnz]);
  std::copy(std::execution::seq, values, values + _nnz, _av.get());

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

mkl_sparse_mat::mkl_sparse_mat(const MKL_INT row, const MKL_INT col,
                               const std::vector<MKL_INT> &ai,
                               const std::vector<MKL_INT> &aj,
                               const std::vector<double> &av,
                               const sparse_index_base_t base) {
  _nrow = row;
  _ncol = col;

  _mkl_base = base;
  _nnz = _mkl_base == SPARSE_INDEX_BASE_ZERO ? ai[_nrow] : ai[_nrow] - 1;

  _ai.reset(new MKL_INT[_nrow + 1]);
  std::copy(ai.begin(), ai.end(), _ai.get());
  _aj.reset(new MKL_INT[_nnz]);
  std::copy(aj.begin(), aj.end(), _aj.get());
  _av.reset(new double[_nnz]);
  std::copy(av.begin(), av.end(), _av.get());

  sp_fill();
}

std::shared_ptr<double[]> mkl_sparse_mat::get_diag() const {
  auto res = std::shared_ptr<double[]>(new double[rows()]);

#pragma omp parallel for
  for (MKL_INT i = 0; i < rows(); i++) {
    auto begin = _ai[i] - _mkl_base;
    auto end = _ai[i + 1] - _mkl_base;
    auto mid = std::find(_aj.get() + begin, _aj.get() + end, i + _mkl_base);
    if (mid == _aj.get() + end) {
      res[i] = 0.;
    } else {
      res[i] = _av[std::distance(_aj.get(), mid)];
    }
  }
  return res;
}

void mkl_sparse_mat::get_adjacency_graph(std::vector<MKL_INT> &xadj,
                                         std::vector<MKL_INT> &adjncy) const {
  xadj.resize(rows() + 1);
  const MKL_INT base = mkl_base();
  xadj[0] = base;
  // Assume all diagonals are occupied
  adjncy.resize(nnz() - rows());

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    auto [start, end] = utils::LoadBalancedPartition(
        _ai.get(), _ai.get() + rows(), tid, nthreads);

    for (auto it = start; it != end; it++) {
      const MKL_INT index = it - _ai.get();
      xadj[index + 1] = _ai[index + 1] - index - 1;
    }
#pragma omp barrier
    auto [start1, end1] = utils::LoadPrefixBalancedPartition(
        _ai.get(), _ai.get() + rows(), tid, nthreads);

    for (auto it = start; it != end; it++) {
      const MKL_INT rowIdx = it - _ai.get() + base;
      MKL_INT pos = xadj[rowIdx - base] - base;
      for (MKL_INT j = *it - base; j != *(it + 1) - base; j++) {
        if (_aj[j] == rowIdx)
          continue;
        adjncy[pos++] = _aj[j];
      }
    }
  }
} // namespace mkl_wrapper

void mkl_sparse_mat::to_one_based() {
  if (_mkl_base == SPARSE_INDEX_BASE_ZERO) {
    {
#pragma omp parallel for
      for (MKL_INT i = 0; i < _nnz; i++) {
        _aj[i] += 1;
      }
#pragma omp parallel for
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
    {
#pragma omp parallel for
      for (MKL_INT i = 0; i < _nnz; i++) {
        _aj[i] -= 1;
      }
#pragma omp parallel for
      for (MKL_INT i = 0; i < _nrow + 1; i++) {
        _ai[i] -= 1;
      }
    }
    _mkl_base = SPARSE_INDEX_BASE_ZERO;
    sp_fill();
  }
}

void mkl_sparse_mat::sp_fill() {
  if (_mkl_mat) {
    mkl_sparse_destroy(_mkl_mat);
    _mkl_mat = nullptr;
  }

  _mkl_stat =
      mkl_sparse_d_create_csr(&_mkl_mat, _mkl_base, _nrow, _ncol, _ai.get(),
                              _ai.get() + 1, _aj.get(), _av.get());
  if (_mkl_stat != SPARSE_STATUS_SUCCESS) {
    std::cerr << "mkl_sparse_mat::sp_fill(): Matrix is not created, state:"
              << _mkl_stat << std::endl;
  }

  _mkl_stat = mkl_sparse_order(_mkl_mat); // ordering in CSR format
  _mkl_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
  _mkl_descr.diag = SPARSE_DIAG_NON_UNIT;
  _mkl_descr.mode = SPARSE_FILL_MODE_FULL;
  optimize();
}

void mkl_sparse_mat::optimize() {
  mkl_sparse_set_mv_hint(_mkl_mat, SPARSE_OPERATION_NON_TRANSPOSE, _mkl_descr,
                         1000);
  mkl_sparse_set_memory_hint(_mkl_mat, SPARSE_MEMORY_AGGRESSIVE);
  // mkl_sparse_optimize(_mkl_mat);
}

void mkl_sparse_mat::prune(const double tol) {
  auto base = _mkl_base;
  if (base == SPARSE_INDEX_BASE_ONE)
    to_zero_based();
  MKL_INT k = 0;
  for (MKL_INT j = 0; j < _nrow; ++j) {
    MKL_INT previousStart = _ai[j];
    _ai[j] = k;
    MKL_INT end = _ai[j + 1];
    for (MKL_INT i = previousStart; i < end; ++i) {
      if (std::abs(_av[i]) > tol) {
        _aj[k] = _aj[i];
        _av[k] = _av[i];
        ++k;
      }
    }
  }
  _ai[_nrow] = k;
  _nnz = k;
  if (base == SPARSE_INDEX_BASE_ONE)
    to_one_based();
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

bool mkl_sparse_mat::transpose_mult_vec(double const *const b,
                                        double *const x) {
  _mkl_stat = mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, _mkl_mat,
                              _mkl_descr, b, 0.0, x);
  return _mkl_stat == SPARSE_STATUS_SUCCESS;
}

void mkl_sparse_mat::print() const {
  std::cout << "nrow: " << _nrow << " ncol: " << _ncol << " nnz: " << _nnz
            << std::endl;
  std::cout << "sparse_index_base_t: " << _mkl_base << std::endl;
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

int mkl_sparse_mat::check() const {
  sparse_checker_error_values check_err_val;
  sparse_struct pt;
  int error = 0;

  sparse_matrix_checker_init(&pt);
  pt.n = _nrow;
  pt.csr_ia = _ai.get();
  pt.csr_ja = _aj.get();
  if (_mkl_base == SPARSE_INDEX_BASE_ZERO)
    pt.indexing = MKL_ZERO_BASED;
  else
    pt.indexing = MKL_ONE_BASED;
  if (_mkl_descr.type == SPARSE_MATRIX_TYPE_GENERAL) {
    pt.matrix_structure = MKL_GENERAL_STRUCTURE;
  } else if (_mkl_descr.type == SPARSE_MATRIX_TYPE_SYMMETRIC) {
    pt.matrix_structure = MKL_STRUCTURAL_SYMMETRIC;
  } else if (_mkl_descr.type == SPARSE_MATRIX_TYPE_TRIANGULAR) {
    if (_mkl_descr.mode == SPARSE_FILL_MODE_LOWER) {
      pt.matrix_structure = MKL_LOWER_TRIANGULAR;
    } else {
      pt.matrix_structure = MKL_UPPER_TRIANGULAR;
    }
  }
  pt.print_style = MKL_C_STYLE;
  pt.message_level = MKL_PRINT;

  check_err_val = sparse_matrix_checker(&pt);

  printf("Matrix check details: (" IFORMAT ", " IFORMAT ", " IFORMAT ")\n",
         pt.check_result[0], pt.check_result[1], pt.check_result[2]);

  if (check_err_val == MKL_SPARSE_CHECKER_NONTRIANGULAR) {
    printf("Matrix check result: MKL_SPARSE_CHECKER_NONTRIANGULAR\n");
    error = 0;
  } else {
    if (check_err_val == MKL_SPARSE_CHECKER_SUCCESS) {
      printf("Matrix check result: MKL_SPARSE_CHECKER_SUCCESS\n");
    }
    if (check_err_val == MKL_SPARSE_CHECKER_NON_MONOTONIC) {
      printf("Matrix check result: MKL_SPARSE_CHECKER_NON_MONOTONIC\n");
    }
    if (check_err_val == MKL_SPARSE_CHECKER_OUT_OF_RANGE) {
      printf("Matrix check result: MKL_SPARSE_CHECKER_OUT_OF_RANGE\n");
    }
    if (check_err_val == MKL_SPARSE_CHECKER_NONORDERED) {
      printf("Matrix check result: MKL_SPARSE_CHECKER_NONORDERED\n");
    }
    error = 1;
  }
  return check_err_val;
}

// https://stackoverflow.com/questions/49395986/compressed-sparse-row-transpose
void mkl_sparse_mat::transpose() {
  mkl_sparse_mat tmp(_ncol + 1, _nrow, _nnz);
  tmp._nrow--;
  // count per column
  for (MKL_INT i = 0; i < _nnz; ++i) {
    ++tmp._ai[_aj[i] + 2];
  }

  // from count per column generate new rowPtr (but shifted)
  for (MKL_INT i = 2; i <= tmp._nrow; ++i) {
    // create incremental sum
    tmp._ai[i] += tmp._ai[i - 1];
  }

  // perform the main part
  for (MKL_INT i = 0; i < _nrow; ++i) {
    for (MKL_INT j = _ai[i]; j < _ai[i + 1]; ++j) {
      // calculate index to transposed matrix at which we should place current
      // element, and at the same time build final rowPtr
      const size_t new_index = tmp._ai[_aj[j] + 1]++;
      tmp._av[new_index] = _av[j];
      tmp._aj[new_index] = i;
    }
  }
  tmp.sp_fill();
  this->swap(tmp);
}

void mkl_sparse_mat::clear() {
  _ai = nullptr;
  _aj = nullptr;
  _av = nullptr;
  _nrow = -1;
  _ncol = -1;
  _nnz = -1;
  if (_mkl_mat) {
    mkl_sparse_destroy(_mkl_mat);
    _mkl_mat = nullptr;
  }
}

MKL_INT mkl_sparse_mat::max_nz() const {
  MKL_INT res = 0;
  for (MKL_INT i = 0; i < _nrow; i++) {
    res = std::max(res, _ai[i + 1] - _ai[i]);
  }
  return res;
}

MKL_INT mkl_sparse_mat::bandwidth() const {

  const MKL_INT base = mkl_base();

  int bw = -1;
#pragma omp parallel reduction(max : bw)
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    auto [start, end] = utils::LoadBalancedPartition(
        _ai.get(), _ai.get() + rows(), tid, nthreads);
    for (auto it = start; it != end; it++) {
      bw = std::max(bw, _aj[*(it + 1) - 1 - base] - _aj[*it - base]);
    }
  } // omp parallel

  return bw;
}

void mkl_sparse_mat::print_svg(std::ostream &out) const {
  const auto m = this->rows();
  const auto n = this->cols();
  out << "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" "
         "viewBox=\"0 0 "
      << n + 2 << " " << m + 2
      << " \">\n"
         "<style type=\"text/css\" >\n"
         "     <![CDATA[\n"
         "      rect.pixel {\n"
         "          fill:   #ff0000;\n"
         "      }\n"
         "    ]]>\n"
         "  </style>\n\n"
         "   <rect width=\""
      << n + 2 << "\" height=\"" << m + 2
      << "\" fill=\"rgb(128, 128, 128)\"/>\n"
         "   <rect x=\"1\" y=\"1\" width=\""
      << n + 0.1 << "\" height=\"" << m + 0.1
      << "\" fill=\"rgb(255, 255, 255)\"/>\n\n";
  const MKL_INT base = _mkl_base;
  for (MKL_INT j = 0; j < _nrow; ++j) {
    MKL_INT previousStart = _ai[j] - base;
    MKL_INT end = _ai[j + 1] - base;
    for (MKL_INT i = previousStart; i < end; ++i) {
      out << "  <rect class=\"pixel\" x=\"" << _aj[i] - base + 1 << "\" y=\""
          << j + 1 << "\" width=\".9\" height=\".9\"/>\n";
    }
  }
  out << "</svg>" << std::endl;
}

void mkl_sparse_mat::print_gnuplot(std::ostream &out) const {
  const MKL_INT base = _mkl_base;
  for (MKL_INT j = 0; j < _nrow; ++j) {
    MKL_INT previousStart = _ai[j] - base;
    MKL_INT end = _ai[j + 1] - base;
    for (MKL_INT i = previousStart; i < end; ++i) {
      out << _aj[i] - base << " " << -j << std::endl;
    }
  }
}

mkl_sparse_mat mkl_sparse_sum(const mkl_sparse_mat &A, const mkl_sparse_mat &B,
                              double c) {
  if (A.mkl_base() != B.mkl_base()) {
    std::cerr << "two inputs of mkl_sparse_sum has different index base\n";
    return mkl_sparse_mat();
  }
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
mkl_sparse_mat mkl_sparse_mult(const mkl_sparse_mat &A, const mkl_sparse_mat &B,
                               const sparse_operation_t opA,
                               const sparse_operation_t opB) {
  if (A.mkl_base() != B.mkl_base()) {
    std::cerr << "two inputs of mkl_sparse_mult has different index base\n";
    return mkl_sparse_mat();
  }
  sparse_matrix_t result;
  auto status =
      mkl_sparse_sp2m(opA, A.mkl_descr(), A.mkl_handler(), opB, B.mkl_descr(),
                      B.mkl_handler(), SPARSE_STAGE_FULL_MULT, &result);
  if (status != SPARSE_STATUS_SUCCESS) {
    std::cerr << "mkl_sparse_sp2m  failed, code: " << status << std::endl;
    return mkl_sparse_mat();
  }
  status = mkl_sparse_order(result);
  if (status != SPARSE_STATUS_SUCCESS) {
    std::cerr << "mkl_sparse_sp2m mkl_sparse_order failed, code: " << status
              << std::endl;
    return mkl_sparse_mat();
  }
  auto res = mkl_sparse_mat(result);
  mkl_sparse_destroy(result);
  return res;
}

// PT*A*P
mkl_sparse_mat mkl_sparse_mult_ptap(mkl_sparse_mat &A, mkl_sparse_mat &P) {
  auto PTA = mkl_sparse_mult(P, A, SPARSE_OPERATION_TRANSPOSE);
  return mkl_sparse_mult(PTA, P);
}

// P*A*PT
mkl_sparse_mat mkl_sparse_mult_papt(mkl_sparse_mat &A, mkl_sparse_mat &P) {
  auto PA = mkl_sparse_mult(P, A);
  return mkl_sparse_mult(PA, P, SPARSE_OPERATION_NON_TRANSPOSE,
                         SPARSE_OPERATION_TRANSPOSE);
}

// c*A+B
mkl_sparse_mat_sym mkl_sparse_sum(const mkl_sparse_mat_sym &A,
                                  const mkl_sparse_mat_sym &B, double c) {

  auto sum =
      mkl_sparse_sum((const mkl_sparse_mat &)A, (const mkl_sparse_mat &)B, c);
  return mkl_sparse_mat_sym(sum.rows(), sum.cols(), sum.get_ai(), sum.get_aj(),
                            sum.get_av(), sum.mkl_base());
}

// PT*A*P
mkl_sparse_mat_sym mkl_sparse_mult_ptap(mkl_sparse_mat_sym &A,
                                        mkl_sparse_mat &P) {

  sparse_matrix_t result;
  auto status = mkl_sparse_sypr(SPARSE_OPERATION_TRANSPOSE, P.mkl_handler(),
                                A.mkl_handler(), A.mkl_descr(), &result,
                                SPARSE_STAGE_FULL_MULT);

  if (status != SPARSE_STATUS_SUCCESS) {
    std::cerr << "mkl_sparse_sypr failed, code: " << status << std::endl;
    return mkl_sparse_mat_sym();
  }
  auto res = mkl_sparse_mat_sym(result);
  mkl_sparse_destroy(result);
  return res;
}

// P*A*PT
mkl_sparse_mat_sym mkl_sparse_mult_papt(mkl_sparse_mat_sym &A,
                                        mkl_sparse_mat &P) {

  sparse_matrix_t result;
  auto status = mkl_sparse_sypr(SPARSE_OPERATION_NON_TRANSPOSE, P.mkl_handler(),
                                A.mkl_handler(), A.mkl_descr(), &result,
                                SPARSE_STAGE_FULL_MULT);

  if (status != SPARSE_STATUS_SUCCESS) {
    std::cerr << "mkl_sparse_sypr failed, code: " << status << std::endl;
    return mkl_sparse_mat_sym();
  }
  auto res = mkl_sparse_mat_sym(result);
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

mkl_sparse_mat_sym::mkl_sparse_mat_sym(const mkl_sparse_mat &A)
    : mkl_sparse_mat() {

  // only take upper triangular

  _nrow = A.rows();
  _ncol = A.cols();
  _mkl_base = A.mkl_base();
  _pd = A.positive_definite();

  _ai.reset(new MKL_INT[_nrow + 1]);
  auto ai = A.get_ai();
  auto aj = A.get_aj();
  auto av = A.get_av();
  _nnz = 0;
  _ai[0] = _mkl_base;

  std::vector<MKL_INT> diag_pos(_nrow); // record the diag pos

#pragma omp parallel for
  for (MKL_INT i = 0; i < _nrow; i++) {
    auto begin = ai[i] - _mkl_base;
    auto end = ai[i + 1] - _mkl_base;
    auto mid = std::find(aj.get() + begin, aj.get() + end, i + _mkl_base);
    if (mid == aj.get() + end) {
      std::cerr << "Could not find diagonal!" << std::endl;
    } else {
      diag_pos[i] = mid - aj.get();
      _ai[i + 1] = end - diag_pos[i];
    }
  }
  std::inclusive_scan(std::execution::seq, _ai.get(), _ai.get() + _nrow + 1,
                      _ai.get(), std::plus<>());
  _nnz = _ai[_nrow] - _mkl_base;

  _aj.reset(new MKL_INT[_nnz]);
  _av.reset(new double[_nnz]);

  MKL_INT ind = 0;

  for (MKL_INT i = 0; i < _nrow; i++) {
    std::copy(std::execution::seq, aj.get() + diag_pos[i],
              aj.get() + ai[i + 1] - _mkl_base, _aj.get() + _ai[i] - _mkl_base);
    std::copy(std::execution::seq, av.get() + diag_pos[i],
              av.get() + ai[i + 1] - _mkl_base, _av.get() + _ai[i] - _mkl_base);
  }
  sp_fill();
}

mkl_sparse_mat_sym::mkl_sparse_mat_sym(const mkl_sparse_mat_sym &A)
    : mkl_sparse_mat(A) {
  sp_fill();
}

mkl_sparse_mat_sym::mkl_sparse_mat_sym(sparse_matrix_t mkl_mat)
    : mkl_sparse_mat(mkl_mat) {
  sp_fill();
}

mkl_sparse_mat_sym::mkl_sparse_mat_sym(const MKL_INT row, const MKL_INT col,
                                       const std::shared_ptr<MKL_INT[]> &ai,
                                       const std::shared_ptr<MKL_INT[]> &aj,
                                       const std::shared_ptr<double[]> &av,
                                       const sparse_index_base_t base)
    : mkl_sparse_mat(row, col, ai, aj, av, base) {
  sp_fill();
}

void mkl_sparse_mat_sym::sp_fill() {
  if (_mkl_mat) {
    mkl_sparse_destroy(_mkl_mat);
    _mkl_mat = nullptr;
  }
  _mkl_stat =
      mkl_sparse_d_create_csr(&_mkl_mat, _mkl_base, _nrow, _ncol, _ai.get(),
                              _ai.get() + 1, _aj.get(), _av.get());
  if (_mkl_stat != SPARSE_STATUS_SUCCESS) {
    std::cerr << "Matrix is not created, state: " << _mkl_stat << std::endl;
  }

  _mkl_stat = mkl_sparse_order(_mkl_mat); // ordering in CSR format
  _mkl_descr.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
  _mkl_descr.diag = SPARSE_DIAG_NON_UNIT;
  _mkl_descr.mode = SPARSE_FILL_MODE_UPPER;

  optimize();
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

mkl_ic0::mkl_ic0(const mkl_sparse_mat &A) : mkl_sparse_mat_sym(A) {
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
    double *ak = val + (col[k] - _mkl_base);
    MKL_INT *rowk = row + (col[k] - _mkl_base);
    MKL_INT Lk = col[k + 1] - col[k];

    // sanity check
    if (rowk[0] - _mkl_base != k) {
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
    tmp[rowk[0] - _mkl_base] = akk;

    // divide column by akk
    for (MKL_INT j = 1; j < Lk; ++j) {
      ak[j] /= akk;
      tmp[rowk[j] - _mkl_base] = ak[j];
    }

    // loop over all other columns
    for (MKL_INT _j = 1; _j < Lk; ++_j) {
      MKL_INT j = rowk[_j] - _mkl_base;
      double tjk = tmp[j];
      if (tjk != 0.0) {
        double *aj = val + col[j] - _mkl_base;
        MKL_INT Lj = col[j + 1] - col[j];
        MKL_INT *rowj = row + col[j] - _mkl_base;

        for (MKL_INT i = 0; i < Lj; i++)
          aj[i] -= tmp[rowj[i] - _mkl_base] * tjk;
      }
    }

    // reset temp buffer
    for (MKL_INT j = 0; j < Lk; ++j)
      tmp[rowk[j] - _mkl_base] = 0.0;
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

bool dense_mat::orthogonalize() {
  std::vector<double> tau(std::min(_m, _n), 0);
  int info = 0;
  info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, _m, _n, _av.get(), _m, tau.data());
  if (info)
    return false;
  info =
      LAPACKE_dorgqr(LAPACK_COL_MAJOR, _m, _n, _n, _av.get(), _m, tau.data());
  if (info)
    return false;
  return true;
}

mkl_sparse_mat dense_mat::to_sparse_trans() const {
  std::shared_ptr<MKL_INT[]> ai(new MKL_INT[_n + 1]);
  std::shared_ptr<MKL_INT[]> aj(new MKL_INT[_m * _n]);
  ai[0] = 0;
  for (MKL_INT i = 1; i <= _n; i++) {
    ai[i] = ai[i - 1] + _m;
  }
  for (MKL_INT i = 0; i < _m * _n; i++) {
    aj[i] = i % _m;
  }
  return mkl_sparse_mat(_n, _m, ai, aj, _av);
}

bool dense_product(const dense_mat &A, const dense_mat &B, dense_mat &C,
                   const CBLAS_TRANSPOSE opA, const CBLAS_TRANSPOSE opB) {
  if (opA == CblasNoTrans) {
    if (opB == CblasNoTrans) {
      if (A.cols() != B.rows()) {
        std::cerr << "matrix size inconsistency in product" << std::endl;
        return false;
      }
      C.resize(A.rows(), B.cols());
    } else {
      if (A.cols() != B.cols()) {
        std::cerr << "matrix size inconsistency in product" << std::endl;
        return false;
      }
      C.resize(A.rows(), B.rows());
    }
  } else {
    if (opB == CblasNoTrans) {
      if (A.rows() != B.rows()) {
        std::cerr << "matrix size inconsistency in product" << std::endl;
        return false;
      }
      C.resize(A.cols(), B.cols());
    } else {
      if (A.rows() != B.cols()) {
        std::cerr << "matrix size inconsistency in product" << std::endl;
        return false;
      }
      C.resize(A.cols(), B.rows());
    }
  }

  cblas_dgemm(CblasColMajor, opA, opB, C.rows(), C.cols(),
              opA == CblasNoTrans ? A.cols() : A.rows(), 1., A.get_av().get(),
              A.rows(), B.get_av().get(), B.rows(), 0., C.get_av().get(),
              C.rows());
  return true;
}

bool mkl_sparse_dense_mat_prod(const mkl_sparse_mat &A, const dense_mat &B,
                               dense_mat &C, const sparse_operation_t opA) {
  if (opA == SPARSE_OPERATION_NON_TRANSPOSE) {
    if (A.cols() != B.rows()) {
      std::cerr << "matrix size inconsistent." << std::endl;
      return false;
    }
  } else {
    if (A.rows() != B.rows()) {
      std::cerr << "matrix size inconsistent." << std::endl;
      return false;
    }
  }

  C.resize(opA == SPARSE_OPERATION_NON_TRANSPOSE ? A.rows() : A.cols(),
           B.cols());
  auto state = mkl_sparse_d_mm(
      opA, 1., A.mkl_handler(), A.mkl_descr(), SPARSE_LAYOUT_COLUMN_MAJOR,
      B.get_av().get(), B.cols(), B.rows(), 0., C.get_av().get(), C.rows());
  if (state != SPARSE_STATUS_SUCCESS) {
    std::cerr << "fail to compute mkl_sparse_d_mm." << std::endl;
    return false;
  }
  return true;
}

// PT*A*P
dense_mat mkl_sparse_mult_ptap(const mkl_sparse_mat &A, const dense_mat &P) {
  dense_mat ATP;
  if (!mkl_sparse_dense_mat_prod(A, P, ATP, SPARSE_OPERATION_TRANSPOSE)) {
    std::cerr << "mkl_sparse_mult_ptap( const mkl_sparse_mat& A, const "
                 "dense_mat& P ) fails\n";
    return dense_mat();
  }
  dense_mat PTAP;
  if (!dense_product(ATP, P, PTAP, CblasTrans)) {
    std::cerr << "mkl_sparse_mult_ptap( const mkl_sparse_mat& A, const "
                 "dense_mat& P ) fails\n";
    return dense_mat();
  }
  return PTAP;
}

// P*A*PT
dense_mat mkl_sparse_mult_papt(const mkl_sparse_mat &A, const dense_mat &P) {
  dense_mat ATP;
  if (!mkl_sparse_dense_mat_prod(A, P, ATP, SPARSE_OPERATION_TRANSPOSE)) {
    std::cerr << "mkl_sparse_mult_papt( const mkl_sparse_mat& A, const "
                 "dense_mat& P ) fails\n";
    return dense_mat();
  }
  dense_mat PAPT;
  if (!dense_product(ATP, P, PAPT, CblasTrans, CblasTrans)) {
    std::cerr << "mkl_sparse_mult_papt( const mkl_sparse_mat& A, const "
                 "dense_mat& P ) fails\n";
    return dense_mat();
  }
  return PAPT;
}

mkl_sparse_mat random_sparse(const MKL_INT row, const MKL_INT nnzRow) {
  mkl_sparse_mat res(row, row, nnzRow * row);

  auto ai = res.get_ai();
  auto aj = res.get_aj();

  for (MKL_INT i = 0; i <= row; i++) {
    ai[i] = i * nnzRow;
  }
  utils::knuth_s rand;
#pragma omp parallel for private(rand)
  for (MKL_INT i = 0; i < row; i++) {
    rand(nnzRow, 0, row, aj.get() + ai[i]);
  }
  return res;
}

std::shared_ptr<MKL_INT[]> permutedAI(const mkl_sparse_mat &A,
                                      MKL_INT const *const pinv) {
  const auto &ai = A.get_ai();

  const MKL_INT base = A.mkl_base();
  const auto rows = A.rows();

  std::shared_ptr<MKL_INT[]> new_ai(new MKL_INT[rows + 1]);
  if (pinv == nullptr) {
    std::copy(std::execution::par, ai.get(), ai.get() + rows + 1, new_ai.get());
    return new_ai;
  }

  std::vector<MKL_INT> localNNZ(omp_get_max_threads() + 1, 0);
  new_ai[0] = 0;
#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    auto [start, end] = utils::LoadBalancedPartition(
        new_ai.get(), new_ai.get() + rows, tid, nthreads);
    for (auto i = start; i < end; i++) {
      size_t rowInd = pinv[i - new_ai.get()] - base;
      MKL_INT nz = ai[rowInd + 1] - ai[rowInd];
      *(i + 1) = (i == start ? 0 : *i) + nz;
      localNNZ[tid + 1] += nz;
    }
#pragma omp barrier
#pragma omp master
    {
      std::inclusive_scan(localNNZ.begin(), localNNZ.end(), localNNZ.begin(),
                          std::plus<>());
    }

#pragma omp barrier
    for (auto i = start + 1; i <= end; i++) {
      *i += localNNZ[tid] + base;
    }
  }
  new_ai[0] = base;

  return new_ai;
}

std::tuple<std::shared_ptr<MKL_INT[]>, std::shared_ptr<MKL_INT[]>,
           std::shared_ptr<double[]>>
permute(const mkl_sparse_mat &A, MKL_INT const *const pinv,
        MKL_INT const *const q) {
  auto new_ai = permutedAI(A, pinv);

  const auto &ai = A.get_ai();
  const auto &aj = A.get_aj();
  const auto &av = A.get_av();

  const MKL_INT base = A.mkl_base();
  const auto rows = A.rows();
  const auto nnz = A.nnz();

  std::shared_ptr<MKL_INT[]> new_aj{new MKL_INT[nnz]};
  std::shared_ptr<double[]> new_av{new double[nnz]};

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    auto [start, end] = utils::LoadPrefixBalancedPartition(
        new_ai.get(), new_ai.get() + rows, tid, nthreads);

    for (auto i = start; i < end; i++) {
      // copy and convert aj and av
      size_t rowInd = pinv ? pinv[i - new_ai.get()] - base : (i - new_ai.get());
      std::transform(
          aj.get() + ai[rowInd] - base, aj.get() + ai[rowInd + 1] - base,
          new_aj.get() + *i - base,
          [q, base](MKL_INT ind) { return q ? q[ind - base] : (ind - base); });
      std::copy(std::execution::seq, av.get() + ai[rowInd] - base,
                av.get() + ai[rowInd + 1] - base, new_av.get() + *i - base);

      if (q == nullptr)
        continue;
      // intersion sort aj and av based on the column index
      auto pos = new_aj.get() + *(i + 1) - base - 1;
      while (pos != new_aj.get() + *i - base) {
        for (auto j = new_aj.get() + *i - base; j != pos; j++) {
          if (*j > *pos) {
            std::swap(*j, *pos);
            std::swap(new_av[j - new_aj.get()], new_av[pos - new_aj.get()]);
          }
        }
        pos--;
      }
    }
  }
  return std::make_tuple(new_ai, new_aj, new_av);
}

std::tuple<std::shared_ptr<MKL_INT[]>, std::shared_ptr<MKL_INT[]>,
           std::shared_ptr<double[]>>
permuteRow(const mkl_sparse_mat &A, MKL_INT const *const pinv) {
  return permute(A, pinv, nullptr);
}

std::tuple<std::shared_ptr<MKL_INT[]>, std::shared_ptr<MKL_INT[]>,
           std::shared_ptr<double[]>>
symPermute(const mkl_sparse_mat &A, MKL_INT const *const pinv) {
  // upper triangular
  auto ai = A.get_ai();
  auto aj = A.get_aj();
  auto av = A.get_av();
  const MKL_INT n = A.rows();

  const MKL_INT base = A.mkl_base();
  const auto rows = A.rows();
  const auto nnz = A.nnz();

  std::shared_ptr<MKL_INT[]> new_ai{new MKL_INT[n + 1]};
  std::shared_ptr<MKL_INT[]> new_aj{new MKL_INT[nnz]};
  std::shared_ptr<double[]> new_av{new double[nnz]};
  new_ai[0] = base;
  std::vector<MKL_INT> ai_prefix;
#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
#pragma omp master
    { ai_prefix = std::vector<MKL_INT>(n * (nthreads + 1), 0); }
#pragma omp barrier

    auto [start, end] = utils::LoadPrefixBalancedPartition(
        ai.get(), ai.get() + n, tid, nthreads);
    MKL_INT new_row, new_col, final_row, final_col, col;
    for (auto i = start; i != end; i++) {
      new_row = pinv ? (pinv[i - ai.get()] - base) : (i - ai.get());
      for (auto j = *i; j != *(i + 1); j++) {
        if (i - ai.get() > j - base)
          continue;
        col = aj[j - base] - base;
        new_col = pinv ? (pinv[col] - base) : col;
        final_row = std::min(new_row, new_col);
        final_col = std::max(new_row, new_col);

        ai_prefix[(tid + 1) * n + final_row]++;
      }
    }
#pragma omp barrier
#pragma omp master
    {
      for (MKL_INT i = 0; i < n; i++) {
        ai_prefix[i] = new_ai[i] - base;
        for (int j = 0; j < nthreads; j++) {
          ai_prefix[(j + 1) * n + i] += ai_prefix[j * n + i];
        }
        new_ai[i + 1] = ai_prefix[nthreads * n + i] + base;
      }
    }

#pragma omp barrier

    for (auto i = start; i != end; i++) {
      new_row = pinv ? (pinv[i - ai.get()] - base) : (i - ai.get());
      for (auto j = *i; j != *(i + 1); j++) {
        if (i - ai.get() > j - base)
          continue;
        col = aj[j - base] - base;
        new_col = pinv ? (pinv[col] - base) : col;
        final_row = std::min(new_row, new_col);
        final_col = std::max(new_row, new_col);
        // continue;
        new_aj[ai_prefix[tid * n + final_row]] = final_col + base;
        new_av[ai_prefix[tid * n + final_row]++] = av[j - base];
      }
    }
#pragma omp barrier
    // TODO: validate if mkl will automatically sort aj
    if (pinv) {
      auto [start_new, end_new] = utils::LoadPrefixBalancedPartition(
          new_ai.get(), new_ai.get() + n, tid, nthreads);

      for (auto i = start_new; i < end_new; i++) {
        // intersion sort aj and av based on the column index
        auto pos = new_aj.get() + *(i + 1) - base - 1;
        while (pos != new_aj.get() + *i - base) {
          for (auto j = new_aj.get() + *i - base; j != pos; j++) {
            if (*j > *pos) {
              MKL_INT tmp = *j;
              *j = *pos;
              *pos = tmp;
            }
          }
          pos--;
        }
      }
    }
  }
  return std::make_tuple(new_ai, new_aj, new_av);
}
} // namespace mkl_wrapper