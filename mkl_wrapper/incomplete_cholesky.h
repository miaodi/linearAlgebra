#pragma once
#include "mkl_sparse_mat.h"

namespace mkl_wrapper {

class incomplete_choleksy_base : public mkl_sparse_mat_sym {
public:
  incomplete_choleksy_base() : mkl_sparse_mat_sym() {}

  virtual bool numeric_factorize(mkl_sparse_mat const *const) { return false; }

  virtual bool symbolic_factorize(mkl_sparse_mat const *const) { return true; }

  virtual bool solve(double const *const b, double *const x) override;

  virtual void optimize() override;

  void shift(const bool shift) { _shift = shift; }

protected:
  template <typename LIST, typename IDX, typename VAL>
  void aij_update(IDX _ai, IDX _aj, VAL _av, MKL_INT j_idx, MKL_INT k,
                  MKL_INT base, const double aki, int &list_size, LIST &list);

protected:
  std::vector<double> _interm_vec;
  double _initial_shift{1e-3};
  int _nrestart{20};
  bool _shift{false};
  std::vector<MKL_INT> _diagPos;
};

// Incomplete Cholesky k level
class incomplete_cholesky_k : public incomplete_choleksy_base {
public:
  incomplete_cholesky_k();

  virtual bool symbolic_factorize(mkl_sparse_mat const *const A) override;

  virtual bool numeric_factorize(mkl_sparse_mat const *const A) override;

  void set_level(const int level) { _level = level; }

protected:
  int _level;
};

// Incomplete Cholesky k level
class incomplete_cholesky_k_2 : public incomplete_choleksy_base {
public:
  incomplete_cholesky_k_2();

  virtual bool symbolic_factorize(mkl_sparse_mat const *const A) override;

  virtual bool numeric_factorize(mkl_sparse_mat const *const A) override;

  void set_level(const int level) { _level = level; }

protected:
  int _level;
};

class incomplete_cholesky_fm : public incomplete_choleksy_base {
public:
  incomplete_cholesky_fm();

  virtual bool symbolic_factorize(mkl_sparse_mat const *const A) override;

  virtual bool numeric_factorize(mkl_sparse_mat const *const A) override;

  void set_lsize(const int lsize) { _lsize = lsize; }

  void set_rsize(const int rsize) { _rsize = rsize; }

protected:
  template <bool buildR> bool numeric_factorize(mkl_sparse_mat const *const A);
  int _lsize{0};
  int _rsize{0};

  // strictly upper triangular
  std::shared_ptr<MKL_INT[]> _ai_r{nullptr}; // Row Pointer for R
  std::shared_ptr<MKL_INT[]> _aj_r{nullptr}; // Column Index for R
  std::shared_ptr<double[]> _av_r{nullptr};  // Value Array for R
};
} // namespace mkl_wrapper