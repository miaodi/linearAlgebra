#include "Reordering.h"
#include "../config.h"
#include "matrix_utils.hpp"
#include "mkl_sparse_mat.h"
#include "utils.h"
#include <benchmark/benchmark.h>
#include <functional>
#include <map>
#include <memory>
#include <omp.h>

static std::unique_ptr<mkl_wrapper::mkl_sparse_mat> mat;

static std::unique_ptr<mkl_wrapper::mkl_sparse_mat> perm_mat;

static std::unique_ptr<mkl_wrapper::mkl_sparse_mat> perm_mat1;

static std::unique_ptr<mkl_wrapper::mkl_sparse_mat_sym> perm_mat_sym;

class Reordering : public benchmark::Fixture {

public:
  // add members as needed

  Reordering() {
    omp_set_num_threads(4);
    if (mat == nullptr) {
      std::string k_mat("../../data/shared/K.bin");
      std::vector<MKL_INT> csr_rows, csr_cols;
      std::vector<double> csr_vals;
      std::cout << "read K\n";
      utils::ReadFromBinaryCSR(k_mat, csr_rows, csr_cols, csr_vals,
                               SPARSE_INDEX_BASE_ONE);

      // std::ifstream f("../tests/data/s3rmt3m3.mtx");
      // f.clear();
      // f.seekg(0, std::ios::beg);
      // std::vector<MKL_INT> csr_rows, csr_cols;
      // std::vector<double> csr_vals;
      // utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);

      const MKL_INT size = csr_rows.size() - 1;
      mat.reset(new mkl_wrapper::mkl_sparse_mat(
          size, size, csr_rows, csr_cols, csr_vals, SPARSE_INDEX_BASE_ONE));
      mat->to_zero_based();

      std::cout << "bandwidth before reordering: " << mat->bandwidth()
                << std::endl;
      std::vector<MKL_INT> inv_perm, perm;
      reordering::SerialCM(mat.get(), inv_perm, perm);
      auto [ai, aj, av] =
          matrix_utils::AllocateCSRData(mat->rows(), mat->nnz());
      matrix_utils::permute(mat->rows(), (int)mat->mkl_base(),
                            mat->get_ai().get(), mat->get_aj().get(),
                            mat->get_av().get(), inv_perm.data(), perm.data(),
                            ai.get(), aj.get(), av.get());
      perm_mat.reset(new mkl_wrapper::mkl_sparse_mat(mat->rows(), mat->cols(),
                                                     ai, aj, av));
      std::cout << "bandwidth after reordering: " << perm_mat->bandwidth()
                << std::endl;

#ifdef USE_METIS_LIB
      std::vector<MKL_INT> inv_perm1, perm1;
      reordering::Metis(mat.get(), inv_perm1, perm1);
      auto [ai1, aj1, av1] =
          matrix_utils::AllocateCSRData(mat->rows(), mat->nnz());
      matrix_utils::permute(mat->rows(), (int)mat->mkl_base(),
                            mat->get_ai().get(), mat->get_aj().get(),
                            mat->get_av().get(), inv_perm1.data(), perm1.data(),
                            ai1.get(), aj1.get(), av1.get());
      perm_mat1.reset(new mkl_wrapper::mkl_sparse_mat(mat->rows(), mat->cols(),
                                                      ai1, aj1, av1));
      std::cout << "bandwidth after reordering: " << perm_mat1->bandwidth()
                << std::endl;
#endif

      auto [ai2, aj2, av2] =
          matrix_utils::AllocateCSRData(mat->rows(), mat->nnz());
      matrix_utils::symPermute(mat->rows(), (int)mat->mkl_base(),
                               mat->get_ai().get(), mat->get_aj().get(),
                               mat->get_av().get(), inv_perm.data(), ai2.get(),
                               aj2.get(), av2.get());
      perm_mat_sym.reset(new mkl_wrapper::mkl_sparse_mat_sym(
          mat->rows(), mat->cols(), ai2, aj2, av2));
      std::cout << "bandwidth after reordering: " << perm_mat_sym->bandwidth()
                << std::endl;
    }
  }
};

BENCHMARK_DEFINE_F(Reordering, Origin)(benchmark::State &state) {
  std::vector<double> x(mat->cols());
  std::vector<double> rhs(mat->rows());
  std::iota(std::begin(rhs), std::end(rhs), 0);
  for (auto _ : state) {
    mat->mult_vec(rhs.data(), x.data());
  }
}

BENCHMARK_REGISTER_F(Reordering, Origin);

BENCHMARK_DEFINE_F(Reordering, RCM)(benchmark::State &state) {
  std::vector<double> x(mat->cols());
  std::vector<double> rhs(mat->rows());
  std::iota(std::begin(rhs), std::end(rhs), 0);
  for (auto _ : state) {
    perm_mat->mult_vec(rhs.data(), x.data());
  }
}

BENCHMARK_REGISTER_F(Reordering, RCM);

#ifdef USE_METIS_LIB
BENCHMARK_DEFINE_F(Reordering, Metis)(benchmark::State &state) {
  std::vector<double> x(mat->cols());
  std::vector<double> rhs(mat->rows());
  std::iota(std::begin(rhs), std::end(rhs), 0);
  for (auto _ : state) {
    perm_mat1->mult_vec(rhs.data(), x.data());
  }
}

BENCHMARK_REGISTER_F(Reordering, Metis);
#endif

BENCHMARK_DEFINE_F(Reordering, RCM_Sym)(benchmark::State &state) {
  std::vector<double> x(mat->cols());
  std::vector<double> rhs(mat->rows());
  std::iota(std::begin(rhs), std::end(rhs), 0);
  for (auto _ : state) {
    perm_mat_sym->mult_vec(rhs.data(), x.data());
  }
}

BENCHMARK_REGISTER_F(Reordering, RCM_Sym);

BENCHMARK_MAIN();