#include "Reordering.h"
#include "../config.h"
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

class Reordering : public benchmark::Fixture {

public:
  // add members as needed

  Reordering() {
    omp_set_num_threads(4);
    if (mat == nullptr) {
      std::string k_mat("../../data/shared/K2.bin");
      std::vector<MKL_INT> k_csr_rows, k_csr_cols;
      std::vector<double> k_csr_vals;
      std::cout << "read K\n";
      utils::ReadFromBinaryCSR(k_mat, k_csr_rows, k_csr_cols, k_csr_vals,
                               SPARSE_INDEX_BASE_ONE);

      //   std::ifstream f("../tests/data/s3rmt3m3.mtx");
      //   f.clear();
      //   f.seekg(0, std::ios::beg);
      //   std::vector<MKL_INT> csr_rows, csr_cols;
      //   std::vector<double> csr_vals;
      //   utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);

      const MKL_INT size = k_csr_rows.size() - 1;
      mat.reset(new mkl_wrapper::mkl_sparse_mat(size, size, k_csr_rows,
                                                k_csr_cols, k_csr_vals,
                                                SPARSE_INDEX_BASE_ONE));
      mat->to_zero_based();

      std::cout << "bandwidth before reordering: " << mat->bandwidth()
                << std::endl;
      auto inv_perm = reordering::SerialCM(mat.get());
      auto perm = utils::inversePermute(inv_perm, mat->mkl_base());
      auto [ai, aj, av] =
          mkl_wrapper::permute(*mat, inv_perm.data(), perm.data());
      perm_mat.reset(new mkl_wrapper::mkl_sparse_mat(mat->rows(), mat->cols(),
                                                     ai, aj, av));
      std::cout << "bandwidth after reordering: " << perm_mat->bandwidth()
                << std::endl;

#ifdef USE_METIS_LIB
      auto inv_perm1 = reordering::Metis(mat.get());
      auto perm1 = utils::inversePermute(inv_perm1, mat->mkl_base());
      auto [ai1, aj1, av1] =
          mkl_wrapper::permute(*mat, inv_perm1.data(), perm1.data());
      perm_mat1.reset(new mkl_wrapper::mkl_sparse_mat(mat->rows(), mat->cols(),
                                                      ai1, aj1, av1));
      std::cout << "bandwidth after reordering: " << perm_mat1->bandwidth()
                << std::endl;
#endif
    }
  }
};

BENCHMARK_DEFINE_F(Reordering, Origin)(benchmark::State &state) {
  std::vector<double> x(mat->cols());
  std::vector<double> rhs(mat->rows());
  std::iota(std::begin(rhs), std::end(rhs), 0); // Fill with 0, 1, ..., 99.
  for (auto _ : state) {
    mat->mult_vec(rhs.data(), x.data());
  }
}

BENCHMARK_REGISTER_F(Reordering, Origin);

BENCHMARK_DEFINE_F(Reordering, RCM)(benchmark::State &state) {
  std::vector<double> x(mat->cols());
  std::vector<double> rhs(mat->rows());
  std::iota(std::begin(rhs), std::end(rhs), 0); // Fill with 0, 1, ..., 99.
  for (auto _ : state) {
    perm_mat->mult_vec(rhs.data(), x.data());
  }
}

BENCHMARK_REGISTER_F(Reordering, RCM);

BENCHMARK_DEFINE_F(Reordering, Metis)(benchmark::State &state) {
  std::vector<double> x(mat->cols());
  std::vector<double> rhs(mat->rows());
  std::iota(std::begin(rhs), std::end(rhs), 0); // Fill with 0, 1, ..., 99.
  for (auto _ : state) {
    perm_mat1->mult_vec(rhs.data(), x.data());
  }
}

BENCHMARK_REGISTER_F(Reordering, Metis);

BENCHMARK_MAIN();