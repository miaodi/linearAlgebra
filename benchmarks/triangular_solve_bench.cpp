#include "../config.h"
#include "incomplete_lu.h"
#include "matrix_utils.hpp"
#include "mkl_sparse_mat.h"
#include "triangle_solve.hpp"
#include <algorithm>
#include <benchmark/benchmark.h>
#include <memory>
#include <numeric>
#include <omp.h>
#include <vector>

static std::unique_ptr<mkl_wrapper::mkl_sparse_mat> mat;
static int _num_threads = 32;
class MyFixture : public benchmark::Fixture {
public:
  MyFixture() {

    // load matrix
    if (mat == nullptr) {
      // std::string k_mat("../../data/shared/K2.bin");
      // std::vector<MKL_INT> csr_rows, csr_cols;
      // std::vector<double> csr_vals;
      // std::cout << "read K\n";
      // utils::ReadFromBinaryCSR(k_mat, csr_rows, csr_cols, csr_vals,
      //                          SPARSE_INDEX_BASE_ONE);

      // const MKL_INT size = csr_rows.size() - 1;
      // mat.reset(new mkl_wrapper::mkl_sparse_mat(
      //     size, size, csr_rows, csr_cols, csr_vals, SPARSE_INDEX_BASE_ONE));
      // mat->to_zero_based();

      std::ifstream f("data/thermal2.mtx");
      f.clear();
      f.seekg(0, std::ios::beg);
      std::vector<MKL_INT> csr_rows, csr_cols;
      std::vector<double> csr_vals;
      utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
      mat.reset(new mkl_wrapper::mkl_sparse_mat(csr_rows.size() - 1,
                                                csr_rows.size() - 1, csr_rows,
                                                csr_cols, csr_vals));

      std::cout << "matrix size: " << mat->rows() << "\n";
    }
  }
};

BENCHMARK_DEFINE_F(MyFixture, SerialForward)(benchmark::State &state) {
  std::vector<double> x(mat->rows(), 0.0);
  std::vector<double> b(mat->rows(), 1.0);

  matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> L, U;
  std::vector<double> D;

  matrix_utils::SplitLDU(mat->rows(), (int)mat->mkl_base(), mat->get_ai().get(),
                         mat->get_aj().get(), mat->get_av().get(), L, D, U);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); i++) {
      matrix_utils::ForwardSubstitution(L.rows, L.Base(), L.ai.get(),
                                        L.aj.get(), L.av.get(), b.data(),
                                        x.data());
    }
  }
}

BENCHMARK_REGISTER_F(MyFixture, SerialForward)->Arg(100)->Arg(1000);

BENCHMARK_DEFINE_F(MyFixture, ParallelForward)(benchmark::State &state) {
  omp_set_num_threads(_num_threads);
  std::vector<double> x(mat->rows(), 0.0);
  std::vector<double> b(mat->rows(), 1.0);

  matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> L, U;
  std::vector<double> D;

  matrix_utils::SplitLDU(mat->rows(), (int)mat->mkl_base(), mat->get_ai().get(),
                         mat->get_aj().get(), mat->get_av().get(), L, D, U);

  std::vector<int> iperm(L.rows);
  std::vector<int> prefix;
  auto lvls = matrix_utils::TopologicalSort2<matrix_utils::TriangularMatrix::L>(
      L.rows, L.Base(), L.ai.get(), L.aj.get(), iperm, prefix);
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); i++) {
      matrix_utils::LevelScheduleForwardSubstitution(
          iperm.data(), prefix.data(), lvls, L.rows, L.Base(), L.ai.get(),
          L.aj.get(), L.av.get(), b.data(), x.data());
    }
  }
}

BENCHMARK_REGISTER_F(MyFixture, ParallelForward)->Arg(100)->Arg(1000);

BENCHMARK_DEFINE_F(MyFixture, MKLForwardSerial)(benchmark::State &state) {
  std::vector<double> x(mat->rows(), 0.0);
  std::vector<double> b(mat->rows(), 1.0);
  omp_set_num_threads(1);

  matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  descr.mode = SPARSE_FILL_MODE_LOWER;
  descr.diag = SPARSE_DIAG_UNIT;

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); i++) {
      mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, mat->mkl_handler(),
                        descr, b.data(), x.data());
    }
  }
}

BENCHMARK_REGISTER_F(MyFixture, MKLForwardSerial)->Arg(100)->Arg(1000);

BENCHMARK_DEFINE_F(MyFixture, MKLForward)(benchmark::State &state) {
  std::vector<double> x(mat->rows(), 0.0);
  std::vector<double> b(mat->rows(), 1.0);
  omp_set_num_threads(_num_threads);

  matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  descr.mode = SPARSE_FILL_MODE_LOWER;
  descr.diag = SPARSE_DIAG_UNIT;

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); i++) {
      mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, mat->mkl_handler(),
                        descr, b.data(), x.data());
    }
  }
}

BENCHMARK_REGISTER_F(MyFixture, MKLForward)->Arg(100)->Arg(1000);

BENCHMARK_DEFINE_F(MyFixture, MKLForwardOpt)(benchmark::State &state) {
  std::vector<double> x(mat->rows(), 0.0);
  std::vector<double> b(mat->rows(), 1.0);
  omp_set_num_threads(_num_threads);

  matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  descr.mode = SPARSE_FILL_MODE_LOWER;
  descr.diag = SPARSE_DIAG_UNIT;

  mkl_sparse_set_sv_hint(mat->mkl_handler(), SPARSE_OPERATION_NON_TRANSPOSE,
                         descr, 1000);
  mkl_sparse_set_memory_hint(mat->mkl_handler(), SPARSE_MEMORY_AGGRESSIVE);
  mkl_sparse_optimize(mat->mkl_handler());

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); i++) {
      mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, mat->mkl_handler(),
                        descr, b.data(), x.data());
    }
  }
}

BENCHMARK_REGISTER_F(MyFixture, MKLForwardOpt)->Arg(100)->Arg(1000);

BENCHMARK_DEFINE_F(MyFixture, CacheParForward_barrier)
(benchmark::State &state) {
  omp_set_num_threads(_num_threads);
  std::vector<double> x(mat->rows(), 0.0);
  std::vector<double> b(mat->rows(), 1.0);

  matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> L, U;
  std::vector<double> D;

  matrix_utils::SplitLDU(mat->rows(), (int)mat->mkl_base(), mat->get_ai().get(),
                         mat->get_aj().get(), mat->get_av().get(), L, D, U);

  matrix_utils::OptimizedTriangularSolve<
      matrix_utils::FBSubstitutionType::Barrier,
      matrix_utils::TriangularMatrix::L, int, int, double>
      forwardsweep(_num_threads);
  forwardsweep.analysis(L.rows, L.Base(), L.ai.get(), L.aj.get(), L.av.get());
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); i++) {
      forwardsweep(b.data(), x.data());
    }
  }
}

BENCHMARK_REGISTER_F(MyFixture, CacheParForward_barrier)->Arg(100)->Arg(1000);

BENCHMARK_DEFINE_F(MyFixture, CacheParForward_nobarrier)
(benchmark::State &state) {
  omp_set_num_threads(_num_threads);
  std::vector<double> x(mat->rows(), 0.0);
  std::vector<double> b(mat->rows(), 1.0);

  matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> L, U;
  std::vector<double> D;

  matrix_utils::SplitLDU(mat->rows(), (int)mat->mkl_base(), mat->get_ai().get(),
                         mat->get_aj().get(), mat->get_av().get(), L, D, U);

  matrix_utils::OptimizedTriangularSolve<
      matrix_utils::FBSubstitutionType::NoBarrier,
      matrix_utils::TriangularMatrix::L, int, int, double>
      forwardsweep(_num_threads);
  forwardsweep.analysis(L.rows, L.Base(), L.ai.get(), L.aj.get(), L.av.get());
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); i++) {
      forwardsweep(b.data(), x.data());
    }
  }
}

BENCHMARK_REGISTER_F(MyFixture, CacheParForward_nobarrier)->Arg(100)->Arg(1000);

BENCHMARK_DEFINE_F(MyFixture, CacheParForward_nobarrier2)
(benchmark::State &state) {
  omp_set_num_threads(_num_threads);
  std::vector<double> x(mat->rows(), 0.0);
  std::vector<double> b(mat->rows(), 1.0);

  matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> L, U;
  std::vector<double> D;

  matrix_utils::SplitLDU(mat->rows(), (int)mat->mkl_base(), mat->get_ai().get(),
                         mat->get_aj().get(), mat->get_av().get(), L, D, U);

  matrix_utils::OptimizedTriangularSolve<
      matrix_utils::FBSubstitutionType::NoBarrierSuperNode,
      matrix_utils::TriangularMatrix::L, int, int, double>
      forwardsweep(_num_threads);
  forwardsweep.analysis(L.rows, L.Base(), L.ai.get(), L.aj.get(), L.av.get());
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); i++) {
      forwardsweep(b.data(), x.data());
    }
  }
}

BENCHMARK_REGISTER_F(MyFixture, CacheParForward_nobarrier2)
    ->Arg(100)
    ->Arg(1000);

BENCHMARK_MAIN();