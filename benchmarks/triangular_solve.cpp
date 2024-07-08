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

      std::ifstream f("/SCRATCH/dimiao/test_zone/matrices/pwtk.mtx");
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
      matrix_utils::ForwardSubstitution(L.rows, L.base, L.ai.get(), L.aj.get(),
                                        L.av.get(), b.data(), x.data());
    }
  }
}

BENCHMARK_REGISTER_F(MyFixture, SerialForward)->Arg(10)->Arg(100);

BENCHMARK_DEFINE_F(MyFixture, ParallelForward)(benchmark::State &state) {

  omp_set_num_threads(8);
  std::vector<double> x(mat->rows(), 0.0);
  std::vector<double> b(mat->rows(), 1.0);

  matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> L, U;
  std::vector<double> D;

  matrix_utils::SplitLDU(mat->rows(), (int)mat->mkl_base(), mat->get_ai().get(),
                         mat->get_aj().get(), mat->get_av().get(), L, D, U);

  std::vector<int> iperm(L.rows);
  std::vector<int> prefix;
  matrix_utils::TopologicalSort2<matrix_utils::TriangularSolve::L>(
      L.rows, L.base, L.ai.get(), L.aj.get(), iperm, prefix);
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); i++) {
      matrix_utils::LevelScheduleForwardSubstitution(
          iperm, prefix, L.rows, L.base, L.ai.get(), L.aj.get(), L.av.get(),
          b.data(), x.data());
    }
  }
}

BENCHMARK_REGISTER_F(MyFixture, ParallelForward)->Arg(10)->Arg(100);

BENCHMARK_MAIN();