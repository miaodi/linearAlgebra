#include "matrix_utils.hpp"
#include "precond_symbolic.hpp"
#include <algorithm>
#include <benchmark/benchmark.h>
#include <memory>
#include <numeric>
#include <omp.h>
#include <vector>

static std::unique_ptr<matrix_utils::CSRMatrixVec<int, int, double>> mat;
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
      mat.reset(new matrix_utils::CSRMatrixVec<int, int, double>());
      std::ifstream f("data/thermal2.mtx");
      f.clear();
      f.seekg(0, std::ios::beg);
      utils::read_matrix_market_csr(f, mat->ai, mat->aj, mat->av);
      mat->rows = mat->ai.size() - 1;
      std::cout << "matrix size: " << mat->rows << "\n";
    }
  }
};

BENCHMARK_DEFINE_F(MyFixture, Serial)(benchmark::State &state) {
  std::vector<double> x(mat->rows, 0.0);
  std::vector<double> b(mat->rows, 1.0);

  matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> U, ICC;
  matrix_utils::SplitTriangle<matrix_utils::TriangularMatrix::U>(
      mat->rows, mat->ai[0], mat->AI(), mat->AJ(), mat->AV(), U);

  for (auto _ : state) {
    matrix_utils::ICCLevelSymbolic(mat->rows, mat->ai[0], U.ai.get(),
                                   U.aj.get(), U.ai.get(), state.range(0), ICC);
  }
}

BENCHMARK_REGISTER_F(MyFixture, Serial)->Arg(1)->Arg(2)->Arg(3);

BENCHMARK_MAIN();