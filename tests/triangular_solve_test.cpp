
#include "incomplete_lu.h"
#include "matrix_utils.hpp"
#include "mkl_sparse_mat.h"
#include "triangle_solve.hpp"
#include "utils.h"
#include <algorithm>
#include <fstream>
#include <gtest/gtest.h>

using namespace mkl_wrapper;
using namespace matrix_utils;

// The fixture for testing class Foo.
class triangular_solve_Test : public testing::Test {
protected:
  std::vector<mkl_wrapper::mkl_sparse_mat> _mats;

  const double _tol = 1e-14;
  const double _MKLtol = 1e-10;

  triangular_solve_Test() {

    std::vector<MKL_INT> csr_rows;
    std::vector<MKL_INT> csr_cols;
    std::vector<double> csr_vals;

    std::ifstream f("data/ex5.mtx"); // https://sparse.tamu.edu/FIDAP/ex5
    utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
    f.close();
    _mats.push_back(mkl_wrapper::mkl_sparse_mat(csr_rows.size() - 1,
                                                csr_rows.size() - 1, csr_rows,
                                                csr_cols, csr_vals));
    f.open("data/nos5.mtx");
    utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
    f.close();
    _mats.push_back(mkl_wrapper::mkl_sparse_mat(csr_rows.size() - 1,
                                                csr_rows.size() - 1, csr_rows,
                                                csr_cols, csr_vals));
    f.open("data/s3rmt3m3.mtx");
    utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
    f.close();
    _mats.push_back(mkl_wrapper::mkl_sparse_mat(csr_rows.size() - 1,
                                                csr_rows.size() - 1, csr_rows,
                                                csr_cols, csr_vals));
    f.open("data/bcsstk17.mtx");
    utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
    f.close();
    _mats.push_back(mkl_wrapper::mkl_sparse_mat(csr_rows.size() - 1,
                                                csr_rows.size() - 1, csr_rows,
                                                csr_cols, csr_vals));
  }

  ~triangular_solve_Test() override {
    // You can do clean-up work that doesn't throw exceptions here.
  }

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  void SetUp() override {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  void TearDown() override {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }

  // Class members declared here can be used by all tests in the test suite
  // for Foo.
};

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

TEST_F(triangular_solve_Test, forward_substitution) {
  for (auto mat : _mats) {
    const MKL_INT size = mat.rows();
    mat.to_zero_based();
    mkl_wrapper::incomplete_lu_k prec;
    prec.set_level(5);
    prec.symbolic_factorize(&mat);
    prec.numeric_factorize(&mat);

    // std::ofstream myfile;
    // myfile.open("prec.svg");
    // prec.print_svg(myfile);
    // myfile.close();

    std::vector<double> b(mat.rows());
    std::fill(std::begin(b), std::end(b), 1.);
    std::vector<double> x_serial(mat.rows(), 0.0);
    std::vector<double> x_par(mat.rows(), 0.0);
    std::vector<double> x_mkl(mat.rows(), 0.0);

    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    descr.mode = SPARSE_FILL_MODE_LOWER;
    descr.diag = SPARSE_DIAG_UNIT;
    mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, prec.mkl_handler(),
                      descr, b.data(), x_mkl.data());

    matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> L, U;
    std::vector<double> D;

    matrix_utils::SplitLDU(prec.rows(), (int)prec.mkl_base(),
                           prec.get_ai().get(), prec.get_aj().get(),
                           prec.get_av().get(), L, D, U);

    matrix_utils::ForwardSubstitution(L.rows, L.Base(), L.ai.get(), L.aj.get(),
                                      L.av.get(), b.data(), x_serial.data());
    for (int i = 0; i < mat.rows(); i++) {
      EXPECT_NEAR(x_serial[i], x_mkl[i], _MKLtol * std::abs(x_mkl[i]));
    }

    std::vector<int> iperm(L.rows);
    std::vector<int> prefix;
    matrix_utils::TopologicalSort2<matrix_utils::TriangularMatrix::L>(
        L.rows, L.Base(), L.ai.get(), L.aj.get(), iperm, prefix);
    matrix_utils::LevelScheduleForwardSubstitution(
        iperm, prefix, L.rows, L.Base(), L.ai.get(), L.aj.get(), L.av.get(),
        b.data(), x_par.data());
    for (int i = 0; i < mat.rows(); i++) {
      EXPECT_NEAR(x_par[i], x_mkl[i], _MKLtol * std::abs(x_mkl[i]));
    }

    auto Lt_data = matrix_utils::AllocateCSRData(L.cols, L.NNZ());
    matrix_utils::ParallelTranspose(
        L.rows, L.cols, L.Base(), L.ai.get(), L.aj.get(), L.av.get(),
        std::get<0>(Lt_data).get(), std::get<1>(Lt_data).get(),
        std::get<2>(Lt_data).get());

    std::vector<double> x_serial_t(mat.rows(), 0.0);

    matrix_utils::ForwardSubstitutionT(
        L.rows, L.Base(), std::get<0>(Lt_data).get(), std::get<1>(Lt_data).get(),
        std::get<2>(Lt_data).get(), b.data(), x_serial_t.data());

    for (int i = 0; i < mat.rows(); i++) {
      EXPECT_NEAR(x_serial_t[i], x_mkl[i], _MKLtol * std::abs(x_mkl[i]));
    }
  }
}

TEST_F(triangular_solve_Test, backward_substitution) {
  for (auto mat : _mats) {
    const MKL_INT size = mat.rows();
    mat.to_zero_based();
    mkl_wrapper::incomplete_lu_k prec;
    prec.set_level(5);
    prec.symbolic_factorize(&mat);
    prec.numeric_factorize(&mat);

    // std::ofstream myfile;
    // myfile.open("prec.svg");
    // prec.print_svg(myfile);
    // myfile.close();

    std::vector<double> b(mat.rows());
    std::fill(std::begin(b), std::end(b), 1.);
    std::vector<double> x_serial(mat.rows(), 0.0);
    std::vector<double> x_par(mat.rows(), 0.0);
    std::vector<double> x_mkl(mat.rows(), 0.0);

    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    descr.mode = SPARSE_FILL_MODE_UPPER;
    descr.diag = SPARSE_DIAG_NON_UNIT;
    mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, prec.mkl_handler(),
                      descr, b.data(), x_mkl.data());

    matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> L, U;
    std::vector<double> D;

    matrix_utils::SplitLDU(prec.rows(), (int)prec.mkl_base(),
                           prec.get_ai().get(), prec.get_aj().get(),
                           prec.get_av().get(), L, D, U);

    matrix_utils::BackwardSubstitution(U.rows, U.Base(), U.ai.get(), U.aj.get(),
                                       U.av.get(), D.data(), b.data(),
                                       x_serial.data());
    for (int i = 0; i < mat.rows(); i++) {
      EXPECT_NEAR(x_serial[i], x_mkl[i], _MKLtol * std::abs(x_mkl[i]));
    }

    std::vector<int> iperm(U.rows);
    std::vector<int> prefix;
    matrix_utils::TopologicalSort2<matrix_utils::TriangularMatrix::U>(
        U.rows, U.Base(), U.ai.get(), U.aj.get(), iperm, prefix);

    matrix_utils::LevelScheduleBackwardSubstitution(
        iperm, prefix, U.rows, U.Base(), U.ai.get(), U.aj.get(), U.av.get(),
        D.data(), b.data(), x_par.data());
    for (int i = 0; i < mat.rows(); i++) {
      EXPECT_NEAR(x_par[i], x_mkl[i], _MKLtol * std::abs(x_mkl[i]));
    }
  }
}

TEST_F(triangular_solve_Test, forward_substitution_optimized) {
  omp_set_num_threads(5);
  for (auto mat : _mats) {
    const MKL_INT size = mat.rows();
    mat.to_zero_based();
    mkl_wrapper::incomplete_lu_k prec;
    prec.set_level(5);
    prec.symbolic_factorize(&mat);
    prec.numeric_factorize(&mat);

    // std::ofstream myfile;
    // myfile.open("prec.svg");
    // prec.print_svg(myfile);
    // myfile.close();

    std::vector<double> b(mat.rows());
    std::fill(std::begin(b), std::end(b), 1.);
    std::vector<double> x(mat.rows(), 0.0);
    std::vector<double> x_serial(mat.rows(), 0.0);

    matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> L, U;
    std::vector<double> D;

    matrix_utils::SplitLDU(prec.rows(), (int)prec.mkl_base(),
                           prec.get_ai().get(), prec.get_aj().get(),
                           prec.get_av().get(), L, D, U);

    matrix_utils::ForwardSubstitution(L.rows, L.Base(), L.ai.get(), L.aj.get(),
                                      L.av.get(), b.data(), x_serial.data());

    matrix_utils::OptimizedTriangularSolve<
        matrix_utils::FBSubstitutionType::Barrier,
        matrix_utils::TriangularMatrix::L, int, int, double>
        forwardsweep_barrier;
    forwardsweep_barrier.analysis(L.rows, L.Base(), L.ai.get(), L.aj.get(),
                                  L.av.get());
    for (int i = 0; i < 100; i++) {
      forwardsweep_barrier(b.data(), x.data());
      for (int i = 0; i < x.size(); i++) {
        EXPECT_NEAR(x[i], x_serial[i], _tol * std::abs(x_serial[i]));
      }
    }

    matrix_utils::OptimizedTriangularSolve<
        matrix_utils::FBSubstitutionType::NoBarrier,
        matrix_utils::TriangularMatrix::L, int, int, double>
        forwardsweep_nobarrier;
    forwardsweep_nobarrier.analysis(L.rows, L.Base(), L.ai.get(), L.aj.get(),
                                    L.av.get());
    for (int i = 0; i < 100; i++) {
      forwardsweep_nobarrier(b.data(), x.data());
      for (int i = 0; i < x.size(); i++) {
        EXPECT_NEAR(x[i], x_serial[i], _tol * std::abs(x_serial[i]));
      }
    }

    matrix_utils::OptimizedTriangularSolve<
        matrix_utils::FBSubstitutionType::NoBarrierSuperNode,
        matrix_utils::TriangularMatrix::L, int, int, double>
        forwardsweep_nobarrier_sn;
    forwardsweep_nobarrier_sn.analysis(L.rows, L.Base(), L.ai.get(), L.aj.get(),
                                       L.av.get());
    for (int i = 0; i < 100; i++) {
      forwardsweep_nobarrier_sn(b.data(), x.data());
      for (int i = 0; i < x.size(); i++) {
        EXPECT_NEAR(x[i], x_serial[i], _tol * std::abs(x_serial[i]));
      }
    }
  }
}

TEST_F(triangular_solve_Test, backward_substitution_optimized) {
  omp_set_num_threads(5);
  for (auto mat : _mats) {
    const MKL_INT size = mat.rows();
    mat.to_zero_based();
    mkl_wrapper::incomplete_lu_k prec;
    prec.set_level(5);
    prec.symbolic_factorize(&mat);
    prec.numeric_factorize(&mat);

    // std::ofstream myfile;
    // myfile.open("prec.svg");
    // prec.print_svg(myfile);
    // myfile.close();

    std::vector<double> b(mat.rows());
    std::fill(std::begin(b), std::end(b), 1.);
    std::vector<double> x(mat.rows(), 0.0);
    std::vector<double> x_serial(mat.rows(), 0.0);

    matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> L, U;
    std::vector<double> D;

    matrix_utils::SplitLDU(prec.rows(), (int)prec.mkl_base(),
                           prec.get_ai().get(), prec.get_aj().get(),
                           prec.get_av().get(), L, D, U);

    matrix_utils::BackwardSubstitution(U.rows, U.Base(), U.ai.get(), U.aj.get(),
                                       U.av.get(), D.data(), b.data(),
                                       x_serial.data());

    matrix_utils::OptimizedTriangularSolve<
        matrix_utils::FBSubstitutionType::Barrier,
        matrix_utils::TriangularMatrix::U, int, int, double>
        forwardsweep_barrier;
    forwardsweep_barrier.analysis(U.rows, U.Base(), U.ai.get(), U.aj.get(),
                                  U.av.get(), D.data());
    for (int i = 0; i < 100; i++) {
      forwardsweep_barrier(b.data(), x.data());
      for (int i = 0; i < x.size(); i++) {
        EXPECT_NEAR(x[i], x_serial[i], _tol * std::abs(x_serial[i]));
      }
    }

    matrix_utils::OptimizedTriangularSolve<
        matrix_utils::FBSubstitutionType::NoBarrier,
        matrix_utils::TriangularMatrix::U, int, int, double>
        forwardsweep_nobarrier;
    forwardsweep_nobarrier.analysis(U.rows, U.Base(), U.ai.get(), U.aj.get(),
                                    U.av.get(), D.data());
    for (int i = 0; i < 100; i++) {
      forwardsweep_nobarrier(b.data(), x.data());
      for (int i = 0; i < x.size(); i++) {
        EXPECT_NEAR(x[i], x_serial[i], _tol * std::abs(x_serial[i]));
      }
    }

    matrix_utils::OptimizedTriangularSolve<
        matrix_utils::FBSubstitutionType::NoBarrierSuperNode,
        matrix_utils::TriangularMatrix::U, int, int, double>
        forwardsweep_nobarrier_sn;
    forwardsweep_nobarrier_sn.analysis(U.rows, U.Base(), U.ai.get(), U.aj.get(),
                                       U.av.get(), D.data());
    for (int i = 0; i < 100; i++) {
      forwardsweep_nobarrier_sn(b.data(), x.data());
      for (int i = 0; i < x.size(); i++) {
        EXPECT_NEAR(x[i], x_serial[i], _tol * std::abs(x_serial[i]));
      }
    }
  }
}