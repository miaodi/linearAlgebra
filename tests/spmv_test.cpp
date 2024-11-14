
#include "incomplete_lu.h"
#include "matrix_utils.hpp"
#include "mkl_sparse_mat.h"
#include "spmv.hpp"
#include "triangle_solve.hpp"
#include "utils.h"
#include <algorithm>
#include <fstream>
#include <gtest/gtest.h>

using namespace mkl_wrapper;
using namespace matrix_utils;

// The fixture for testing class Foo.
class spmv_Test : public testing::Test {
protected:
  std::vector<mkl_wrapper::mkl_sparse_mat> _mats;

  const double _tol = 1e-14;
  const double _MKLtol = 1e-13;

  spmv_Test() {

    std::vector<MKL_INT> csr_rows;
    std::vector<MKL_INT> csr_cols;
    std::vector<double> csr_vals;

    std::ifstream f;
    f.open("data/ex5.mtx"); // https://sparse.tamu.edu/FIDAP/ex5
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

    f.open("data/rdist1.mtx");
    utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
    f.close();
    _mats.push_back(mkl_wrapper::mkl_sparse_mat(csr_rows.size() - 1,
                                                csr_rows.size() - 1, csr_rows,
                                                csr_cols, csr_vals));
  }

  ~spmv_Test() override {
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

TEST_F(spmv_Test, serial_spmv) {
  for (auto &mat : _mats) {
    const MKL_INT size = mat.rows();

    std::vector<double> b(mat.rows());
    std::fill(std::begin(b), std::end(b), 1.);
    std::vector<double> x(mat.rows(), 0.0);
    std::vector<double> x_mkl(mat.rows(), 0.0);
    std::vector<double> diff(mat.rows(), 0.0);

    mat.mult_vec(b.data(), x_mkl.data());

    SerialSPMV spmv;
    spmv(mat.rows(), (int)mat.mkl_base(), mat.get_ai().get(),
         mat.get_aj().get(), mat.get_av().get(), b.data(), x.data(), 1., 0.);

    for (int i = 0; i < mat.rows(); i++)
      diff[i] = x[i] - x_mkl[i];

    const double diff_l2 = cblas_dnrm2(diff.size(), diff.data(), 1);
    const double mkl_l2 = cblas_dnrm2(x_mkl.size(), x_mkl.data(), 1);
    EXPECT_NEAR(diff_l2 / mkl_l2, 0, _MKLtol);
    // std::cout << "error: " << diff_l2 / mkl_l2 << std::endl;

    for (int i = 0; i < mat.rows(); i++) {
      EXPECT_NEAR(x[i], x_mkl[i], _MKLtol * std::abs(x_mkl[i]));
    }
  }
}

TEST_F(spmv_Test, parallel_spmv) {
  omp_set_num_threads(10);

  for (auto &mat : _mats) {
    const MKL_INT size = mat.rows();

    std::vector<double> b(mat.rows());
    std::fill(std::begin(b), std::end(b), 1.);
    std::vector<double> x(mat.rows(), 0.0);
    std::vector<double> x_serial(mat.rows(), 0.0);
    std::vector<double> diff(mat.rows(), 0.0);

    SerialSPMV spmv;
    spmv(mat.rows(), (int)mat.mkl_base(), mat.get_ai().get(),
         mat.get_aj().get(), mat.get_av().get(), b.data(), x_serial.data(), 1.,
         0.);

    ParallelSPMV p_spmv;
    p_spmv(mat.rows(), (int)mat.mkl_base(), mat.get_ai().get(),
           mat.get_aj().get(), mat.get_av().get(), b.data(), x.data(), 1., 0.);
    for (int i = 0; i < mat.rows(); i++)
      diff[i] = x[i] - x_serial[i];

    const double diff_l2 = cblas_dnrm2(diff.size(), diff.data(), 1);
    const double serial_l2 = cblas_dnrm2(x_serial.size(), x_serial.data(), 1);
    EXPECT_NEAR(diff_l2 / serial_l2, 0, _tol);

    for (int i = 0; i < mat.rows(); i++) {
      EXPECT_NEAR(x[i], x_serial[i], _tol * std::abs(x_serial[i]));
    }
  }
}

TEST_F(spmv_Test, seg_spmv) {
  for (auto &mat : _mats) {
    const MKL_INT size = mat.rows();

    std::vector<double> b(mat.rows());
    std::fill(std::begin(b), std::end(b), 1.);
    std::vector<double> x(mat.rows(), 0.0);
    std::vector<double> x_serial(mat.rows(), 0.0);
    std::vector<double> diff(mat.rows(), 0.0);

    SerialSPMV spmv;
    spmv(mat.rows(), (int)mat.mkl_base(), mat.get_ai().get(),
         mat.get_aj().get(), mat.get_av().get(), b.data(), x_serial.data(), 1.,
         0.);

    SegSumSPMV ss_spmv(10);
    ss_spmv.preprocess(mat.rows(), (int)mat.mkl_base(), mat.get_ai().get(),
                       mat.get_aj().get(), mat.get_av().get());
    ss_spmv(mat.rows(), (int)mat.mkl_base(), mat.get_ai().get(),
            mat.get_aj().get(), mat.get_av().get(), b.data(), x.data(), 1., 0.);

    for (int i = 0; i < mat.rows(); i++)
      diff[i] = x[i] - x_serial[i];

    const double diff_l2 = cblas_dnrm2(diff.size(), diff.data(), 1);
    const double serial_l2 = cblas_dnrm2(x_serial.size(), x_serial.data(), 1);
    // EXPECT_NEAR(diff_l2 / serial_l2, 0, _tol);

    std::cout << "error: " << diff_l2 / serial_l2 << std::endl;
    // for (int i = 0; i < mat.rows(); i++) {
    //   EXPECT_NEAR(x[i], x_serial[i], _tol * std::abs(x_serial[i]));
    // }
  }
}

TEST_F(spmv_Test, ALBUS_spmv) {
  for (auto &mat : _mats) {
    const MKL_INT size = mat.rows();
    // std::cout << "size: " << size << std::endl;

    std::vector<double> b(mat.rows());
    std::fill(std::begin(b), std::end(b), 1.);
    std::vector<double> x(mat.rows(), 0.0);
    std::vector<double> x_serial(mat.rows(), 0.0);
    std::vector<double> diff(mat.rows(), 0.0);

    SerialSPMV spmv;
    spmv(mat.rows(), (int)mat.mkl_base(), mat.get_ai().get(),
         mat.get_aj().get(), mat.get_av().get(), b.data(), x_serial.data(), 1.,
         0.);

    ALBUSSPMV albus_spmv(7);
    albus_spmv.preprocess(mat.rows(), (int)mat.mkl_base(), mat.get_ai().get(),
                          mat.get_aj().get(), mat.get_av().get());
    albus_spmv(mat.rows(), (int)mat.mkl_base(), mat.get_ai().get(),
               mat.get_aj().get(), mat.get_av().get(), b.data(), x.data(), 1.,
               0.);

    for (int i = 0; i < mat.rows(); i++)
      diff[i] = x[i] - x_serial[i];

    const double diff_l2 = cblas_dnrm2(diff.size(), diff.data(), 1);
    const double serial_l2 = cblas_dnrm2(x_serial.size(), x_serial.data(), 1);
    // EXPECT_NEAR(diff_l2 / serial_l2, 0, _tol);

    std::cout << "error: " << diff_l2 / serial_l2 << std::endl;
    // for (int i = 0; i < mat.rows(); i++) {
    //   EXPECT_NEAR(x[i], x_serial[i], _tol * std::abs(x_serial[i]));
    // }
    // break;
  }
}