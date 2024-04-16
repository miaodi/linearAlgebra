#include "BFS.h"
#include "mkl_sparse_mat.h"
#include "utils.h"
#include <algorithm>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <omp.h>
#include <random>

// Demonstrate some basic assertions.
TEST(bfs, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
}

TEST(bfs, serial) {
  // https://dl.acm.org/cms/attachment/039ee79d-efce-4a81-8a76-ed21ffbd1a5b/f1.jpg
  std::shared_ptr<MKL_INT[]> aiA(
      new MKL_INT[10]{0, 3, 5, 8, 12, 16, 20, 24, 27, 28});
  std::shared_ptr<MKL_INT[]> ajA(new MKL_INT[28]{1, 2, 3, 0, 2, 0, 1, 3, 0, 2,
                                                 4, 5, 3, 5, 6, 7, 3, 4, 6, 7,
                                                 4, 5, 7, 8, 4, 5, 6, 6});
  std::shared_ptr<double[]> avA(new double[28]);

  mkl_wrapper::mkl_sparse_mat A(9, 9, aiA, ajA, avA);
  MKL_INT level;
  auto levels = reordering::BFS(&A, 0, level);
  EXPECT_EQ(level, 5);
  std::vector<MKL_INT> ref{0, 1, 1, 1, 2, 2, 3, 3, 4};
  for (size_t i = 0; i < ref.size(); i++) {
    EXPECT_EQ(levels[i], ref[i]);
  }
}

TEST(bfs, parallel) {
  // https://dl.acm.org/cms/attachment/039ee79d-efce-4a81-8a76-ed21ffbd1a5b/f1.jpg
  std::shared_ptr<MKL_INT[]> aiA(
      new MKL_INT[10]{0, 3, 5, 8, 12, 16, 20, 24, 27, 28});
  std::shared_ptr<MKL_INT[]> ajA(new MKL_INT[28]{1, 2, 3, 0, 2, 0, 1, 3, 0, 2,
                                                 4, 5, 3, 5, 6, 7, 3, 4, 6, 7,
                                                 4, 5, 7, 8, 4, 5, 6, 6});
  std::shared_ptr<double[]> avA(new double[28]);

  mkl_wrapper::mkl_sparse_mat A(9, 9, aiA, ajA, avA);
  MKL_INT level;
  auto levels = reordering::PBFS(&A, 0, level);

  // std::copy(levels.get(), levels.get() + 9,
  //           std::ostream_iterator<int>(std::cout, " "));
  EXPECT_EQ(level, 5);
  std::vector<MKL_INT> ref{0, 1, 1, 1, 2, 2, 3, 3, 4};
  for (size_t i = 0; i < ref.size(); i++) {
    EXPECT_EQ(levels[i], ref[i]);
  }
}

TEST(bfs, serial_vs_parallel) {
  std::vector<std::string> files{"data/ex5.mtx", "data/rdist1.mtx"};
  for (const auto &fn : files) {
    std::ifstream f(fn);
    f.clear();
    f.seekg(0, std::ios::beg);
    std::vector<MKL_INT> csr_rows, csr_cols;
    std::vector<double> csr_vals;
    utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
    mkl_wrapper::mkl_sparse_mat mat(csr_rows.size() - 1, csr_rows.size() - 1,
                                    csr_rows, csr_cols, csr_vals);

    // std::default_random_engine generator;
    // std::uniform_int_distribution<int> distribution(0, mat.rows() - 1);
    MKL_INT level_serial, level_parallel;
    for (int t = 1; t <= 8; t++) {
      omp_set_num_threads(t);
      for (int s = 0; s < mat.rows(); s++) {
        auto levels_serial = reordering::BFS(&mat, s, level_serial);
        auto levels_parallel = reordering::PBFS(&mat, s, level_parallel);
        EXPECT_EQ(level_serial, level_parallel);
        for (size_t i = 0; i < mat.rows(); i++)
          EXPECT_EQ(levels_serial[i], levels_parallel[i]);
      }
    }
  }
}