#include "BFS.h"
#include "mkl_sparse_mat.h"
#include <algorithm>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

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

  std::copy(levels.get(), levels.get() + 9,
            std::ostream_iterator<int>(std::cout, " "));
  // EXPECT_EQ(level, 5);
  // std::vector<MKL_INT> ref{0, 1, 1, 1, 2, 2, 3, 3, 4};
  // for (size_t i = 0; i < ref.size(); i++) {
  //   EXPECT_EQ(levels[i], ref[i]);
  // }
}
