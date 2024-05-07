#include "BFS.h"
#include "Reordering.h"
#include "mkl_sparse_mat.h"
#include <algorithm>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <omp.h>
#include <random>
#include <unordered_set>
#include "utils.h"

TEST(bfs, serial) {
  // https://dl.acm.org/cms/attachment/039ee79d-efce-4a81-8a76-ed21ffbd1a5b/f1.jpg
  std::shared_ptr<MKL_INT[]> aiA(
      new MKL_INT[10]{0, 3, 5, 8, 12, 16, 20, 24, 27, 28});
  std::shared_ptr<MKL_INT[]> ajA(new MKL_INT[28]{1, 2, 3, 0, 2, 0, 1, 3, 0, 2,
                                                 4, 5, 3, 5, 6, 7, 3, 4, 6, 7,
                                                 4, 5, 7, 8, 4, 5, 6, 6});
  std::shared_ptr<double[]> avA(new double[28]);

  mkl_wrapper::mkl_sparse_mat A(9, 9, aiA, ajA, avA);

  reordering::BFS bfs(reordering::BFS_Fn<false>);
  bfs(&A, 0);
  EXPECT_EQ(bfs.getHeight(), 5);
  std::vector<MKL_INT> ref{0, 1, 1, 1, 2, 2, 3, 3, 4};
  for (size_t i = 0; i < ref.size(); i++) {
    EXPECT_EQ(bfs.getLevels()[i], ref[i]);
  }

  reordering::BFS bfs2(reordering::BFS_Fn<true>);
  bfs2(&A, 0);

  EXPECT_EQ(bfs2.getLastLevel().size(), 1);
  EXPECT_EQ(bfs2.getLastLevel()[0], 8);
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
  reordering::BFS bfs(reordering::PBFS_Fn<false, true>);
  bfs(&A, 0);

  // std::copy(levels.get(), levels.get() + 9,
  //           std::ostream_iterator<int>(std::cout, " "));
  EXPECT_EQ(bfs.getHeight(), 5);
  std::vector<MKL_INT> ref{0, 1, 1, 1, 2, 2, 3, 3, 4};
  for (size_t i = 0; i < ref.size(); i++) {
    EXPECT_EQ(bfs.getLevels()[i], ref[i]);
  }

  reordering::BFS bfs2(reordering::PBFS_Fn<false, false>);
  bfs2(&A, 0);
  EXPECT_EQ(bfs2.getHeight(), 5);
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

    for (int s = 0; s < mat.rows(); s++) {
      reordering::BFS bfs(reordering::BFS_Fn<true>);
      bfs(&mat, s);
      const auto &bfs_lastLevel = bfs.getLastLevel();
      std::unordered_set<MKL_INT> bfs_set(bfs_lastLevel.begin(),
                                          bfs_lastLevel.end());
      reordering::BFS pbfs(reordering::PBFS_Fn<true, true>);
      reordering::BFS pbfs2(reordering::PBFS_Fn<true, false>);
      for (int t = 1; t <= 7; t++) {
        omp_set_num_threads(t);
        pbfs(&mat, s);
        pbfs2(&mat, s);
        EXPECT_EQ(pbfs.getHeight(), bfs.getHeight());
        for (size_t i = 0; i < mat.rows(); i++)
          EXPECT_EQ(pbfs.getLevels()[i], bfs.getLevels()[i]);

        const auto &pbfs_lastLevel = pbfs.getLastLevel();
        std::unordered_set<MKL_INT> pbfs_set(pbfs_lastLevel.begin(),
                                             pbfs_lastLevel.end());
        EXPECT_EQ(bfs_set, pbfs_set);

        EXPECT_EQ(pbfs.getHeight(), pbfs2.getHeight());

        EXPECT_EQ(pbfs2.getLastLevel().size(), bfs.getLastLevel().size());
      }
    }
  }
}