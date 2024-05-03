#include "UnionFind.h"
#include "mkl_sparse_mat.h"
#include <gtest/gtest.h>
#include <omp.h>
#include <unordered_map>

// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}

TEST(UnionFind, rank_vs_rem) {
  auto mat = mkl_wrapper::random_sparse(1000, 100);
  auto parents_rank = reordering::UnionFindRank(&mat);
  auto parents_rem = reordering::UnionFindRem(&mat);
  std::unordered_map<MKL_INT, MKL_INT> rank_to_rem;
  for (int i = 0; i < mat.rows(); i++) {
    if (rank_to_rem.find(parents_rank[i]) == rank_to_rem.end()) {
      rank_to_rem[parents_rank[i]] = parents_rem[i];
    } else {
      EXPECT_EQ(rank_to_rem[parents_rank[i]], parents_rem[i]);
    }
  }
}

TEST(UnionFind, rem_vs_parrem) {
  omp_set_num_threads(8);
  auto mat = mkl_wrapper::random_sparse(10000, 1000);
  auto parents_rem = reordering::UnionFindRem(&mat);
  for (int i = 0; i < 100; i++) {
    auto parants_parrem = reordering::ParUnionFindRem(&mat);
    std::unordered_map<MKL_INT, MKL_INT> rank_to_rem;
    for (int i = 0; i < mat.rows(); i++) {
      if (rank_to_rem.find(parants_parrem[i]) == rank_to_rem.end()) {
        rank_to_rem[parants_parrem[i]] = parents_rem[i];
      } else {
        EXPECT_EQ(rank_to_rem[parants_parrem[i]], parents_rem[i]);
      }
    }
  }
}
