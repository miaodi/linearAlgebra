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

// TEST(UnionFind, rank_vs_rem) {
//   for (int i = 0; i < 100; i++) {
//     auto mat = mkl_wrapper::random_sparse(1000, 1);
//     auto parents_rank = reordering::UnionFindRank(&mat);
//     auto parents_rem = reordering::UnionFindRem(&mat);
//     std::unordered_map<MKL_INT, MKL_INT> rank_to_rem;
//     std::unordered_map<MKL_INT, MKL_INT> rem_to_rank;
//     for (int i = 0; i < mat.rows(); i++) {
//       if (rank_to_rem.find(reordering::Find(parents_rem, i)) ==
//           rank_to_rem.end()) {
//         rank_to_rem[reordering::Find(parents_rem, i)] =
//             reordering::Find(parents_rank, i);
//       } else {
//         EXPECT_EQ(rank_to_rem[reordering::Find(parents_rem, i)],
//                   reordering::Find(parents_rank, i));
//       }
//       if (rem_to_rank.find(reordering::Find(parents_rank, i)) ==
//           rem_to_rank.end()) {
//         rem_to_rank[reordering::Find(parents_rank, i)] =
//             reordering::Find(parents_rem, i);
//       } else {
//         EXPECT_EQ(rem_to_rank[reordering::Find(parents_rank, i)],
//                   reordering::Find(parents_rem, i));
//       }
//     }
//   }
// }

TEST(UnionFind, rem_vs_parrem) {
  omp_set_num_threads(8);
  for (int i = 0; i < 100; i++) {
    auto mat = mkl_wrapper::random_sparse(10, 1);
    auto parents_rem = reordering::UnionFindRem(&mat);
    for (int i = 0; i < 100; i++) {
      auto parants_parrem = reordering::ParUnionFindRem(&mat);
      std::unordered_map<MKL_INT, MKL_INT> rank_to_rem;
      std::unordered_map<MKL_INT, MKL_INT> rem_to_rank;
      for (int i = 0; i < mat.rows(); i++) {
        if (rank_to_rem.find(reordering::Find(parents_rem, i)) ==
            rank_to_rem.end()) {
          rank_to_rem[reordering::Find(parents_rem, i)] =
              reordering::Find(parants_parrem, i);
        } else {
          EXPECT_EQ(rank_to_rem[reordering::Find(parents_rem, i)],
                    reordering::Find(parants_parrem, i));
        }
        if (rem_to_rank.find(reordering::Find(parants_parrem, i)) ==
            rem_to_rank.end()) {
          rem_to_rank[reordering::Find(parants_parrem, i)] =
              reordering::Find(parents_rem, i);
        } else {
          EXPECT_EQ(rem_to_rank[reordering::Find(parants_parrem, i)],
                    reordering::Find(parents_rem, i));
        }
      }
      // std::cout << rank_to_rem.size() << " " << rem_to_rank.size() <<
      // std::endl; EXPECT_EQ(rank_to_rem.size(), rem_to_rank.size());
    }
  }
}

// TEST(UnionFind, rem_vs_parrank) {
//   omp_set_num_threads(8);
//   for (int i = 0; i < 100; i++) {
//     auto mat = mkl_wrapper::random_sparse(1000, 1);
//     auto parents_rem = reordering::UnionFindRem(&mat);
//     for (int i = 0; i < 100; i++) {
//       reordering::DisjointSets ds(&mat);
//       ds.execute();
//       std::unordered_map<MKL_INT, MKL_INT> rem_to_parrank;
//       std::unordered_map<MKL_INT, MKL_INT> parrank_to_rem;
//       for (int i = 0; i < mat.rows(); i++) {
//         if (rem_to_parrank.find(reordering::Find(parents_rem, i)) ==
//             rem_to_parrank.end()) {
//           rem_to_parrank[reordering::Find(parents_rem, i)] = ds.find(i);
//         } else {
//           EXPECT_EQ(rem_to_parrank[reordering::Find(parents_rem, i)],
//                     ds.find(i));
//         }
//         if (parrank_to_rem.find(ds.find(i)) == parrank_to_rem.end()) {
//           parrank_to_rem[ds.find(i)] = reordering::Find(parents_rem, i);
//         } else {
//           EXPECT_EQ(parrank_to_rem[ds.find(i)],
//                     reordering::Find(parents_rem, i));
//         }
//       }
//     }
//   }
// }
