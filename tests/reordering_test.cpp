#include "../config.h"
#include "Reordering.h"
#include "UnionFind.h"
#include "matrix_utils.hpp"
#include "utils.h"
#include "io.hpp"
#include <algorithm>
#include <deque>
#include <fstream>
#include <gtest/gtest.h>
#include <omp.h>
#include <unordered_map>
#include <set>
#include "permutation.hpp"

TEST(MinDegreeNode, serial_basic) {
  std::vector<int> degrees = {5, 2, 8, 1, 6, 3};
  std::vector<int> nodes = {0, 1, 2, 3, 4, 5};
  const int base = 0;
  
  auto result = reordering::MinDegreeNode(degrees.data(), base, nodes.begin(), nodes.end(), 1);
  
  // Node 3 has minimum degree 1
  EXPECT_EQ(result.first, 3);
  EXPECT_EQ(result.second, 1);
}

TEST(MinDegreeNode, serial_with_base_offset) {
  std::vector<int> degrees = {5, 2, 8, 1, 6, 3};
  std::vector<int> nodes = {10, 11, 12, 13, 14, 15};  // base-10 indexing
  const int base = 10;
  
  auto result = reordering::MinDegreeNode(degrees.data(), base, nodes.begin(), nodes.end(), 1);
  
  // Node 13 (index 3 in degrees) has minimum degree 1
  EXPECT_EQ(result.first, 13);
  EXPECT_EQ(result.second, 1);
}

TEST(MinDegreeNode, parallel_vs_serial) {
  std::vector<int> degrees = {15, 7, 23, 4, 18, 9, 2, 31, 11, 5};
  std::vector<int> nodes(10);
  std::iota(nodes.begin(), nodes.end(), 0);
  const int base = 0;
  
  auto serial_result = reordering::MinDegreeNode(degrees.data(), base, nodes.begin(), nodes.end(), 1);
  
  for (int nthreads = 2; nthreads <= 8; nthreads *= 2) {
    auto parallel_result = reordering::MinDegreeNode(degrees.data(), base, nodes.begin(), nodes.end(), nthreads);
    EXPECT_EQ(serial_result.first, parallel_result.first);
    EXPECT_EQ(serial_result.second, parallel_result.second);
  }
}

TEST(MinDegreeNode, empty_range) {
  std::vector<int> degrees = {5, 2, 8};
  std::vector<int> nodes;
  const int base = 0;
  
  auto result = reordering::MinDegreeNode(degrees.data(), base, nodes.begin(), nodes.end(), 1);
  
  EXPECT_EQ(result.first, std::numeric_limits<int>::max());
  EXPECT_EQ(result.second, std::numeric_limits<int>::max());
}

TEST(MinDegreeNode, single_node) {
  std::vector<int> degrees = {42};
  std::vector<int> nodes = {0};
  const int base = 0;
  
  auto result = reordering::MinDegreeNode(degrees.data(), base, nodes.begin(), nodes.end(), 1);
  
  EXPECT_EQ(result.first, 0);
  EXPECT_EQ(result.second, 42);
}

TEST(MinDegreeNode, tie_breaking) {
  // Multiple nodes with same minimum degree - should return first one
  std::vector<int> degrees = {5, 2, 8, 2, 6, 2};
  std::vector<int> nodes = {0, 1, 2, 3, 4, 5};
  const int base = 0;
  
  auto result = reordering::MinDegreeNode(degrees.data(), base, nodes.begin(), nodes.end(), 1);
  
  // Should find one of the nodes with degree 2 (nodes 1, 3, or 5)
  EXPECT_EQ(result.second, 2);
  EXPECT_TRUE(result.first == 1 || result.first == 3 || result.first == 5);
}

// TEST(global_min_degree, parallel_vs_serial) {
//   std::vector<std::string> files{"data/ex5.mtx", "data/rdist1.mtx"};
//   for (const auto &fn : files) {
//     std::ifstream f(fn);
//     f.clear();
//     f.seekg(0, std::ios::beg);
//     std::vector<MKL_INT> csr_rows, csr_cols;
//     std::vector<double> csr_vals;
//     utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
//     mkl_wrapper::mkl_sparse_mat mat(csr_rows.size() - 1, csr_rows.size() - 1,
//                                     csr_rows, csr_cols, csr_vals);

//     std::vector<MKL_INT> degrees;
//     reordering::NodeDegree(&mat, degrees);
//     auto res = reordering::MinDegreeNode(
//         degrees, mat.mkl_base(),
//         std::views::iota(0 + mat.mkl_base(), mat.rows() + mat.mkl_base()));

//     for (int t = 1; t <= 8; t++) {
//       omp_set_num_threads(t);
//       std::vector<MKL_INT> pdegrees;
//       reordering::PNodeDegree(&mat, pdegrees);
//       auto res1 = reordering::PMinDegreeNode(
//           pdegrees, mat.mkl_base(),
//           std::views::iota(0 + mat.mkl_base(), mat.rows() + mat.mkl_base()));
//       EXPECT_EQ(degrees, pdegrees);
//       EXPECT_EQ(res, res1);
//     }
//   }
// }

// TEST(global_min_degree, base0_vs_base1) {
//   std::vector<std::string> files{"data/ex5.mtx", "data/rdist1.mtx"};
//   for (const auto &fn : files) {
//     std::ifstream f(fn);
//     f.clear();
//     f.seekg(0, std::ios::beg);
//     std::vector<MKL_INT> csr_rows, csr_cols;
//     std::vector<double> csr_vals;
//     utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
//     mkl_wrapper::mkl_sparse_mat mat(csr_rows.size() - 1, csr_rows.size() - 1,
//                                     csr_rows, csr_cols, csr_vals);
//     std::vector<MKL_INT> degrees0;
//     reordering::NodeDegree(&mat, degrees0);
//     auto res = reordering::MinDegreeNode(
//         degrees0, mat.mkl_base(),
//         std::views::iota(0 + mat.mkl_base(), mat.rows() + mat.mkl_base()));

//     mat.to_one_based();
//     std::vector<MKL_INT> degrees1;
//     reordering::NodeDegree(&mat, degrees1);
//     auto res1 = reordering::MinDegreeNode(
//         degrees1, mat.mkl_base(),
//         std::views::iota(0 + mat.mkl_base(), mat.rows() + mat.mkl_base()));
//     res1.first -= 1; // convert to base 0
//     EXPECT_EQ(res, res1);
//   }
// }

// TEST(component_min_degree, compare_with_sliding_window_size_10) {
//   std::vector<std::string> files{"data/ex5.mtx", "data/rdist1.mtx"};
//   for (const auto &fn : files) {
//     std::ifstream f(fn);
//     f.clear();
//     f.seekg(0, std::ios::beg);
//     std::vector<MKL_INT> csr_rows, csr_cols;
//     std::vector<double> csr_vals;
//     utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
//     mkl_wrapper::mkl_sparse_mat mat(csr_rows.size() - 1, csr_rows.size() - 1,
//                                     csr_rows, csr_cols, csr_vals);
//     const MKL_INT base = mat.mkl_base();
//     auto ai = mat.get_ai();
//     std::deque<std::pair<MKL_INT, MKL_INT>> window;
//     std::vector<MKL_INT> degrees;
//     reordering::NodeDegree(&mat, degrees);
//     for (int i = 0; i < mat.rows(); i++) {

//       while (!window.empty() && window.back().second > ai[i + 1] - ai[i])
//         window.pop_back();
//       window.push_back(std::make_pair(i + base, ai[i + 1] - ai[i]));

//       while (window.front().first <= i - 10 + base)
//         window.pop_front();
//       if (i >= 9) {
//         auto res = reordering::MinDegreeNode(
//             degrees, mat.mkl_base(),
//             std::views::iota(i - 9 + base, i + 1 + base));
//         EXPECT_EQ(res, window.front());
//       }
//     }
//   }
// }

// TEST(reordering, pseudoDiameter) {
//   std::vector<std::string> files{"data/ex5.mtx", "data/rdist1.mtx"};
//   std::vector<MKL_INT> degrees;
//   for (const auto &fn : files) {
//     std::ifstream f(fn);
//     f.clear();
//     f.seekg(0, std::ios::beg);
//     std::vector<MKL_INT> csr_rows, csr_cols;
//     std::vector<double> csr_vals;
//     utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
//     mkl_wrapper::mkl_sparse_mat mat(csr_rows.size() - 1, csr_rows.size() - 1,
//                                     csr_rows, csr_cols, csr_vals);

//     std::vector<MKL_INT> degrees;
//     reordering::NodeDegree(&mat, degrees);
//     MKL_INT source, target;
//     std::cout << "diameter: "
//               << reordering::PseudoDiameter(
//                      &mat, degrees,
//                      std::views::iota(0 + mat.mkl_base(),
//                                       mat.rows() + mat.mkl_base()),
//                      source, target)
//               << " " << source << " " << target << std::endl;
//   }
// }

TEST(UnionFind, rank_vs_rem) {
  for (int iter = 0; iter < 100; iter++) {
    // Create random sparse matrix with raw CSR
    const int rows = 1000;
    const int cols = 1000;
    const int nnz = 1000;
    const int base = 0;
    
    std::vector<int> ai(rows + 1);
    std::vector<int> aj(nnz);
    
    // Generate row pointers
    ai[0] = base;
    for (int i = 1; i <= rows; i++) {
      ai[i] = ai[i - 1] + (nnz / rows) + (i <= (nnz % rows) ? 1 : 0);
    }
    
    // Generate random column indices
    matrix_utils::RandomCSR<int, int, double>(rows, cols, ai.data(), aj.data(), nullptr);
    
    std::vector<int> parents_rank(rows);
    std::vector<int> parents_rem(rows);
    reordering::UnionFindRank(rows, ai.data(), aj.data(), parents_rank.data());
    reordering::UnionFindRem(rows, ai.data(), aj.data(), parents_rem.data());
    std::unordered_map<int, int> rank_to_rem;
    std::unordered_map<int, int> rem_to_rank;
    for (int i = 0; i < rows; i++) {
      if (rank_to_rem.find(reordering::Find(parents_rem.data(), i)) ==
          rank_to_rem.end()) {
        rank_to_rem[reordering::Find(parents_rem.data(), i)] =
            reordering::Find(parents_rank.data(), i);
      } else {
        EXPECT_EQ(rank_to_rem[reordering::Find(parents_rem.data(), i)],
                  reordering::Find(parents_rank.data(), i));
      }
      if (rem_to_rank.find(reordering::Find(parents_rank.data(), i)) ==
          rem_to_rank.end()) {
        rem_to_rank[reordering::Find(parents_rank.data(), i)] =
            reordering::Find(parents_rem.data(), i);
      } else {
        EXPECT_EQ(rem_to_rank[reordering::Find(parents_rank.data(), i)],
                  reordering::Find(parents_rem.data(), i));
      }
    }
  }
}

TEST(UnionFind, rem_vs_parrem)
{
    for (int nthreads = 1; nthreads <= 8; nthreads++)
    {
        for (int iter = 0; iter < 100; iter++)
        {
            // Create random sparse matrix with raw CSR
            const int rows = 1000;
            const int cols = 1000;
            const int nnz = 2000;
            const int base = 0;

            std::vector<int> ai(rows + 1);
            std::vector<int> aj(nnz);

            // Generate row pointers
            ai[0] = base;
            for (int i = 1; i <= rows; i++)
            {
                ai[i] = ai[i - 1] + (nnz / rows) + (i <= (nnz % rows) ? 1 : 0);
            }

            // Generate random column indices
            matrix_utils::RandomCSR<int, int, double>(rows, cols, ai.data(), aj.data(), nullptr);

            std::vector<int> parents_rem(rows);
            reordering::UnionFindRem(rows, ai.data(), aj.data(), parents_rem.data());
            for (int j = 0; j < 100; j++)
            {
                std::vector<int> parants_parrem(rows);
                reordering::ParUnionFindRem(rows, ai.data(), aj.data(), parants_parrem.data(), nthreads);
                std::unordered_map<int, int> rank_to_rem;
                std::unordered_map<int, int> rem_to_rank;
                for (int i = 0; i < rows; i++)
                {
                    if (rank_to_rem.find(reordering::Find(parents_rem.data(), i)) == rank_to_rem.end())
                    {
                        rank_to_rem[reordering::Find(parents_rem.data(), i)] =
                            reordering::Find(parants_parrem.data(), i);
                    }
                    else
                    {
                        EXPECT_EQ(rank_to_rem[reordering::Find(parents_rem.data(), i)],
                                  reordering::Find(parants_parrem.data(), i));
                    }
                    if (rem_to_rank.find(reordering::Find(parants_parrem.data(), i)) == rem_to_rank.end())
                    {
                        rem_to_rank[reordering::Find(parants_parrem.data(), i)] =
                            reordering::Find(parents_rem.data(), i);
                    }
                    else
                    {
                        EXPECT_EQ(rem_to_rank[reordering::Find(parants_parrem.data(), i)],
                                  reordering::Find(parents_rem.data(), i));
                    }
                }
            }
        }
    }
}

TEST(UnionFind, parrem_base)
{
    for (int nthreads = 1; nthreads <= 8; nthreads++)
    {
        for (int iter = 0; iter < 100; iter++)
        {
            // Create random sparse matrix with raw CSR
            const int rows = 1000;
            const int cols = 1000;
            const int nnz = 2000;

            std::vector<int> ai(rows + 1);
            std::vector<int> aj(nnz);

            // Test with 0-based indexing
            int base0 = 0;
            ai[0] = base0;
            for (int i = 1; i <= rows; i++)
            {
                ai[i] = ai[i - 1] + (nnz / rows) + (i <= (nnz % rows) ? 1 : 0);
            }
            matrix_utils::RandomCSR<int, int, double>(rows, cols, ai.data(), aj.data(), nullptr);

            std::vector<int> parants_parrem(rows);
            reordering::ParUnionFindRem(rows, ai.data(), aj.data(), parants_parrem.data(), nthreads);

            // Convert to 1-based indexing
            int base1 = 1;
            for (int i = 0; i <= rows; i++)
            {
                ai[i] += (base1 - base0);
            }
            for (int i = 0; i < nnz; i++)
            {
                aj[i] += (base1 - base0);
            }

            std::vector<int> parants_parrem1(rows);
            reordering::ParUnionFindRem(rows, ai.data(), aj.data(), parants_parrem1.data(), nthreads);

            std::unordered_map<int, int> zero_to_one;
            std::unordered_map<int, int> one_to_zero;
            for (int i = 0; i < rows; i++)
            {
                if (zero_to_one.find(reordering::Find(parants_parrem.data(), i)) == zero_to_one.end())
                {
                    zero_to_one[reordering::Find(parants_parrem.data(), i)] =
                        reordering::Find(parants_parrem1.data(), i);
                }
                else
                {
                    EXPECT_EQ(zero_to_one[reordering::Find(parants_parrem.data(), i)],
                              reordering::Find(parants_parrem1.data(), i));
                }
                if (one_to_zero.find(reordering::Find(parants_parrem1.data(), i)) == one_to_zero.end())
                {
                    one_to_zero[reordering::Find(parants_parrem1.data(), i)] =
                        reordering::Find(parants_parrem.data(), i);
                }
                else
                {
                    EXPECT_EQ(one_to_zero[reordering::Find(parants_parrem1.data(), i)],
                              reordering::Find(parants_parrem.data(), i));
                }
            }
        }
    }
}

TEST(UnionFind, rem_vs_parrank)
{
    for (int nthreads = 1; nthreads <= 8; nthreads++)
    {
        for (int iter = 0; iter < 100; iter++)
        {
            // Create random sparse matrix with raw CSR
            const int rows = 1000;
            const int cols = 1000;
            const int nnz = 1000;
            const int base = 0;

            std::vector<int> ai(rows + 1);
            std::vector<int> aj(nnz);

            // Generate row pointers
            ai[0] = base;
            for (int i = 1; i <= rows; i++)
            {
                ai[i] = ai[i - 1] + (nnz / rows) + (i <= (nnz % rows) ? 1 : 0);
            }

            // Generate random column indices
            matrix_utils::RandomCSR<int, int, double>(rows, cols, ai.data(), aj.data(), nullptr);

            std::vector<int> parents_rem(rows);
            reordering::UnionFindRem(rows, ai.data(), aj.data(), parents_rem.data());
            for (int j = 0; j < 100; j++)
            {
                reordering::DisjointSets ds{static_cast<uint32_t>(rows)};
                ds.execute<int, int>(rows, ai.data(), aj.data());
                std::unordered_map<int, int> rem_to_parrank;
                std::unordered_map<int, int> parrank_to_rem;
                for (int i = 0; i < rows; i++)
                {
                    if (rem_to_parrank.find(reordering::Find(parents_rem.data(), i)) ==
                        rem_to_parrank.end())
                    {
                        rem_to_parrank[reordering::Find(parents_rem.data(), i)] = ds.find(i);
                    }
                    else
                    {
                        EXPECT_EQ(rem_to_parrank[reordering::Find(parents_rem.data(), i)], ds.find(i));
                    }
                    if (parrank_to_rem.find(ds.find(i)) == parrank_to_rem.end())
                    {
                        parrank_to_rem[ds.find(i)] = reordering::Find(parents_rem.data(), i);
                    }
                    else
                    {
                        EXPECT_EQ(parrank_to_rem[ds.find(i)], reordering::Find(parents_rem.data(), i));
                    }
                }
            }
        }
    }
}

// TEST(Reordering, SerialCM) {
//   omp_set_num_threads(3);
//   std::vector<std::string> files{"data/ex5.mtx", "data/s3rmt3m3.mtx"};
//   std::ofstream myfile;
//   for (const auto &fn : files) {
//     std::ifstream f(fn);
//     f.clear();
//     f.seekg(0, std::ios::beg);
//     std::vector<MKL_INT> csr_rows, csr_cols;
//     std::vector<double> csr_vals;
//     utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);

//     mkl_wrapper::mkl_sparse_mat mat(csr_rows.size() - 1, csr_rows.size() - 1,
//                                     csr_rows, csr_cols, csr_vals);

//     std::cout << "bandwidth before rcm reordering: " << mat.bandwidth()
//               << std::endl;
//     std::vector<MKL_INT> inv_perm, perm;
//     reordering::SerialCM(&mat, inv_perm, perm);
//     EXPECT_EQ(matrix_utils::isPermutation(mat.rows(), static_cast<int>(mat.mkl_base()),
//                                           inv_perm.data()),
//               true);

//     auto [ai, aj, av] = matrix_utils::AllocateCSRData(mat.rows(), mat.nnz());
//     matrix_utils::permuteMat(mat.rows(), mat.cols(), inv_perm.data(),
//                           perm.data(), mat.get_ai().get(),
//                           mat.get_aj().get(), mat.get_av().get(), ai.get(), aj.get(),
//                           av.get());
//     mkl_wrapper::mkl_sparse_mat perm_mat(mat.rows(), mat.cols(), ai, aj, av);
//     std::cout << "bandwidth after rcm reordering: " << perm_mat.bandwidth()
//               << std::endl;

//     // myfile.open("mat_perm_rcm.svg");
//     // perm_mat.print_svg(myfile);
//     // myfile.close();

//     mat.to_one_based();

//     std::vector<MKL_INT> inv_perm1, perm1;
//     reordering::SerialCM(&mat, inv_perm1, perm1);
//     EXPECT_EQ(matrix_utils::isPermutation(mat.rows(), static_cast<int>(mat.mkl_base()),
//                                           inv_perm1.data()),
//               true);
//     for (int i = 0; i < mat.rows(); i++) {
//       EXPECT_EQ(inv_perm[i], inv_perm1[i] - 1);
//     }

//     std::vector<MKL_INT> inv_perm2, perm2;
//     reordering::SerialCM(&mat, inv_perm2, perm2);
//     for (int i = 0; i < mat.rows(); i++) {
//       EXPECT_EQ(inv_perm1[i], inv_perm2[i]);
//     }

// #ifdef USE_METIS_LIB
//     std::vector<MKL_INT> nd_inv_perm, nd_perm;
//     reordering::MetisND(&mat, nd_inv_perm, nd_perm);
//     EXPECT_EQ( matrix_utils::isPermutation(
//                    mat.rows(), static_cast<int>( mat.mkl_base() ), nd_inv_perm.data() ),
//                true );
//     auto [ai1, aj1, av1] = matrix_utils::AllocateCSRData(mat.rows(), mat.nnz());
//     matrix_utils::permuteMat(mat.rows(), mat.cols(), nd_inv_perm.data(),
//                           nd_perm.data(), mat.get_ai().get(),
//                           mat.get_aj().get(), mat.get_av().get(), ai1.get(),
//                           aj1.get(), av1.get());
//     mkl_wrapper::mkl_sparse_mat perm_mat1(mat.rows(), mat.cols(), ai1, aj1, av1,
//                                           SPARSE_INDEX_BASE_ONE);
//     std::cout << "bandwidth after metis reordering: " << perm_mat1.bandwidth()
//               << std::endl;
//     // myfile.open("mat_perm_metis.svg");
//     // perm_mat1.print_svg(myfile);
//     // myfile.close();
// #endif
//   }
// }

TEST(UnionFind, ComponentsStat_basic) {
  // Create a random sparse graph and run union-find to get realistic parents
  const int rows = 500;
  const int cols = 500;
  const int nnz = 2000;
  const int base = 0;
  
  // Allocate CSR arrays
  std::vector<int> ai(rows + 1);
  std::vector<int> aj(nnz);
  
  // Generate random row pointers with roughly uniform distribution
  ai[0] = base;
  int entries_per_row = nnz / rows;
  int remainder = nnz % rows;
  for (int i = 1; i <= rows; i++) {
    ai[i] = ai[i - 1] + entries_per_row + (i <= remainder ? 1 : 0);
  }
  
  // Generate random column indices
  matrix_utils::RandomCSR<int, int, double>(rows, cols, ai.data(), aj.data(), nullptr);
  
  // Run union-find to create a realistic parents array
  std::vector<int> parents(rows);
  reordering::ParUnionFindRem(rows, ai.data(), aj.data(), parents.data(), 1);
  
  // Compute component statistics
  std::vector<int> compRoots, sortedComp, compPrefSum;
  reordering::ComponentsStat(parents.data(), rows, base, compRoots, sortedComp, compPrefSum);
  
  // Basic sanity checks
  EXPECT_GT(compRoots.size(), 0);
  EXPECT_EQ(sortedComp.size(), rows);
  EXPECT_EQ(compPrefSum.size(), compRoots.size() + 1);
  EXPECT_EQ(compPrefSum[0], 0);
  EXPECT_EQ(compPrefSum[compRoots.size()], rows);
  
  // Verify that each node appears exactly once in sortedComp
  std::vector<bool> seen(rows, false);
  for (auto node : sortedComp) {
    EXPECT_GE(node, 0);
    EXPECT_LT(node, rows);
    EXPECT_FALSE(seen[node]) << "Node " << node << " appears multiple times";
    seen[node] = true;
  }
  
  // Verify all nodes were seen
  for (int i = 0; i < rows; i++) {
    EXPECT_TRUE(seen[i]) << "Node " << i << " not found in sortedComp";
  }
  
  // Verify component grouping: all nodes in same component should have same root
  for (size_t compIdx = 0; compIdx < compRoots.size(); compIdx++) {
    int compStart = compPrefSum[compIdx];
    int compEnd = compPrefSum[compIdx + 1];
    int expectedRoot = compRoots[compIdx];
    
    for (int i = compStart; i < compEnd; i++) {
      int node = sortedComp[i];
      int actualRoot = reordering::Find(parents.data(), node);
      EXPECT_EQ(actualRoot, expectedRoot) 
          << "Node " << node << " has root " << actualRoot 
          << " but is in component with root " << expectedRoot;
    }
  }
  
  // Verify prefix sum consistency
  for (size_t i = 0; i < compRoots.size(); i++) {
    int compSize = compPrefSum[i + 1] - compPrefSum[i];
    EXPECT_GT(compSize, 0) << "Component " << i << " has zero size";
  }
}

TEST(UnionFind, ComponentsStat_with_base1) {
  // Test with 1-based indexing
  std::vector<int> parents = {0, 0, 2, 2};
  const int size = 4;
  const int base = 1;
  
  std::vector<int> compRoots, sortedComp, compPrefSum;
  
  reordering::ComponentsStat(parents.data(), size, base, compRoots, sortedComp, compPrefSum);
  
  // Should have 2 components
  EXPECT_EQ(compRoots.size(), 2);
  
  // All nodes in sortedComp should be 1-based
  for (auto node : sortedComp) {
    EXPECT_GE(node, base);
    EXPECT_LE(node, size);
  }
  
  // Verify prefix sum
  EXPECT_EQ(compPrefSum[0], 0);
  EXPECT_EQ(compPrefSum[2], size);
}

TEST(UnionFind, ComponentsStat_from_graph) {
  // Create a random sparse graph and run union-find
  const int rows = 100;
  const int cols = 100;
  const int nnz = 500;
  const int base = 0;
  
  // Allocate CSR arrays
  std::vector<int> ai(rows + 1);
  std::vector<int> aj(nnz);
  
  // Generate random row pointers with roughly uniform distribution
  ai[0] = base;
  int entries_per_row = nnz / rows;
  int remainder = nnz % rows;
  for (int i = 1; i <= rows; i++) {
    ai[i] = ai[i - 1] + entries_per_row + (i <= remainder ? 1 : 0);
  }
  
  // Generate random column indices
  matrix_utils::RandomCSR<int, int, double>(rows, cols, ai.data(), aj.data(), nullptr);
  
  // Run union-find
  std::vector<int> parents(rows);
  reordering::ParUnionFindRem(rows, ai.data(), aj.data(), parents.data(), 1);
  
  // Compute component statistics
  std::vector<int> compRoots, sortedComp, compPrefSum;
  reordering::ComponentsStat(parents.data(), rows, base, compRoots, sortedComp, compPrefSum);
  
  // Basic sanity checks
  EXPECT_GT(compRoots.size(), 0);
  EXPECT_EQ(sortedComp.size(), rows);
  EXPECT_EQ(compPrefSum.size(), compRoots.size() + 1);
  EXPECT_EQ(compPrefSum[0], 0);
  EXPECT_EQ(compPrefSum[compRoots.size()], rows);
  
  // Verify that each node appears exactly once in sortedComp
  std::vector<bool> seen(rows, false);
  for (auto node : sortedComp) {
    EXPECT_GE(node, 0);
    EXPECT_LT(node, rows);
    EXPECT_FALSE(seen[node]) << "Node " << node << " appears multiple times";
    seen[node] = true;
  }
  
  // Verify all nodes were seen
  for (int i = 0; i < rows; i++) {
    EXPECT_TRUE(seen[i]) << "Node " << i << " not found in sortedComp";
  }
  
  // Verify component grouping: all nodes in same component should have same root
  for (size_t compIdx = 0; compIdx < compRoots.size(); compIdx++) {
    int compStart = compPrefSum[compIdx];
    int compEnd = compPrefSum[compIdx + 1];
    int expectedRoot = compRoots[compIdx];
    
    for (int i = compStart; i < compEnd; i++) {
      int node = sortedComp[i];
      int actualRoot = reordering::Find(parents.data(), node);
      EXPECT_EQ(actualRoot, expectedRoot) 
          << "Node " << node << " has root " << actualRoot 
          << " but is in component with root " << expectedRoot;
    }
  }
  
  // Verify prefix sum consistency
  for (size_t i = 0; i < compRoots.size(); i++) {
    int compSize = compPrefSum[i + 1] - compPrefSum[i];
    EXPECT_GT(compSize, 0) << "Component " << i << " has zero size";
  }
}

TEST(UnionFind, ComponentsStat_parallel) {
  // Test with multiple thread counts to verify parallel correctness
  const int rows = 1000;
  const int cols = 1000;
  const int nnz = 10000;
  const int base = 0;
  
  // Allocate CSR arrays
  std::vector<int> ai(rows + 1);
  std::vector<int> aj(nnz);
  
  // Generate random row pointers
  ai[0] = base;
  int entries_per_row = nnz / rows;
  int remainder = nnz % rows;
  for (int i = 1; i <= rows; i++) {
    ai[i] = ai[i - 1] + entries_per_row + (i <= remainder ? 1 : 0);
  }
  
  // Generate random column indices
  matrix_utils::RandomCSR<int, int, double>(rows, cols, ai.data(), aj.data(), nullptr);
  
  // Run union-find once
  std::vector<int> parents(rows);
  reordering::ParUnionFindRem(rows, ai.data(), aj.data(), parents.data(), 1);
  
  // Run with 1 thread (baseline)
  std::vector<int> compRoots1, sortedComp1, compPrefSum1;
  reordering::ComponentsStat(parents.data(), rows, base, compRoots1, sortedComp1, compPrefSum1, 1);
  
  // Run with multiple threads
  for (int nthreads : {2, 4, 8}) {
    std::vector<int> compRoots, sortedComp, compPrefSum;
    reordering::ComponentsStat(parents.data(), rows, base, compRoots, sortedComp, compPrefSum, nthreads);
    
    // Number of components should be the same
    EXPECT_EQ(compRoots.size(), compRoots1.size()) 
        << "Different number of components with " << nthreads << " threads";
    
    // Same roots (possibly in different order)
    std::set<int> roots1(compRoots1.begin(), compRoots1.end());
    std::set<int> roots(compRoots.begin(), compRoots.end());
    EXPECT_EQ(roots, roots1) << "Different root sets with " << nthreads << " threads";
    
    // Same component sizes (possibly in different order)
    std::multiset<int> sizes1, sizes;
    for (size_t i = 0; i < compRoots1.size(); i++) {
      sizes1.insert(compPrefSum1[i + 1] - compPrefSum1[i]);
    }
    for (size_t i = 0; i < compRoots.size(); i++) {
      sizes.insert(compPrefSum[i + 1] - compPrefSum[i]);
    }
    EXPECT_EQ(sizes, sizes1) << "Different component sizes with " << nthreads << " threads";
  }
}

TEST(UnionFind, ComponentsStat_block_diagonal_permutation) {
  // Create a block diagonal matrix with known structure:
  // Block 0: 100 nodes (0-99)
  // Block 1: 150 nodes (100-249)
  // Block 2: 200 nodes (250-449)
  // Block 3: 50 nodes (450-499)
  
  const int n = 500;
  const std::vector<int> block_sizes = {100, 150, 200, 50};
  const int num_blocks = block_sizes.size();
  
  // Build block boundaries
  std::vector<int> block_start(num_blocks + 1);
  block_start[0] = 0;
  for (int b = 0; b < num_blocks; b++) {
    block_start[b + 1] = block_start[b] + block_sizes[b];
  }
  
  // Generate block diagonal matrix CSR structure
  std::vector<int> ai(n + 1, 0);
  std::vector<int> aj;
  
  std::mt19937 rng(42);
  
  // First pass: count edges per row
  for (int block = 0; block < num_blocks; block++) {
    int start = block_start[block];
    int end = block_start[block + 1];
    
    // Add spanning tree edges (ensures connectivity)
    for (int i = start + 1; i < end; i++) {
      std::uniform_int_distribution<int> dist(start, i - 1);
      int j = dist(rng);
      ai[i + 1]++; // Edge i->j
      ai[j + 1]++; // Edge j->i (symmetric)
    }
    
    // Add extra random edges within block (symmetric)
    int extra_edges = (end - start) * 2;
    rng.seed(42 + block * 1000);
    for (int e = 0; e < extra_edges; e++) {
      std::uniform_int_distribution<int> dist(start, end - 1);
      int i = dist(rng);
      int j = dist(rng);
      if (i != j) {
        ai[i + 1]++; // Edge i->j
        ai[j + 1]++; // Edge j->i (symmetric)
      }
    }
  }
  
  // Prefix sum
  for (int i = 0; i < n; i++) {
    ai[i + 1] += ai[i];
  }
  
  aj.resize(ai[n]);
  std::vector<int> pos = ai; // Current position for each row
  
  // Second pass: fill column indices
  rng.seed(42);
  for (int block = 0; block < num_blocks; block++) {
    int start = block_start[block];
    int end = block_start[block + 1];
    
    // Spanning tree edges (symmetric)
    for (int i = start + 1; i < end; i++) {
      std::uniform_int_distribution<int> dist(start, i - 1);
      int j = dist(rng);
      aj[pos[i]++] = j; // Edge i->j
      aj[pos[j]++] = i; // Edge j->i (symmetric)
    }
    
    // Extra edges (symmetric)
    rng.seed(42 + block * 1000);
    int extra_edges = (end - start) * 2;
    for (int e = 0; e < extra_edges; e++) {
      std::uniform_int_distribution<int> dist(start, end - 1);
      int i = dist(rng);
      int j = dist(rng);
      if (i != j && pos[i] < ai[i + 1] && pos[j] < ai[j + 1]) {
        aj[pos[i]++] = j; // Edge i->j
        aj[pos[j]++] = i; // Edge j->i (symmetric)
      }
    }
  }
  
  // // Sort column indices for each row
  // for (int i = 0; i < n; i++) {
  //   std::sort(aj.begin() + ai[i], aj.begin() + ai[i + 1]);
  // }
  
  // Create random permutation
  rng.seed(123);
  std::vector<int> perm(n);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), rng);
  
  // Create inverse permutation
  std::vector<int> inv_perm(n);
  matrix_utils::invPerm(n, 0, perm.data(), inv_perm.data());
  
  // Write original block diagonal matrix to SVG
  {
    std::ofstream svg_out("block_diagonal_original.svg");
    matrix_utils::writeSVG(n, n, ai.data(), aj.data(), svg_out);
  }
  
  // Apply permutation: B = P * A * P^T using permuteMat
  std::vector<int> ai_perm(n + 1);
  std::vector<int> aj_perm(ai[n]);
  
  matrix_utils::permuteMat(n, n, perm.data(), inv_perm.data(),
                           ai.data(), aj.data(), 
                           ai_perm.data(), aj_perm.data(), 4);
  
  // Write permuted matrix to SVG
  {
    std::ofstream svg_out("block_diagonal_permuted.svg");
    matrix_utils::writeSVG(n, n, ai_perm.data(), aj_perm.data(), svg_out);
  }
  
  // Run union-find on permuted matrix
  std::vector<int> parents(n);
  reordering::ParUnionFindRem(n, ai_perm.data(), aj_perm.data(), parents.data(), 4);
  
  // Get component statistics
  std::vector<int> compRoots, sortedComp, compPrefSum;
  reordering::ComponentsStat(parents.data(), n, 0, compRoots, sortedComp, compPrefSum, 4);
  
  // Verify correct number of components
  EXPECT_EQ(compRoots.size(), num_blocks) 
      << "Should find " << num_blocks << " components";
  
  // Verify component sizes match block sizes (order may differ)
  std::vector<int> found_sizes;
  for (size_t i = 0; i < compRoots.size(); i++) {
    found_sizes.push_back(compPrefSum[i + 1] - compPrefSum[i]);
  }
  std::sort(found_sizes.begin(), found_sizes.end());
  auto sorted_block_sizes = block_sizes;
  std::sort(sorted_block_sizes.begin(), sorted_block_sizes.end());
  
  EXPECT_EQ(found_sizes, sorted_block_sizes) 
      << "Component sizes should match block sizes";
  
  // Verify each component contains nodes from exactly one original block
  for (size_t comp = 0; comp < compRoots.size(); comp++) {
    int comp_start = compPrefSum[comp];
    int comp_end = compPrefSum[comp + 1];
    
    std::vector<int> block_membership(num_blocks, 0);
    for (int idx = comp_start; idx < comp_end; idx++) {
      int node = sortedComp[idx];
      int orig_node = perm[node];
      
      // Find which block orig_node belonged to
      for (int b = 0; b < num_blocks; b++) {
        if (orig_node >= block_start[b] && orig_node < block_start[b + 1]) {
          block_membership[b]++;
          break;
        }
      }
    }
    
    // All nodes in this component should come from same original block
    int non_zero_blocks = std::count_if(block_membership.begin(), block_membership.end(),
                                        [](int c) { return c > 0; });
    
    EXPECT_EQ(non_zero_blocks, 1) 
        << "Component " << comp << " has nodes from " << non_zero_blocks << " blocks (expected 1)";
  }
  
  // Verify reconstruction: applying inverse permutation to sortedComp
  // should give us back contiguous blocks
  std::vector<int> reconstructed(n);
  for (int i = 0; i < n; i++) {
    reconstructed[i] = perm[sortedComp[i]];
  }
  
  // Within each component, nodes should form a contiguous range
  for (size_t comp = 0; comp < compRoots.size(); comp++) {
    int comp_start = compPrefSum[comp];
    int comp_end = compPrefSum[comp + 1];
    
    std::vector<int> comp_nodes(reconstructed.begin() + comp_start,
                                 reconstructed.begin() + comp_end);
    std::sort(comp_nodes.begin(), comp_nodes.end());
    
    // Verify contiguous range
    for (size_t i = 1; i < comp_nodes.size(); i++) {
      EXPECT_EQ(comp_nodes[i], comp_nodes[i - 1] + 1)
          << "Component " << comp << " nodes not contiguous at position " << i;
    }
  }
}
