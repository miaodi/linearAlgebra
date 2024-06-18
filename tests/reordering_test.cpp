#include "../config.h"
#include "Reordering.h"
#include "UnionFind.h"
#include "mkl_sparse_mat.h"
#include "utils.h"
#include <algorithm>
#include <deque>
#include <gtest/gtest.h>
#include <omp.h>
#include <unordered_map>

TEST(global_min_degree, parallel_vs_serial) {
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

    std::vector<MKL_INT> degrees;
    reordering::NodeDegree(&mat, degrees);
    auto res = reordering::MinDegreeNode(
        degrees, mat.mkl_base(),
        std::views::iota(0 + mat.mkl_base(), mat.rows() + mat.mkl_base()));

    for (int t = 1; t <= 8; t++) {
      omp_set_num_threads(t);
      std::vector<MKL_INT> pdegrees;
      reordering::PNodeDegree(&mat, pdegrees);
      auto res1 = reordering::PMinDegreeNode(
          pdegrees, mat.mkl_base(),
          std::views::iota(0 + mat.mkl_base(), mat.rows() + mat.mkl_base()));
      EXPECT_EQ(degrees, pdegrees);
      EXPECT_EQ(res, res1);
    }
  }
}

TEST(global_min_degree, base0_vs_base1) {
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
    std::vector<MKL_INT> degrees0;
    reordering::NodeDegree(&mat, degrees0);
    auto res = reordering::MinDegreeNode(
        degrees0, mat.mkl_base(),
        std::views::iota(0 + mat.mkl_base(), mat.rows() + mat.mkl_base()));

    mat.to_one_based();
    std::vector<MKL_INT> degrees1;
    reordering::NodeDegree(&mat, degrees1);
    auto res1 = reordering::MinDegreeNode(
        degrees1, mat.mkl_base(),
        std::views::iota(0 + mat.mkl_base(), mat.rows() + mat.mkl_base()));
    res1.first -= 1; // convert to base 0
    EXPECT_EQ(res, res1);
  }
}

TEST(component_min_degree, compare_with_sliding_window_size_10) {
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
    const MKL_INT base = mat.mkl_base();
    auto ai = mat.get_ai();
    std::deque<std::pair<MKL_INT, MKL_INT>> window;
    std::vector<MKL_INT> degrees;
    reordering::NodeDegree(&mat, degrees);
    for (int i = 0; i < mat.rows(); i++) {

      while (!window.empty() && window.back().second > ai[i + 1] - ai[i])
        window.pop_back();
      window.push_back(std::make_pair(i + base, ai[i + 1] - ai[i]));

      while (window.front().first <= i - 10 + base)
        window.pop_front();
      if (i >= 9) {
        auto res = reordering::MinDegreeNode(
            degrees, mat.mkl_base(),
            std::views::iota(i - 9 + base, i + 1 + base));
        EXPECT_EQ(res, window.front());
      }
    }
  }
}

TEST(reordering, pseudoDiameter) {
  std::vector<std::string> files{"data/ex5.mtx", "data/rdist1.mtx"};
  std::vector<MKL_INT> degrees;
  for (const auto &fn : files) {
    std::ifstream f(fn);
    f.clear();
    f.seekg(0, std::ios::beg);
    std::vector<MKL_INT> csr_rows, csr_cols;
    std::vector<double> csr_vals;
    utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
    mkl_wrapper::mkl_sparse_mat mat(csr_rows.size() - 1, csr_rows.size() - 1,
                                    csr_rows, csr_cols, csr_vals);

    std::vector<MKL_INT> degrees;
    reordering::NodeDegree(&mat, degrees);
    MKL_INT source, target;
    std::cout << "diameter: "
              << reordering::PseudoDiameter(
                     &mat, degrees,
                     std::views::iota(0 + mat.mkl_base(),
                                      mat.rows() + mat.mkl_base()),
                     source, target)
              << " " << source << " " << target << std::endl;
  }
}

TEST(UnionFind, rank_vs_rem) {
  for (int i = 0; i < 100; i++) {
    auto mat = mkl_wrapper::random_sparse(1000, 1);
    auto parents_rank = reordering::UnionFindRank(&mat);
    auto parents_rem = reordering::UnionFindRem(&mat);
    std::unordered_map<MKL_INT, MKL_INT> rank_to_rem;
    std::unordered_map<MKL_INT, MKL_INT> rem_to_rank;
    for (int i = 0; i < mat.rows(); i++) {
      if (rank_to_rem.find(reordering::Find(parents_rem, i)) ==
          rank_to_rem.end()) {
        rank_to_rem[reordering::Find(parents_rem, i)] =
            reordering::Find(parents_rank, i);
      } else {
        EXPECT_EQ(rank_to_rem[reordering::Find(parents_rem, i)],
                  reordering::Find(parents_rank, i));
      }
      if (rem_to_rank.find(reordering::Find(parents_rank, i)) ==
          rem_to_rank.end()) {
        rem_to_rank[reordering::Find(parents_rank, i)] =
            reordering::Find(parents_rem, i);
      } else {
        EXPECT_EQ(rem_to_rank[reordering::Find(parents_rank, i)],
                  reordering::Find(parents_rem, i));
      }
    }
  }
}

TEST(UnionFind, rem_vs_parrem) {
  omp_set_num_threads(5);
  for (int i = 0; i < 100; i++) {
    auto mat = mkl_wrapper::random_sparse(1000, 2);
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
      // std::endl;
      // EXPECT_EQ(rank_to_rem.size(), rem_to_rank.size());
      // if (rank_to_rem.size() != rem_to_rank.size()) {
      //   mat.print();
      //   for (int i = 0; i < mat.rows(); i++) {
      //     std::cout << i << " " << reordering::Find(parents_rem, i) << " "
      // <<
      //     reordering::Find(parants_parrem, i)
      //               << std::endl;
      //   }
      //   std::cout << std::endl;
      //   break;
      // }
    }
  }
}

TEST(UnionFind, parrem_base) {
  omp_set_num_threads(5);
  for (int i = 0; i < 100; i++) {
    auto mat = mkl_wrapper::random_sparse(1000, 2);
    mat.to_zero_based();
    auto parants_parrem = reordering::ParUnionFindRem(&mat);
    mat.to_one_based();
    auto parants_parrem1 = reordering::ParUnionFindRem(&mat);

    std::unordered_map<MKL_INT, MKL_INT> zero_to_one;
    std::unordered_map<MKL_INT, MKL_INT> one_to_zero;
    for (int i = 0; i < mat.rows(); i++) {
      if (zero_to_one.find(reordering::Find(parants_parrem, i)) ==
          zero_to_one.end()) {
        zero_to_one[reordering::Find(parants_parrem, i)] =
            reordering::Find(parants_parrem1, i);
      } else {
        EXPECT_EQ(zero_to_one[reordering::Find(parants_parrem1, i)],
                  reordering::Find(parants_parrem1, i));
      }
      if (one_to_zero.find(reordering::Find(parants_parrem1, i)) ==
          one_to_zero.end()) {
        one_to_zero[reordering::Find(parants_parrem1, i)] =
            reordering::Find(parants_parrem, i);
      } else {
        EXPECT_EQ(one_to_zero[reordering::Find(parants_parrem1, i)],
                  reordering::Find(parants_parrem, i));
      }
    }
  }
}

TEST(UnionFind, rem_vs_parrank) {
  omp_set_num_threads(5);
  for (int i = 0; i < 100; i++) {
    auto mat = mkl_wrapper::random_sparse(1000, 1);
    auto parents_rem = reordering::UnionFindRem(&mat);
    for (int i = 0; i < 100; i++) {
      reordering::DisjointSets ds(&mat);
      ds.execute();
      std::unordered_map<MKL_INT, MKL_INT> rem_to_parrank;
      std::unordered_map<MKL_INT, MKL_INT> parrank_to_rem;
      for (int i = 0; i < mat.rows(); i++) {
        if (rem_to_parrank.find(reordering::Find(parents_rem, i)) ==
            rem_to_parrank.end()) {
          rem_to_parrank[reordering::Find(parents_rem, i)] = ds.find(i);
        } else {
          EXPECT_EQ(rem_to_parrank[reordering::Find(parents_rem, i)],
                    ds.find(i));
        }
        if (parrank_to_rem.find(ds.find(i)) == parrank_to_rem.end()) {
          parrank_to_rem[ds.find(i)] = reordering::Find(parents_rem, i);
        } else {
          EXPECT_EQ(parrank_to_rem[ds.find(i)],
                    reordering::Find(parents_rem, i));
        }
      }
    }
  }
}

TEST(Reordering, SerialCM) {
  omp_set_num_threads(1);
  std::vector<std::string> files{"data/ex5.mtx", "data/s3rmt3m3.mtx"};
  std::ofstream myfile;
  for (const auto &fn : files) {
    std::ifstream f(fn);
    f.clear();
    f.seekg(0, std::ios::beg);
    std::vector<MKL_INT> csr_rows, csr_cols;
    std::vector<double> csr_vals;
    utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
    mkl_wrapper::mkl_sparse_mat mat(csr_rows.size() - 1, csr_rows.size() - 1,
                                    csr_rows, csr_cols, csr_vals);

    std::cout << "bandwidth before rcm reordering: " << mat.bandwidth()
              << std::endl;
    std::vector<MKL_INT> inv_perm, perm;
    reordering::SerialCM(&mat, inv_perm, perm);
    auto [ai, aj, av] = mkl_wrapper::permute(mat, inv_perm.data(), perm.data());
    mkl_wrapper::mkl_sparse_mat perm_mat(mat.rows(), mat.cols(), ai, aj, av);
    std::cout << "bandwidth after rcm reordering: " << perm_mat.bandwidth()
              << std::endl;

    myfile.open("mat_perm_rcm.svg");
    perm_mat.print_svg(myfile);
    myfile.close();
    mat.to_one_based();
    std::vector<MKL_INT> inv_perm1, perm1;
    reordering::SerialCM(&mat, inv_perm1, perm1);
    for (int i = 0; i < mat.rows(); i++) {
      EXPECT_EQ(inv_perm[i], inv_perm1[i] - 1);
    }

#ifdef USE_METIS_LIB
    std::vector<MKL_INT> nd_inv_perm, nd_perm;
    reordering::Metis(&mat, nd_inv_perm, nd_perm);
    std::cout << (utils::isPermutation(nd_inv_perm, mat.mkl_base()))
              << std::endl;
    auto [ai1, aj1, av1] =
        mkl_wrapper::permute(mat, nd_inv_perm.data(), nd_perm.data());
    mkl_wrapper::mkl_sparse_mat perm_mat1(mat.rows(), mat.cols(), ai1, aj1, av1,
                                          SPARSE_INDEX_BASE_ONE);
    std::cout << "bandwidth after metis reordering: " << perm_mat1.bandwidth()
              << std::endl;
    myfile.open("mat_perm_metis.svg");
    perm_mat1.print_svg(myfile);
    myfile.close();
#endif
  }
}