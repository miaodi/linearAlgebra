#include "../config.h"
#include "Reordering.h"
#include "UnionFind.h"
#include "mkl_sparse_mat.h"
#include "utils.h"
#include <gtest/gtest.h>
#include <omp.h>
#include <unordered_map>

TEST(reordering, min_degree_node) {
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
    auto res = reordering::MinDegreeNode(&mat);

    for (int t = 1; t <= 8; t++) {
      omp_set_num_threads(t);
      auto res2 = reordering::PMinDegreeNode(&mat);
      EXPECT_EQ(res.first, res2.first);
      EXPECT_EQ(res.second, res2.second);
    }
  }
}

TEST(reordering, pseudoDiameter) {
  std::vector<std::string> files{"../benchmarks/data/ldoor.mtx"};
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

    auto parants_parrem = reordering::ParUnionFindRem(&mat);
    std::cout << reordering::CountComponents(parants_parrem, 0) << std::endl;
    MKL_INT source, target;
    reordering::PseudoDiameter(&mat, source, target, degrees);
    std::cout << source << " " << target << std::endl;
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
      //     std::cout << i << " " << reordering::Find(parents_rem, i) << " " <<
      //     reordering::Find(parants_parrem, i)
      //               << std::endl;
      //   }
      //   std::cout << std::endl;
      //   break;
      // }
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
  // std::string k_mat("../../data/shared/K2.bin");
  // std::vector<MKL_INT> k_csr_rows, k_csr_cols;
  // std::vector<double> k_csr_vals;
  // std::cout << "read K\n";
  // utils::ReadFromBinaryCSR(k_mat, k_csr_rows, k_csr_cols, k_csr_vals,
  //                          SPARSE_INDEX_BASE_ONE);
  // std::shared_ptr<MKL_INT[]> k_csr_rows_ptr(k_csr_rows.data(),
  //                                           [](MKL_INT[]) {});
  // std::shared_ptr<MKL_INT[]> k_csr_cols_ptr(k_csr_cols.data(),
  //                                           [](MKL_INT[]) {});
  // std::shared_ptr<double[]> k_csr_vals_ptr(k_csr_vals.data(), [](double[])
  // {});

  // const MKL_INT size = k_csr_rows.size() - 1;
  // mkl_wrapper::mkl_sparse_mat mat(size, size, k_csr_rows_ptr, k_csr_cols_ptr,
  //                                 k_csr_vals_ptr, SPARSE_INDEX_BASE_ONE);
  // mat.to_zero_based();

  std::ifstream f("data/ex5.mtx");
  f.clear();
  f.seekg(0, std::ios::beg);
  std::vector<MKL_INT> csr_rows, csr_cols;
  std::vector<double> csr_vals;
  utils::read_matrix_market_csr(f, csr_rows, csr_cols, csr_vals);
  mkl_wrapper::mkl_sparse_mat mat(csr_rows.size() - 1, csr_rows.size() - 1,
                                  csr_rows, csr_cols, csr_vals);

  std::cout << "bandwidth before reordering: " << mat.bandwidth() << std::endl;
  std::ofstream myfile;
  myfile.open("mat.svg");
  mat.print_svg(myfile);
  myfile.close();
  auto inv_perm = reordering::SerialCM(&mat);
  std::cout << (utils::isPermutation(inv_perm, mat.mkl_base())) << std::endl;
  auto perm = utils::inversePermute(inv_perm, mat.mkl_base());
  auto [ai, aj, av] = mkl_wrapper::permute(mat, inv_perm.data(), perm.data());
  mkl_wrapper::mkl_sparse_mat perm_mat(mat.rows(), mat.cols(), ai, aj, av);
  std::cout << "bandwidth after rcm reordering: " << perm_mat.bandwidth()
            << std::endl;
  myfile.open("mat_perm.svg");
  perm_mat.print_svg(myfile);
  myfile.close();
#ifdef USE_METIS_LIB
  auto inv_perm1 = reordering::Metis(&mat);
  std::cout << (utils::isPermutation(inv_perm1, mat.mkl_base())) << std::endl;
  auto perm1 = utils::inversePermute(inv_perm1, mat.mkl_base());
  auto [ai1, aj1, av1] =
      mkl_wrapper::permute(mat, inv_perm1.data(), perm1.data());
  mkl_wrapper::mkl_sparse_mat perm_mat1(mat.rows(), mat.cols(), ai1, aj1, av1);
  std::cout << "bandwidth after metis reordering: " << perm_mat1.bandwidth()
            << std::endl;
  myfile.open("mat_perm_metis.svg");
  perm_mat1.print_svg(myfile);
  myfile.close();
#endif
  auto symMat = mkl_wrapper::mkl_sparse_mat_sym(mat);
  myfile.open("mat_sym.svg");
  symMat.print_svg(myfile);
  myfile.close();

  auto [ai2, aj2, av2] = mkl_wrapper::permute(symMat, perm.data());
  mkl_wrapper::mkl_sparse_mat_sym perm_mat_sym(mat.rows(), mat.cols(), ai2, aj2,
                                               av2);
  myfile.open("perm_mat_sym.svg");
  perm_mat_sym.print_svg(myfile);
  myfile.close();

  symMat.print();
  perm_mat_sym.print();
}