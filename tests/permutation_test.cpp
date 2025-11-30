#include "../sparse_mat_op/permutation.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <iomanip>
#include <memory>
#include <random>
#include <vector>

TEST(permutation, is_permutation) {
  std::vector<int> perm{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  EXPECT_TRUE(matrix_utils::isPermutation(static_cast<int>(perm.size()), 0,
                                          perm.data()));
  perm[0] = 1;
  EXPECT_FALSE(matrix_utils::isPermutation(static_cast<int>(perm.size()), 0,
                                           perm.data()));
}

TEST(permutation, is_permutation_threads) {
  std::vector<int> perm{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  for (int nthreads : {1, 2, 4, 8}) {
    EXPECT_TRUE(matrix_utils::isPermutation(static_cast<int>(perm.size()), 0,
                                            perm.data(), nthreads));
  }
  perm[0] = 1;
  for (int nthreads : {1, 2, 4, 8}) {
    EXPECT_FALSE(matrix_utils::isPermutation(static_cast<int>(perm.size()), 0,
                                             perm.data(), nthreads));
  }
}

TEST(permutation, rand_perm) {
  std::vector<int> perm(10);
  for (int i = 0; i < 100; i++) {
    matrix_utils::randPerm(static_cast<int>(perm.size()), 0, perm.data());
    EXPECT_TRUE(matrix_utils::isPermutation(static_cast<int>(perm.size()), 0,
                                            perm.data()));
    matrix_utils::randPerm(static_cast<int>(perm.size()), 1, perm.data());
    EXPECT_TRUE(matrix_utils::isPermutation(static_cast<int>(perm.size()), 1,
                                            perm.data()));
    perm[0] += 1;
    EXPECT_FALSE(matrix_utils::isPermutation(static_cast<int>(perm.size()), 1,
                                             perm.data()));
  }
}

TEST(permutation, inv_perm) {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(1, 100000);
  size_t size = dist(gen);
  int base = 0;
  std::vector<int> perm(size);
  matrix_utils::randPerm(static_cast<int>(perm.size()), base, perm.data());
  std::vector<int> iperm(perm.size());
  matrix_utils::invPerm(static_cast<int>(perm.size()), base, perm.data(),
                        iperm.data());
  for (size_t i = 0; i < size; i++) {
    EXPECT_EQ(iperm[perm[i]], i);
    EXPECT_EQ(perm[iperm[i]], i);
  }
  base = 1;
  matrix_utils::randPerm(static_cast<int>(perm.size()), base, perm.data());
  matrix_utils::invPerm(static_cast<int>(perm.size()), base, perm.data(),
                        iperm.data());
  for (size_t i = 0; i < size; i++) {
    EXPECT_EQ(iperm[perm[i] - base], i + base);
    EXPECT_EQ(perm[iperm[i] - base], i + base);
  }
}

TEST(permutation, inv_perm_threads) {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(1, 100000);
  size_t size = dist(gen);
  
  for (int nthreads : {1, 2, 4, 8}) {
    int base = 0;
    std::vector<int> perm(size);
    matrix_utils::randPerm(static_cast<int>(perm.size()), base, perm.data());
    std::vector<int> iperm(perm.size());
    matrix_utils::invPerm(static_cast<int>(perm.size()), base, perm.data(),
                          iperm.data(), nthreads);
    for (size_t i = 0; i < size; i++) {
      EXPECT_EQ(iperm[perm[i]], i);
      EXPECT_EQ(perm[iperm[i]], i);
    }
    base = 1;
    matrix_utils::randPerm(static_cast<int>(perm.size()), base, perm.data());
    matrix_utils::invPerm(static_cast<int>(perm.size()), base, perm.data(),
                          iperm.data(), nthreads);
    for (size_t i = 0; i < size; i++) {
      EXPECT_EQ(iperm[perm[i] - base], i + base);
      EXPECT_EQ(perm[iperm[i] - base], i + base);
    }
  }
}

TEST(permutation, perm_vec) {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(1, 100000);

  size_t size = dist(gen);
  int base = 0;
  std::vector<int> perm(size);
  std::vector<int> vec(size);
  std::iota(vec.begin(), vec.end(), 0);
  matrix_utils::randPerm(static_cast<int>(perm.size()), base, perm.data());
  std::vector<int> perm_vec(size);
  matrix_utils::permVec(static_cast<int>(perm.size()), base, vec.data(),
                        perm.data(), perm_vec.data());
  for (size_t i = 0; i < size; i++) {
    EXPECT_EQ(perm_vec[i], vec[perm[i] - base]);
  }
}

TEST(permutation, perm_vec_threads) {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(1, 100000);

  size_t size = dist(gen);
  int base = 0;
  std::vector<int> perm(size);
  std::vector<int> vec(size);
  std::iota(vec.begin(), vec.end(), 0);
  matrix_utils::randPerm(static_cast<int>(perm.size()), base, perm.data());
  
  for (int nthreads : {1, 2, 4, 8}) {
    std::vector<int> perm_vec(size);
    matrix_utils::permVec(static_cast<int>(perm.size()), base, vec.data(),
                          perm.data(), perm_vec.data(), nthreads);
    for (size_t i = 0; i < size; i++) {
      EXPECT_EQ(perm_vec[i], vec[perm[i] - base]);
    }
  }
}

TEST(permutation, inv_perm_vec_threads) {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(1, 100000);

  size_t size = dist(gen);
  int base = 0;
  std::vector<int> perm(size);
  std::vector<int> vec(size);
  std::iota(vec.begin(), vec.end(), 0);
  matrix_utils::randPerm(static_cast<int>(perm.size()), base, perm.data());
  
  for (int nthreads : {1, 2, 4, 8}) {
    std::vector<int> inv_perm_vec(size);
    matrix_utils::invPermVec(static_cast<int>(perm.size()), base, vec.data(),
                             perm.data(), inv_perm_vec.data(), nthreads);
    for (size_t i = 0; i < size; i++) {
      EXPECT_EQ(inv_perm_vec[perm[i] - base], vec[i]);
    }
  }
}

TEST(permutation, perm_row_ptr) {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(1, 100000);
  std::uniform_int_distribution<> dist2(0, 100);
  size_t size = dist(gen);
  for (auto base : {0, 1}) {
    std::vector<int> perm(size);
    std::vector<int> ai(size + 1);
    ai[0] = base;
    for (auto i = 0; i < size; i++) {
      ai[i + 1] = ai[i] + dist2(gen);
    }
    matrix_utils::randPerm(static_cast<int>(perm.size()), base, perm.data());
    std::vector<int> perm_ai(size + 1);
    matrix_utils::permRowPtr(static_cast<int>(perm.size()), ai.data(),
                             perm.data(), perm_ai.data());
    std::vector<int> row_size(size);
    std::vector<int> perm_row_size(size);
    for (size_t i = 0; i < size; i++) {
      row_size[i] = ai[i + 1] - ai[i];
      perm_row_size[i] = perm_ai[i + 1] - perm_ai[i];
    }
    for (size_t i = 0; i < size; i++) {
      EXPECT_EQ(row_size[perm[i] - base], perm_row_size[i]);
    }
  }
}

TEST(permutation, perm_row_ptr_threads) {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(1, 100000);
  std::uniform_int_distribution<> dist2(0, 100);
  size_t size = dist(gen);
  
  for (int nthreads : {1, 2, 4, 8}) {
    for (auto base : {0, 1}) {
      std::vector<int> perm(size);
      std::vector<int> ai(size + 1);
      ai[0] = base;
      for (auto i = 0; i < size; i++) {
        ai[i + 1] = ai[i] + dist2(gen);
      }
      matrix_utils::randPerm(static_cast<int>(perm.size()), base, perm.data());
      std::vector<int> perm_ai(size + 1);
      matrix_utils::permRowPtr(static_cast<int>(perm.size()), ai.data(),
                               perm.data(), perm_ai.data(), nthreads);
      std::vector<int> row_size(size);
      std::vector<int> perm_row_size(size);
      for (size_t i = 0; i < size; i++) {
        row_size[i] = ai[i + 1] - ai[i];
        perm_row_size[i] = perm_ai[i + 1] - perm_ai[i];
      }
      for (size_t i = 0; i < size; i++) {
        EXPECT_EQ(row_size[perm[i] - base], perm_row_size[i]);
      }
    }
  }
}
