#include "mkl_sparse_mat.h"
#include <gtest/gtest.h>
#include <memory>
// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}
using namespace mkl_wrapper;

/*
A = 1 2 3      B = 1 0 0
    0 4 5          2 4 0
    0 0 6          3 5 6
*/
TEST(sparse_matrix, add) {
  std::shared_ptr<MKL_INT[]> aiA(new MKL_INT[4]{0, 3, 5, 6});
  std::shared_ptr<MKL_INT[]> ajA(new MKL_INT[6]{0, 1, 2, 1, 2, 2});
  std::shared_ptr<double[]> avA(new double[6]{1, 2, 3, 4, 5, 6});

  std::shared_ptr<MKL_INT[]> aiB(new MKL_INT[4]{0, 1, 3, 6});
  std::shared_ptr<MKL_INT[]> ajB(new MKL_INT[6]{0, 0, 1, 0, 1, 2});
  std::shared_ptr<double[]> avB(new double[6]{1, 2, 4, 3, 5, 6});

  std::shared_ptr<MKL_INT[]> aiC(new MKL_INT[4]{0, 3, 6, 9});
  std::shared_ptr<MKL_INT[]> ajC(new MKL_INT[9]{0, 1, 2, 0, 1, 2, 0, 1, 2});
  std::shared_ptr<double[]> avC(new double[9]{2, 2, 3, 2, 8, 5, 3, 5, 12});

  mkl_wrapper::mkl_sparse_mat A(3, 3, aiA, ajA, avA);
  mkl_wrapper::mkl_sparse_mat B(3, 3, aiB, ajB, avB);
  auto C = mkl_sparse_sum(A, B);

  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(aiC[i], C.get_ai()[i]);
  }

  for (int i = 0; i < 9; i++) {
    EXPECT_EQ(ajC[i], C.get_aj()[i]);
    EXPECT_EQ(avC[i], C.get_av()[i]);
  }
}

/*
A = 1 2 3      B = 1 0 0
    0 4 5          2 4 0
    0 0 6          3 5 6
*/
TEST(sparse_matrix, mult_mat) {
  std::shared_ptr<MKL_INT[]> aiA(new MKL_INT[4]{0, 3, 5, 6});
  std::shared_ptr<MKL_INT[]> ajA(new MKL_INT[6]{0, 1, 2, 1, 2, 2});
  std::shared_ptr<double[]> avA(new double[6]{1, 2, 3, 4, 5, 6});

  std::shared_ptr<MKL_INT[]> aiB(new MKL_INT[4]{0, 1, 3, 6});
  std::shared_ptr<MKL_INT[]> ajB(new MKL_INT[6]{0, 0, 1, 0, 1, 2});
  std::shared_ptr<double[]> avB(new double[6]{1, 2, 4, 3, 5, 6});

  std::shared_ptr<MKL_INT[]> aiC(new MKL_INT[4]{0, 3, 6, 9});
  std::shared_ptr<MKL_INT[]> ajC(new MKL_INT[9]{0, 1, 2, 0, 1, 2, 0, 1, 2});
  std::shared_ptr<double[]> avC(
      new double[9]{14, 23, 18, 23, 41, 30, 18, 30, 36});

  mkl_wrapper::mkl_sparse_mat A(3, 3, aiA, ajA, avA);
  mkl_wrapper::mkl_sparse_mat B(3, 3, aiB, ajB, avB);
  auto C = mkl_sparse_mult(A, B);

  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(aiC[i], C.get_ai()[i]);
  }

  for (int i = 0; i < 9; i++) {
    EXPECT_EQ(ajC[i], C.get_aj()[i]);
    EXPECT_EQ(avC[i], C.get_av()[i]);
  }

  std::shared_ptr<MKL_INT[]> aiCT(new MKL_INT[4]{0, 1, 3, 6});
  std::shared_ptr<MKL_INT[]> ajCT(new MKL_INT[6]{0, 0, 1, 0, 1, 2});
  std::shared_ptr<double[]> avCT(new double[6]{1, 10, 16, 31, 50, 36});

  auto CT = mkl_sparse_mult(A, B, SPARSE_OPERATION_TRANSPOSE);

  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(aiCT[i], CT.get_ai()[i]);
  }

  for (int i = 0; i < 6; i++) {
    EXPECT_EQ(ajCT[i], CT.get_aj()[i]);
    EXPECT_EQ(avCT[i], CT.get_av()[i]);
  }
}

/*
A = 1 2 3     
    0 4 5    
    0 0 6    
*/
TEST(sparse_matrix, mult_vec) {
  std::shared_ptr<MKL_INT[]> aiA(new MKL_INT[4]{0, 3, 5, 6});
  std::shared_ptr<MKL_INT[]> ajA(new MKL_INT[6]{0, 1, 2, 1, 2, 2});
  std::shared_ptr<double[]> avA(new double[6]{1, 2, 3, 4, 5, 6});

  mkl_wrapper::mkl_sparse_mat A(3, 3, aiA, ajA, avA);

  std::vector<double> rhs{1, 2, 3};
  std::vector<double> x(3);
  A.mult_vec(rhs.data(), x.data());
  for(auto i:x){
    std::cout<<i<<std::endl;
  }
  A.to_one_based();
  A.mult_vec(rhs.data(), x.data());
  for(auto i:x){
    std::cout<<i<<std::endl;
  }
}