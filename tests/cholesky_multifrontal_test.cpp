#include "cholesky_multifrontal.hpp"
#include "cholesky_symbolic.hpp"
#include "matrix_utils.hpp"
#include "tree.hpp"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

using CSRMatrixType = matrix_utils::CSRMatrix<int, int, double>;

namespace {

CSRMatrixType makeFullSymmetricMatrix(const int base) {
  CSRMatrixType matrix;
  matrix.rows = 5;
  matrix.cols = 5;
  matrix.ResizeAI(6);
  matrix.ResizeAJ(17);
  matrix.ResizeAV(17);

  const std::vector<int> ai{base,      base + 3,  base + 6,
                            base + 10, base + 14, base + 17};
  const std::vector<int> aj{base,     base + 1, base + 2, base,
                            base + 1, base + 3, base,     base + 2,
                            base + 3, base + 4, base + 1, base + 2,
                            base + 3, base + 4, base + 2, base + 3,
                            base + 4};
  const std::vector<double> av{6.0, 0.2, 0.1, 0.2, 7.0, 0.3,
                               0.1, 8.0, 0.4, 0.2, 0.3, 0.4,
                               9.0, 0.5, 0.2, 0.5, 10.0};

  std::copy(ai.begin(), ai.end(), matrix.AI());
  std::copy(aj.begin(), aj.end(), matrix.AJ());
  std::copy(av.begin(), av.end(), matrix.AV());
  return matrix;
}

CSRMatrixType makeUpperMatrixFromFull(const CSRMatrixType &full) {
  const int base = full.Base();
  CSRMatrixType upper;
  upper.rows = full.rows;
  upper.cols = full.cols;

  std::vector<int> ai(static_cast<std::size_t>(full.rows) + 1, base);
  std::vector<int> aj;
  std::vector<double> av;
  for (int row = 0; row < full.rows; row++) {
    for (int pos = full.AI()[row] - base; pos < full.AI()[row + 1] - base;
         pos++) {
      if (full.AJ()[pos] >= row + base) {
        aj.push_back(full.AJ()[pos]);
        av.push_back(full.AV()[pos]);
      }
    }
    ai[static_cast<std::size_t>(row) + 1] =
        static_cast<int>(aj.size()) + base;
  }

  upper.ResizeAI(ai.size());
  upper.ResizeAJ(aj.size());
  upper.ResizeAV(av.size());
  std::copy(ai.begin(), ai.end(), upper.AI());
  std::copy(aj.begin(), aj.end(), upper.AJ());
  std::copy(av.begin(), av.end(), upper.AV());
  return upper;
}

Eigen::MatrixXd toDenseSymmetric(const CSRMatrixType &matrix) {
  const int base = matrix.Base();
  Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(matrix.rows, matrix.cols);
  for (int row = 0; row < matrix.rows; row++) {
    for (int pos = matrix.AI()[row] - base; pos < matrix.AI()[row + 1] - base;
         pos++) {
      const int col = matrix.AJ()[pos] - base;
      dense(row, col) = matrix.AV()[pos];
      dense(col, row) = matrix.AV()[pos];
    }
  }
  return dense;
}

Eigen::MatrixXd toDenseLowerFactor(const CSRMatrixType &L) {
  const int base = L.Base();
  Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(L.rows, L.cols);
  for (int col = 0; col < L.cols; col++) {
    for (int pos = L.AI()[col] - base; pos < L.AI()[col + 1] - base; pos++) {
      dense(L.AJ()[pos] - base, col) = L.AV()[pos];
    }
  }
  return dense;
}

void expectFactorReconstructsMatrix(const CSRMatrixType &A,
                                    const CSRMatrixType &L) {
  const Eigen::MatrixXd dense_a = toDenseSymmetric(A);
  const Eigen::MatrixXd dense_l = toDenseLowerFactor(L);
  const Eigen::MatrixXd reconstructed = dense_l * dense_l.transpose();
  ASSERT_EQ(reconstructed.rows(), dense_a.rows());
  ASSERT_EQ(reconstructed.cols(), dense_a.cols());
  for (int row = 0; row < dense_a.rows(); row++) {
    for (int col = 0; col < dense_a.cols(); col++) {
      EXPECT_NEAR(reconstructed(row, col), dense_a(row, col), 1e-12)
          << "entry (" << row << ", " << col << ")";
    }
  }
}

} // namespace

TEST(MultifrontalCholesky, ReusesSymbolicV3ForFullCsr) {
  const auto matrix = makeFullSymmetricMatrix(0);
  std::vector<int> parent(matrix.rows);
  std::vector<int> ancestor(matrix.rows);
  graph::eliminationTree(matrix.rows, matrix.AI(), matrix.AJ(), parent.data(),
                         ancestor.data());

  factorization::SymbolicCholeskyColV3<CSRMatrixType> symbolic(1);
  CSRMatrixType L;
  ASSERT_TRUE(symbolic.apply(matrix.rows, matrix.AI(), matrix.AJ(),
                             parent.data(), L));

  std::vector<int> diagpos(matrix.rows);
  ASSERT_TRUE(matrix_utils::Diagonal(matrix.rows, matrix.AI(), matrix.AJ(),
                                     matrix.AV(), diagpos.data(),
                                     static_cast<double *>(nullptr)));

  factorization::MultifrontalCholesky<CSRMatrixType> numeric;
  ASSERT_TRUE(numeric.apply(matrix.rows, diagpos.data(), matrix.AI() + 1,
                            matrix.AJ(), matrix.AV(),
                            symbolic.eliminationTree(), L));

  expectFactorReconstructsMatrix(matrix, L);
}

TEST(MultifrontalCholesky, ReusesSymbolicV3ForUpperCsr) {
  const auto full = makeFullSymmetricMatrix(0);
  const auto upper = makeUpperMatrixFromFull(full);
  std::vector<int> parent(full.rows);
  std::vector<int> ancestor(full.rows);
  graph::eliminationTree(full.rows, full.AI(), full.AJ(), parent.data(),
                         ancestor.data());

  factorization::SymbolicCholeskyColV3<CSRMatrixType> symbolic(1);
  CSRMatrixType L;
  ASSERT_TRUE(symbolic.apply(upper.rows, upper.AI(), upper.AJ(), parent.data(),
                             L));

  factorization::MultifrontalCholesky<CSRMatrixType> numeric;
  ASSERT_TRUE(numeric.apply(upper.rows, upper.AI(), upper.AI() + 1,
                            upper.AJ(), upper.AV(), symbolic.eliminationTree(),
                            L));

  expectFactorReconstructsMatrix(upper, L);
}

TEST(MultifrontalCholesky, AcceptsExplicitBeginPointers) {
  const auto matrix = makeFullSymmetricMatrix(1);
  std::vector<int> parent(matrix.rows);
  std::vector<int> ancestor(matrix.rows);
  graph::eliminationTree(matrix.rows, matrix.AI(), matrix.AJ(), parent.data(),
                         ancestor.data());

  factorization::SymbolicCholeskyColV3<CSRMatrixType> symbolic(1);
  CSRMatrixType L;
  ASSERT_TRUE(symbolic.apply(matrix.rows, matrix.AI(), matrix.AJ(),
                             parent.data(), L));

  std::vector<int> diagpos(matrix.rows);
  ASSERT_TRUE(matrix_utils::Diagonal(matrix.rows, matrix.AI(), matrix.AJ(),
                                     matrix.AV(), diagpos.data(),
                                     static_cast<double *>(nullptr)));

  factorization::MultifrontalCholesky<CSRMatrixType> numeric;
  ASSERT_TRUE(numeric.apply(matrix.rows, diagpos.data(), matrix.AI() + 1,
                            matrix.AJ(), matrix.AV(), symbolic.eliminationTree(),
                            L));

  expectFactorReconstructsMatrix(matrix, L);
}
