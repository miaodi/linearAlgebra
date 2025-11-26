
#include "ObjectPool.hpp"
#include "incomplete_lu.h"
#include "matrix_utils.hpp"
#include "mkl_sparse_handle.h"
#include "mkl_sparse_mat.h"
#include "permutation.hpp"
#include "sp_ops.hpp"
#include "triangle_solve.hpp"
#include "utils.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <omp.h>

TEST(transpose_and_partranspose, base0) {
  auto mat = mkl_wrapper::random_sparse(100, 16);
  mat.randomVals();
  auto t_csr = matrix_utils::AllocateCSRData(mat.cols(), mat.nnz());
  matrix_utils::SerialTranspose(
      mat.rows(), mat.cols(), mat.get_ai().get(),
      mat.get_aj().get(), mat.get_av().get(), std::get<0>(t_csr).get(),
      std::get<1>(t_csr).get(), std::get<2>(t_csr).get());

  auto tt_csr = matrix_utils::AllocateCSRData(mat.cols(), mat.nnz());
  matrix_utils::SerialTranspose(
      mat.cols(), mat.rows(), std::get<0>(t_csr).get(),
      std::get<1>(t_csr).get(), std::get<2>(t_csr).get(),
      std::get<0>(tt_csr).get(), std::get<1>(tt_csr).get(),
      std::get<2>(tt_csr).get());

  for (int i = 0; i <= mat.rows(); i++) {
    EXPECT_EQ(std::get<0>(tt_csr)[i], mat.get_ai()[i]);
  }
  for (size_t i = 0; i < mat.nnz(); i++) {
    EXPECT_EQ(std::get<1>(tt_csr)[i], mat.get_aj()[i]);
    EXPECT_EQ(std::get<2>(tt_csr)[i], mat.get_av()[i]);
  }

  auto ptt_csr = matrix_utils::AllocateCSRData(mat.cols(), mat.nnz());
  matrix_utils::ParallelTranspose(
      mat.cols(), mat.rows(), std::get<0>(t_csr).get(),
      std::get<1>(t_csr).get(), std::get<2>(t_csr).get(),
      std::get<0>(ptt_csr).get(), std::get<1>(ptt_csr).get(),
      std::get<2>(ptt_csr).get());

  for (int i = 0; i <= mat.rows(); i++) {
    EXPECT_EQ(std::get<0>(tt_csr)[i], std::get<0>(ptt_csr)[i]);
  }
  for (size_t i = 0; i < mat.nnz(); i++) {
    EXPECT_EQ(std::get<1>(tt_csr)[i], std::get<1>(ptt_csr)[i]);
    EXPECT_EQ(std::get<2>(tt_csr)[i], std::get<2>(ptt_csr)[i]);
  }

  auto pt2_csr = matrix_utils::AllocateCSRData(mat.cols(), mat.nnz());
  matrix_utils::ParallelTranspose2(
      mat.rows(), mat.cols(), mat.get_ai().get(),
      mat.get_aj().get(), mat.get_av().get(), std::get<0>(pt2_csr).get(),
      std::get<1>(pt2_csr).get(), std::get<2>(pt2_csr).get());

  for (int i = 0; i <= mat.rows(); i++) {
    EXPECT_EQ(std::get<0>(t_csr)[i], std::get<0>(pt2_csr)[i]);
  }
  for (size_t i = 0; i < mat.nnz(); i++) {
    EXPECT_EQ(std::get<1>(t_csr)[i], std::get<1>(pt2_csr)[i]);
    EXPECT_EQ(std::get<2>(t_csr)[i], std::get<2>(pt2_csr)[i]);
  }
}

TEST(transpose_and_partranspose, base1) {
  auto mat = mkl_wrapper::random_sparse(10, 3);
  mat.randomVals();
  mat.to_one_based();

  //   std::ofstream myfile;
  //   myfile.open("origin.svg");
  //   mat.print_svg(myfile);
  //   myfile.close();

  auto t_csr = matrix_utils::AllocateCSRData(mat.cols(), mat.nnz());
  matrix_utils::SerialTranspose(
      mat.rows(), mat.cols(), mat.get_ai().get(),
      mat.get_aj().get(), mat.get_av().get(), std::get<0>(t_csr).get(),
      std::get<1>(t_csr).get(), std::get<2>(t_csr).get());

  auto tt_csr = matrix_utils::AllocateCSRData(mat.cols(), mat.nnz());
  matrix_utils::SerialTranspose(
      mat.cols(), mat.rows(), std::get<0>(t_csr).get(),
      std::get<1>(t_csr).get(), std::get<2>(t_csr).get(),
      std::get<0>(tt_csr).get(), std::get<1>(tt_csr).get(),
      std::get<2>(tt_csr).get());

  for (int i = 0; i <= mat.rows(); i++) {
    EXPECT_EQ(std::get<0>(tt_csr)[i], mat.get_ai()[i]);
  }
  for (size_t i = 0; i < mat.nnz(); i++) {
    EXPECT_EQ(std::get<1>(tt_csr)[i], mat.get_aj()[i]);
    EXPECT_EQ(std::get<2>(tt_csr)[i], mat.get_av()[i]);
  }

  auto pt_csr = matrix_utils::AllocateCSRData(mat.cols(), mat.nnz());
  matrix_utils::ParallelTranspose(
      mat.rows(), mat.cols(), mat.get_ai().get(),
      mat.get_aj().get(), mat.get_av().get(), std::get<0>(pt_csr).get(),
      std::get<1>(pt_csr).get(), std::get<2>(pt_csr).get());

  for (int i = 0; i <= mat.rows(); i++) {
    EXPECT_EQ(std::get<0>(t_csr)[i], std::get<0>(pt_csr)[i]);
  }
  for (size_t i = 0; i < mat.nnz(); i++) {
    EXPECT_EQ(std::get<1>(t_csr)[i], std::get<1>(pt_csr)[i]);
    EXPECT_EQ(std::get<2>(t_csr)[i], std::get<2>(pt_csr)[i]);
  }

  auto pt2_csr = matrix_utils::AllocateCSRData(mat.cols(), mat.nnz());
  matrix_utils::ParallelTranspose2(
      mat.rows(), mat.cols(), mat.get_ai().get(),
      mat.get_aj().get(), mat.get_av().get(), std::get<0>(pt2_csr).get(),
      std::get<1>(pt2_csr).get(), std::get<2>(pt2_csr).get());

  // for (int i = 0; i <= mat.rows(); i++) {
  //   EXPECT_EQ(std::get<0>(t_csr)[i], std::get<0>(pt2_csr)[i]);
  // }
  // for (size_t i = 0; i < mat.nnz(); i++) {
  //   EXPECT_EQ(std::get<1>(t_csr)[i], std::get<1>(pt2_csr)[i]);
  //   EXPECT_EQ(std::get<2>(t_csr)[i], std::get<2>(pt2_csr)[i]);
  // }
  //   mkl_wrapper::mkl_sparse_mat t_mat(mat.cols(), mat.rows(),
  //   std::get<0>(pt_csr),
  //                                     std::get<1>(pt_csr),
  //                                     std::get<2>(pt_csr),
  //                                     SPARSE_INDEX_BASE_ONE);
  //   myfile.open("transpose.svg");
  //   t_mat.print_svg(myfile);
  //   myfile.close();
}

TEST(transpose_and_partranspose, no_av) {
  auto mat = mkl_wrapper::random_sparse(10, 3);
  mat.randomVals();
  mat.to_one_based();

  //   std::ofstream myfile;
  //   myfile.open("origin.svg");
  //   mat.print_svg(myfile);
  //   myfile.close();

  auto t_csr = matrix_utils::AllocateCSRData(mat.cols(), mat.nnz());
  matrix_utils::SerialTranspose(
      mat.rows(), mat.cols(), mat.get_ai().get(),
      mat.get_aj().get(), (double *)nullptr, std::get<0>(t_csr).get(),
      std::get<1>(t_csr).get(), std::get<2>(t_csr).get());
}

TEST(SplitLDU, base0) {
  omp_set_num_threads(5);
  auto mat = mkl_wrapper::random_sparse(1000, 32);
  mat.randomVals();

  matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> L, U;
  std::vector<double> D;
  matrix_utils::SplitLDU(mat.rows(), (int)mat.mkl_base(), mat.get_ai().get(),
                         mat.get_aj().get(), mat.get_av().get(), L, D, U);
  mkl_wrapper::mkl_sparse_mat matL(mat.rows(), mat.rows(), L.ai, L.aj, L.av);
  mkl_wrapper::mkl_sparse_mat matU(mat.rows(), mat.rows(), U.ai, U.aj, U.av);
  auto tmp = mkl_wrapper::mkl_sparse_sum(matU, mat, -1.);
  auto matD = mkl_wrapper::mkl_sparse_sum(matL, tmp, -1.);
  matD.prune(1e-11);

  std::vector<double> ones(mat.rows(), 1);
  std::vector<double> diag(mat.rows());

  matD.mult_vec(ones.data(), diag.data());
  for (int i = 0; i < mat.rows(); i++) {
    EXPECT_NEAR(diag[i], D[i], 2e-11);
  }

  // std::ofstream myfile;
  // myfile.open("origin.svg");
  // mat.print_svg(myfile);
  // myfile.close();

  // myfile.open("L.svg");
  // matL.print_svg(myfile);
  // myfile.close();

  // myfile.open("U.svg");
  // matU.print_svg(myfile);
  // myfile.close();

  // myfile.open("D.svg");
  // matD.print_svg(myfile);
  // myfile.close();
}

TEST(SplitLDU, base1) {
  omp_set_num_threads(5);
  auto mat = mkl_wrapper::random_sparse(1000, 32);
  mat.randomVals();
  mat.to_one_based();

  matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> L, U;
  std::vector<double> D;
  matrix_utils::SplitLDU(mat.rows(), (int)mat.mkl_base(), mat.get_ai().get(),
                         mat.get_aj().get(), mat.get_av().get(), L, D, U);
  mkl_wrapper::mkl_sparse_mat matL(mat.rows(), mat.rows(), L.ai, L.aj, L.av,
                                   SPARSE_INDEX_BASE_ONE);
  mkl_wrapper::mkl_sparse_mat matU(mat.rows(), mat.rows(), U.ai, U.aj, U.av,
                                   SPARSE_INDEX_BASE_ONE);
  auto tmp = mkl_wrapper::mkl_sparse_sum(matU, mat, -1.);
  auto matD = mkl_wrapper::mkl_sparse_sum(matL, tmp, -1.);
  matD.prune(1e-11);

  std::vector<double> ones(mat.rows(), 1);
  std::vector<double> diag(mat.rows());

  matD.mult_vec(ones.data(), diag.data());
  for (int i = 0; i < mat.rows(); i++) {
    EXPECT_NEAR(diag[i], D[i], 2e-11);
  }

  // std::ofstream myfile;
  // myfile.open("origin.svg");
  // mat.print_svg(myfile);
  // myfile.close();

  // myfile.open("L.svg");
  // matL.print_svg(myfile);
  // myfile.close();

  // myfile.open("U.svg");
  // matU.print_svg(myfile);
  // myfile.close();

  // myfile.open("D.svg");
  // matD.print_svg(myfile);
  // myfile.close();
}

TEST(LowerTri, small) {
  omp_set_num_threads(5);
  for (int i = 0; i < 20; i++) {
    auto mat = mkl_wrapper::random_sparse(50, 13);
    mat.randomVals();
    if (i % 2 == 0)
      mat.to_one_based();
    else
      mat.to_zero_based();
    MKL_INT base = mat.mkl_base();
    matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> T;
    matrix_utils::SplitTriangle<matrix_utils::TriangularMatrix::L>(
        mat.rows(), base, mat.get_ai().get(), mat.get_aj().get(),
        mat.get_av().get(), T);

    mkl_wrapper::mkl_sparse_mat triangle(mat.rows(), mat.rows(), T.ai, T.aj,
                                         T.av, mat.mkl_base());
    for (int i = 0; i < triangle.rows(); i++) {
      if (triangle.get_ai()[i] != triangle.get_ai()[i + 1]) {
        int last_index = triangle.get_aj()[triangle.get_ai()[i + 1] - base - 1];
        EXPECT_TRUE(last_index - base <= i);
      }
    }
    for (int i = 0; i < triangle.rows(); i++) {
      for (int j = triangle.get_ai()[i] - base;
           j < triangle.get_ai()[i + 1] - base; j++) {
        int mat_j =
            j - (triangle.get_ai()[i] - base) + (mat.get_ai()[i] - base);
        EXPECT_EQ(triangle.get_aj()[j], mat.get_aj()[mat_j]);
        EXPECT_EQ(triangle.get_av()[j], mat.get_av()[mat_j]);
      }
    }
  }
}

TEST(UpperTri, small) {
  omp_set_num_threads(5);
  for (int i = 0; i < 20; i++) {
    auto mat = mkl_wrapper::random_sparse(50, 13);
    mat.randomVals();
    if (i % 2 == 0)
      mat.to_one_based();
    else
      mat.to_zero_based();
    MKL_INT base = mat.mkl_base();
    matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> T;
    matrix_utils::SplitTriangle<matrix_utils::TriangularMatrix::U>(
        mat.rows(), base, mat.get_ai().get(), mat.get_aj().get(),
        mat.get_av().get(), T);

    mkl_wrapper::mkl_sparse_mat triangle(mat.rows(), mat.rows(), T.ai, T.aj,
                                         T.av, mat.mkl_base());
    for (int i = 0; i < triangle.rows(); i++) {
      if (triangle.get_ai()[i] != triangle.get_ai()[i + 1]) {
        EXPECT_TRUE(triangle.get_aj()[triangle.get_ai()[i] - base] - base >= i);
      }
    }
    for (int i = 0; i < triangle.rows(); i++) {
      for (int j = 1; j <= triangle.get_ai()[i + 1] - triangle.get_ai()[i];
           j++) {
        EXPECT_EQ(triangle.get_aj()[triangle.get_ai()[i + 1] - base - j],
                  mat.get_aj()[mat.get_ai()[i + 1] - base - j]);
        EXPECT_EQ(triangle.get_av()[triangle.get_ai()[i + 1] - base - j],
                  mat.get_av()[mat.get_ai()[i + 1] - base - j]);
      }
    }
  }
}

TEST(UpperTrigToFull, small) {
  omp_set_num_threads(5);
  for (int i = 0; i < 20; i++) {
    auto mat = mkl_wrapper::random_sparse(50, 13);
    mat.randomVals();
    if (i % 2 == 0)
      mat.to_one_based();
    else
      mat.to_zero_based();
    matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> U, F;
    matrix_utils::SplitTriangle<matrix_utils::TriangularMatrix::U>(
        mat.rows(), mat.mkl_base(), mat.get_ai().get(), mat.get_aj().get(),
        mat.get_av().get(), U);
    matrix_utils::TriangularToFull<matrix_utils::TriangularMatrix::U>(
        U.rows, U.Base(), U.AI(), U.AJ(), U.AV(), F);

    mkl_wrapper::mkl_sparse_mat full(mat.rows(), mat.rows(), F.ai, F.aj, F.av,
                                     mat.mkl_base());
    mkl_wrapper::mkl_sparse_mat transpose_full = full;
    transpose_full.transpose();
    for (int i = 0; i <= full.rows(); i++) {
      EXPECT_EQ(full.get_ai()[i], transpose_full.get_ai()[i]);
    }
  }
}

TEST(UpperTrigToFull, medium) {
  omp_set_num_threads(10);
  for (int i = 0; i < 10; i++) {
    auto mat = mkl_wrapper::random_sparse(10000, 30);
    mat.randomVals();
    if (i % 2 == 0)
      mat.to_one_based();
    else
      mat.to_zero_based();
    matrix_utils::CSRMatrix<MKL_INT, MKL_INT, double> U, F;
    matrix_utils::SplitTriangle<matrix_utils::TriangularMatrix::U>(
        mat.rows(), mat.mkl_base(), mat.get_ai().get(), mat.get_aj().get(),
        mat.get_av().get(), U);
    matrix_utils::TriangularToFull<matrix_utils::TriangularMatrix::U>(
        U.rows, U.Base(), U.AI(), U.AJ(), U.AV(), F);

    mkl_wrapper::mkl_sparse_mat full(mat.rows(), mat.rows(), F.ai, F.aj, F.av,
                                     mat.mkl_base());
    mkl_wrapper::mkl_sparse_mat transpose_full = full;
    transpose_full.transpose();
    for (int i = 0; i <= full.rows(); i++) {
      EXPECT_EQ(full.get_ai()[i], transpose_full.get_ai()[i]);
    }
  }
}

TEST(ObjectPool, vector) {
  utils::ObjectPool<std::vector<int>> pool;
  using DataType = decltype(pool)::value_type;
  pool.setObjectPrep([](DataType *obj) {
    obj->reserve(20);
    obj->clear();
  });
  auto obj1 = pool.acquire();
  EXPECT_EQ(obj1->capacity(), 20);
  EXPECT_EQ(pool.size(), 0);
  {
    auto obj2 = pool.acquire();
    EXPECT_EQ(obj2->capacity(), 20);
    EXPECT_EQ(pool.size(), 0);
    {
      auto obj3 = pool.acquire();
      EXPECT_EQ(obj3->capacity(), 20);
      EXPECT_EQ(pool.size(), 0);
      obj3->reserve(50);
    }
    EXPECT_EQ(pool.size(), 1);
    {
      auto obj4 = pool.acquire();
      EXPECT_EQ(obj4->capacity(), 50);
      EXPECT_EQ(pool.size(), 0);
    }
  }
}

TEST(TopologicalSort, L)
{
    const int size = 1000;
    const int nnz_per_row = 20;
    for ( int base = 0; base <= 1; base++ )
    {
        matrix_utils::CSRMatrix<std::int32_t, std::int32_t, double> L;
        matrix_utils::RandomL( size, base, nnz_per_row, L );
        // test random L
        for ( int i = 0; i < size; i++ )
        {
            EXPECT_TRUE( std::is_sorted( L.AJ() + L.AI()[i] - base,
                                         L.AJ() + L.AI()[i + 1] - base ) );
        }
        std::vector<std::int32_t> perm( size );
        std::vector<std::int32_t> prefix( size + 1 );
        matrix_utils::KahnSerial<std::int32_t, std::int32_t> kahn;
        std::int32_t level =
            kahn( size, L.AI(), L.AJ(), perm.data(), prefix.data(), false );
        EXPECT_EQ( prefix[0], base );
        std::map<std::int32_t, std::int32_t> idx_map;
        for ( int i = 0; i < size; i++ )
        {
            idx_map[perm[i] - base] = i;
        }
        // check topological order
        for ( int i = 0; i < size; i++ )
        {
            for ( int j = L.AI()[i] - base; j < L.AI()[i + 1] - base; j++ )
            {
                EXPECT_TRUE( idx_map[L.AJ()[j] - base] < idx_map[i] );
            }
        }
        EXPECT_EQ( prefix[level] - base, size );

        // test parallel kahn
        std::vector<std::int32_t> perm_parallel( size );
        std::vector<std::int32_t> prefix_parallel( size + 1 );
        matrix_utils::KahnParallel<std::int32_t, std::int32_t> kahn_parallel(
            5 );
        std::int32_t level_parallel = kahn_parallel(
            size, L.AI(), L.AJ(), perm_parallel.data(), prefix_parallel.data(), false );
        EXPECT_EQ( prefix_parallel[0], base );
        EXPECT_EQ( prefix_parallel[level_parallel] - base, size );
        EXPECT_EQ( level, level_parallel );
        for ( int i = 0; i < level; i++ )
        {
            EXPECT_EQ( prefix[i + 1], prefix_parallel[i + 1] );
            std::sort( perm.data() + prefix[i] - base, perm.data() + prefix[i + 1] - base );
            std::sort( perm_parallel.data() + prefix_parallel[i] - base,
                       perm_parallel.data() + prefix_parallel[i + 1] - base );
            for ( int j = prefix[i] - base; j < prefix[i + 1] - base; j++ )
            {
                EXPECT_EQ( perm[j], perm_parallel[j] );
            }
        }

        // test topological sort
        std::vector<std::int32_t> perm_2( size );
        std::vector<std::int32_t> prefix_2( size + 1 );
        matrix_utils::TopologicalSort2<int, int, matrix_utils::TriangularMatrix::L> topSort;
        std::int32_t level_2 =
            topSort( size, L.AI(), L.AJ(), perm_2.data(), prefix_2.data(), false );
        EXPECT_EQ( prefix_2[0], base );
        EXPECT_EQ( prefix_2[level_2] - base, size );
        EXPECT_EQ( level, level_2 );

        // check topological order
        idx_map.clear();
        for ( int i = 0; i < size; i++ )
        {
            idx_map[perm_2[i] - base] = i;
        }

        for ( int i = 0; i < size; i++ )
        {
            for ( int j = L.AI()[i] - base; j < L.AI()[i + 1] - base; j++ )
            {
                EXPECT_TRUE( idx_map[L.AJ()[j] - base] < idx_map[i] );
            }
        }

        for ( int i = 0; i < level; i++ )
        {
            EXPECT_EQ( prefix[i + 1], prefix_2[i + 1] );
            std::sort( perm_2.data() + prefix_2[i] - base,
                       perm_2.data() + prefix_2[i + 1] - base );
            for ( int j = prefix[i] - base; j < prefix[i + 1] - base; j++ )
            {
                EXPECT_EQ( perm[j], perm_2[j] );
            }
        }
    }
}

TEST(TopologicalSort, U)
{
    const int size = 1003;
    const int nnz_per_row = 20;
    for ( int base = 0; base <= 1; base++ )
    {
        matrix_utils::CSRMatrix<std::int32_t, std::int32_t, double> U;
        matrix_utils::RandomU( size, base, nnz_per_row, U );
        // test random U
        for ( int i = 0; i < size; i++ )
        {
            EXPECT_TRUE( std::is_sorted( U.AJ() + U.AI()[i] - base,
                                         U.AJ() + U.AI()[i + 1] - base ) );
        }
        std::vector<std::int32_t> perm( size );
        std::vector<std::int32_t> prefix( size + 1 );
        matrix_utils::KahnSerial<std::int32_t, std::int32_t> kahn;
        std::int32_t level =
            kahn.operator()( size, U.AI(), U.AJ(), perm.data(), prefix.data(), false );
        EXPECT_EQ( prefix[0], base );
        std::map<std::int32_t, std::int32_t> idx_map;
        for ( int i = 0; i < size; i++ )
        {
            idx_map[perm[i] - base] = i;
        }
        // check reverse topological order
        for ( int i = 0; i < size; i++ )
        {
            for ( int j = U.AI()[i] - base; j < U.AI()[i + 1] - base; j++ )
            {
                EXPECT_TRUE( idx_map[U.AJ()[j] - base] < idx_map[i] );
            }
        }
        EXPECT_EQ( prefix[level] - base, size );

        // test parallel kahn
        std::vector<std::int32_t> perm_parallel( size );
        std::vector<std::int32_t> prefix_parallel( size + 1 );
        matrix_utils::KahnParallel<std::int32_t, std::int32_t> kahn_parallel(
            5 );
        std::int32_t level_parallel = kahn_parallel.operator()(
            size, U.AI(), U.AJ(), perm_parallel.data(), prefix_parallel.data(), false );
        EXPECT_EQ( prefix_parallel[0], base );
        EXPECT_EQ( prefix_parallel[level_parallel] - base, size );
        EXPECT_EQ( level, level_parallel );
        for ( int i = 0; i < level; i++ )
        {
            EXPECT_EQ( prefix[i + 1], prefix_parallel[i + 1] );
            std::sort( perm.data() + prefix[i] - base, perm.data() + prefix[i + 1] - base );
            std::sort( perm_parallel.data() + prefix_parallel[i] - base,
                       perm_parallel.data() + prefix_parallel[i + 1] - base );
            for ( int j = prefix[i] - base; j < prefix[i + 1] - base; j++ )
            {
                EXPECT_EQ( perm[j], perm_parallel[j] );
            }
        }

        // test topological sort
        std::vector<std::int32_t> perm_2( size );
        std::vector<std::int32_t> prefix_2( size + 1 );
        matrix_utils::TopologicalSort2<int, int, matrix_utils::TriangularMatrix::U> topSort;
        std::int32_t level_2 = topSort.operator()(
            size, U.AI(), U.AJ(), perm_2.data(), prefix_2.data(), false );
        EXPECT_EQ( prefix_2[0], base );
        EXPECT_EQ( prefix_2[level_2] - base, size );
        EXPECT_EQ( level, level_2 );

        // check order consistency
        idx_map.clear();
        for ( int i = 0; i < size; i++ )
        {
            idx_map[perm_2[i] - base] = i;
        }

        for ( int i = 0; i < size; i++ )
        {
            for ( int j = U.AI()[i] - base; j < U.AI()[i + 1] - base; j++ )
            {
                EXPECT_TRUE( idx_map[U.AJ()[j] - base] < idx_map[i] );
            }
        }

        for ( int i = 0; i < level; i++ )
        {
            EXPECT_EQ( prefix[i + 1], prefix_2[i + 1] );
            std::sort( perm_2.data() + prefix_2[i] - base,
                       perm_2.data() + prefix_2[i + 1] - base );
            for ( int j = prefix[i] - base; j < prefix[i + 1] - base; j++ )
            {
                EXPECT_EQ( perm[j], perm_2[j] );
            }
        }
    }
}
