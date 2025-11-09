
#include "Cholesky.hpp"
#include "sp_ops.hpp"
#include "config.h"
#include "io.hpp"
#include "matrix_utils.hpp"
#include "permutation.hpp"
#include "precond.hpp"
#include "utils.h"
#include <cxxopts.hpp>
#include <fstream>
#include <numeric>
#include <string>

int main(int argc, char **argv) {
  cxxopts::Options options("Cholesky Example", "Example of using Cholesky");
  options.add_options()(
      "f,filename", "Matrix Market file to read",
      cxxopts::value<std::string>()->default_value("../data/symm_example.mtx"));
  options.add_options()("l,level", "Level of ILU",
                        cxxopts::value<int>()->default_value("0"));
  auto result = options.parse(argc, argv);
  std::string filename = result["filename"].as<std::string>();
  int level = result["level"].as<int>();

  std::ifstream f(filename);
  f.clear();
  f.seekg(0, std::ios::beg);
  matrix_utils::CSRMatrix<int, int, double> csr_matrix;
  matrix_utils::readMatrixMarket(f, csr_matrix);

  std::ofstream out0("mat_csr.svg");
  matrix_utils::writeSVG(csr_matrix.rows, csr_matrix.cols, csr_matrix.AI(),
                         csr_matrix.AJ(), out0);
  out0.close();
  std::vector<int> parent(csr_matrix.rows);
  std::vector<int> ancestor(csr_matrix.rows);

  factorization::EliminationTree(csr_matrix.rows, csr_matrix.AI(),
                                 csr_matrix.AJ(), parent.data(),
                                 ancestor.data());
  // for (size_t i = 0; i < csr_matrix.rows; i++) {
  //   std::cout << i + 1 << "->" << parent[i] + 1 << std::endl;
  // }

  // std::vector<int> row_size(csr_matrix.rows);
  // factorization::RowCount(csr_matrix.rows, csr_matrix.AI(), csr_matrix.AJ(),
  //                         parent.data(), row_size.data(), ancestor.data());
  // for (size_t i = 0; i < csr_matrix.rows; i++) {
  //   std::cout << i + 1 << "->" << row_size[i] << std::endl;
  // }
#ifdef USE_BOOST_LIB
  utils::printEliminationTree(csr_matrix.rows, csr_matrix.AI()[0],
                              parent.data(), "tree.dot");
#endif

  std::vector<int> perm(csr_matrix.rows);
  std::vector<int> iperm(csr_matrix.rows);
  std::vector<int> permed_parent(csr_matrix.rows);
  factorization::PostOrderNoRecur<int> postorder;
  postorder(csr_matrix.rows, 0, parent.data(), permed_parent.data(),
            perm.data(), iperm.data());
  //   {

  //     factorization::PostOrderNoRecur<int> postorder2;
  //     std::vector<int> perm2(csr_matrix.rows);
  //     std::vector<int> iperm2(csr_matrix.rows);
  //     std::vector<int> permed_parent2(csr_matrix.rows);
  //     postorder2(csr_matrix.rows, 0, parent.data(), permed_parent2.data(),
  //                perm2.data(), iperm2.data());

  // #ifdef USE_BOOST_LIB
  //     utils::printEliminationTree(csr_matrix.rows, csr_matrix.AI()[0],
  //                                 permed_parent2.data(),
  //                                 "post_order2_tree.dot");
  // #endif
  //   }
  matrix_utils::CSRMatrix<int, int, double> csr_matrix_perm;
  csr_matrix_perm.ResizeAI(csr_matrix.rows + 1);
  csr_matrix_perm.ResizeAJ(csr_matrix.NNZ());
  csr_matrix_perm.ResizeAV(csr_matrix.NNZ());
  csr_matrix_perm.ResizeDiagonal(csr_matrix.rows);
  csr_matrix_perm.rows = csr_matrix.rows;
  csr_matrix_perm.cols = csr_matrix.cols;
  matrix_utils::permuteMat<int, int, double>(csr_matrix.rows, csr_matrix.cols, perm.data(),
                           iperm.data(), csr_matrix.AI(), csr_matrix.AJ(), csr_matrix.AV(),
                           csr_matrix_perm.AI(), csr_matrix_perm.AJ(), csr_matrix_perm.AV());

  std::ofstream out1("mat_csr_perm.svg");
  matrix_utils::writeSVG(csr_matrix_perm.rows, csr_matrix_perm.cols,
                         csr_matrix_perm.AI(), csr_matrix_perm.AJ(), out1);
  out1.close();
  // #ifdef USE_BOOST_LIB
  //   utils::printEliminationTree(csr_matrix.rows, csr_matrix.AI()[0],
  //                               permed_parent.data(), "post_order_tree.dot");
  // #endif
  // {
  //   matrix_utils::CSRMatrix<int, int, double> csr_matrix_perm2;
  //   csr_matrix_perm2.ResizeAI(csr_matrix.rows + 1);
  //   csr_matrix_perm2.ResizeAJ(csr_matrix.NNZ());
  //   csr_matrix_perm2.ResizeAV(csr_matrix.NNZ());
  //   csr_matrix_perm2.ResizeDiagonal(csr_matrix.rows);
  //   csr_matrix_perm2.rows = csr_matrix.rows;
  //   csr_matrix_perm2.cols = csr_matrix.cols;
  //   matrix_utils::permuteMat(csr_matrix.rows, csr_matrix.cols, perm.data(),
  //                            iperm.data(), csr_matrix.AI(), csr_matrix.AJ(),
  //                            csr_matrix_perm2.AI(), csr_matrix_perm2.AJ(),
  //                            csr_matrix.AV(), csr_matrix_perm2.AV());
  //   std::cout << "symbolic cholesky col" << std::endl;
  //   factorization::SymbolicCholeskyCol<decltype(csr_matrix_perm2)>
  //       cholesky_symbol(5);
  //   cholesky_symbol(csr_matrix.rows, csr_matrix_perm2.AI(),
  //                   csr_matrix_perm2.AJ(), permed_parent.data(),
  //                   csr_matrix_perm2);
  // }
  // std::cout << "cholesky symbol" << std::endl;
  // matrix_utils::CSRMatrix<int, int, double> skeleton_graph, L;
  // factorization::SkeletonGraph<decltype(skeleton_graph)> sk_generator(4);
  // sk_generator(csr_matrix.rows, csr_matrix_perm.AI(), csr_matrix_perm.AJ(),
  //              permed_parent.data(), skeleton_graph);
  // factorization::SymbolicCholesky<decltype(L)> cholesky_symbol(4);
  // cholesky_symbol(csr_matrix.rows, csr_matrix_perm.AI(),
  // csr_matrix_perm.AJ(),
  //                 permed_parent.data(), skeleton_graph.AI(),
  //                 skeleton_graph.AJ(), L);
  // // std::cout << "L: " << std::endl;
  // // for (int i = 0; i < L.rows + 1; i++) {
  // //   std::cout << L.AI()[i] << " ";
  // // }
  // // std::cout << std::endl;
  // // for (int i = 0; i < L.NNZ(); i++) {
  // //   std::cout << L.AJ()[i] << " ";
  // // }
  // std::cout << std::endl;
  // std::ofstream out2("L.svg");
  // matrix_utils::writeSVG(L.rows, L.cols, L.AI(), L.AJ(), out2);
  // out2.close();

  // // std::vector<int> row_count(csr_matrix.rows + 1);
  // // std::vector<int> col_count(csr_matrix.rows + 1);
  // // row_count[0] = col_count[0] = csr_matrix_perm.Base();
  // // std::vector<int> mark(csr_matrix.rows);
  // // factorization::NNZCount(csr_matrix.rows, csr_matrix_perm.AI(),
  // //                         csr_matrix_perm.AJ(), permed_parent.data(),
  // //                         row_count.data() + 1, col_count.data() + 1,
  // //                         mark.data());
  // // std::inclusive_scan(col_count.begin(), col_count.end(),
  // col_count.begin());

  // {
  //   csr_matrix_perm.ResizeDiagonal(csr_matrix.rows);
  //   std::cout << matrix_utils::Diagonal(
  //                    csr_matrix.rows, csr_matrix_perm.AI(),
  //                    csr_matrix_perm.AJ(), csr_matrix_perm.AV(),
  //                    csr_matrix_perm.Diagonal(), static_cast<double
  //                    *>(nullptr))
  //             << std::endl;

  //   matrix_utils::CSRMatrix<int, int, double> icc, icc_transpose;
  //   matrix_utils::ICCLevelSymbolic2(csr_matrix_perm.rows,
  //   csr_matrix_perm.AI(),
  //                                   csr_matrix_perm.AJ(),
  //                                   csr_matrix_perm.Diagonal(), 100, icc);
  //   icc_transpose.ResizeAI(icc.rows + 1);
  //   icc_transpose.ResizeAJ(icc.NNZ());
  //   icc_transpose.ResizeAV(icc.NNZ());
  //   icc_transpose.ResizeDiagonal(icc.rows);
  //   matrix_utils::ParallelTranspose2(icc.rows, icc.cols, 0, icc.AI(),
  //   icc.AJ(),
  //                                    icc.AV(), icc_transpose.AI(),
  //                                    icc_transpose.AJ(), icc_transpose.AV());
  //   for (int i = 0; i < icc.rows + 1; i++) {
  //     if (L.AI()[i] != icc_transpose.AI()[i]) {
  //       std::cout << "AI[" << i << "] is not equal" << std::endl;
  //     }
  //   }
  //   for (int i = 0; i < icc.NNZ(); i++) {
  //     if (L.AJ()[i] != icc_transpose.AJ()[i]) {
  //       std::cout << "AJ[" << i << "] is not equal" << std::endl;
  //     }
  //   }
  //   // std::cout << std::endl;
  //   // for (int i = 0; i < icc.NNZ(); i++) {
  //   //   std::cout << icc_transpose.AJ()[i] << " ";
  //   // }
  //   // std::cout << std::endl;
  // }
  {
    std::cout << "icc symbolic" << std::endl;
    matrix_utils::CSRMatrix<int, int, double> L;
    matrix_utils::ICCLevelSymbolicParallel<
        matrix_utils::CSRMatrix<int, int, double>>
        icc_symbolic(20);
    icc_symbolic(csr_matrix_perm.rows, csr_matrix_perm.AI(),
                 csr_matrix_perm.AJ(), level, L);
    std::cout << "icc symbolic done" << std::endl;

    // std::ofstream out1("iccL.svg");
    // matrix_utils::writeSVG(L.rows, L.cols, L.AI(), L.AJ(), out1);
    // out1.close();
    // std::cout << "icc numeric" << std::endl;
    // matrix_utils::ICCLevelNumericFixedPoint<
    //     matrix_utils::CSRMatrix<int, int, double>>
    //     icc_numeric(10);
    // icc_numeric(csr_matrix_perm.rows, csr_matrix_perm.AI(),
    //             csr_matrix_perm.AJ(), csr_matrix_perm.AV(), L);
    // std::cout << "icc numeric done" << std::endl;
    // std::ofstream out3("iccL.mtx");
    // matrix_utils::writeMatrixMarket(L, out3);
    // out3.close();
  }
  return 0;
}