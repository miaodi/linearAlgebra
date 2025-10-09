#include "UnsymmReordering.hpp"
#include "io.hpp"
#include "matrix_utils.hpp"
#include "permutation.hpp"
#include <cxxopts.hpp>
#include <fstream>
#include <mkl.h>
#include <string>
#include <vector>
//--------------------------------------------------------------------
// Build CSC (column→rows) from CSR (row→columns)
void buildCSC_fromCSR(int n,
                      const int *row_ptr, // length n+1
                      const int *col_ind, // length nnz
                      int *col_ptr,       // output length n+1
                      int *row_of_col     // output length nnz
) {
  int nnz = row_ptr[n];
  // zero counts
  for (int j = 0; j <= n; ++j)
    col_ptr[j] = 0;

  // Pass 1: count nnz per column
  for (int p = 0; p < nnz; ++p) {
    int j = col_ind[p];
    ++col_ptr[j + 1];
  }
  // Prefix sum
  for (int j = 0; j < n; ++j)
    col_ptr[j + 1] += col_ptr[j];

  // Temp copy for insertion heads
  std::vector<int> next(col_ptr, col_ptr + n + 1);

  // Pass 2: fill row indices for each column
  for (int i = 0; i < n; ++i) {
    for (int p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
      int j = col_ind[p];
      int q = next[j]++;
      row_of_col[q] = i;
    }
  }
}
//--------------------------------------------------------------------
// Flip edges along augmenting path
void augment_path(int i, int j,
                  const int *parent_col, // length n
                  int *match_row,        // length n
                  int *match_col         // length n
) {
  while (true) {
    int i_prev = match_col[j]; // old mate of column j (-1 at root)
    match_row[i] = j;
    match_col[j] = i;

    int jp = parent_col[j];
    if (jp == -1)
      break; // reached root

    // continue flipping upward
    i = i_prev;
    j = jp;
  }
}

//--------------------------------------------------------------------
// Search for one augmenting path from an unmatched column j0
bool find_augmenting_path_from_root(int n, int j0, const int *col_ptr,
                                    const int *row_of_col, int *match_row,
                                    int *match_col) {
  // stack-allocated auxiliaries
  std::vector<char> seen_row(n, 0);
  std::vector<int> parent_col(n, -1);

  int j = j0;
  while (j != -1) {
    // (1) look-ahead: any unmatched row neighbor?
    int i_unmatched = -1;
    for (int q = col_ptr[j]; q < col_ptr[j + 1]; ++q) {
      int i = row_of_col[q];
      if (match_row[i] == -1) {
        i_unmatched = i;
        break;
      }
    }
    if (i_unmatched != -1) {
      augment_path(i_unmatched, j, parent_col.data(), match_row, match_col);
      return true;
    }

    // (2) else extend tree via a matched row not yet visited
    int i_next = -1;
    for (int q = col_ptr[j]; q < col_ptr[j + 1]; ++q) {
      int i = row_of_col[q];
      if (match_row[i] != -1 && !seen_row[i]) {
        i_next = i;
        break;
      }
    }

    if (i_next != -1) {
      seen_row[i_next] = 1;
      int j1 = match_row[i_next];
      if (parent_col[j1] == -1)
        parent_col[j1] = j;
      j = j1; // descend
    } else {
      j = parent_col[j]; // backtrack
    }
  }
  return false; // none found
}

//--------------------------------------------------------------------
// Public API
// Returns size of matching; outputs match_row & match_col (-1 if unmatched)
int maximum_cardinality_matching_csr(int n, const int *row_ptr,
                                     const int *col_ind, int *match_row,
                                     int *match_col) {
  int nnz = row_ptr[n];

  // allocate CSC
  std::vector<int> col_ptr(n + 1);
  std::vector<int> row_of_col(nnz);
  buildCSC_fromCSR(n, row_ptr, col_ind, col_ptr.data(), row_of_col.data());

  // init matches to -1
  for (int i = 0; i < n; ++i) {
    match_row[i] = -1;
    match_col[i] = -1;
  }

  // main loop: try augmenting from every unmatched column
  for (int j0 = 0; j0 < n; ++j0) {
    if (match_col[j0] == -1) {
      find_augmenting_path_from_root(n, j0, col_ptr.data(), row_of_col.data(),
                                     match_row, match_col);
    }
  }

  // count matched columns
  int cardinality = 0;
  for (int j = 0; j < n; ++j)
    if (match_col[j] != -1)
      ++cardinality;
  return cardinality;
}

int main(int argc, char **argv) {

  cxxopts::Options options("GMRES Example",
                           "Example of using GMRES with a CSR matrix");
  options.add_options()(
      "f,filename", "Matrix Market file to read",
      cxxopts::value<std::string>()->default_value("../tests/data/ex5.mtx"))(
      "l,level", "ILU level",
      cxxopts::value<int>()->default_value("0"))("h,help", "Print usage");
  auto result = options.parse(argc, argv);
  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }
  std::string filename = result["filename"].as<std::string>();
  int level = result["level"].as<int>();
  std::ifstream f(filename);
  f.clear();
  f.seekg(0, std::ios::beg);
  matrix_utils::CSRMatrix<int, int, double> csr_matrix, ilu_matrix;
  matrix_utils::readMatrixMarket(f, csr_matrix);
  std::cout << "size: " << csr_matrix.rows << std::endl;
  std::ofstream out0("mat_csr.svg");
  matrix_utils::writeSVG(csr_matrix.rows, csr_matrix.cols, csr_matrix.AI(),
                         csr_matrix.AJ(), out0);
  out0.close();

  std::vector<int> matching_row(csr_matrix.rows);
  std::vector<int> matching_col(csr_matrix.rows);
  reordering::MaximumMatching(csr_matrix.rows, csr_matrix.AI(), csr_matrix.AJ(),
                              matching_row.data(), matching_col.data());
  for (int i = 0; i < csr_matrix.rows; i++) {
    std::cout << matching_row[i] << " ";
  }
  std::cout << std::endl;
  for (int i = 0; i < csr_matrix.rows; i++) {
    std::cout << matching_col[i] << " ";
  }
  std::cout << std::endl;

  // Row permute the matrix according to matching_row
  matrix_utils::CSRMatrix<int, int, double> permuted_matrix;

  permuted_matrix.rows = csr_matrix.rows;
  permuted_matrix.cols = csr_matrix.cols;
  permuted_matrix.ResizeAI(csr_matrix.rows + 1);
  permuted_matrix.ResizeAJ(csr_matrix.NNZ());
  permuted_matrix.ResizeAV(csr_matrix.NNZ());
  matrix_utils::permuteMat(csr_matrix.rows, csr_matrix.cols, (int *)nullptr,
                           matching_col.data(), csr_matrix.AI(),
                           csr_matrix.AJ(), permuted_matrix.AI(),
                           permuted_matrix.AJ());

  std::ofstream out1("mat_csr_rowperm.svg");
  matrix_utils::writeSVG(permuted_matrix.rows, permuted_matrix.cols,
                         permuted_matrix.AI(), permuted_matrix.AJ(), out1);
  out1.close();

  // maximum_cardinality_matching_csr(csr_matrix.rows, csr_matrix.AI(),
  //                                  csr_matrix.AJ(), matching_row.data(),
  //                                  matching_col.data());
  // for (int i = 0; i < csr_matrix.rows; i++) {
  //   std::cout << matching_row[i] << " ";
  // }
  // std::cout << std::endl;
  // for (int i = 0; i < csr_matrix.rows; i++) {
  //   std::cout << matching_col[i] << " ";
  // }
  // std::cout << std::endl;
  {
    reordering::HungarianAlgorithm<int, int, double> hungarian;
    std::vector<int> matching_row(csr_matrix.rows);
    std::vector<int> matching_col(csr_matrix.rows);
    std::vector<double> potential_row(csr_matrix.rows);
    std::vector<double> potential_col(csr_matrix.rows);
    hungarian(csr_matrix.rows, csr_matrix.AI(), csr_matrix.AJ(),
              csr_matrix.AV(), matching_row.data(), matching_col.data(),
              potential_row.data(), potential_col.data());

    for (int i = 0; i < csr_matrix.rows; i++) {
      std::cout << matching_row[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < csr_matrix.rows; i++) {
      std::cout << potential_row[i] << " ";
    }
    std::cout << std::endl;
  }
  return 0;
}