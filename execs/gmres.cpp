#include "io.hpp"
#include "iterative_solver.hpp"
#include "matrix_utils.hpp"
#include "precond.hpp"
#include "spmv.hpp"
#include <fstream>
#include <string>
#include <vector>
#include <mkl.h>

int main() {

  std::string filename = "../tests/data/ex5.mtx";

  std::ifstream f(filename);
  f.clear();
  f.seekg(0, std::ios::beg);
  matrix_utils::CSRMatrix<int, int, double> csr_matrix;
  matrix_utils::readMatrixMarket(f, csr_matrix);

  std::ofstream out0("mat_csr.svg");
  matrix_utils::writeSVG(csr_matrix.rows, csr_matrix.cols, csr_matrix.AI(),
                         csr_matrix.AJ(), out0);
  out0.close();

  using CSRTYPE = typename matrix_utils::CSRMatrix<int, int, double>;
  matrix_utils::SPMV<CSRTYPE, matrix_utils::SerialSPMV> spmv;
  spmv.setMatrix(&csr_matrix);
  spmv.preprocess();

  matrix_utils::IdentityPrec<double> identity_prec(csr_matrix.rows);

  std::vector<double> b(csr_matrix.rows, 1.0);
  std::vector<double> x(csr_matrix.rows, 1.0);

  iterative_solver::GMRES<double> gmres_solver;
  gmres_solver(&spmv, &identity_prec, b.data(), x.data());
  for (auto i : x) {
    std::cout << i << " ";
  }
  std::cout << std::endl;

  {
    // Upper triangular matrix A (row-major)
    double A[9] = {2, 3, 0, 0, 4, 0, 0, 0, 0}; // 2x2 matrix: [2 3; 0 4]
    double b[2] = {5, 8};       // Right-hand side
    int n = 2;
    int incx = 1;

    // Solve A*x = b (A is upper triangular)
    // The solution will overwrite b
    cblas_dtrsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, A, 3,
                b, incx);
    std::cout << "Solution x:\n";
    for (int i = 0; i < n; ++i) {
      std::cout << b[i] << "\n";
    }
  }
  return 0;
}