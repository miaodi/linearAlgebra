#include "io.hpp"
#include "iterative_solver.hpp"
#include "matrix_utils.hpp"
#include "precond.hpp"
#include "sparse_mat_traits.hpp"
#include "spmv.hpp"
#include "triangle_solve.hpp"
#include <cxxopts.hpp>
#include <fstream>
#include <mkl.h>
#include <string>
#include <vector>

template <matrix_utils::ResizableDiagonalType CSRMatrixType> class ILUPrec {
public:
  using ROWTYPE = typename CSRMatrixType::ROWTYPE;
  using COLTYPE = typename CSRMatrixType::COLTYPE;
  using VALTYPE = typename CSRMatrixType::VALTYPE;
  ILUPrec(const COLTYPE size, const CSRMatrixType &ilu)
      : _size(size), _ilu(ilu), tmp(size) {}

  COLTYPE size() const { return _size; }

  bool operator()(VALTYPE const *const b, VALTYPE *const x) const {
    const auto base = _ilu.AI()[0];
    matrix_utils::ForwardSubstitution(_size, base, _ilu.AI(), _ilu.Diagonal(),
                                      _ilu.AJ(), _ilu.AV(), b, tmp.data());
    matrix_utils::BackwardSubstitution(_size, base, _ilu.Diagonal(),
                                       _ilu.AI() + 1, _ilu.AJ(), _ilu.AV(),
                                       tmp.data(), x);
    return true;
  }
  COLTYPE _size;
  const CSRMatrixType &_ilu;
  mutable std::vector<VALTYPE> tmp;
};

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
  bool success = false;
  std::cout << "Symbolic ILU factorization..." << std::endl;
  matrix_utils::ILULevelSymbolic<decltype(ilu_matrix)> ilu;
  success =
      ilu(csr_matrix.rows, csr_matrix.AI(), csr_matrix.AJ(), level, ilu_matrix);
  if (!success) {
    std::cout << "Symbolic ILU factorization failed." << std::endl;
    return -1;
  }
  std::cout << "Symbolic ILU factorization done. nnz: " << ilu_matrix.NNZ()
            << std::endl;
  std::cout << "Numeric ILU factorization..." << std::endl;
  success = matrix_utils::ILULevelNumeric(csr_matrix.rows, csr_matrix.AI(),
                                          csr_matrix.AJ(), csr_matrix.AV(),
                                          level, ilu_matrix);
  if (!success) {
    std::cout << "Numeric ILU factorization failed." << std::endl;
    return -1;
  }
  std::cout << "ILU factorization done." << std::endl;
  std::ofstream out1("ilu_csr.svg");
  matrix_utils::writeSVG(ilu_matrix.rows, ilu_matrix.cols, ilu_matrix.AI(),
                         ilu_matrix.AJ(), out1);
  out1.close();

  // spmv operator
  std::cout<<"spmv operator..."<<std::endl;
  using CSRTYPE = typename matrix_utils::CSRMatrix<int, int, double>;
  matrix_utils::SPMV<CSRTYPE, matrix_utils::SerialSPMV> spmv;
  spmv.setMatrix(&csr_matrix);
  spmv.preprocess();
  std::cout<<"spmv operator done."<<std::endl;

  // precond operator
  std::cout<<"precond operator..."<<std::endl;
  ILUPrec<decltype(ilu_matrix)> ilu_prec(csr_matrix.rows, ilu_matrix);
  std::cout<<"precond operator done."<<std::endl;

  std::vector<double> b(csr_matrix.rows, 1.0);
  std::vector<double> x(csr_matrix.rows, 0.0);

  std::cout<<"GMRES..."<<std::endl;
  iterative_solver::GMRES<double> gmres_solver;
  gmres_solver.setMaxIter(10000000);
  gmres_solver.setRelTol(1e-8);
  gmres_solver.setRestart(100);
  gmres_solver(&spmv, &ilu_prec, b.data(), x.data());
  std::cout<<"GMRES done."<<std::endl;

  // {
  //   // Upper triangular matrix A (row-major)
  //   double A[9] = {2, 3, 0, 0, 4, 0, 0, 0, 0}; // 2x2 matrix: [2 3; 0 4]
  //   double b[2] = {5, 8};                      // Right-hand side
  //   int n = 2;
  //   int incx = 1;

  //   // Solve A*x = b (A is upper triangular)
  //   // The solution will overwrite b
  //   cblas_dtrsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, A,
  //   3,
  //               b, incx);
  //   std::cout << "Solution x:\n";
  //   for (int i = 0; i < n; ++i) {
  //     std::cout << b[i] << "\n";
  //   }
  // }
  return 0;
}