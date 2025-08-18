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
  std::cout << "spmv operator..." << std::endl;
  using CSRTYPE = typename matrix_utils::CSRMatrix<int, int, double>;
  matrix_utils::SPMV<CSRTYPE, matrix_utils::SerialSPMV> spmv;
  spmv.setMatrix(&csr_matrix);
  spmv.preprocess();
  std::cout << "spmv operator done." << std::endl;

  // precond operator
  std::cout << "precond operator..." << std::endl;
  ILUPrec<decltype(ilu_matrix)> ilu_prec(csr_matrix.rows, ilu_matrix);
  std::cout << "precond operator done." << std::endl;

  {
    matrix_utils::SplitLU<decltype(ilu_matrix)> splitlu(8);
    matrix_utils::CSRMatrix<int, int, double> L, U;

    std::cout << "SplitLU..." << std::endl;
    splitlu(ilu_matrix.rows, ilu_matrix.AI(), ilu_matrix.Diagonal(),
            ilu_matrix.AJ(), ilu_matrix.AV(), L, U);
    std::cout << "SplitLU done." << std::endl;

    std::ofstream outL("L_csr.svg");
    matrix_utils::writeSVG(L.rows, L.cols, L.AI(), L.AJ(), outL);
    outL.close();
    std::ofstream outU("U_csr.svg");
    matrix_utils::writeSVG(U.rows, U.cols, U.AI(), U.AJ(), outU);
    outU.close();

    auto base = L.AI()[0];
    std::cout << "base: " << base << std::endl;
    matrix_utils::KahnParallel<int, int> kahn(8);
    // matrix_utils::KahnSerial<int, int> kahn;
    matrix_utils::TopologicalSort2<int, int> topSort;
    std::vector<int> perm(L.rows);
    std::vector<int> prefix(L.rows + 1);
    int level = topSort(matrix_utils::TriangularMatrix::L, L.rows, L.AI(), L.AJ(),
                     perm.data(), prefix.data());
    std::cout << "Kahn done." << std::endl;
    std::cout << "Level: " << level << std::endl;
    for (int i = 0; i < level; i++) {
      for (int j = prefix[i] - base; j < prefix[i + 1] - base; j++) {
        std::cout << perm[j] - base << " ";
      }
      std::cout << std::endl;
    }

    // std::vector<int> permU(U.rows);
    // std::vector<int> prefixU(U.rows + 1);
    // int levelU = kahn(matrix_utils::TriangularMatrix::U, U.rows, U.AI(), U.AJ(),
    //                   permU.data(), prefixU.data());
    // std::cout << "Kahn done." << std::endl;
    // std::cout << "Level: " << levelU << std::endl;
    // for (int i = 0; i < levelU; i++) {
    //   for (int j = prefixU[i] - base; j < prefixU[i + 1] - base; j++) {
    //     std::cout << permU[j] - base << " ";
    //   }
    //   std::cout << std::endl;
    // }
  }

  // std::vector<double> b(csr_matrix.rows, 1.0);
  // std::vector<double> x(csr_matrix.rows, 0.0);

  // std::cout<<"GMRES..."<<std::endl;
  // iterative_solver::GMRES<double> gmres_solver;
  // gmres_solver.setMaxIter(10000000);
  // gmres_solver.setRelTol(1e-8);
  // gmres_solver.setRestart(100);
  // gmres_solver(&spmv, &ilu_prec, b.data(), x.data());
  // std::cout<<"GMRES done."<<std::endl;
  return 0;
}