#include "matrix_utils.hpp"

namespace matrix_utils {
template class CSRMatrix<int, int, double>;
template class CSRMatrixVec<int, int, double>;

template void SerialTranspose<int, int, double>(const int rows, const int cols,
                                                const int base, int const *ai,
                                                int const *aj, double const *av,
                                                int *ai_transpose,
                                                int *aj_transpose,
                                                double *av_transpose);

template void ParallelTranspose<int, int, double>(
    const int rows, const int cols, const int base, int const *ai,
    int const *aj, double const *av, int *ai_transpose, int *aj_transpose,
    double *av_transpose);

template void ParallelTranspose2<int, int, double>(
    const int rows, const int cols, const int base, int const *ai,
    int const *aj, double const *av, int *ai_transpose, int *aj_transpose,
    double *av_transpose);

template void permutedAI<int, int>(const int rows, const int base,
                                   int const *ai, int const *iperm,
                                   int *permed_ai);

template void permute<int, int, double>(const int rows, const int base,
                                        int const *ai, int const *aj,
                                        double const *av, int const *iperm,
                                        int const *perm, int *permed_ai,
                                        int *permed_aj, double *permed_av);

template void permuteRow<int, int, double>(const int rows, const int base,
                                           int const *ai, int const *aj,
                                           double const *av, int const *iperm,
                                           int *permed_ai, int *permed_aj,
                                           double *permed_av);

template void symPermute<int, int, double>(const int rows, const int base,
                                           int const *ai, int const *aj,
                                           double const *av, int const *iperm,
                                           int *permed_ai, int *permed_aj,
                                           double *permed_av);

template <typename ROWTYPE, typename COLTYPE>
template <TriangularMatrix TS>
COLTYPE TopologicalSort<ROWTYPE, COLTYPE>::operator()<TS>(const COLTYPE nodes,
                                                          ROWTYPE const *ai,
                                                          COLTYPE const *aj,
                                                          COLTYPE *iperm,
                                                          COLTYPE *prefix) {
  _degrees.resize(nodes);
  const auto base = ai[0];
  const auto nnz = ai[nodes] - base;

  _t_ai.resize(nodes + 1);
  _t_aj.resize(nnz);

  //   reverse graph
  ParallelTranspose2(nodes, nodes, base, ai, aj, (double *)nullptr,
                     _t_ai.data(), _t_aj.data(), (double *)nullptr);

  if constexpr (TS == TriangularMatrix::L) {
    _start = 0;
    _end = nodes;
    _inc = 1;
  } else {
    _start = nodes - 1;
    _end = -1;
    _inc = -1;
  }

  //   node degrees
  for (COLTYPE i = _start; i != _end; i += _inc) {
    _degrees[i] = ai[i + 1] - ai[i];
    if(_degrees[i]==)
  }
}

template int TopologicalSort<TriangularMatrix::L, int, int, std::vector<int>>(
    const int nodes, int const *ai, int const *aj, std::vector<int> &iperm,
    std::vector<int> &prefix);

template int TopologicalSort<TriangularMatrix::U, int, int, std::vector<int>>(
    const int nodes, int const *ai, int const *aj, std::vector<int> &iperm,
    std::vector<int> &prefix);

template int TopologicalSort2<TriangularMatrix::L, int, int, std::vector<int>>(
    const int nodes, int const *ai, int const *aj, std::vector<int> &iperm,
    std::vector<int> &prefix);

template int TopologicalSort2<TriangularMatrix::U, int, int, std::vector<int>>(
    const int nodes, int const *ai, int const *aj, std::vector<int> &iperm,
    std::vector<int> &prefix);

template bool Diagonal<int, int, double>(const int rows, const int base,
                                         int const *ai, int const *aj,
                                         double const *av, int *diagpos,
                                         double *diag, const bool invert);

template void SplitLDU(const int rows, const int base, int const *ai,
                       int const *aj, double const *av,
                       CSRMatrix<int, int, double> &L, std::vector<double> &D,
                       CSRMatrix<int, int, double> &U);

template void SplitTriangle<TriangularMatrix::U, int, int, double,
                            CSRMatrix<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    double const *av, CSRMatrix<int, int, double> &U);

template void SplitTriangle<TriangularMatrix::U, int, int, double,
                            CSRMatrixVec<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    double const *av, CSRMatrixVec<int, int, double> &U);

template void SplitTriangle<TriangularMatrix::L, int, int, double,
                            CSRMatrix<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    double const *av, CSRMatrix<int, int, double> &U);

template void SplitTriangle<TriangularMatrix::L, int, int, double,
                            CSRMatrixVec<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    double const *av, CSRMatrixVec<int, int, double> &U);

template void TriangularToFull<TriangularMatrix::U, int, int, double,
                               CSRMatrix<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    double const *av, CSRMatrix<int, int, double> &F);

template void TriangularToFull<TriangularMatrix::U, int, int, double,
                               CSRMatrixVec<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    double const *av, CSRMatrixVec<int, int, double> &F);

template void Block<int, int, double, CSRMatrixVec<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    double const *av, const int i, const int j, const int p, const int q,
    CSRMatrixVec<int, int, double> &);
} // namespace matrix_utils