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

template int TopologicalSort<TriangularMatrix::L, int, int, std::vector<int>>(
    const int nodes, const int base, int const *ai, int const *aj,
    std::vector<int> &iperm, std::vector<int> &prefix);

template int TopologicalSort<TriangularMatrix::U, int, int, std::vector<int>>(
    const int nodes, const int base, int const *ai, int const *aj,
    std::vector<int> &iperm, std::vector<int> &prefix);

template int TopologicalSort2<TriangularMatrix::L, int, int, std::vector<int>>(
    const int nodes, const int base, int const *ai, int const *aj,
    std::vector<int> &iperm, std::vector<int> &prefix);

template int TopologicalSort2<TriangularMatrix::U, int, int, std::vector<int>>(
    const int nodes, const int base, int const *ai, int const *aj,
    std::vector<int> &iperm, std::vector<int> &prefix);

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

template void TriangularToFull<TriangularMatrix::U, int, int, double,
                               CSRMatrix<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    double const *av, CSRMatrix<int, int, double> &F);

template void TriangularToFull<TriangularMatrix::U, int, int, double,
                               CSRMatrixVec<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    double const *av, CSRMatrixVec<int, int, double> &F);
} // namespace matrix_utils