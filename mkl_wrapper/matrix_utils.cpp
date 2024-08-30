#include "matrix_utils.hpp"

namespace matrix_utils {
template class CSRMatrix<int, int, double>;

template void SerialTranspose<int, int, int, double>(
    const int rows, const int cols, const int base, int const *ai,
    int const *aj, double const *av, int *ai_transpose, int *aj_transpose,
    double *av_transpose);

template void ParallelTranspose<int, int, int, double>(
    const int rows, const int cols, const int base, int const *ai,
    int const *aj, double const *av, int *ai_transpose, int *aj_transpose,
    double *av_transpose);

template void ParallelTranspose2<int, int, int, double>(
    const int rows, const int cols, const int base, int const *ai,
    int const *aj, double const *av, int *ai_transpose, int *aj_transpose,
    double *av_transpose);

template void SplitLDU(const int rows, const int base, int const *ai,
                       int const *aj, double const *av,
                       CSRMatrix<int, int, double> &L, std::vector<double> &D,
                       CSRMatrix<int, int, double> &U);
} // namespace matrix_utils