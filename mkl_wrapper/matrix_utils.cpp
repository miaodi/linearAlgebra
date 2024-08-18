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
} // namespace matrix_utils