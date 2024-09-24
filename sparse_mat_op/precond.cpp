#include "precond.hpp"
#include "matrix_utils.hpp"

namespace matrix_utils {
template void ICCLevel0SymSymbolic<int, int, CSRMatrix<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    CSRMatrix<int, int, double> &icc);

template void ICCLevelSymbolic0<int, int, CSRMatrix<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    int const *diag_pos, const int lvl, CSRMatrix<int, int, double> &icc);

template void ICCLevelSymbolic1<int, int, CSRMatrix<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    int const *diag_pos, const int lvl, CSRMatrix<int, int, double> &icc);

template void ICCLevelSymbolic2<int, int, CSRMatrix<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    int const *diag_pos, const int lvl, CSRMatrix<int, int, double> &icc);

template void ICCLevelSymbolic3<int, int, CSRMatrix<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    int const *diag_pos, const int lvl, CSRMatrix<int, int, double> &icc);

template void ICCLevelNumeric<int, int, CSRMatrix<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    int const *diag_pos, const int lvl, CSRMatrix<int, int, double> &icc);
} // namespace matrix_utils