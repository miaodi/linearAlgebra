#include "precond_symbolic.hpp"
#include "matrix_utils.hpp"

namespace matrix_utils {
template void ICCLevel0SymSymbolic<int, int, CSRMatrix<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    CSRMatrix<int, int, double> &icc);

template void ICCLevelSymbolic<int, int, CSRMatrix<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    int const *diag_pos, const int lvl, CSRMatrix<int, int, double> &icc);

template void ICCLevelVecSymbolic<int, int, CSRMatrix<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    int const *diag_pos, const int lvl, CSRMatrix<int, int, double> &icc);
} // namespace matrix_utils