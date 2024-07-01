#include "matrix_utils.hpp"

namespace matrix_utils {
template class CSRMatrix<int, int, double>;

template auto
SerialTranspose<int, int, int, double>(const int rows, const int cols,
                                       const int base, int const *ai,
                                       int const *aj, double const *av);

template auto
ParallelTranspose<int, int, int, double>(const int rows, const int cols,
                                         const int base, int const *ai,
                                         int const *aj, double const *av);

template auto
ParallelTranspose2<int, int, int, double>(const int rows, const int cols,
                                          const int base, int const *ai,
                                          int const *aj, double const *av);
} // namespace matrix_utils