#include "symmetric_ops.hpp"

namespace matrix_utils
{
// Explicit template instantiations
template void AATSymbolic<int, int, true>(const int, int const*, int const*, int*);
template void AATSymbolic<int, int, false>(const int, int const*, int const*, int*);

template void AATNumeric<int, int, true>(const int, int const*, int const*, int const*, int*);
template void AATNumeric<int, int, false>(const int, int const*, int const*, int const*, int*);
} // namespace matrix_utils
