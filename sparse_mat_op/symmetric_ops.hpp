#pragma once

#include "matrix_utils.hpp"
#include "sparse_mat_traits.hpp"
#include "utils.h"
#include <numeric>
#include <vector>

namespace matrix_utils {

template <typename ROWTYPE, typename COLTYPE, bool KEEPDIAG = true>
void AATSymbolic(const COLTYPE size, ROWTYPE const *ai, COLTYPE const *aj,
                 ROWTYPE *ai_AAT);

template <typename ROWTYPE, typename COLTYPE, bool KEEPDIAG = true>
void AATNumeric(const COLTYPE size, ROWTYPE const *ai, COLTYPE const *aj,
                ROWTYPE const *ai_AAT, COLTYPE *aj_AAT);

} // namespace matrix_utils
