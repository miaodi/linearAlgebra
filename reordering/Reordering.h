#pragma once
#include "circularbuffer.hpp"
#include <functional>
#include <memory>
#include <mkl_types.h>
#include <utility>

namespace mkl_wrapper {
class mkl_sparse_mat;
}

namespace reordering {
// returns node index and degree
std::pair<MKL_INT, MKL_INT>
MinDegreeNode(mkl_wrapper::mkl_sparse_mat const *const mat);
// returns node index and degree
std::pair<MKL_INT, MKL_INT>
PMinDegreeNode(mkl_wrapper::mkl_sparse_mat const *const mat);

// input
// returns source and target node indices
void PseudoDiameter(mkl_wrapper::mkl_sparse_mat const *const mat,
                    MKL_INT &source, MKL_INT &target);
} // namespace reordering