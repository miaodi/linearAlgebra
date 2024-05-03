#pragma once

#include <utility>
#include <vector>
#include <mkl_types.h>

namespace mkl_wrapper {
class mkl_sparse_mat;
}

namespace reordering {
std::vector<MKL_INT>
UnionFindRank(mkl_wrapper::mkl_sparse_mat const *const mat);

std::vector<MKL_INT> UnionFindRem(mkl_wrapper::mkl_sparse_mat const *const mat);

std::vector<MKL_INT>
ParUnionFindRem(mkl_wrapper::mkl_sparse_mat const *const mat);
} // namespace reordering