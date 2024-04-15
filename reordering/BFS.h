#pragma once
#include <memory>
#include <mkl_types.h>

namespace mkl_wrapper {
class mkl_sparse_mat;
}

namespace reordering {
// return levels
std::shared_ptr<MKL_INT[]> BFS(mkl_wrapper::mkl_sparse_mat const *const mat,
                               int source, MKL_INT &level);

// return levels
std::shared_ptr<MKL_INT[]> PBFS(mkl_wrapper::mkl_sparse_mat const *const mat,
                                int source, MKL_INT &level);
} // namespace reordering