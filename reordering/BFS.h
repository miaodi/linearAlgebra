#pragma once
#include <memory>
#include <mkl_types.h>

namespace mkl_wrapper {
class mkl_sparse_mat;
}

namespace reordering {
// return levels
std::shared_ptr<MKL_INT[]> BFS_serial(mkl_wrapper::mkl_sparse_mat *mat,
                                      int source, int &level);
} // namespace reordering