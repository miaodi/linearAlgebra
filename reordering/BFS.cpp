#include "BFS.h"
#include "mkl_sparse_mat.h"
#include <algorithm>

namespace reordering {
// return levels
std::shared_ptr<MKL_INT[]> BFS_serial(mkl_wrapper::mkl_sparse_mat *mat,
                                      int source, int &level) {
  auto res = std::shared_ptr<MKL_INT[]>(new MKL_INT[mat->rows()]);
  std::fill_n(res.get(), mat->rows(), -1);

  
  return res;
  
}
} // namespace reordering