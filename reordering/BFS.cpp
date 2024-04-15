#include "BFS.h"
#include "circularbuffer.hpp"
#include "mkl_sparse_mat.h"
#include <algorithm>
#include <execution>
#include <iostream>
#include <omp.h>

namespace reordering {
// return levels
std::shared_ptr<MKL_INT[]> BFS(mkl_wrapper::mkl_sparse_mat const *const mat,
                               int source, MKL_INT &level) {
  auto res = std::shared_ptr<MKL_INT[]>(new MKL_INT[mat->rows()]);
  std::fill_n(res.get(), mat->rows(), -1);
  auto ai = mat->get_ai();
  auto aj = mat->get_aj();

  utils::CircularBuffer<MKL_INT> cb(
      std::max(1, static_cast<MKL_INT>(mat->rows() * .2)));
  cb.push(source - mat->mkl_base());
  level = 0;
  res[source - mat->mkl_base()] = level;
  while (!cb.isEmpty()) {
    auto u = cb.first();
    cb.shift();
    level = res[u] + 1;
    for (MKL_INT i = ai[u]; i < ai[u + 1]; i++) {
      auto v = aj[i] - mat->mkl_base();
      if (res[v] == -1) {
        res[v] = level;
        if (!cb.available())
          cb.resize(cb.size() * 2);
        cb.push(v);
      }
    }
  }
  return res;
}

// return levels
std::shared_ptr<MKL_INT[]> PBFS(mkl_wrapper::mkl_sparse_mat const *const mat,
                                int source, MKL_INT &level) {

  auto res = std::shared_ptr<MKL_INT[]>(new MKL_INT[mat->rows()]);
  std::fill_n(std::execution::par_unseq, res.get(), mat->rows(), -1);
  auto ai = mat->get_ai();
  auto aj = mat->get_aj();

  int max_threads = omp_get_max_threads();
  // std::vector<int> 
  // #pragma omp parallel
  //   {
  //     int nthreads = omp_get_num_threads();
  //     std::cout << nthreads << std::endl;
  //   }
  return res;
} // namespace reordering
} // namespace reordering