#include "Reordering.h"
#include "mkl_sparse_mat.h"
#include <algorithm>
#include <execution>
#include <iostream>
#include <limits>
#include <omp.h>

namespace reordering {
std::pair<MKL_INT, MKL_INT>
MinDegreeNode(mkl_wrapper::mkl_sparse_mat const *const mat) {
  std::pair<MKL_INT, MKL_INT> res(-1, std::numeric_limits<MKL_INT>::max());
  auto ai = mat->get_ai();
  auto aj = mat->get_aj();
  for (MKL_INT i = 0; i < mat->rows(); i++) {
    MKL_INT degree = ai[i + 1] - ai[i];
    if (degree < res.second) {
      res.first = i + mat->mkl_base();
      res.second = degree;
    }
  }
  return res;
}

void PairReduce(std::pair<MKL_INT, MKL_INT> &inout,
                const std::pair<MKL_INT, MKL_INT> &in) {
  if (in.second < inout.second) {
    inout = in;
  } else if (in.second == inout.second) {
    inout.first = std::min(in.first, inout.first);
  }
}
#pragma omp declare reduction(                                  \
                              PairReduce :                      \
                              std::pair<MKL_INT, MKL_INT> :     \
                              PairReduce(omp_out, omp_in)       \
                             )                                  \
                    initializer (omp_priv=omp_orig)

std::pair<MKL_INT, MKL_INT>
PMinDegreeNode(mkl_wrapper::mkl_sparse_mat const *const mat) {
  std::pair<MKL_INT, MKL_INT> res(-1, std::numeric_limits<MKL_INT>::max());
  auto ai = mat->get_ai();
  auto aj = mat->get_aj();
#pragma omp parallel for reduction(PairReduce : res)
  for (MKL_INT i = 0; i < mat->rows(); i++) {
    PairReduce(res, std::make_pair(i + mat->mkl_base(), ai[i + 1] - ai[i]));
  }
  return res;
}
} // namespace reordering