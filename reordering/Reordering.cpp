#include "Reordering.h"
#include "BFS.h"
#include "mkl_sparse_mat.h"
#include <algorithm>
#include <execution>
#include <iostream>
#include <limits>
#include <omp.h>

namespace reordering {
std::pair<MKL_INT, MKL_INT>
MinDegreeNode(mkl_wrapper::mkl_sparse_mat const *const mat,
              std::vector<MKL_INT> *degrees) {
  std::pair<MKL_INT, MKL_INT> res(-1, std::numeric_limits<MKL_INT>::max());
  if (degrees)
    degrees->resize(mat->rows());
  auto ai = mat->get_ai();
  auto aj = mat->get_aj();
  for (MKL_INT i = 0; i < mat->rows(); i++) {
    MKL_INT degree = ai[i + 1] - ai[i];
    if (degrees)
      (*degrees)[i] = degree;
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
PMinDegreeNode(mkl_wrapper::mkl_sparse_mat const *const mat,
               std::vector<MKL_INT> *degrees) {
  std::pair<MKL_INT, MKL_INT> res(-1, std::numeric_limits<MKL_INT>::max());
  if (degrees)
    degrees->resize(mat->rows());
  auto ai = mat->get_ai();
  auto aj = mat->get_aj();
#pragma omp parallel for reduction(PairReduce : res)
  for (MKL_INT i = 0; i < mat->rows(); i++) {
    if (degrees)
      (*degrees)[i] = ai[i + 1] - ai[i];
    PairReduce(res, std::make_pair(i + mat->mkl_base(), ai[i + 1] - ai[i]));
  }
  res.first += mat->mkl_base();
  return res;
}

void PseudoDiameter(mkl_wrapper::mkl_sparse_mat const *const mat,
                    MKL_INT &source, MKL_INT &target) {
  std::vector<MKL_INT> degrees;
  source = MinDegreeNode(mat, &degrees).first;
  target = -1;
  std::vector<MKL_INT> choosen;
  auto ai = mat->get_ai();
  auto aj = mat->get_aj();
  MKL_INT diameter;
  MKL_INT forwardWidth;
  MKL_INT backwardWidth;
  while (target == -1) {
    choosen.resize(0);
    BFS bfs(reordering::PBFS_Fn<true, false>);
    bfs(mat, source);
    diameter = bfs.getHeight();
    forwardWidth = bfs.getHeight();
    while (choosen.size() < 5) {
      int minDeg = std::numeric_limits<int>::max();
      int sel = -1;
      for (auto i : bfs.getLastLevel()) {
        i = i - mat->mkl_base();
        if (degrees[i] < minDeg) {
          minDeg = degrees[i];
          sel = i;
        }
      }
      if (minDeg == std::numeric_limits<int>::max())
        break;

      choosen.push_back(sel);
      degrees[sel] = std::numeric_limits<int>::max();
      for (MKL_INT i = ai[sel]; i < ai[sel + 1]; i++) {
        degrees[aj[i]] = std::numeric_limits<int>::max();
      }
    }
    backwardWidth = std::numeric_limits<int>::max();
    for (auto i : choosen) {
      bfs.setShortCut(backwardWidth);
      if (!bfs(mat, i))
        continue;
      if (diameter < bfs.getHeight() && bfs.getWidth() < backwardWidth) {
        source = i + mat->mkl_base();
        break;
      } else if (bfs.getWidth() < backwardWidth) {

        backwardWidth = bfs.getWidth();
        target = i + mat->mkl_base();
      }
    }
  }
  std::cout << "diameter: " << diameter << std::endl;
  if (forwardWidth > backwardWidth)
    std::swap(source, target);
}
} // namespace reordering