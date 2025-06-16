#pragma once
#include "../config.h"
#include "BFS.h"
#include "circularbuffer.hpp"
#include "mkl_sparse_mat.h"
#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mkl_types.h>
#include <omp.h>
#include <ranges>
#include <utility>
#include <vector>

namespace mkl_wrapper {
class mkl_sparse_mat;
}

namespace reordering {

void NodeDegree(mkl_wrapper::mkl_sparse_mat const *const mat,
                std::vector<MKL_INT> &degrees);

void PNodeDegree(mkl_wrapper::mkl_sparse_mat const *const mat,
                 std::vector<MKL_INT> &degrees);
// returns node index and degree
template <typename View>
std::pair<MKL_INT, MKL_INT> MinDegreeNode(const std::vector<MKL_INT> &degrees,
                                          const MKL_INT base, View &&view) {
  std::pair<MKL_INT, MKL_INT> res(-1, std::numeric_limits<MKL_INT>::max());
  for (const auto i : view) {
    if (degrees[i - base] < res.second) {
      res.first = i;
      res.second = degrees[i - base];
    }
  }
  return res;
}

template <typename T>
void PairReduce(std::pair<T, T> &inout, const std::pair<T, T> &in) {
  if (in.second < inout.second) {
    inout = in;
  } else if (in.second == inout.second) {
    inout.first = std::min(in.first, inout.first);
  }
}

// returns node index and degree;
template <typename View>
std::pair<MKL_INT, MKL_INT> PMinDegreeNode(const std::vector<MKL_INT> &degrees,
                                           const MKL_INT base, View &&view) {
#pragma omp declare reduction(                                                 \
        pairreduce : std::pair<MKL_INT, MKL_INT> : PairReduce<MKL_INT>(        \
                omp_out, omp_in)) initializer(omp_priv = omp_orig)

  std::pair<MKL_INT, MKL_INT> res(-1, std::numeric_limits<MKL_INT>::max());
#pragma omp parallel for reduction(pairreduce : res)
  for (const auto i : view) {
    PairReduce(res, std::make_pair(i, degrees[i - base]));
  }
  return res;
}

// input view dof for a component
// returns source and target node indices and diameter
// NOTE: the relevant degrees will be modified
// https://github.com/dralves/sp1-sp2-galois/blob/1597f1f510cc1aa75f5595f0d42f5701dfc34a91/lonestar/experimental/cuthill/serial/cuthill.cpp#L815
// duff1989use The use of profile reduction algorithms with a frontal code
template <typename View>
double PseudoDiameter(mkl_wrapper::mkl_sparse_mat const *const mat,
                      std::vector<MKL_INT> &degrees, View &&view,
                      MKL_INT &source, MKL_INT &target) {
  source = MinDegreeNode(degrees, mat->mkl_base(), view).first;
  target = -1;
  std::vector<MKL_INT> choosen;
  auto ai = mat->get_ai();
  auto aj = mat->get_aj();
  const MKL_INT base = mat->mkl_base();
  MKL_INT diameter;
  MKL_INT forwardWidth;
  MKL_INT backwardWidth;
  while (target == -1) {
    choosen.resize(0);
    BFS bfs(reordering::PBFS_Fn<true, false>);
    bfs(mat, source);
    diameter = bfs.getHeight();
    forwardWidth = bfs.getWidth();

    // First five strategy
    while (choosen.size() < 5) {
      int minDeg = std::numeric_limits<int>::max();
      int sel = -1;
      for (auto i : bfs.getLastLevel()) {
        if (degrees[i - base] < minDeg) {
          minDeg = degrees[i - base];
          sel = i;
        } else if (degrees[i - base] ==
                   minDeg) { // make sure multi threading result is consistent
          sel = std::min(sel, i);
        }
      }
      if (minDeg == std::numeric_limits<int>::max())
        break;

      choosen.push_back(sel);
      degrees[sel - base] =
          std::numeric_limits<int>::max(); // mark-off selected node
      for (MKL_INT i = ai[sel - base] - base; i < ai[sel - base + 1] - base;
           i++) {
        degrees[aj[i] - base] =
            std::numeric_limits<int>::max(); // avoiding any node with a
                                             // neighbour that had been tested
      }
    }
    backwardWidth = std::numeric_limits<int>::max();
    for (auto i : choosen) {
      bfs.setShortCut(backwardWidth);
      if (!bfs(mat, i)) // short circuited
        continue;
      if (diameter < bfs.getHeight() && bfs.getWidth() < backwardWidth) {
        source = i;
        break;
      } else if (bfs.getWidth() < backwardWidth) {

        backwardWidth = bfs.getWidth();
        target = i;
      }
    }
  }
  if (forwardWidth > backwardWidth)
    std::swap(source, target);
  return diameter;
}

// TODO: implement parallel one
void SerialCM(mkl_wrapper::mkl_sparse_mat const *const mat,
              std::vector<MKL_INT> &iperm, std::vector<MKL_INT> &perm);

#ifdef USE_METIS_LIB
// nested dissection from metis
void Metis(mkl_wrapper::mkl_sparse_mat const *const mat,
           std::vector<MKL_INT> &iperm, std::vector<MKL_INT> &perm);
#endif
} // namespace reordering