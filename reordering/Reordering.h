#pragma once
#include "config.h"
#include "BFS.h"
#include "mkl_sparse_mat.h"
#include <algorithm>
#include <limits>
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
/// @brief Configuration options for METIS nested dissection
struct MetisNDOptions {
  /// @brief Number of different separators to try (1-10+, default: 1)
  /// Higher values may produce better quality orderings but take longer
  int nseps = 1;
  
  /// @brief Number of refinement iterations (default: 10)
  /// Higher values may produce better quality orderings but take longer
  int niter = 10;
  
  /// @brief Random seed for reproducibility (default: -1 for random)
  /// Set to a fixed value (e.g., 0, 42) for reproducible results
  int seed = -1;
  
  /// @brief Compress graph by removing self-loops and duplicate edges (default: true)
  bool compress = true;
  
  /// @brief Order connected components of the graph separately (default: false)
  /// Useful for disconnected graphs
  bool ccorder = false;
  
  /// @brief Coarsening type (default: 1 = METIS_CTYPE_SHEM - sorted heavy-edge matching)
  /// 0 = METIS_CTYPE_RM (random matching)
  /// 1 = METIS_CTYPE_SHEM (sorted heavy-edge matching, usually better)
  int ctype = 1;
  
  /// @brief Refinement type (default: 3 = METIS_RTYPE_SEP1SIDED)
  /// 0 = METIS_RTYPE_FM (Fiduccia-Mattheyses)
  /// 1 = METIS_RTYPE_GREEDY
  /// 2 = METIS_RTYPE_SEP2SIDED (2-sided separator refinement)
  /// 3 = METIS_RTYPE_SEP1SIDED (1-sided separator refinement, usually best for ND)
  int rtype = 3;
  
  /// @brief Debug level (default: 0 = no output)
  /// Higher values produce more diagnostic output
  int dbglvl = 0;
};

// nested dissection from metis
void MetisND(mkl_wrapper::mkl_sparse_mat const *const mat,
             std::vector<MKL_INT> &iperm, std::vector<MKL_INT> &perm,
             const MetisNDOptions& opts = MetisNDOptions());

/// @brief METIS nested dissection reordering for general CSR matrices
/// @tparam ROWTYPE Type for row pointers (xadj array) - supports int32_t or int64_t
/// @tparam COLTYPE Type for column indices (adjncy array) - supports int32_t or int64_t
/// @param nrows Number of rows in the matrix
/// @param ncols Number of columns in the matrix (must equal nrows for square matrix)
/// @param xadj Row pointer array of size (nrows + 1), zero-based, with diagonals removed
/// @param adjncy Column index array, zero-based, with diagonals removed
/// @param iperm Inverse permutation array: iperm[i] = k means new row i comes from old row k
/// @param perm Permutation array: perm[i] = k means old row i goes to new row k
/// @param opts METIS options (optional, uses defaults if not provided)
/// @return 0 on success, non-zero on failure
template <typename ROWTYPE, typename COLTYPE>
int MetisND(const COLTYPE nrows, const COLTYPE ncols,
            const ROWTYPE* xadj, const COLTYPE* adjncy,
            COLTYPE* iperm, COLTYPE* perm,
            const MetisNDOptions& opts = MetisNDOptions());
#endif


} // namespace reordering