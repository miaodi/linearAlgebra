#include "Reordering.h"
#include "BFS.h"
#include "UnionFind.h"
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
                    MKL_INT &source, MKL_INT &target,
                    std::vector<MKL_INT> &degrees) {
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

std::vector<MKL_INT> SerialCM(mkl_wrapper::mkl_sparse_mat const *const mat) {
  //   MKL_INT source, target;
  //   std::vector<MKL_INT> degrees;
  //   PseudoDiameter(mat, source, target, degrees);

  //   struct Node {
  //     MKL_INT base_zero_index;
  //     MKL_INT level;
  //     MKL_INT degree;
  //   };

  //   const MKL_INT base = mat->mkl_base();
  //   std::vector<Node> nodes(mat->cols());
  //   // initialize nodes
  // #pragma omp parallel for
  //   for (MKL_INT i = 0; i < mat->cols(); i++) {
  //     nodes[i].base_zero_index = i;
  //     nodes[i].degree = degrees[i];
  //     nodes[i].level = std::numeric_limits<MKL_INT>::max();
  //   }

  //   std::vector<Node *> cur;
  //   std::vector<Node *> next;
  //   std::vector<MKL_INT> inv_perm(mat->cols());
  //   nodes[source - base].level = 0;
  //   cur.push_back(&nodes[source - base]);
  //   MKL_INT nextId = 0;

  //   const auto &ai = mat->get_ai();
  //   const auto &aj = mat->get_aj();

  //   while (cur.size()) {
  //     size_t base_zero_index = 0;
  //     for (const auto &n : cur) {
  //       inv_perm[nextId++] = n->base_zero_index + base;
  //       for (MKL_INT j = ai[n->base_zero_index]; j < ai[n->base_zero_index +
  //       1];
  //            j++) {
  //         if (nodes[aj[j] - base].level > nodes[n->base_zero_index].level +
  //         1) {
  //           nodes[aj[j] - base].level = nodes[n->base_zero_index].level + 1;
  //           next.push_back(&nodes[aj[j] - base]);
  //         }
  //       }
  //       std::sort(
  //           next.begin() + base_zero_index, next.end(),
  //           [](const auto &a, const auto &b) { return a->degree < b->degree;
  //           });
  //       base_zero_index = next.size();
  //     }
  //     cur.clear();
  //     std::swap(cur, next);
  //   }

  // TODO: need to assert rows=cols
  std::vector<MKL_INT> degrees;
  PMinDegreeNode(mat, &degrees);
  std::vector<MKL_INT> inv_perm(mat->cols());

  auto parents = reordering::ParUnionFindRem(mat);

  std::vector<MKL_INT> compRoots;
  std::vector<MKL_INT> sortedComp;
  std::vector<MKL_INT> compPrefSum;

  const MKL_INT base = mat->mkl_base();
  const auto &ai = mat->get_ai();
  const auto &aj = mat->get_aj();

  auto find_min_deg = [&degrees, base](auto begin, auto end) {
    MKL_INT deg = std::numeric_limits<MKL_INT>::max();
    MKL_INT pos = -1;
    for (auto it = begin; it != end; it++) {
      if (deg > degrees[*it - base]) {
        pos = *it;
        deg = degrees[*it - base];
      }
    }
    return pos;
  };

  reordering::ComponentsStat(parents, mat->mkl_base(), compRoots, sortedComp,
                             compPrefSum);
  for (int c = 0; c < compRoots.size(); c++) {
    MKL_INT offset = compPrefSum[c];
    MKL_INT i = sortedComp[offset];
    if (compPrefSum[c + 1] - compPrefSum[c] == 1) {
      inv_perm[offset] = i;
      continue;
    } else if (compPrefSum[c + 1] - compPrefSum[c] == 2) {
      inv_perm[offset] = i;
      inv_perm[offset + 1] = sortedComp[offset + 1];
      continue;
    }

    i = find_min_deg(sortedComp.cbegin() + compPrefSum[c],
                     sortedComp.cbegin() + compPrefSum[c + 1]);
    reordering::BFS bfs(reordering::BFS_Fn<false>);
    bfs(mat, i);
    auto &levels = bfs.getLevels();
    const auto height = bfs.getHeight();
    std::cout << "height: " << height << std::endl;

    std::vector<MKL_INT> prefix(height + 1, 0);
    for (MKL_INT p = compPrefSum[c]; p != compPrefSum[c + 1]; p++) {
      prefix[levels[sortedComp[p] - base] + 1]++;
    }
    for (MKL_INT l = 0; l < height; l++) {
      prefix[l + 1] += prefix[l];
    }
    for (auto p : prefix) {
      std::cout << p << " ";
    }
    std::cout << std::endl;
    std::vector<MKL_INT> children;
    children.reserve(bfs.getWidth());
    MKL_INT e = offset;
    inv_perm[e++] = i;
    std::cout << "bfs.getWidth(): " << bfs.getWidth() << std::endl;
    for (MKL_INT l = 0; l < height; l++) {
      for (MKL_INT r = prefix[l]; r != prefix[l + 1]; r++) {
        MKL_INT u = inv_perm[r + offset];
        for (MKL_INT j = ai[u - base]; j != ai[u - base + 1]; j++) {
          MKL_INT v = aj[j - base] - base;
          if (levels[v] == l + 1) {
            children.push_back(v);
            levels[v] = -1;
          }
        }

        std::cout << "children.size: " << children.size() << std::endl;
        std::sort(children.begin(), children.end(),
                  [&degrees, base](const MKL_INT a, const MKL_INT b) {
                    if (degrees[a - base] == degrees[b - base])
                      return a < b;
                    return degrees[a - base] < degrees[b - base];
                  });
        // std::cout << "children size: " << children.size() << std::endl;
        for (auto i : children) {
          inv_perm[e++] = i + base;
          std::cout << i - base << " " << degrees[i - base] << std::endl;
        }
        children.resize(0);
      }
    }
  }
  std::reverse(std::execution::par_unseq, inv_perm.begin(), inv_perm.end());
  return inv_perm;
}
} // namespace reordering