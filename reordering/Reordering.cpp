#include "Reordering.h"
#include "UnionFind.h"
#include "utils.h"
#include <algorithm>
#include <execution>
#include <iostream>
#include <span>
#ifdef USE_METIS_LIB
#include <metis.h>
#endif

namespace reordering {

void NodeDegree(mkl_wrapper::mkl_sparse_mat const *const mat,
                std::vector<MKL_INT> &degrees) {
  degrees.resize(mat->rows());
  const MKL_INT base = mat->mkl_base();
  auto ai = mat->get_ai();

  for (MKL_INT i = 0; i < mat->rows(); i++) {
    degrees[i] = ai[i + 1] - ai[i];
  }
}

void PNodeDegree(mkl_wrapper::mkl_sparse_mat const *const mat,
                 std::vector<MKL_INT> &degrees) {
  degrees.resize(mat->rows());
  const MKL_INT base = mat->mkl_base();
  auto ai = mat->get_ai();

#pragma omp parallel for
  for (MKL_INT i = 0; i < mat->rows(); i++) {
    degrees[i] = ai[i + 1] - ai[i];
  }
}

void SerialCM(mkl_wrapper::mkl_sparse_mat const *const mat,
              std::vector<MKL_INT> &iperm, std::vector<MKL_INT> &perm) {
  // TODO: need to assert rows=cols
  std::vector<MKL_INT> degrees;
  PNodeDegree(mat, degrees); // get degrees of all nodes
  iperm.resize(mat->cols());

  auto parents = reordering::ParUnionFindRem(mat);

  std::vector<MKL_INT> compRoots;
  std::vector<MKL_INT> sortedComp;
  std::vector<MKL_INT> compPrefSum;

  const MKL_INT base = mat->mkl_base();
  const auto &ai = mat->get_ai();
  const auto &aj = mat->get_aj();

  reordering::ComponentsStat(parents, base, compRoots, sortedComp, compPrefSum);
  MKL_INT offset;
  MKL_INT source, target;
  MKL_INT e;
  std::vector<MKL_INT> prefix;
  std::vector<MKL_INT> children;

  for (int c = 0; c < compRoots.size(); c++) {
    offset = compPrefSum[c];
    // special treatment for components of size 1 and 2
    if (compPrefSum[c + 1] - compPrefSum[c] == 1) {
      iperm[offset] = sortedComp[offset];
      continue;
    } else if (compPrefSum[c + 1] - compPrefSum[c] == 2) {
      iperm[offset] = sortedComp[offset];
      iperm[offset + 1] = sortedComp[offset + 1];
      continue;
    }

    // select source node
    // source = reordering::MinDegreeNode(
    //              degrees, base,
    //              std::span(sortedComp.cbegin() + compPrefSum[c],
    //                        sortedComp.cbegin() + compPrefSum[c + 1]))
    //              .first -
    //          base;
    reordering::PseudoDiameter(
        mat, degrees,
        std::span(sortedComp.cbegin() + compPrefSum[c],
                  sortedComp.cbegin() + compPrefSum[c + 1]),
        source, target);
    e = offset;
    iperm[e++] = source;

    reordering::BFS bfs(reordering::BFS_Fn<false>);
    bfs(mat, source);
    auto &levels = bfs.getLevels();
    const auto height = bfs.getHeight();
    // std::cout << "height: " << height << std::endl;

    prefix.resize(height + 1);
    std::fill(prefix.begin(), prefix.end(), 0);
    for (MKL_INT p = compPrefSum[c]; p != compPrefSum[c + 1]; p++) {
      prefix[levels[sortedComp[p] - base] + 1]++;
    }
    for (MKL_INT l = 0; l < height; l++) {
      prefix[l + 1] += prefix[l];
    }

    children.reserve(bfs.getWidth());
    for (MKL_INT l = 0; l < height; l++) {
      for (MKL_INT r = prefix[l]; r != prefix[l + 1]; r++) {
        children.resize(0);
        MKL_INT u = iperm[r + offset] - base;
        for (MKL_INT j = ai[u] - base; j != ai[u + 1] - base; j++) {
          MKL_INT v = aj[j] - base;
          if (levels[v] == l + 1) {
            children.push_back(v);
            levels[v] = -1; // TODO: optimization is needed
          }
        }

        // pick nodes with the smallest degree
        std::sort(children.begin(), children.end(),
                  [&degrees](const MKL_INT a, const MKL_INT b) {
                    if (degrees[a] == degrees[b])
                      return a < b;
                    return degrees[a] < degrees[b];
                  });
#pragma ivdep
#pragma vector always
        for (size_t i = 0; i < children.size(); i++) {
          iperm[e + i] = children[i] + base;
        }
        e += children.size();
        // for (auto i : children)
        //   iperm[e++] = i + base;
      }
    }
  }
  std::reverse(std::execution::par_unseq, iperm.begin(), iperm.end());
  utils::inversePermute(perm, iperm, base);
}

#ifdef USE_METIS_LIB
void Metis(mkl_wrapper::mkl_sparse_mat const *const mat,
           std::vector<MKL_INT> &iperm, std::vector<MKL_INT> &perm) {

  iperm.resize(mat->cols());
  perm.resize(mat->cols());
  std::vector<MKL_INT> xadj;
  std::vector<MKL_INT> adjncy;
  mat->get_adjacency_graph(xadj, adjncy);

  std::vector<idx_t> options(METIS_NOPTIONS);
  METIS_SetDefaultOptions(options.data());
  options[METIS_OPTION_NUMBERING] = static_cast<MKL_INT>(mat->mkl_base());
  MKL_INT nvtxs = mat->rows();
  // perm[i] = k -> perm[i, k] = 1 -> C(i,*) = perm dot A(k,*)
  METIS_NodeND(&nvtxs, xadj.data(), adjncy.data(), NULL, options.data(),
               iperm.data(), perm.data());
}
#endif

template <typename COLTYPE>
template <typename ROWTYPE>
QuotientGraph<COLTYPE>::QuotientGraph(const COLTYPE nnodes, ROWTYPE const *ai,
                                      COLTYPE const *aj):_nodes(nnodes)
{
  const ROWTYPE base = ai[0];
  for (COLTYPE i = 0; i < nnodes; i++) {
    _nodes[i].id = i;
    _nodes[i].reserve(ai[i + 1] - ai[i]);
    for (ROWTYPE j_idx = ai[i] - base; j_idx < ai[i + 1] - base; j_idx++) {
      COLTYPE j = aj[j_idx] - base;
      if (j == i)
        continue;
      _nodes[i].adjacent_variables.push_back(j);
    }
    _nodes[i].degree = _nodes[i].adjacent_variables.size();
    _nodes[i].simple_variables.push_back(i);
  }
}



template class QuotientGraph<int>;
} // namespace reordering