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

template <typename ROWTYPE, typename COLTYPE>
int MetisND(const COLTYPE nrows, const COLTYPE ncols,
            const ROWTYPE* ai, const COLTYPE* aj,
            COLTYPE* iperm, COLTYPE* perm, const MetisNDOptions& opts) {
  
  // METIS requires square matrix
  if (nrows != ncols) {
    return -1;
  }
  
  // METIS uses idx_t (typically int32 or int64 depending on build)
  // Build adjacency list without self-loops
  const ROWTYPE base = ai[0];
  std::vector<idx_t> xadj(nrows + 1);
  xadj[0] = base;
  
  // Count non-diagonal entries per row directly into xadj
  for (COLTYPE i = 0; i < nrows; ++i) {
    idx_t count = 0;
    for (ROWTYPE j = ai[i]; j < ai[i + 1]; ++j) {
      if (aj[j - base] != i + base) {  // Skip diagonal
        count++;
      }
    }
    xadj[i + 1] = xadj[i] + count;
  }
  
  // Fill adjncy without diagonal entries
  const idx_t nnz = xadj[nrows] - xadj[0];
  std::vector<idx_t> adjncy(nnz);
  idx_t pos = 0;
  for (COLTYPE i = 0; i < nrows; ++i) {
    for (ROWTYPE j = ai[i]; j < ai[i + 1]; ++j) {
      COLTYPE col = aj[j - base];
      if (col != i + base) {  // Skip diagonal
        adjncy[pos++] = static_cast<idx_t>(col);
      }
    }
  }
  
  // Prepare output arrays
  std::vector<idx_t> iperm_metis(nrows);
  std::vector<idx_t> perm_metis(nrows);
  
  // Set METIS options
  std::vector<idx_t> options(METIS_NOPTIONS);
  METIS_SetDefaultOptions(options.data());
  options[METIS_OPTION_NUMBERING] = base;
  options[METIS_OPTION_NSEPS] = opts.nseps;
  options[METIS_OPTION_NITER] = opts.niter;
  options[METIS_OPTION_SEED] = opts.seed;
  options[METIS_OPTION_COMPRESS] = opts.compress ? 1 : 0;
  options[METIS_OPTION_CCORDER] = opts.ccorder ? 1 : 0;
  options[METIS_OPTION_CTYPE] = opts.ctype;
  options[METIS_OPTION_RTYPE] = opts.rtype;
  options[METIS_OPTION_DBGLVL] = opts.dbglvl;
  
  // Call METIS nested dissection
  idx_t nvtxs = static_cast<idx_t>(nrows);
  int result = METIS_NodeND(&nvtxs, xadj.data(), adjncy.data(), NULL, 
                             options.data(), perm_metis.data(), iperm_metis.data());
  
  // Convert results back to requested type
  for (COLTYPE i = 0; i < nrows; ++i) {
    iperm[i] = static_cast<COLTYPE>(iperm_metis[i]);
    perm[i] = static_cast<COLTYPE>(perm_metis[i]);
  }
  
  return (result == METIS_OK) ? 0 : -1;
}

// Explicit template instantiations for common type combinations
template int MetisND<int32_t, int32_t>(const int32_t, const int32_t,
                                        const int32_t*, const int32_t*,
                                        int32_t*, int32_t*, const MetisNDOptions&);
template int MetisND<int64_t, int64_t>(const int64_t, const int64_t,
                                        const int64_t*, const int64_t*,
                                        int64_t*, int64_t*, const MetisNDOptions&);
template int MetisND<int64_t, int32_t>(const int32_t, const int32_t,
                                        const int64_t*, const int32_t*,
                                        int32_t*, int32_t*, const MetisNDOptions&);
#endif
} // namespace reordering