#include "Reordering.h"
#include "UnionFind.h"
#include "utils.h"
#include <algorithm>
#include <execution>
#include <iostream>
#include <span>
#ifdef USE_METIS_LIB
#ifdef USE_MTMETIS
#include <mtmetis.h>
#else
#include <metis.h>
#endif
#endif

namespace reordering {

// Template implementation
template <typename ROWTYPE, typename COLTYPE>
void NodeDegree(COLTYPE rows, const ROWTYPE* ai, COLTYPE* degrees, int numthreads)
{
    if (numthreads <= 0)
    {
        numthreads = omp_get_max_threads();
    }

// Parallel version with SIMD vectorization
#pragma omp parallel for simd num_threads(numthreads)
    for (COLTYPE i = 0; i < rows; i++)
    {
        degrees[i] = ai[i + 1] - ai[i];
    }
}

// Explicit instantiations
template void NodeDegree<int, int>(int rows, const int* ai, int* degrees, int numthreads);
template void NodeDegree<long, int>(int rows, const long* ai, int* degrees, int numthreads);
template void NodeDegree<long, long>(long rows, const long* ai, long* degrees, int numthreads);

// void SerialCM(mkl_wrapper::mkl_sparse_mat const *const mat,
//               std::vector<MKL_INT> &iperm, std::vector<MKL_INT> &perm) {
//   // TODO: need to assert rows=cols
//   std::vector<MKL_INT> degrees;
//   PNodeDegree(mat, degrees); // get degrees of all nodes
//   iperm.resize(mat->cols());

//   auto parents = reordering::ParUnionFindRem(mat);

//   std::vector<MKL_INT> compRoots;
//   std::vector<MKL_INT> sortedComp;
//   std::vector<MKL_INT> compPrefSum;

//   const MKL_INT base = mat->mkl_base();
//   const auto &ai = mat->get_ai();
//   const auto &aj = mat->get_aj();

//   reordering::ComponentsStat(parents, base, compRoots, sortedComp, compPrefSum);
//   MKL_INT offset;
//   MKL_INT source, target;
//   MKL_INT e;
//   std::vector<MKL_INT> prefix;
//   std::vector<MKL_INT> children;

//   for (int c = 0; c < compRoots.size(); c++) {
//     offset = compPrefSum[c];
//     // special treatment for components of size 1 and 2
//     if (compPrefSum[c + 1] - compPrefSum[c] == 1) {
//       iperm[offset] = sortedComp[offset];
//       continue;
//     } else if (compPrefSum[c + 1] - compPrefSum[c] == 2) {
//       iperm[offset] = sortedComp[offset];
//       iperm[offset + 1] = sortedComp[offset + 1];
//       continue;
//     }

//     // select source node
//     // source = reordering::MinDegreeNode(
//     //              degrees, base,
//     //              std::span(sortedComp.cbegin() + compPrefSum[c],
//     //                        sortedComp.cbegin() + compPrefSum[c + 1]))
//     //              .first -
//     //          base;
//     reordering::PseudoDiameter(
//         mat, degrees,
//         std::span(sortedComp.cbegin() + compPrefSum[c],
//                   sortedComp.cbegin() + compPrefSum[c + 1]),
//         source, target);
//     e = offset;
//     iperm[e++] = source;

//     reordering::BFS bfs(reordering::BFS_Fn<false>);
//     bfs(mat, source);
//     auto &levels = bfs.getLevels();
//     const auto height = bfs.getHeight();
//     // std::cout << "height: " << height << std::endl;

//     prefix.resize(height + 1);
//     std::fill(prefix.begin(), prefix.end(), 0);
//     for (MKL_INT p = compPrefSum[c]; p != compPrefSum[c + 1]; p++) {
//       prefix[levels[sortedComp[p] - base] + 1]++;
//     }
//     for (MKL_INT l = 0; l < height; l++) {
//       prefix[l + 1] += prefix[l];
//     }

//     children.reserve(bfs.getWidth());
//     for (MKL_INT l = 0; l < height; l++) {
//       for (MKL_INT r = prefix[l]; r != prefix[l + 1]; r++) {
//         children.resize(0);
//         MKL_INT u = iperm[r + offset] - base;
//         for (MKL_INT j = ai[u] - base; j != ai[u + 1] - base; j++) {
//           MKL_INT v = aj[j] - base;
//           if (levels[v] == l + 1) {
//             children.push_back(v);
//             levels[v] = -1; // TODO: optimization is needed
//           }
//         }

//         // pick nodes with the smallest degree
//         std::sort(children.begin(), children.end(),
//                   [&degrees](const MKL_INT a, const MKL_INT b) {
//                     if (degrees[a] == degrees[b])
//                       return a < b;
//                     return degrees[a] < degrees[b];
//                   });
// #pragma ivdep
// #pragma vector always
//         for (size_t i = 0; i < children.size(); i++) {
//           iperm[e + i] = children[i] + base;
//         }
//         e += children.size();
//         // for (auto i : children)
//         //   iperm[e++] = i + base;
//       }
//     }
//   }
//   std::reverse(std::execution::par_unseq, iperm.begin(), iperm.end());
//   utils::inversePermute(perm, iperm, base);
// }

#ifdef USE_METIS_LIB
template <typename ROWTYPE, typename COLTYPE>
int MetisND(const COLTYPE nrows, const COLTYPE ncols,
            const ROWTYPE* xadj, const COLTYPE* adjncy,
            COLTYPE* iperm, COLTYPE* perm, const MetisNDOptions& opts) {
  
  // METIS requires square matrix
  if (nrows != ncols) {
    return -1;
  }
  
#ifdef USE_MTMETIS
  using metis_idx_t = mtmetis_vtx_type;
  using metis_adj_t = mtmetis_adj_type;
  using metis_pid_t = mtmetis_pid_type;
  constexpr int SUCCESS_CODE = MTMETIS_SUCCESS;
#else
  using metis_idx_t = idx_t;
  using metis_adj_t = idx_t;
  using metis_pid_t = idx_t;
  constexpr int SUCCESS_CODE = METIS_OK;
#endif
  
  // Copy xadj and adjncy to METIS types (assuming input is already zero-based with diagonals removed)
  const metis_adj_t nnz = xadj[nrows];
  std::vector<metis_adj_t> xadj_metis(xadj, xadj + nrows + 1);
  std::vector<metis_idx_t> adjncy_metis(adjncy, adjncy + nnz);
  
  // Prepare output arrays
  std::vector<metis_pid_t> iperm_metis(nrows);
  std::vector<metis_pid_t> perm_metis(nrows);
  
  metis_idx_t nvtxs = static_cast<metis_idx_t>(nrows);
  int result;
  
#ifdef USE_MTMETIS
  double options[MTMETIS_NOPTIONS];
  result = MTMETIS_NodeND(&nvtxs, xadj_metis.data(), adjncy_metis.data(), NULL,
                          options, perm_metis.data(), iperm_metis.data());
#else
  std::vector<idx_t> options(METIS_NOPTIONS);
  METIS_SetDefaultOptions(options.data());
  options[METIS_OPTION_NUMBERING] = 0;  // Zero-based indexing
  options[METIS_OPTION_NSEPS] = opts.nseps;
  options[METIS_OPTION_NITER] = opts.niter;
  options[METIS_OPTION_SEED] = opts.seed;
  options[METIS_OPTION_COMPRESS] = opts.compress ? 1 : 0;
  options[METIS_OPTION_CCORDER] = opts.ccorder ? 1 : 0;
  options[METIS_OPTION_CTYPE] = opts.ctype;
  options[METIS_OPTION_RTYPE] = opts.rtype;
  options[METIS_OPTION_DBGLVL] = opts.dbglvl;
  
  result = METIS_NodeND(&nvtxs, xadj_metis.data(), adjncy_metis.data(), NULL, 
                        options.data(), perm_metis.data(), iperm_metis.data());
#endif
  
  // Convert results back to requested type
  for (COLTYPE i = 0; i < nrows; ++i) {
    iperm[i] = static_cast<COLTYPE>(iperm_metis[i]);
    perm[i] = static_cast<COLTYPE>(perm_metis[i]);
  }
  
  return (result == SUCCESS_CODE) ? 0 : -1;
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