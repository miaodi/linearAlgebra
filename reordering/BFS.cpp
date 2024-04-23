#include "BFS.h"
#include "BitVector.h"
#include "mkl_sparse_mat.h"
#include <algorithm>
#include <execution>
#include <iostream>
#include <omp.h>

namespace reordering {
template <bool LASTLEVEL>
bool BFS_Fn(mkl_wrapper::mkl_sparse_mat const *const mat, int source,
            int shortCutWidth, MKL_INT &height, MKL_INT &width,
            std::vector<MKL_INT> &levels, std::vector<MKL_INT> &lastLevel) {
  levels.resize(mat->rows());
  lastLevel.resize(0);
  height = 0;
  std::fill_n(levels.begin(), levels.size(), -1);
  auto ai = mat->get_ai();
  auto aj = mat->get_aj();

  utils::CircularBuffer<MKL_INT> cb(
      std::max(1, static_cast<MKL_INT>(mat->rows() * .2)));
  cb.push(source - mat->mkl_base());
  levels[source - mat->mkl_base()] = 0;
  if constexpr (LASTLEVEL)
    lastLevel.push_back(source);
  int widthCounter = 1;
  while (!cb.isEmpty()) {
    auto u = cb.first();
    cb.shift();
    for (MKL_INT i = ai[u]; i < ai[u + 1]; i++) {
      auto v = aj[i] - mat->mkl_base();
      if (levels[v] == -1) {
        if (height < levels[u] + 1) {
          height = levels[u] + 1;
          width = std::max(width, widthCounter);
          widthCounter = 0;
          if constexpr (LASTLEVEL) {
            lastLevel.resize(0);
          }
        }
        levels[v] = height;
        if constexpr (LASTLEVEL)
          lastLevel.push_back(v + mat->mkl_base());
        if (!cb.available())
          cb.resizePreserve(cb.size() * 2);
        cb.push(v);
        if (++widthCounter >= shortCutWidth)
          return false;
        ;
      }
    }
  }
  height++;
  return true;
}

template <bool LASTLEVEL, bool RECORDLEVEL>
bool PBFS_Fn(mkl_wrapper::mkl_sparse_mat const *const mat, int source,
             int shortCutWidth, MKL_INT &height, MKL_INT &width,
             std::vector<MKL_INT> &levels, std::vector<MKL_INT> &lastLevel) {
  if constexpr (RECORDLEVEL) {
    levels.resize(mat->rows());
    std::fill_n(std::execution::par_unseq, levels.begin(), levels.size(), -1);
  }
  auto ai = mat->get_ai();
  auto aj = mat->get_aj();
  bool stat = true;
  int max_threads = omp_get_max_threads();
  static std::vector<std::vector<MKL_INT>> bvc;
  static std::vector<std::vector<MKL_INT>> bvn;
  bvc.resize(max_threads);
  bvn.resize(max_threads);

  // std::vector<bool> visited(mat->rows(), false);
  utils::BitVector visited(mat->rows());
  std::vector<MKL_INT> count_per_thread(max_threads + 1, 0);
  std::vector<MKL_INT> count_per_thread_prev(max_threads + 1, 0);
  height = 0;
  if constexpr (RECORDLEVEL) {
    levels[source - mat->mkl_base()] = 0;
  }
  bvn[0].push_back(source - mat->mkl_base());
  // visited[source - mat->mkl_base()] = true;
  visited.set(source - mat->mkl_base());
  count_per_thread[1] = 1;
  MKL_INT total_work;
  MKL_INT total_work_prev;
  int nthreads;
  std::vector<std::pair<int, int>> chunck_pos_pairs(max_threads + 1);
  chunck_pos_pairs[0] = std::make_pair(0, 0);
#pragma omp parallel shared(total_work, nthreads)
  {
    nthreads = omp_get_num_threads();
    const int tid = omp_get_thread_num();
    while (true) {
#pragma omp barrier
#pragma omp master
      {
        std::swap(bvn, bvc);
        std::inclusive_scan(count_per_thread.begin(), count_per_thread.end(),
                            count_per_thread.begin());
        if constexpr (LASTLEVEL) {
          total_work_prev = total_work;
        }
        total_work = count_per_thread[nthreads];
        width = std::max(width, total_work);
        int pos = 0;
        MKL_INT target = 0;
        for (int i = 0; i < nthreads; i++) {
          target +=
              total_work / nthreads + ((total_work % nthreads) > i ? 1 : 0);
          while (count_per_thread[pos + 1] < target)
            pos++;
          chunck_pos_pairs[i + 1] =
              std::make_pair(pos, target - count_per_thread[pos]);
        }
        if constexpr (LASTLEVEL) {
          if (total_work == 0) {
            lastLevel.resize(total_work_prev);
          } else {
            std::swap(count_per_thread, count_per_thread_prev);
          }
        }
        height++;
      }
#pragma omp barrier
      if (total_work == 0) {
        if constexpr (LASTLEVEL) {
          for (size_t i = 0; i < bvn[tid].size(); i++) {
            *(lastLevel.data() + count_per_thread_prev[tid] + i) =
                bvn[tid][i] + mat->mkl_base();
          }
        }
        bvn[tid].resize(0);
        break;
      } else if (total_work >= shortCutWidth) {
        stat = false;
        bvn[tid].resize(0);
        break;
      }
      bvn[tid].resize(0);
      // bvn[tid].clear();
      for (int i = chunck_pos_pairs[tid].first;
           i <= chunck_pos_pairs[tid + 1].first; i++) {
        int start = (i == chunck_pos_pairs[tid].first)
                        ? chunck_pos_pairs[tid].second
                        : 0;
        int end = (i == chunck_pos_pairs[tid + 1].first)
                      ? chunck_pos_pairs[tid + 1].second
                      : bvc[i].size();
        for (int j = start; j < end; j++) {
          for (MKL_INT k = ai[bvc[i][j]]; k < ai[bvc[i][j] + 1]; k++) {
            auto v = aj[k] - mat->mkl_base();
            if constexpr (RECORDLEVEL) {
              if (!visited.get(v)) {
                // visited[v] = true;
                visited.set(v);
                if (levels[v] == -1) {
                  levels[v] = height;
                  bvn[tid].push_back(v);
                }
              }
            } else {
              if (visited.testAndSet(v)) {
                bvn[tid].push_back(v);
              }
            }
          }
        }
      }
      count_per_thread[tid + 1] = bvn[tid].size();
    }
  }
  height--;
  return stat;
}

template bool BFS_Fn<true>(mkl_wrapper::mkl_sparse_mat const *const mat,
                           int source, int shortCut, MKL_INT &level,
                           MKL_INT &width, std::vector<MKL_INT> &levels,
                           std::vector<MKL_INT> &lastLevel);

template bool BFS_Fn<false>(mkl_wrapper::mkl_sparse_mat const *const mat,
                            int source, int shortCut, MKL_INT &level,
                            MKL_INT &width, std::vector<MKL_INT> &levels,
                            std::vector<MKL_INT> &lastLevel);

template bool PBFS_Fn<true, true>(mkl_wrapper::mkl_sparse_mat const *const mat,
                                  int source, int shortCut, MKL_INT &level,
                                  MKL_INT &width, std::vector<MKL_INT> &levels,
                                  std::vector<MKL_INT> &lastLevel);

template bool PBFS_Fn<true, false>(mkl_wrapper::mkl_sparse_mat const *const mat,
                                   int source, int shortCut, MKL_INT &level,
                                   MKL_INT &width, std::vector<MKL_INT> &levels,
                                   std::vector<MKL_INT> &lastLevel);

template bool PBFS_Fn<false, true>(mkl_wrapper::mkl_sparse_mat const *const mat,
                                   int source, int shortCut, MKL_INT &level,
                                   MKL_INT &width, std::vector<MKL_INT> &levels,
                                   std::vector<MKL_INT> &lastLevel);

template bool
PBFS_Fn<false, false>(mkl_wrapper::mkl_sparse_mat const *const mat, int source,
                      int shortCut, MKL_INT &level, MKL_INT &width,
                      std::vector<MKL_INT> &levels,
                      std::vector<MKL_INT> &lastLevel);
} // namespace reordering