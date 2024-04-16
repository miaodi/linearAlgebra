#include "BFS.h"
#include "BitVector.h"
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
  std::vector<std::vector<MKL_INT>> bvc(max_threads);
  std::vector<std::vector<MKL_INT>> bvn(max_threads);

  // std::vector<bool> visited(mat->rows(), false);
  utils::BitVector visited(mat->rows());
  std::vector<MKL_INT> count_per_thread(max_threads + 1, 0);

  bvn[0].push_back(source - mat->mkl_base());
  level = 0;
  res[source - mat->mkl_base()] = level;
  // visited[source - mat->mkl_base()] = true;
  visited.set(source - mat->mkl_base());
  count_per_thread[1] = 1;
  MKL_INT total_work;
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
        total_work = count_per_thread[nthreads];
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
        level++;
        // std::cout << level << " " << total_work << std::endl;
      }
#pragma omp barrier
      if (total_work == 0)
        break;
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
            if (!visited.get(v)) {
              // visited[v] = true;
              visited.set(v);
              if (res[v] == -1) {
                res[v] = level;
                bvn[tid].push_back(v);
              }
            }
          }
        }
      }
      count_per_thread[tid + 1] = bvn[tid].size();
    }
  }
  level--;
  return res;
}
} // namespace reordering