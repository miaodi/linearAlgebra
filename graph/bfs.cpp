#include "bfs.hpp"
#include "BitVector.hpp"
#include <algorithm>
#include <execution>
#include <iostream>
#include <omp.h>
#include "circularbuffer.hpp"

namespace graph {

template <typename ROWTYPE, typename COLTYPE, bool LASTLEVEL, bool SHORTCUT>
bool BFSFunc(COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj, COLTYPE source, COLTYPE shortCutWidth,
             COLTYPE& height, COLTYPE& width, std::vector<COLTYPE>& levels, std::vector<COLTYPE>& lastLevel)
{
    levels.resize(rows);
    height = 0;
    width = 0;
    const COLTYPE INVALID = std::numeric_limits<COLTYPE>::max();
    std::fill_n(levels.begin(), levels.size(), INVALID);
    const COLTYPE base = ai[0]; // Get base indexing from first element
    const int capacity = 256;
    utils::CircularBuffer<COLTYPE> cb(capacity);
    lastLevel.reserve(capacity);
    lastLevel.clear();
    cb.push_back(source - base);
    levels[source - base] = 0;
    if constexpr (LASTLEVEL)
        lastLevel.push_back(source);

    COLTYPE nodes_left_in_level = cb.size();
    COLTYPE widthCounter = 1;
    while (nodes_left_in_level)
    {
        const auto u = cb.pop_front();
        for (ROWTYPE i = ai[u] - base; i < ai[u + 1] - base; i++)
        {
            auto v = aj[i] - base;
            if (levels[v] == INVALID)
            {
                levels[v] = height + 1;
                if constexpr (LASTLEVEL)
                    lastLevel.push_back(v + base);
                cb.push_back(v);
                if constexpr (SHORTCUT)
                {
                    if (++widthCounter >= shortCutWidth)
                        return false;
                }
                else
                {
                    ++widthCounter;
                }
            }
        }
        if (--nodes_left_in_level == 0)
        {
            height++;
            width = std::max(width, widthCounter);
            widthCounter = 0;
            if constexpr (LASTLEVEL)
            {
                lastLevel.clear();
            }
            nodes_left_in_level = cb.size();
            if (nodes_left_in_level == 0)
            {
                break;
            }
        }
    }
    return true;
}

template <typename ROWTYPE, typename COLTYPE, bool LASTLEVEL, bool RECORDLEVEL, bool SHORTCUT>
bool PBFSFunc(COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj,
              COLTYPE source, COLTYPE shortCutWidth, COLTYPE& height,
              COLTYPE& width, std::vector<COLTYPE>& levels,
              std::vector<COLTYPE>& lastLevel) {
  if constexpr (RECORDLEVEL) {
    levels.resize(rows);
    std::fill_n(std::execution::par_unseq, levels.begin(), levels.size(), -1);
  }
  
  const COLTYPE base = ai[0]; // Get base indexing from first element
  bool stat = true;
  int max_threads = omp_get_max_threads();
  
  static std::vector<std::vector<COLTYPE>> bvc;
  static std::vector<std::vector<COLTYPE>> bvn;
  bvc.resize(max_threads);
  bvn.resize(max_threads);

  utils::BitVector visited(rows);
  std::vector<COLTYPE> count_per_thread(max_threads + 1, 0);
  std::vector<COLTYPE> count_per_thread_prev(max_threads + 1, 0);
  
  height = 0;
  width = 0;
  
  if constexpr (RECORDLEVEL) {
    levels[source - base] = 0;
  }
  
  bvn[0].push_back(source - base);
  visited.set(source - base);
  count_per_thread[1] = 1;
  
  COLTYPE total_work;
  COLTYPE total_work_prev;
  int nthreads;
  std::vector<std::pair<int, int>> chunk_pos_pairs(max_threads + 1);
  chunk_pos_pairs[0] = std::make_pair(0, 0);
  
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
        COLTYPE target = 0;
        for (int i = 0; i < nthreads; i++) {
          target += total_work / nthreads + ((total_work % nthreads) > i ? 1 : 0);
          while (count_per_thread[pos + 1] < target)
            pos++;
          chunk_pos_pairs[i + 1] = std::make_pair(pos, target - count_per_thread[pos]);
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
            *(lastLevel.data() + count_per_thread_prev[tid] + i) = bvn[tid][i] + base;
          }
        }
        bvn[tid].resize(0);
        break;
      }
      if constexpr (SHORTCUT) {
        if (total_work >= shortCutWidth) {
          stat = false;
          bvn[tid].resize(0);
          break;
        }
      }
      
      bvn[tid].resize(0);
      
      for (int i = chunk_pos_pairs[tid].first; i <= chunk_pos_pairs[tid + 1].first; i++) {
        int start = (i == chunk_pos_pairs[tid].first) ? chunk_pos_pairs[tid].second : 0;
        int end = (i == chunk_pos_pairs[tid + 1].first) ? chunk_pos_pairs[tid + 1].second
                                                          : bvc[i].size();
        for (int j = start; j < end; j++) {
          for (ROWTYPE k = ai[bvc[i][j]] - base; k < ai[bvc[i][j] + 1] - base; k++) {
            auto v = aj[k] - base;
            if constexpr (RECORDLEVEL) {
              if (!visited.get(v)) {
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

// Explicit template instantiations for common types
template bool BFSFunc<int, int, true, true>(int rows, int const* ai, int const* aj,
                                             int source, int shortCut, int& level,
                                             int& width, std::vector<int>& levels,
                                             std::vector<int>& lastLevel);

template bool BFSFunc<int, int, false, true>(int rows, int const* ai, int const* aj,
                                              int source, int shortCut, int& level,
                                              int& width, std::vector<int>& levels,
                                              std::vector<int>& lastLevel);

template bool BFSFunc<int, int, true, false>(int rows, int const* ai, int const* aj,
                                              int source, int shortCut, int& level,
                                              int& width, std::vector<int>& levels,
                                              std::vector<int>& lastLevel);

template bool BFSFunc<int, int, false, false>(int rows, int const* ai, int const* aj,
                                               int source, int shortCut, int& level,
                                               int& width, std::vector<int>& levels,
                                               std::vector<int>& lastLevel);

template bool PBFSFunc<int, int, true, true, true>(int rows, int const* ai, int const* aj,
                                                    int source, int shortCut, int& level,
                                                    int& width, std::vector<int>& levels,
                                                    std::vector<int>& lastLevel);

template bool PBFSFunc<int, int, true, false, true>(int rows, int const* ai, int const* aj,
                                                     int source, int shortCut, int& level,
                                                     int& width, std::vector<int>& levels,
                                                     std::vector<int>& lastLevel);

template bool PBFSFunc<int, int, false, true, true>(int rows, int const* ai, int const* aj,
                                                     int source, int shortCut, int& level,
                                                     int& width, std::vector<int>& levels,
                                                     std::vector<int>& lastLevel);

template bool PBFSFunc<int, int, false, false, true>(int rows, int const* ai, int const* aj,
                                                      int source, int shortCut, int& level,
                                                      int& width, std::vector<int>& levels,
                                                      std::vector<int>& lastLevel);

template bool PBFSFunc<int, int, true, true, false>(int rows, int const* ai, int const* aj,
                                                     int source, int shortCut, int& level,
                                                     int& width, std::vector<int>& levels,
                                                     std::vector<int>& lastLevel);

template bool PBFSFunc<int, int, true, false, false>(int rows, int const* ai, int const* aj,
                                                      int source, int shortCut, int& level,
                                                      int& width, std::vector<int>& levels,
                                                      std::vector<int>& lastLevel);

template bool PBFSFunc<int, int, false, true, false>(int rows, int const* ai, int const* aj,
                                                      int source, int shortCut, int& level,
                                                      int& width, std::vector<int>& levels,
                                                      std::vector<int>& lastLevel);

template bool PBFSFunc<int, int, false, false, false>(int rows, int const* ai, int const* aj,
                                                       int source, int shortCut, int& level,
                                                       int& width, std::vector<int>& levels,
                                                       std::vector<int>& lastLevel);

// int64_t instantiations
template bool BFSFunc<int64_t, int64_t, true, true>(int64_t rows, int64_t const* ai,
                                                     int64_t const* aj, int64_t source,
                                                     int64_t shortCut, int64_t& level,
                                                     int64_t& width, std::vector<int64_t>& levels,
                                                     std::vector<int64_t>& lastLevel);

template bool BFSFunc<int64_t, int64_t, false, true>(int64_t rows, int64_t const* ai,
                                                      int64_t const* aj, int64_t source,
                                                      int64_t shortCut, int64_t& level,
                                                      int64_t& width, std::vector<int64_t>& levels,
                                                      std::vector<int64_t>& lastLevel);

template bool BFSFunc<int64_t, int64_t, true, false>(int64_t rows, int64_t const* ai,
                                                      int64_t const* aj, int64_t source,
                                                      int64_t shortCut, int64_t& level,
                                                      int64_t& width, std::vector<int64_t>& levels,
                                                      std::vector<int64_t>& lastLevel);

template bool BFSFunc<int64_t, int64_t, false, false>(int64_t rows, int64_t const* ai,
                                                       int64_t const* aj, int64_t source,
                                                       int64_t shortCut, int64_t& level,
                                                       int64_t& width, std::vector<int64_t>& levels,
                                                       std::vector<int64_t>& lastLevel);

template bool PBFSFunc<int64_t, int64_t, true, true, true>(int64_t rows, int64_t const* ai,
                                                            int64_t const* aj, int64_t source,
                                                            int64_t shortCut, int64_t& level,
                                                            int64_t& width,
                                                            std::vector<int64_t>& levels,
                                                            std::vector<int64_t>& lastLevel);

template bool PBFSFunc<int64_t, int64_t, true, false, true>(int64_t rows, int64_t const* ai,
                                                             int64_t const* aj, int64_t source,
                                                             int64_t shortCut, int64_t& level,
                                                             int64_t& width,
                                                             std::vector<int64_t>& levels,
                                                             std::vector<int64_t>& lastLevel);

template bool PBFSFunc<int64_t, int64_t, false, true, true>(int64_t rows, int64_t const* ai,
                                                             int64_t const* aj, int64_t source,
                                                             int64_t shortCut, int64_t& level,
                                                             int64_t& width,
                                                             std::vector<int64_t>& levels,
                                                             std::vector<int64_t>& lastLevel);

template bool PBFSFunc<int64_t, int64_t, false, false, true>(int64_t rows, int64_t const* ai,
                                                              int64_t const* aj, int64_t source,
                                                              int64_t shortCut, int64_t& level,
                                                              int64_t& width,
                                                              std::vector<int64_t>& levels,
                                                              std::vector<int64_t>& lastLevel);

template bool PBFSFunc<int64_t, int64_t, true, true, false>(int64_t rows, int64_t const* ai,
                                                             int64_t const* aj, int64_t source,
                                                             int64_t shortCut, int64_t& level,
                                                             int64_t& width,
                                                             std::vector<int64_t>& levels,
                                                             std::vector<int64_t>& lastLevel);

template bool PBFSFunc<int64_t, int64_t, true, false, false>(int64_t rows, int64_t const* ai,
                                                              int64_t const* aj, int64_t source,
                                                              int64_t shortCut, int64_t& level,
                                                              int64_t& width,
                                                              std::vector<int64_t>& levels,
                                                              std::vector<int64_t>& lastLevel);

template bool PBFSFunc<int64_t, int64_t, false, true, false>(int64_t rows, int64_t const* ai,
                                                              int64_t const* aj, int64_t source,
                                                              int64_t shortCut, int64_t& level,
                                                              int64_t& width,
                                                              std::vector<int64_t>& levels,
                                                              std::vector<int64_t>& lastLevel);

template bool PBFSFunc<int64_t, int64_t, false, false, false>(int64_t rows, int64_t const* ai,
                                                               int64_t const* aj, int64_t source,
                                                               int64_t shortCut, int64_t& level,
                                                               int64_t& width,
                                                               std::vector<int64_t>& levels,
                                                               std::vector<int64_t>& lastLevel);

} // namespace graph
