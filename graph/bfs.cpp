#include "bfs.hpp"
#include "BitVector.hpp"
#include <algorithm>
#include <execution>
#include <iostream>
#include <ranges>
#include <omp.h>
#include "utils.h"

namespace graph {

template <typename ROWTYPE, typename COLTYPE, bool LASTLEVEL, bool TRACK>
bool BFSFunc(COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj, COLTYPE source, COLTYPE shortCutWidth,
             COLTYPE& height, COLTYPE& width, std::vector<COLTYPE>& levels, std::vector<COLTYPE>& lastLevel)
{
    levels.resize(rows);
    height = 0;
    width = 0;
    const COLTYPE INVALID = std::numeric_limits<COLTYPE>::max();
    std::fill_n(levels.begin(), levels.size(), INVALID);
    const COLTYPE base = ai[0]; // Get base indexing from first element
    
    // Use two frontiers for alternating levels
    std::vector<COLTYPE> frontier_a, frontier_b;
    frontier_a.reserve(256);
    frontier_b.reserve(256);
    
    std::vector<COLTYPE>* current_frontier = &frontier_a;
    std::vector<COLTYPE>* next_frontier = &frontier_b;
    
    current_frontier->clear();
    current_frontier->push_back(source - base);
    levels[source - base] = 0;

    COLTYPE widthCounter = 1;
    while (!current_frontier->empty())
    {
        next_frontier->clear();
        if constexpr (TRACK)
            width = std::max(width, static_cast<COLTYPE>(current_frontier->size()));
        
        // Process current level
        for (const auto u : *current_frontier)
        {
            for (ROWTYPE i = ai[u] - base; i < ai[u + 1] - base; i++)
            {
                auto v = aj[i] - base;
                if (levels[v] == INVALID)
                {
                    levels[v] = height + 1;
                    next_frontier->push_back(v);
                    if constexpr (TRACK)
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
        }
        
        // Move to next level
        height++;
        
        // Swap frontier pointers
        std::swap(current_frontier, next_frontier);
    }
    
    // After loop exits, current_frontier is empty and next_frontier has the last level
    // Copy last level if requested
    if constexpr (LASTLEVEL) {
        lastLevel.clear();
        lastLevel.reserve(next_frontier->size());
        for (auto v : *next_frontier) {
            lastLevel.push_back(v + base);
        }
    } else {
        lastLevel.clear();
    }
    
    // If not tracking, ensure width remains 0
    if constexpr (!TRACK) {
        width = 0;
    }
    return true;
}
template<typename T>
T atomic_fetch_set(T* base, std::size_t i, T new_val) {
  std::atomic_ref<T> ref(base[i]);
  return ref.exchange(new_val, std::memory_order_relaxed);
}

template <typename ROWTYPE, typename COLTYPE, bool LASTLEVEL, bool TRACK>
bool PBFSFunc(COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj, COLTYPE source,
              COLTYPE shortCutWidth, COLTYPE& height, COLTYPE& width, std::vector<COLTYPE>& levels,
              std::vector<COLTYPE>& lastLevel, int nthreads)
{
    levels.resize(rows);
    const COLTYPE INVALID = std::numeric_limits<COLTYPE>::max();
    std::fill(levels.begin(), levels.end(), INVALID);

    const COLTYPE base = ai[0];
    bool stat = true;

    std::vector<std::vector<COLTYPE>> wavefront_cur;
    std::vector<std::vector<COLTYPE>> wavefront_next;
    wavefront_cur.resize(nthreads);
    wavefront_next.resize(nthreads);
    height = 0;
    width = 0;

    levels[source - base] = 0;

    wavefront_cur[0].push_back(source - base);

#pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();

        while (true)
        {
            auto joined_local = wavefront_cur | std::views::join;
            const COLTYPE total_work = static_cast<COLTYPE>(std::ranges::distance(joined_local));
            if (total_work == 0)
            {
                break;
            }
            wavefront_next[tid].clear();
            if constexpr (TRACK)
            {
              if (shortCutWidth <= total_work)
              {
                  stat = false;
                  break;
              }
              width = std::max(width, total_work);
            }
            wavefront_next[tid].resize(0);
            const auto [chunk_start, chunk_end] = utils::LoadBalancedPartitionPos(total_work, tid, nthreads);
            auto partition_view = joined_local | std::views::drop(chunk_start) |
                                  std::views::take(chunk_end - chunk_start);

            for (const auto u : partition_view)
            {
                for (ROWTYPE k = ai[u] - base; k < ai[u + 1] - base; k++)
                {
                    auto v = aj[k] - base;
                    if constexpr (TRACK)
                    {
                        if (levels[v] == INVALID)
                        {
                            auto old_val = atomic_fetch_set(levels.data(), v, height + 1);
                            if (old_val == INVALID)
                            {
                                wavefront_next[tid].push_back(v);
                            }
                        }
                    }
                    else
                    {
                        if (levels[v] == INVALID)
                        {
                            levels[v] = height + 1;
                            wavefront_next[tid].push_back(v);
                        }
                    }
                }
            }
#pragma omp barrier
#pragma omp single
            {
                std::swap(wavefront_next, wavefront_cur);
                height++;
            }
        }
    }

    if constexpr (TRACK)
    {
        if (!stat)
            return false;
    }

    if constexpr (LASTLEVEL)
    {
        lastLevel.clear();
        for (const auto& vec : wavefront_next)
        {
            for (auto v : vec)
            {
                lastLevel.push_back(v + base);
            }
        }
        if constexpr (!TRACK)
        {
            std::sort(lastLevel.begin(), lastLevel.end());
            lastLevel.erase(std::unique(lastLevel.begin(), lastLevel.end()), lastLevel.end());
        }
    } else {
        lastLevel.clear();
    }
    
    // If not tracking, ensure width remains 0
    if constexpr (!TRACK) {
        width = 0;
    }
    return true;
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

template bool PBFSFunc<int, int, true, true>(int rows, int const* ai, int const* aj,
                                              int source, int shortCut, int& level,
                                              int& width, std::vector<int>& levels,
                                              std::vector<int>& lastLevel, int);

template bool PBFSFunc<int, int, true, false>(int rows, int const* ai, int const* aj,
                                               int source, int shortCut, int& level,
                                               int& width, std::vector<int>& levels,
                                               std::vector<int>& lastLevel, int);

template bool PBFSFunc<int, int, false, true>(int rows, int const* ai, int const* aj,
                                               int source, int shortCut, int& level,
                                               int& width, std::vector<int>& levels,
                                               std::vector<int>& lastLevel, int);

template bool PBFSFunc<int, int, false, false>(int rows, int const* ai, int const* aj,
                                                int source, int shortCut, int& level,
                                                int& width, std::vector<int>& levels,
                                                std::vector<int>& lastLevel, int);

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

template bool PBFSFunc<int64_t, int64_t, true, true>(int64_t rows, int64_t const* ai,
                                                      int64_t const* aj, int64_t source,
                                                      int64_t shortCut, int64_t& level,
                                                      int64_t& width,
                                                      std::vector<int64_t>& levels,
                                                      std::vector<int64_t>& lastLevel, int);

template bool PBFSFunc<int64_t, int64_t, true, false>(int64_t rows, int64_t const* ai,
                                                       int64_t const* aj, int64_t source,
                                                       int64_t shortCut, int64_t& level,
                                                       int64_t& width,
                                                       std::vector<int64_t>& levels,
                                                       std::vector<int64_t>& lastLevel, int);

template bool PBFSFunc<int64_t, int64_t, false, true>(int64_t rows, int64_t const* ai,
                                                       int64_t const* aj, int64_t source,
                                                       int64_t shortCut, int64_t& level,
                                                       int64_t& width,
                                                       std::vector<int64_t>& levels,
                                                       std::vector<int64_t>& lastLevel, int);

template bool PBFSFunc<int64_t, int64_t, false, false>(int64_t rows, int64_t const* ai,
                                                        int64_t const* aj, int64_t source,
                                                        int64_t shortCut, int64_t& level,
                                                        int64_t& width,
                                                        std::vector<int64_t>& levels,
                                                        std::vector<int64_t>& lastLevel, int);

} // namespace graph
