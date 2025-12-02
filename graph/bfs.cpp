#include "bfs.hpp"
#include "BitVector.hpp"
#include <algorithm>
#include <execution>
#include <iostream>
#include <omp.h>

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
    
    // Use lastLevel as one frontier, and a local vector as the other
    std::vector<COLTYPE> next_frontier;
    next_frontier.reserve(256);
    
    lastLevel.clear();
    lastLevel.reserve(256);
    lastLevel.push_back(source - base);
    levels[source - base] = 0;

    COLTYPE widthCounter = 1;
    while (!lastLevel.empty())
    {
        width = std::max(width, static_cast<COLTYPE>(lastLevel.size()));
        
        // Process current level
        for (const auto u : lastLevel)
        {
            for (ROWTYPE i = ai[u] - base; i < ai[u + 1] - base; i++)
            {
                auto v = aj[i] - base;
                if (levels[v] == INVALID)
                {
                    levels[v] = height + 1;
                    next_frontier.push_back(v);
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
        }
        
        // Move to next level
        height++;
        
        // Swap frontiers
        lastLevel.clear();
        std::swap(lastLevel, next_frontier);
    }
    
    // Add base offset back to lastLevel if needed
    if constexpr (LASTLEVEL) {
        for (auto& v : lastLevel) {
            v += base;
        }
    }
    
    return true;
}

template <typename ROWTYPE, typename COLTYPE, bool LASTLEVEL, bool SHORTCUT>
bool PBFSFunc(COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj, COLTYPE source,
              COLTYPE shortCutWidth, COLTYPE& height, COLTYPE& width, std::vector<COLTYPE>& levels,
              std::vector<COLTYPE>& lastLevel, int nthreads)
{
    levels.resize(rows);
    const COLTYPE INVALID = std::numeric_limits<COLTYPE>::max();
    std::fill_n(std::execution::par_unseq, levels.begin(), levels.size(), INVALID);

    const COLTYPE base = ai[0]; // Get base indexing from first element
    bool stat = true;

    std::vector<std::vector<COLTYPE>> wavefront_cur;
    std::vector<std::vector<COLTYPE>> wavefront_next;
    wavefront_cur.resize(nthreads);
    wavefront_next.resize(nthreads);

    utils::BitVector visited(rows);
    std::vector<COLTYPE> count_per_thread(nthreads + 1, 0);
    std::vector<COLTYPE> count_per_thread_prev(nthreads + 1, 0);

    height = 0;
    width = 0;

    levels[source - base] = 0;

    wavefront_next[0].push_back(source - base);
    visited.set(source - base);
    count_per_thread[1] = 1;

    COLTYPE total_work;
    COLTYPE total_work_prev;
    std::vector<std::pair<int, int>> chunk_pos_pairs(nthreads + 1);
    chunk_pos_pairs[0] = std::make_pair(0, 0);

#pragma omp parallel num_threads(nthreads) shared(total_work)
    {
        const int tid = omp_get_thread_num();

        while (true)
        {
#pragma omp barrier
#pragma omp master
            {
                std::swap(wavefront_next, wavefront_cur);
                std::inclusive_scan(count_per_thread.begin(), count_per_thread.end(),
                                    count_per_thread.begin());
                if constexpr (LASTLEVEL)
                {
                    total_work_prev = total_work;
                }
                total_work = count_per_thread[nthreads];
                width = std::max(width, total_work);

                int pos = 0;
                COLTYPE target = 0;
                for (int i = 0; i < nthreads; i++)
                {
                    target += total_work / nthreads + ((total_work % nthreads) > i ? 1 : 0);
                    while (count_per_thread[pos + 1] < target)
                        pos++;
                    chunk_pos_pairs[i + 1] = std::make_pair(pos, target - count_per_thread[pos]);
                }

                if constexpr (LASTLEVEL)
                {
                    if (total_work == 0)
                    {
                        lastLevel.resize(total_work_prev);
                    }
                    else
                    {
                        std::swap(count_per_thread, count_per_thread_prev);
                    }
                }
                height++;
            }
#pragma omp barrier

            if (total_work == 0)
            {
                if constexpr (LASTLEVEL)
                {
                    for (size_t i = 0; i < wavefront_next[tid].size(); i++)
                    {
                        *(lastLevel.data() + count_per_thread_prev[tid] + i) = wavefront_next[tid][i] + base;
                    }
                }
                wavefront_next[tid].resize(0);
                break;
            }
            if constexpr (SHORTCUT)
            {
                if (total_work >= shortCutWidth)
                {
                    stat = false;
                    wavefront_next[tid].resize(0);
                    break;
                }
            }

            wavefront_next[tid].resize(0);

            for (int i = chunk_pos_pairs[tid].first; i <= chunk_pos_pairs[tid + 1].first; i++)
            {
                int start = (i == chunk_pos_pairs[tid].first) ? chunk_pos_pairs[tid].second : 0;
                int end = (i == chunk_pos_pairs[tid + 1].first) ? chunk_pos_pairs[tid + 1].second
                                                                : wavefront_cur[i].size();
                for (int j = start; j < end; j++)
                {
                    for (ROWTYPE k = ai[wavefront_cur[i][j]] - base; k < ai[wavefront_cur[i][j] + 1] - base; k++)
                    {
                        auto v = aj[k] - base;
                        if (!visited.get(v))
                        {
                            visited.set(v);
                            if (levels[v] == INVALID)
                            {
                                levels[v] = height;
                                wavefront_next[tid].push_back(v);
                            }
                        }
                    }
                }
            }
            count_per_thread[tid + 1] = wavefront_next[tid].size();
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
