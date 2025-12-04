#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <omp.h>

namespace utils {

// Parallel sort with two phases:
// Phase 1: Partition array into nthreads chunks, each thread sorts its chunk
// Phase 2: Hierarchical merge with n/2, n/4, ..., 1 threads
template<typename RandomIt, typename Compare>
void sort(RandomIt first, RandomIt last, Compare comp, int nthreads) {
    if (nthreads <= 0) {
        nthreads = omp_get_max_threads();
    }
    
    auto size = std::distance(first, last);
    if (size <= 1) return;
    
    // For very small arrays, just use std::sort
    if (size < nthreads * 1000) {
        std::sort(first, last, comp);
        return;
    }
    
    // Single parallel region for both phases to avoid thread creation overhead
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        
        // Phase 1: Partition and sort each chunk in parallel
        {
            // Calculate this thread's chunk boundaries
            size_t chunk_size = size / num_threads;
            size_t start = tid * chunk_size;
            size_t end = (tid == num_threads - 1) ? size : (tid + 1) * chunk_size;
            
            // Sort this thread's chunk
            std::sort(first + start, first + end, comp);
        }
        
        // Barrier: wait for all threads to finish sorting
        #pragma omp barrier
        
        // Phase 2: Hierarchical merge
        // Start with num_threads sorted chunks, merge pairs until we have 1 sorted array
        size_t chunk_size = size / num_threads;
        size_t num_chunks = num_threads;
        
        while (num_chunks > 1) {
            size_t active_threads = num_chunks / 2;
            
            // Only active threads participate in this merge level
            if (static_cast<size_t>(tid) < active_threads) {
                size_t left_start = tid * 2 * chunk_size;
                size_t mid = left_start + chunk_size;
                size_t right_end = (static_cast<size_t>(tid) == active_threads - 1 && num_chunks % 2 == 0) 
                                   ? static_cast<size_t>(size)
                                   : std::min(mid + chunk_size, static_cast<size_t>(size));
                
                if (mid < static_cast<size_t>(size) && right_end > mid) {
                    std::inplace_merge(first + left_start, first + mid, first + right_end, comp);
                }
            }
            
            // Barrier: wait for all merges at this level to complete
            #pragma omp barrier
            
            // Update for next iteration
            chunk_size *= 2;
            num_chunks = (num_chunks + 1) / 2; // Ceiling division
        }
    }
}

// Overload with default comparator
template<typename RandomIt>
void sort(RandomIt first, RandomIt last, int nthreads) {
    using value_type = typename std::iterator_traits<RandomIt>::value_type;
    sort(first, last, std::less<value_type>(), nthreads);
}

} // namespace parallel
