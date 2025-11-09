#include "sp_ops.hpp"
#include "utils.h"
#include <atomic>
#include <cstring>

namespace matrix_utils
{

template <typename ROWTYPE, typename COLTYPE, bool KEEPDIAG>
void AATSymbolic( const COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj, ROWTYPE* ai_AAT )
{
    const int base = ai[0];
    ai_AAT[0] = base;

    std::memset( ai_AAT + 1, 0, size * sizeof( ROWTYPE ) );

    std::vector<ROWTYPE> start_pos( size );
    ROWTYPE j, k;
    for ( COLTYPE i = 0; i < size; i++ )
    {
        start_pos[i] = ai[i] - base;
        for ( j = ai[i] - base; j < ai[i + 1] - base; j++ )
        {
            COLTYPE col = aj[j] - base;
            if ( col > i )
            {
                break; // skip upper triangle
            }
            else if ( col == i )
            {
                if constexpr ( KEEPDIAG )
                    ai_AAT[i + 1]++;
                j++;
                break;
            }

            ai_AAT[col + 1]++; // increment the row size for AAT[col, i]
            ai_AAT[i + 1]++;   // increment the row size for AAT[i, col]
            for ( k = start_pos[col]; k < ai[col + 1] - base; k++ )
            {
                COLTYPE col2 = aj[k] - base;
                if ( col2 == i )
                {
                    k++;
                    break;
                }
                else if ( col2 > i )
                {
                    break;
                }
                ai_AAT[col2 + 1]++; // increment the row size for AAT[col2, col]
                ai_AAT[col + 1]++;  // increment the row size for AAT[col, col2]
            }
            start_pos[col] = k;
        }
        start_pos[i] = j;
    }
    for ( COLTYPE i = 0; i < size; i++ )
    {
        for ( ROWTYPE j = start_pos[i]; j < ai[i + 1] - base; j++ )
        {
            COLTYPE col = aj[j] - base;
            ai_AAT[col + 1]++; // increment the row size for AAT[col, i]
            ai_AAT[i + 1]++;   // increment the row size for AAT[i, col]
        }
    }

    // Convert counts to CSR row pointers using parallel prefix sum
    // Single parallel region with 3 phases to minimize OpenMP overhead
    const int num_threads = omp_get_max_threads();
    std::vector<ROWTYPE> thread_sums( num_threads + 1, 0 );

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        const COLTYPE chunk_size = ( size + nthreads ) / nthreads;
        const COLTYPE start = tid * chunk_size + 1;
        const COLTYPE end =
            std::min( start + chunk_size, static_cast<COLTYPE>( size + 1 ) );

        // Phase 1: Parallel local prefix sums per thread
        if ( start < end )
        {
            for ( COLTYPE i = start; i < end; i++ )
            {
                ai_AAT[i] += ai_AAT[i - 1];
            }
            // Store the last value for global adjustment
            thread_sums[tid + 1] = ai_AAT[end - 1];
        }

// Barrier: Wait for all threads to finish Phase 1
#pragma omp barrier

// Phase 2: Sequential scan of thread sums (done by thread 0 only)
#pragma omp single
        {
            for ( int i = 1; i <= nthreads; i++ )
            {
                thread_sums[i] += thread_sums[i - 1];
            }
        }
        // Implicit barrier at end of single

        // Phase 3: Parallel adjustment of each chunk
        if ( tid > 0 && start < end )
        {
            const ROWTYPE offset = thread_sums[tid];
            for ( COLTYPE i = start; i < end; i++ )
            {
                ai_AAT[i] += offset;
            }
        }
    }
}

template <typename ROWTYPE, typename COLTYPE, bool KEEPDIAG>
void AATNumeric( const COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj, ROWTYPE const* ai_AAT, COLTYPE* aj_AAT )
{
    const ROWTYPE base = ai[0];

    std::vector<ROWTYPE> start_pos( size );
    std::vector<ROWTYPE> start_pos_AAT( size );
    ROWTYPE j, k;
    for ( COLTYPE i = 0; i < size; i++ )
    {
        start_pos[i] = ai[i] - base;
        start_pos_AAT[i] = ai_AAT[i] - base;
        for ( j = ai[i] - base; j < ai[i + 1] - base; j++ )
        {
            COLTYPE col = aj[j] - base;
            if ( col > i )
            {
                break; // skip upper triangle
            }
            else if ( col == i )
            {
                if constexpr ( KEEPDIAG )
                {
                    aj_AAT[start_pos_AAT[i]++] = i + base;
                }
                j++;
                break;
            }
            aj_AAT[start_pos_AAT[col]++] = i + base; // AAT[col, i]
            aj_AAT[start_pos_AAT[i]++] = col + base; // AAT[i, col]
            for ( k = start_pos[col]; k < ai[col + 1] - base; k++ )
            {
                COLTYPE col2 = aj[k] - base;
                if ( col2 == i )
                {
                    k++;
                    break;
                }
                else if ( col2 > i )
                {
                    break;
                }
                aj_AAT[start_pos_AAT[col2]++] = col + base; // AAT[col2, col]
                aj_AAT[start_pos_AAT[col]++] = col2 + base; // AAT[col, col2]
            }
            start_pos[col] = k;
        }
        start_pos[i] = j;
    }
    for ( COLTYPE i = 0; i < size; i++ )
    {
        for ( ROWTYPE j = start_pos[i]; j < ai[i + 1] - base; j++ )
        {
            COLTYPE col = aj[j] - base;
            aj_AAT[start_pos_AAT[col]++] = i + base; // AAT[col, i]
            aj_AAT[start_pos_AAT[i]++] = col + base; // AAT[i, col]
        }
        std::sort( aj_AAT + ai_AAT[i] - base,
                   aj_AAT + ai_AAT[i + 1] - base ); // sort the row
    }
}

// Implementation of APlusATStruct struct methods
template <typename ROWTYPE, typename COLTYPE, bool KEEPDIAG>
void APlusATStruct<ROWTYPE, COLTYPE, KEEPDIAG>::operator()( const COLTYPE size,
                                                            ROWTYPE const* ai,
                                                            COLTYPE const* aj,
                                                            ROWTYPE* ai_APlusAT,
                                                            COLTYPE* aj_APlusAT )
{
    // Phase 1: Symbolic prefix (compute row pointers)
    prefix( size, ai, aj );

    // Phase 2: Fill, sort, compact, and copy results
    fillAndCompact( size, ai, aj, ai_APlusAT, aj_APlusAT );
}

template <typename ROWTYPE, typename COLTYPE, bool KEEPDIAG>
void APlusATStruct<ROWTYPE, COLTYPE, KEEPDIAG>::prefix( const COLTYPE size,
                                                        ROWTYPE const* ai,
                                                        COLTYPE const* aj )
{
    const ROWTYPE base = ai[0];

    // Allocate/resize result row pointers
    _APAT.ai.resize( size + 1 );
    ROWTYPE* ai_APlusAT = _APAT.ai.data();

    // Initialize row sizes to zero
    std::memset( ai_APlusAT, 0, ( size + 1 ) * sizeof( ROWTYPE ) );

    _thread_sums.resize( _nthreads + 1 );
    _thread_sums[0] = base;

#pragma omp parallel num_threads( _nthreads )
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();

        // Phase 1: Count entries for each row (parallelized with atomic operations)
        // Use LoadPrefixBalancedPartitionPos for load-balanced work distribution
        auto [row_start, row_end] =
            utils::LoadPrefixBalancedPartitionPos( ai, ai + size, tid, nthreads );

        for ( COLTYPE i = row_start; i < row_end; i++ )
        {
            ROWTYPE local_count = 0; // Count for current row i (no race condition)

            for ( ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; j++ )
            {
                COLTYPE col = aj[j] - base;

                if ( col == i )
                {
                    // Diagonal entry: appears once in A+A^T (if KEEPDIAG is true)
                    if constexpr ( KEEPDIAG )
                    {
                        local_count++;
                    }
                }
                else
                {
                    // Off-diagonal: A[i,col] contributes to row i (local, no race)
                    local_count++;

// A[col,i] contributes to row col (different row, needs atomic)
#pragma omp atomic
                    ai_APlusAT[col + 1]++;
                }
            }
#pragma omp atomic
            // Write local count to row i (no race condition for this row)
            ai_APlusAT[i + 1] += local_count;
        }

#pragma omp barrier

        // Phase 2: Parallel prefix sum - Local scan per thread
        auto [start, end] = utils::LoadBalancedPartitionPos( size, tid, nthreads );
        ROWTYPE local_sum = 0;
        for ( COLTYPE i = start; i < end; i++ )
        {
            local_sum += ai_APlusAT[i + 1];
        }
        // Store the last value for global adjustment
        _thread_sums[tid + 1] = local_sum;

#pragma omp barrier

#pragma omp single
        {
            for ( int i = 1; i <= nthreads; i++ )
            {
                _thread_sums[i] += _thread_sums[i - 1];
            }
            ai_APlusAT[size] = _thread_sums[nthreads];
        }

        const ROWTYPE offset = _thread_sums[tid];
        ai_APlusAT[start] = offset;
        for ( COLTYPE i = start; i < end - 1; i++ )
        {
            ai_APlusAT[i + 1] += ai_APlusAT[i];
        }
    }
}

template <typename ROWTYPE, typename COLTYPE, bool KEEPDIAG>
void APlusATStruct<ROWTYPE, COLTYPE, KEEPDIAG>::fillAndCompact( const COLTYPE size,
                                                                ROWTYPE const* ai,
                                                                COLTYPE const* aj,
                                                                ROWTYPE* ai_APlusAT,
                                                                COLTYPE* aj_APlusAT )
{
    const ROWTYPE base = ai[0];
    ROWTYPE* ai_internal = _APAT.ai.data();

    // Allocate space for column indices based on symbolic phase
    const ROWTYPE nnz_with_duplicates = ai_internal[size] - base;
    _APAT.aj.resize( nnz_with_duplicates );
    COLTYPE* aj_internal = _APAT.aj.data();

    // Reallocate _row_pos if needed with 1.25x growth factor
    if ( static_cast<size_t>( size ) > _row_pos_capacity )
    {
        _row_pos_capacity = static_cast<size_t>( size * 1.25 );
        _row_pos = std::make_unique<std::atomic<ROWTYPE>[]>( _row_pos_capacity );
    }

    for ( COLTYPE i = 0; i < size; i++ )
    {
        _row_pos[i].store( ai_internal[i] - base, std::memory_order_relaxed );
    }
    _thread_sums[0] = base;

#pragma omp parallel num_threads( _nthreads )
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();

        // Phase 1: Fill in column indices (parallelized with load-balanced work distribution)
        auto [row_start, row_end] =
            utils::LoadPrefixBalancedPartitionPos( ai, ai + size, tid, nthreads );

        for ( COLTYPE i = row_start; i < row_end; i++ )
        {
            for ( ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; j++ )
            {
                COLTYPE col = aj[j] - base;

                if ( col == i )
                {
                    if constexpr ( KEEPDIAG )
                    {
                        ROWTYPE pos = _row_pos[i].fetch_add( 1, std::memory_order_relaxed );
                        aj_internal[pos] = i + base;
                    }
                }
                else
                {
                    ROWTYPE pos_i = _row_pos[i].fetch_add( 1, std::memory_order_relaxed );
                    aj_internal[pos_i] = col + base;

                    ROWTYPE pos_col =
                        _row_pos[col].fetch_add( 1, std::memory_order_relaxed );
                    aj_internal[pos_col] = i + base;
                }
            }
        }

#pragma omp barrier

        // Phase 2: Sort and deduplicate each row's column indices (reuse load-balanced partitioning)

        for ( COLTYPE i = row_start; i < row_end; i++ )
        {
            ROWTYPE row_start_pos = ai_internal[i] - base;
            ROWTYPE row_end_pos = ai_internal[i + 1] - base;

            std::sort( aj_internal + row_start_pos, aj_internal + row_end_pos );

            // Use std::unique to remove duplicates and get the new end iterator
            auto new_end = std::unique( aj_internal + row_start_pos, aj_internal + row_end_pos );

            // Store the actual NNZ count for this row
            ai_APlusAT[i + 1] = new_end - ( aj_internal + row_start_pos );
        }

#pragma omp barrier

        // Local scan per thread
        auto [start, end] = utils::LoadBalancedPartitionPos( size, tid, nthreads );

        ROWTYPE local_sum = 0;
        for ( COLTYPE i = start; i < end; i++ )
        {
            local_sum += ai_APlusAT[i + 1];
        }
        // Store the last value for global adjustment
        _thread_sums[tid + 1] = local_sum;

#pragma omp barrier

#pragma omp single
        {
            for ( int i = 1; i <= nthreads; i++ )
            {
                _thread_sums[i] += _thread_sums[i - 1];
            }
            ai_APlusAT[size] = _thread_sums[nthreads];
        }

        const ROWTYPE offset = _thread_sums[tid];
        ai_APlusAT[start] = offset;
        for ( COLTYPE i = start; i < end - 1; i++ )
        {
            ai_APlusAT[i + 1] += ai_APlusAT[i];
        }

#pragma omp barrier

        // Phase 4: Copy unique column indices from internal buffer to output

        for ( COLTYPE i = row_start; i < row_end; i++ )
        {
            ROWTYPE old_start = ai_internal[i] - base;
            ROWTYPE new_start = ai_APlusAT[i] - base;
            ROWTYPE count = ai_APlusAT[i + 1] - ai_APlusAT[i];

            std::memcpy( aj_APlusAT + new_start, aj_internal + old_start,
                         count * sizeof( COLTYPE ) );
        }
    }
}

template <ResizableCSRMatrixType CSRMatrixType>
void partitionCSR1x2(const typename CSRMatrixType::COLTYPE rows,
                        const typename CSRMatrixType::COLTYPE cols,
                        typename CSRMatrixType::ROWTYPE const* ai,  
                        typename CSRMatrixType::COLTYPE const* aj,
                        typename CSRMatrixType::VALTYPE const* av,
                        const typename CSRMatrixType::COLTYPE col_split,
                        const typename CSRMatrixType::ROWTYPE base,
                        CSRMatrixType& A1,
                        CSRMatrixType& A2,
                        const int nthreads)
{
    using ROWTYPE = typename CSRMatrixType::ROWTYPE;
    using COLTYPE = typename CSRMatrixType::COLTYPE;
    
    // Set dimensions
    A1.rows = rows;
    A1.cols = col_split;
    A2.rows = rows;
    A2.cols = cols - col_split;

    // Resize row pointers using ResizableCSRMatrixType interface
    ROWTYPE* ai1 = A1.ResizeAI(rows + 1);
    ROWTYPE* ai2 = A2.ResizeAI(rows + 1);
    
    ai1[0] = base;
    ai2[0] = base;
    
    // First pass: count entries in each block and store split positions
    std::vector<ROWTYPE> split_pos(rows, 0);
    std::vector<ROWTYPE> thread_sums1(nthreads + 1, 0);
    std::vector<ROWTYPE> thread_sums2(nthreads + 1, 0);
    // Declare pointers outside parallel region
    COLTYPE* aj1 = nullptr;
    COLTYPE* aj2 = nullptr;
    
    using VALTYPE = typename CSRMatrixType::VALTYPE;
    VALTYPE* av1 = nullptr;
    VALTYPE* av2 = nullptr;
    
#pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        
        auto [start, end] = utils::LoadPrefixBalancedPartitionPos(ai, ai + rows, tid, nthreads);
        
        // First pass: count entries in each block and store split positions
        for (COLTYPE i = start; i < end; i++) {
            // Use lower_bound to find split point since aj is sorted
            auto split_it = std::lower_bound(aj + ai[i] - base, aj + ai[i + 1] - base, col_split + base);
            split_pos[i] = split_it - aj;  // Store absolute position
            ai1[i + 1] = split_it - (aj + ai[i] - base);
            ai2[i + 1] = (ai[i + 1] - ai[i]) - ai1[i + 1];
        }
        
#pragma omp barrier
        
        // Build row pointers using parallel prefix sum with O(nthreads) serial work
        // Phase 1: Local prefix sum per thread
        auto [chunk_start, chunk_end] = utils::LoadBalancedPartitionPos(rows, tid, nthreads);
        
        ROWTYPE local_sum1 = 0;
        ROWTYPE local_sum2 = 0;
        for (COLTYPE i = chunk_start; i < chunk_end; i++) {
            local_sum1 += ai1[i + 1];
            local_sum2 += ai2[i + 1];
        }
        
        // Use thread_local storage for thread sums (reuse split_pos vector space)
        
        thread_sums1[tid + 1] = local_sum1;
        thread_sums2[tid + 1] = local_sum2;
        
#pragma omp barrier
        
        // Phase 2: Sequential scan of thread sums - O(nthreads) serial work
#pragma omp single
        {
            thread_sums1[0] = base;
            thread_sums2[0] = base;
            for (int i = 1; i <= nthreads; i++) {
                thread_sums1[i] += thread_sums1[i - 1];
                thread_sums2[i] += thread_sums2[i - 1];
            }
            
            // Allocate column and value arrays using ResizableCSRMatrixType interface
            aj1 = A1.ResizeAJ(thread_sums1[nthreads] - base);
            aj2 = A2.ResizeAJ(thread_sums2[nthreads] - base);
            av1 = A1.ResizeAV(thread_sums1[nthreads] - base);
            av2 = A2.ResizeAV(thread_sums2[nthreads] - base);
        }
        
#pragma omp barrier
        
        // Phase 3: Parallel adjustment of each chunk
        const ROWTYPE offset1 = thread_sums1[tid];
        const ROWTYPE offset2 = thread_sums2[tid];
        
        ai1[chunk_start] = offset1;
        ai2[chunk_start] = offset2;
        for (COLTYPE i = chunk_start; i < chunk_end - 1; i++) {
            ai1[i + 1] += ai1[i];
            ai2[i + 1] += ai2[i];
        }
        if (chunk_end == rows) {
            ai1[rows] = thread_sums1[nthreads];
            ai2[rows] = thread_sums2[nthreads];
        }
        
#pragma omp barrier
        
        // Second pass: fill column indices and values using stored split positions
        for (COLTYPE i = start; i < end; i++) {
            // Use stored split position
            ROWTYPE split = split_pos[i];
            
            // Copy left block (columns < col_split)
            std::copy(aj + ai[i] - base, aj + split, aj1 + ai1[i] - base);
            std::copy(av + ai[i] - base, av + split, av1 + ai1[i] - base);
            
            // Copy right block (columns >= col_split) with adjusted column indices
            // Subtract col_split to shift column indices, keeping the base offset
            ROWTYPE pos2 = ai2[i] - base;
            for (ROWTYPE j = split; j < ai[i + 1] - base; ++j) {
                aj2[pos2] = aj[j] - col_split;
                av2[pos2] = av[j];
                pos2++;
            }
        }
    }
}

template <ResizableCSRMatrixType CSRMatrixType>
void partitionCSR2x2(const typename CSRMatrixType::COLTYPE rows,
                     const typename CSRMatrixType::COLTYPE cols,
                     typename CSRMatrixType::ROWTYPE const* ai,
                     typename CSRMatrixType::COLTYPE const* aj,
                     typename CSRMatrixType::VALTYPE const* av,
                     const typename CSRMatrixType::COLTYPE row_split,
                     const typename CSRMatrixType::COLTYPE col_split,
                     CSRMatrixType& A11,
                     CSRMatrixType& A12,
                     CSRMatrixType& A21,
                     CSRMatrixType& A22,
                     const int nthreads)
{
    using ROWTYPE = typename CSRMatrixType::ROWTYPE;
    using COLTYPE = typename CSRMatrixType::COLTYPE;
    
    const ROWTYPE base = ai[0];
    
    // Partition upper block [rows 0 to row_split-1] into left and right
    partitionCSR1x2<CSRMatrixType>(row_split, cols, ai, aj, av, col_split, base, A11, A12, nthreads);
    
    // Partition lower block [rows row_split to rows-1] into left and right
    partitionCSR1x2<CSRMatrixType>(rows - row_split, cols, ai + row_split, aj, av, col_split, base, A21, A22, nthreads);
}

// Macro for instantiation to reduce boilerplate
#define INSTANTIATE_SPARSE_OPS(ROWTYPE, COLTYPE) \
    template void AATSymbolic<ROWTYPE, COLTYPE, true>(const COLTYPE, ROWTYPE const*, COLTYPE const*, ROWTYPE*); \
    template void AATSymbolic<ROWTYPE, COLTYPE, false>(const COLTYPE, ROWTYPE const*, COLTYPE const*, ROWTYPE*); \
    template void AATNumeric<ROWTYPE, COLTYPE, true>(const COLTYPE, ROWTYPE const*, COLTYPE const*, ROWTYPE const*, COLTYPE*); \
    template void AATNumeric<ROWTYPE, COLTYPE, false>(const COLTYPE, ROWTYPE const*, COLTYPE const*, ROWTYPE const*, COLTYPE*); \
    template struct APlusATStruct<ROWTYPE, COLTYPE, true>; \
    template struct APlusATStruct<ROWTYPE, COLTYPE, false>; \
    template void partitionCSR1x2<CSRMatrixVec<ROWTYPE, COLTYPE, double>>( \
        const COLTYPE, const COLTYPE, ROWTYPE const*, COLTYPE const*, double const*, const COLTYPE, const ROWTYPE, \
        CSRMatrixVec<ROWTYPE, COLTYPE, double>&, CSRMatrixVec<ROWTYPE, COLTYPE, double>&, const int); \
    template void partitionCSR1x2<CSRMatrixVec<ROWTYPE, COLTYPE, float>>( \
        const COLTYPE, const COLTYPE, ROWTYPE const*, COLTYPE const*, float const*, const COLTYPE, const ROWTYPE, \
        CSRMatrixVec<ROWTYPE, COLTYPE, float>&, CSRMatrixVec<ROWTYPE, COLTYPE, float>&, const int); \
    template void partitionCSR2x2<CSRMatrixVec<ROWTYPE, COLTYPE, double>>( \
        const COLTYPE, const COLTYPE, ROWTYPE const*, COLTYPE const*, double const*, const COLTYPE, const COLTYPE, \
        CSRMatrixVec<ROWTYPE, COLTYPE, double>&, CSRMatrixVec<ROWTYPE, COLTYPE, double>&, \
        CSRMatrixVec<ROWTYPE, COLTYPE, double>&, CSRMatrixVec<ROWTYPE, COLTYPE, double>&, const int); \
    template void partitionCSR2x2<CSRMatrixVec<ROWTYPE, COLTYPE, float>>( \
        const COLTYPE, const COLTYPE, ROWTYPE const*, COLTYPE const*, float const*, const COLTYPE, const COLTYPE, \
        CSRMatrixVec<ROWTYPE, COLTYPE, float>&, CSRMatrixVec<ROWTYPE, COLTYPE, float>&, \
        CSRMatrixVec<ROWTYPE, COLTYPE, float>&, CSRMatrixVec<ROWTYPE, COLTYPE, float>&, const int);

// Explicit template instantiations
INSTANTIATE_SPARSE_OPS(int32_t, int32_t)
INSTANTIATE_SPARSE_OPS(int64_t, int64_t)

#undef INSTANTIATE_SPARSE_OPS

} // namespace matrix_utils
