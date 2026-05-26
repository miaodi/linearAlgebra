#include "spadd.hpp"
#include "matrix_utils.hpp"
#include <algorithm>
#include <stdexcept>
#include <vector>
#include "utils.h"

namespace matrix_utils
{

template <ResizableCSR CSRMatrixType>
void SpADD<CSRMatrixType>::analysis( const COLTYPE A_rows,
                                     const COLTYPE A_cols,
                                     const ROWTYPE* A_ai,
                                     const COLTYPE* A_aj,
                                     const COLTYPE B_rows,
                                     const COLTYPE B_cols,
                                     const ROWTYPE* B_ai,
                                     const COLTYPE* B_aj,
                                     CSRMatrixType& C )
{
    // Check dimensions
    if ( A_rows != B_rows || A_cols != B_cols )
    {
        throw std::invalid_argument( "Matrix dimensions do not match for addition" );
    }

    const ROWTYPE A_base = A_ai[0];
    const ROWTYPE B_base = B_ai[0];
    const ROWTYPE C_base = A_base; // Use same base as A

    // Set output dimensions
    C.rows = A_rows;
    C.cols = A_cols;
    C.ResizeAI( A_rows + 1 );

    ROWTYPE* C_ai = C.AI();

    // Initialize row pointers
    C_ai[0] = C_base;
    const int chunk_size = 32;
#pragma omp parallel num_threads( _nthreads )
    {
        // Thread-local workspace for tracking columns
        std::vector<char> col_mask( A_cols, 0 );

#pragma omp for schedule( dynamic, chunk_size )
        for ( COLTYPE i = 0; i < A_rows; i++ )
        {
            ROWTYPE nnz_count = 0;

            // Add columns from row i of A
            for ( ROWTYPE ja = A_ai[i] - A_base; ja < A_ai[i + 1] - A_base; ja++ )
            {
                const COLTYPE col = A_aj[ja] - A_base;
                col_mask[col] = 1;
                nnz_count++;
            }

            // Add columns from row i of B
            for ( ROWTYPE jb = B_ai[i] - B_base; jb < B_ai[i + 1] - B_base; jb++ )
            {
                const COLTYPE col = B_aj[jb] - B_base;
                const char is_new = !col_mask[col];
                col_mask[col] = 1;
                nnz_count += is_new;
            }

            // Store count for this row
            C_ai[i + 1] = nnz_count;

            // Reset mask for next row
            for ( ROWTYPE ja = A_ai[i] - A_base; ja < A_ai[i + 1] - A_base; ja++ )
            {
                col_mask[A_aj[ja] - A_base] = 0;
            }
            for ( ROWTYPE jb = B_ai[i] - B_base; jb < B_ai[i + 1] - B_base; jb++ )
            {
                col_mask[B_aj[jb] - B_base] = 0;
            }
        }
    }

    // Convert counts to row pointers (prefix sum)
    utils::ParallelPrefixSumInplace( _nthreads, C_ai, C_ai + A_rows + 1 );

    // Allocate column indices
    const ROWTYPE C_nnz = C_ai[A_rows] - C_base;
    C.ResizeAJ( C_nnz );

    COLTYPE* C_aj = C.AJ();

    // Build merged column structure
#pragma omp parallel num_threads( _nthreads )
    {
#ifdef SPADD_TWO_POINTER_MERGE
        // Two-pointer merge version: merge A and B column indices directly
#pragma omp for schedule( dynamic, chunk_size )
        for ( COLTYPE i = 0; i < A_rows; i++ )
        {
            ROWTYPE ja = A_ai[i] - A_base;
            ROWTYPE jb = B_ai[i] - B_base;
            const ROWTYPE ja_end = A_ai[i + 1] - A_base;
            const ROWTYPE jb_end = B_ai[i + 1] - B_base;
            ROWTYPE pos = C_ai[i] - C_base;

            // Merge A and B columns in sorted order
            while ( ja < ja_end && jb < jb_end )
            {
                const COLTYPE col_a = A_aj[ja] - A_base;
                const COLTYPE col_b = B_aj[jb] - B_base;

                if ( col_a < col_b )
                {
                    C_aj[pos++] = col_a + C_base;
                    ja++;
                }
                else if ( col_b < col_a )
                {
                    C_aj[pos++] = col_b + C_base;
                    jb++;
                }
                else // col_a == col_b
                {
                    C_aj[pos++] = col_a + C_base;
                    ja++;
                    jb++;
                }
            }

            // Add remaining columns from A
            while ( ja < ja_end )
            {
                C_aj[pos++] = A_aj[ja] - A_base + C_base;
                ja++;
            }

            // Add remaining columns from B
            while ( jb < jb_end )
            {
                C_aj[pos++] = B_aj[jb] - B_base + C_base;
                jb++;
            }
        }
#else
        // Hash map version: use column mask for tracking and then sort
        std::vector<char> col_mask( A_cols, 0 );
        std::vector<COLTYPE> temp_cols;
        temp_cols.reserve( A_cols );

#pragma omp for schedule( dynamic, chunk_size )
        for ( COLTYPE i = 0; i < A_rows; i++ )
        {
            temp_cols.clear();

            // Collect columns from A
            for ( ROWTYPE ja = A_ai[i] - A_base; ja < A_ai[i + 1] - A_base; ja++ )
            {
                const COLTYPE col = A_aj[ja] - A_base;
                col_mask[col] = 1;
                temp_cols.push_back( col );
            }

            // Collect new columns from B
            for ( ROWTYPE jb = B_ai[i] - B_base; jb < B_ai[i + 1] - B_base; jb++ )
            {
                const COLTYPE col = B_aj[jb] - B_base;
                if ( !col_mask[col] )
                {
                    temp_cols.push_back( col );
                }
            }

            // Sort and write column indices
            std::sort( temp_cols.begin(), temp_cols.end() );
            ROWTYPE pos = C_ai[i] - C_base;
            for ( const COLTYPE col : temp_cols )
            {
                C_aj[pos++] = col + C_base;
            }

            // Reset mask
            for ( ROWTYPE ja = A_ai[i] - A_base; ja < A_ai[i + 1] - A_base; ja++ )
            {
                col_mask[A_aj[ja] - A_base] = 0;
            }
        }
#endif
    }
}

template <ResizableCSR CSRMatrixType>
void SpADD<CSRMatrixType>::operator()( const COLTYPE A_rows,
                                       const COLTYPE A_cols,
                                       const ROWTYPE* A_ai,
                                       const COLTYPE* A_aj,
                                       const VALTYPE* A_av,
                                       const VALTYPE alpha,
                                       const COLTYPE B_rows,
                                       const COLTYPE B_cols,
                                       const ROWTYPE* B_ai,
                                       const COLTYPE* B_aj,
                                       const VALTYPE* B_av,
                                       const VALTYPE beta,
                                       CSRMatrixType& C )
{
    // Check dimensions
    if ( A_rows != B_rows || A_cols != B_cols )
    {
        throw std::invalid_argument( "Matrix dimensions do not match for addition" );
    }

    const ROWTYPE A_base = A_ai[0];
    const ROWTYPE B_base = B_ai[0];
    const ROWTYPE C_base = C.AI()[0];
    const ROWTYPE C_nnz = C.AI()[C.rows] - C_base;

    // Allocate values array
    C.ResizeAV( C_nnz );

    ROWTYPE* C_ai = C.AI();
    COLTYPE* C_aj = C.AJ();
    VALTYPE* C_av = C.AV();

    const int chunk_size = 32;
    // Compute values based on existing structure
#pragma omp parallel num_threads( _nthreads )
    {
#ifdef SPADD_TWO_POINTER_MERGE
        // Two-pointer merge version: merge A and B values while traversing C structure
#pragma omp for schedule( dynamic, chunk_size )
        for ( COLTYPE i = 0; i < A_rows; i++ )
        {
            ROWTYPE ja = A_ai[i] - A_base;
            ROWTYPE jb = B_ai[i] - B_base;
            const ROWTYPE ja_end = A_ai[i + 1] - A_base;
            const ROWTYPE jb_end = B_ai[i + 1] - B_base;

            for ( ROWTYPE jc = C_ai[i] - C_base; jc < C_ai[i + 1] - C_base; jc++ )
            {
                const COLTYPE col = C_aj[jc] - C_base;
                VALTYPE val = 0;

                // Check if current column is in A
                if ( ja < ja_end && A_aj[ja] - A_base == col )
                {
                    val += alpha * A_av[ja];
                    ja++;
                }

                // Check if current column is in B
                if ( jb < jb_end && B_aj[jb] - B_base == col )
                {
                    val += beta * B_av[jb];
                    jb++;
                }

                C_av[jc] = val;
            }
        }
#else
        // Hash map version: use value map for random access
        std::vector<VALTYPE> val_map( A_cols, 0 );

#pragma omp for schedule( dynamic, chunk_size )
        for ( COLTYPE i = 0; i < A_rows; i++ )
        {
            // Accumulate values from A
            for ( ROWTYPE ja = A_ai[i] - A_base; ja < A_ai[i + 1] - A_base; ja++ )
            {
                const COLTYPE col = A_aj[ja] - A_base;
                val_map[col] = alpha * A_av[ja];
            }

            // Accumulate values from B
            for ( ROWTYPE jb = B_ai[i] - B_base; jb < B_ai[i + 1] - B_base; jb++ )
            {
                const COLTYPE col = B_aj[jb] - B_base;
                val_map[col] += beta * B_av[jb];
            }

            // Write values in the order determined by C's column structure
            for ( ROWTYPE jc = C_ai[i] - C_base; jc < C_ai[i + 1] - C_base; jc++ )
            {
                const COLTYPE col = C_aj[jc] - C_base;
                C_av[jc] = val_map[col];
            }

            // Reset value map
            for ( ROWTYPE ja = A_ai[i] - A_base; ja < A_ai[i + 1] - A_base; ja++ )
            {
                val_map[A_aj[ja] - A_base] = 0;
            }
            for ( ROWTYPE jb = B_ai[i] - B_base; jb < B_ai[i + 1] - B_base; jb++ )
            {
                val_map[B_aj[jb] - B_base] = 0;
            }
        }
#endif
    }
}

// Explicit instantiation for common types
template struct SpADD<CSRMatrixVec<int, int, double>>;
template struct SpADD<CSRMatrixVec<int64_t, int64_t, double>>;
template struct SpADD<CSRMatrixVec<int, int, float>>;
template struct SpADD<CSRMatrixVec<int64_t, int64_t, float>>;
template struct SpADD<CSRMatrixVec<int, int, int8_t>>;
template struct SpADD<CSRMatrixVec<int64_t, int64_t, int8_t>>;
template struct SpADD<CSRMatrix<int, int, double>>;

} // namespace matrix_utils
