#include "spgemm.hpp"
#include "matrix_utils.hpp"
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace matrix_utils
{

template <ResizableCSRMatrixType CSRMatrixType>
void SpGEMM<CSRMatrixType>::analysis( const COLTYPE A_rows,
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
    if ( A_cols != B_rows )
    {
        throw std::invalid_argument( "Matrix dimensions do not match for multiplication" );
    }

    const ROWTYPE A_base = A_ai[0];
    const ROWTYPE B_base = B_ai[0];
    const ROWTYPE C_base = A_base; // Use same base as A

    // Set output dimensions
    C.rows = A_rows;
    C.cols = B_cols;
    C.ResizeAI( A_rows + 1 );

    ROWTYPE* C_ai = C.AI();

    // Initialize row pointers
    C_ai[0] = C_base;

#pragma omp parallel num_threads( _nthreads )
    {
        // Thread-local workspace for tracking columns
        std::vector<bool> col_mask( B_cols, false );

#pragma omp for schedule( dynamic )
        for ( COLTYPE i = 0; i < A_rows; i++ )
        {
            ROWTYPE nnz_count = 0;
            
            // For each non-zero in row i of A
            for ( ROWTYPE ja = A_ai[i] - A_base; ja < A_ai[i + 1] - A_base; ja++ )
            {
                const COLTYPE k = A_aj[ja] - A_base; // Column in A, row in B

                // Add all columns from row k of B
                for ( ROWTYPE jb = B_ai[k] - B_base; jb < B_ai[k + 1] - B_base; jb++ )
                {
                    const COLTYPE col = B_aj[jb] - B_base;
                    if ( !col_mask[col] )
                    {
                        col_mask[col] = true;
                        nnz_count++;
                    }
                }
            }

            // Store count for this row
            C_ai[i + 1] = nnz_count;

            // Reset mask for next row
            for ( ROWTYPE ja = A_ai[i] - A_base; ja < A_ai[i + 1] - A_base; ja++ )
            {
                const COLTYPE k = A_aj[ja] - A_base;
                for ( ROWTYPE jb = B_ai[k] - B_base; jb < B_ai[k + 1] - B_base; jb++ )
                {
                    col_mask[B_aj[jb] - B_base] = false;
                }
            }
        }
    }

    // Convert counts to row pointers (prefix sum)
    for ( COLTYPE i = 0; i < A_rows; i++ )
    {
        C_ai[i + 1] += C_ai[i];
    }
}

template <ResizableCSRMatrixType CSRMatrixType>
void SpGEMM<CSRMatrixType>::operator()( const COLTYPE A_rows,
                                         const COLTYPE A_cols,
                                         const ROWTYPE* A_ai,
                                         const COLTYPE* A_aj,
                                         const VALTYPE* A_av,
                                         const COLTYPE B_rows,
                                         const COLTYPE B_cols,
                                         const ROWTYPE* B_ai,
                                         const COLTYPE* B_aj,
                                         const VALTYPE* B_av,
                                         CSRMatrixType& C )
{
    // Check dimensions
    if ( A_cols != B_rows )
    {
        throw std::invalid_argument( "Matrix dimensions do not match for multiplication" );
    }

    const ROWTYPE A_base = A_ai[0];
    const ROWTYPE B_base = B_ai[0];

    // Allocate column indices and values
    const ROWTYPE C_base = C.AI()[0];
    const ROWTYPE C_nnz = C.AI()[C.rows] - C_base;
    C.ResizeAJ( C_nnz );
    C.ResizeAV( C_nnz );

    ROWTYPE* C_ai = C.AI();
    COLTYPE* C_aj = C.AJ();
    VALTYPE* C_av = C.AV();

    // Compute values
#pragma omp parallel num_threads( _nthreads )
    {
        // Thread-local workspace
        std::unordered_map<COLTYPE, VALTYPE> accumulator;
        std::vector<COLTYPE> temp_cols;
        temp_cols.reserve( B_cols );

#pragma omp for schedule( dynamic )
        for ( COLTYPE i = 0; i < A_rows; i++ )
        {
            accumulator.clear();
            temp_cols.clear();

            // For each non-zero in row i of A
            for ( ROWTYPE ja = A_ai[i] - A_base; ja < A_ai[i + 1] - A_base; ja++ )
            {
                const COLTYPE k = A_aj[ja] - A_base;
                const VALTYPE a_val = A_av[ja];

                // Multiply with row k of B and accumulate
                for ( ROWTYPE jb = B_ai[k] - B_base; jb < B_ai[k + 1] - B_base; jb++ )
                {
                    const COLTYPE col = B_aj[jb] - B_base;
                    const VALTYPE b_val = B_av[jb];

                    auto it = accumulator.find( col );
                    if ( it != accumulator.end() )
                    {
                        it->second += a_val * b_val;
                    }
                    else
                    {
                        accumulator[col] = a_val * b_val;
                        temp_cols.push_back( col );
                    }
                }
            }

            // Sort columns and write to output
            std::sort( temp_cols.begin(), temp_cols.end() );

            ROWTYPE pos = C_ai[i] - C_base;
            for ( const COLTYPE col : temp_cols )
            {
                C_aj[pos] = col + C_base;
                C_av[pos] = accumulator[col];
                pos++;
            }
        }
    }
}

// Explicit instantiation for common types
template struct SpGEMM<CSRMatrixVec<int, int, double>>;
template struct SpGEMM<CSRMatrixVec<int64_t, int64_t, double>>;
template struct SpGEMM<CSRMatrixVec<int, int, float>>;
template struct SpGEMM<CSRMatrixVec<int64_t, int64_t, float>>;

} // namespace matrix_utils
