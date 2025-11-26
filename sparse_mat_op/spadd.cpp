#include "spadd.hpp"
#include "matrix_utils.hpp"
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace matrix_utils
{

template <ResizableCSRMatrixType CSRMatrixType>
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

#pragma omp parallel num_threads( _nthreads )
    {
        // Thread-local workspace for tracking columns
        std::vector<bool> col_mask( A_cols, false );

#pragma omp for schedule( dynamic )
        for ( COLTYPE i = 0; i < A_rows; i++ )
        {
            ROWTYPE nnz_count = 0;
            
            // Add columns from row i of A
            for ( ROWTYPE ja = A_ai[i] - A_base; ja < A_ai[i + 1] - A_base; ja++ )
            {
                const COLTYPE col = A_aj[ja] - A_base;
                if ( !col_mask[col] )
                {
                    col_mask[col] = true;
                    nnz_count++;
                }
            }

            // Add columns from row i of B
            for ( ROWTYPE jb = B_ai[i] - B_base; jb < B_ai[i + 1] - B_base; jb++ )
            {
                const COLTYPE col = B_aj[jb] - B_base;
                if ( !col_mask[col] )
                {
                    col_mask[col] = true;
                    nnz_count++;
                }
            }

            // Store count for this row
            C_ai[i + 1] = nnz_count;

            // Reset mask for next row
            for ( ROWTYPE ja = A_ai[i] - A_base; ja < A_ai[i + 1] - A_base; ja++ )
            {
                col_mask[A_aj[ja] - A_base] = false;
            }
            for ( ROWTYPE jb = B_ai[i] - B_base; jb < B_ai[i + 1] - B_base; jb++ )
            {
                col_mask[B_aj[jb] - B_base] = false;
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
        std::vector<std::pair<COLTYPE, VALTYPE>> temp_entries;
        temp_entries.reserve( A_cols );

#pragma omp for schedule( dynamic )
        for ( COLTYPE i = 0; i < A_rows; i++ )
        {
            temp_entries.clear();

            // Add entries from row i of A
            for ( ROWTYPE ja = A_ai[i] - A_base; ja < A_ai[i + 1] - A_base; ja++ )
            {
                const COLTYPE col = A_aj[ja] - A_base;
                const VALTYPE val = alpha * A_av[ja];
                temp_entries.push_back( { col, val } );
            }

            // Add entries from row i of B
            for ( ROWTYPE jb = B_ai[i] - B_base; jb < B_ai[i + 1] - B_base; jb++ )
            {
                const COLTYPE col = B_aj[jb] - B_base;
                const VALTYPE val = beta * B_av[jb];
                
                // Check if this column already exists from A
                bool found = false;
                for ( auto& entry : temp_entries )
                {
                    if ( entry.first == col )
                    {
                        entry.second += val;
                        found = true;
                        break;
                    }
                }
                
                if ( !found )
                {
                    temp_entries.push_back( { col, val } );
                }
            }

            // Sort by column index
            std::sort( temp_entries.begin(), temp_entries.end(),
                      []( const auto& a, const auto& b ) { return a.first < b.first; } );

            // Write to output
            ROWTYPE pos = C_ai[i] - C_base;
            for ( const auto& entry : temp_entries )
            {
                C_aj[pos] = entry.first + C_base;
                C_av[pos] = entry.second;
                pos++;
            }
        }
    }
}

// Explicit instantiation for common types
template struct SpADD<CSRMatrixVec<int, int, double>>;
template struct SpADD<CSRMatrixVec<int64_t, int64_t, double>>;
template struct SpADD<CSRMatrixVec<int, int, float>>;
template struct SpADD<CSRMatrixVec<int64_t, int64_t, float>>;
template struct SpADD<CSRMatrix<int, int, double>>;

} // namespace matrix_utils
