#include "cuda_sptrsv.cuh"
#include <cmath>
#include <string>
#include <limits>

namespace matrix_utils::sparse_cuda
{

namespace
{
inline void cuda_check( cudaError_t error, const char* message )
{
    if ( error != cudaSuccess )
    {
        throw std::runtime_error( std::string( "CUDA error: " ) + message + " - " + cudaGetErrorString( error ) );
    }
}

template <typename T>
__device__ __forceinline__ T device_nan();

template <>
__device__ __forceinline__ float device_nan<float>()
{
    return nanf( "" );
}

template <>
__device__ __forceinline__ double device_nan<double>()
{
    return nan( "" );
}
} // namespace

/**
 * @brief Kernel to initialize a vector with NaN values
 */
template <typename VALTYPE>
__global__ void init_nan_kernel( VALTYPE* d_x, int n )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tid < n )
    {
        d_x[tid] = device_nan<VALTYPE>();
    }
}

/**
 * @brief Unified SpTRSV kernel for both lower and upper triangular solve
 *
 * Each thread solves one row:
 * - Lower: x[i] = (b[i] - sum(L[i,j] * x[j] for j < i)) / L[i,i]
 * - Upper: x[i] = (b[i] - sum(U[i,j] * x[j] for j > i)) / U[i,i]
 *
 * Threads spin-wait on NaN values in dependencies.
 *
 * @tparam ROWTYPE Row pointer type
 * @tparam COLTYPE Column index type
 * @tparam VALTYPE Value type
 * @tparam IS_LOWER If true, solve lower triangular; if false, solve upper triangular
 * @tparam UNIT_DIAG Whether diagonal is assumed to be 1.0
 * @param d_ia Device row pointers
 * @param d_ja Device column indices
 * @param d_av Device values
 * @param d_b Device right-hand side
 * @param d_x Device solution (initialized to NaN)
 * @param n Matrix dimension
 */
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, bool IS_LOWER, bool UNIT_DIAG>
__global__ void sptrsv_kernel( const ROWTYPE* __restrict__ d_ia,
                               const COLTYPE* __restrict__ d_ja,
                               const VALTYPE* __restrict__ d_av,
                               const VALTYPE* __restrict__ d_b,
                               VALTYPE* __restrict__ d_x,
                               COLTYPE n )
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if ( row >= n )
    {
        return;
    }

    // Get row bounds
    ROWTYPE row_start = d_ia[row];
    ROWTYPE row_end = d_ia[row + 1];

    // Compute: x[row] = (b[row] - sum(A[row,j] * x[j])) / A[row,row]
    VALTYPE sum = 0.0;
    VALTYPE diag_val = 1.0;

    for ( ROWTYPE idx = row_start; idx < row_end; ++idx )
    {
        COLTYPE col = d_ja[idx];
        VALTYPE val = d_av[idx];

        // Check if this is an off-diagonal dependency (compile-time constant)
        constexpr bool is_dependency_check = IS_LOWER ? true : false; // Will be optimized away
        bool is_dependency = IS_LOWER ? ( col < row ) : ( col > row );

        if ( is_dependency )
        {
            // Off-diagonal element: spin-wait until x[col] is computed
            VALTYPE x_col;
            do
            {
                // Volatile read to prevent compiler caching
                x_col = *( (volatile VALTYPE*)&d_x[col] );
                // Spin while x[col] is NaN
            } while ( isnan( x_col ) );

            sum += val * x_col;
        }
        else if ( col == row )
        {
            // Diagonal element
            if constexpr ( !UNIT_DIAG )
            {
                diag_val = val;
            }
        }
    }

    // Compute solution for this row
    VALTYPE x_row = ( d_b[row] - sum ) / diag_val;

    // Write result with memory fence to ensure visibility to all threads
    d_x[row] = x_row;
    __threadfence();
}

// ============================================================================
// SpTRSVOperator implementation
// ============================================================================

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
SpTRSVOperator<ROWTYPE, COLTYPE, VALTYPE>::SpTRSVOperator( COLTYPE n,
                                                           const ROWTYPE* d_ia,
                                                           const COLTYPE* d_ja,
                                                           const VALTYPE* d_av,
                                                           TriangularType tri_type,
                                                           DiagonalType diag_type )
    : _n( n ), _d_ia( d_ia ), _d_ja( d_ja ), _d_av( d_av ), _tri_type( tri_type ), _diag_type( diag_type )
{
    if ( !d_ia || !d_ja || !d_av )
    {
        throw std::runtime_error( "SpTRSVOperator: invalid device pointers" );
    }
    if ( n <= 0 )
    {
        throw std::runtime_error( "SpTRSVOperator: invalid matrix dimension" );
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void SpTRSVOperator<ROWTYPE, COLTYPE, VALTYPE>::solve( const VALTYPE* d_b, VALTYPE* d_x )
{
    if ( !d_b || !d_x )
    {
        throw std::runtime_error( "SpTRSVOperator::solve: invalid device pointers" );
    }

    // Step 1: Initialize solution vector to NaN
    constexpr int block_size = 256;
    int num_blocks = ( _n + block_size - 1 ) / block_size;

    init_nan_kernel<<<num_blocks, block_size>>>( d_x, _n );
    cuda_check( cudaGetLastError(), "init_nan_kernel launch failed" );

    // Step 2: Launch unified SpTRSV kernel with compile-time template parameters
    if ( _tri_type == TriangularType::Lower )
    {
        if ( _diag_type == DiagonalType::Unit )
        {
            sptrsv_kernel<ROWTYPE, COLTYPE, VALTYPE, true, true>
                <<<num_blocks, block_size>>>( _d_ia, _d_ja, _d_av, d_b, d_x, _n );
        }
        else
        {
            sptrsv_kernel<ROWTYPE, COLTYPE, VALTYPE, true, false>
                <<<num_blocks, block_size>>>( _d_ia, _d_ja, _d_av, d_b, d_x, _n );
        }
    }
    else
    {
        if ( _diag_type == DiagonalType::Unit )
        {
            sptrsv_kernel<ROWTYPE, COLTYPE, VALTYPE, false, true>
                <<<num_blocks, block_size>>>( _d_ia, _d_ja, _d_av, d_b, d_x, _n );
        }
        else
        {
            sptrsv_kernel<ROWTYPE, COLTYPE, VALTYPE, false, false>
                <<<num_blocks, block_size>>>( _d_ia, _d_ja, _d_av, d_b, d_x, _n );
        }
    }
    cuda_check( cudaGetLastError(), "sptrsv_kernel launch failed" );

    // Synchronize to ensure all threads complete
    cuda_check( cudaDeviceSynchronize(), "SpTRSV kernel execution failed" );
}

// ============================================================================
// Convenience functions
// ============================================================================

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void sptrsv_lower( COLTYPE n,
                   const ROWTYPE* d_ia,
                   const COLTYPE* d_ja,
                   const VALTYPE* d_av,
                   const VALTYPE* d_b,
                   VALTYPE* d_x,
                   DiagonalType diag_type )
{
    SpTRSVOperator<ROWTYPE, COLTYPE, VALTYPE> op( n, d_ia, d_ja, d_av, TriangularType::Lower, diag_type );
    op.solve( d_b, d_x );
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void sptrsv_upper( COLTYPE n,
                   const ROWTYPE* d_ia,
                   const COLTYPE* d_ja,
                   const VALTYPE* d_av,
                   const VALTYPE* d_b,
                   VALTYPE* d_x,
                   DiagonalType diag_type )
{
    SpTRSVOperator<ROWTYPE, COLTYPE, VALTYPE> op( n, d_ia, d_ja, d_av, TriangularType::Upper, diag_type );
    op.solve( d_b, d_x );
}

// ============================================================================
// Explicit template instantiations
// ============================================================================

// SpTRSVOperator instantiations
template class SpTRSVOperator<int, int, double>;
template class SpTRSVOperator<int, int, float>;
template class SpTRSVOperator<int64_t, int64_t, double>;
template class SpTRSVOperator<int64_t, int64_t, float>;

// Kernel instantiations for all combinations (IS_LOWER, UNIT_DIAG)
// int, int, double
template __global__ void sptrsv_kernel<int, int, double, true, true>( const int*,
                                                                      const int*,
                                                                      const double*,
                                                                      const double*,
                                                                      double*,
                                                                      int );
template __global__ void sptrsv_kernel<int, int, double, true, false>( const int*,
                                                                       const int*,
                                                                       const double*,
                                                                       const double*,
                                                                       double*,
                                                                       int );
template __global__ void sptrsv_kernel<int, int, double, false, true>( const int*,
                                                                       const int*,
                                                                       const double*,
                                                                       const double*,
                                                                       double*,
                                                                       int );
template __global__ void sptrsv_kernel<int, int, double, false, false>( const int*,
                                                                        const int*,
                                                                        const double*,
                                                                        const double*,
                                                                        double*,
                                                                        int );

// int, int, float
template __global__ void sptrsv_kernel<int, int, float, true, true>( const int*,
                                                                     const int*,
                                                                     const float*,
                                                                     const float*,
                                                                     float*,
                                                                     int );
template __global__ void sptrsv_kernel<int, int, float, true, false>( const int*,
                                                                      const int*,
                                                                      const float*,
                                                                      const float*,
                                                                      float*,
                                                                      int );
template __global__ void sptrsv_kernel<int, int, float, false, true>( const int*,
                                                                      const int*,
                                                                      const float*,
                                                                      const float*,
                                                                      float*,
                                                                      int );
template __global__ void sptrsv_kernel<int, int, float, false, false>( const int*,
                                                                       const int*,
                                                                       const float*,
                                                                       const float*,
                                                                       float*,
                                                                       int );

// int64_t, int64_t, double
template __global__ void sptrsv_kernel<int64_t, int64_t, double, true, true>( const int64_t*,
                                                                              const int64_t*,
                                                                              const double*,
                                                                              const double*,
                                                                              double*,
                                                                              int64_t );
template __global__ void sptrsv_kernel<int64_t, int64_t, double, true, false>( const int64_t*,
                                                                               const int64_t*,
                                                                               const double*,
                                                                               const double*,
                                                                               double*,
                                                                               int64_t );
template __global__ void sptrsv_kernel<int64_t, int64_t, double, false, true>( const int64_t*,
                                                                               const int64_t*,
                                                                               const double*,
                                                                               const double*,
                                                                               double*,
                                                                               int64_t );
template __global__ void sptrsv_kernel<int64_t, int64_t, double, false, false>( const int64_t*,
                                                                                const int64_t*,
                                                                                const double*,
                                                                                const double*,
                                                                                double*,
                                                                                int64_t );

// int64_t, int64_t, float
template __global__ void sptrsv_kernel<int64_t, int64_t, float, true, true>( const int64_t*,
                                                                             const int64_t*,
                                                                             const float*,
                                                                             const float*,
                                                                             float*,
                                                                             int64_t );
template __global__ void sptrsv_kernel<int64_t, int64_t, float, true, false>( const int64_t*,
                                                                              const int64_t*,
                                                                              const float*,
                                                                              const float*,
                                                                              float*,
                                                                              int64_t );
template __global__ void sptrsv_kernel<int64_t, int64_t, float, false, true>( const int64_t*,
                                                                              const int64_t*,
                                                                              const float*,
                                                                              const float*,
                                                                              float*,
                                                                              int64_t );
template __global__ void sptrsv_kernel<int64_t, int64_t, float, false, false>( const int64_t*,
                                                                               const int64_t*,
                                                                               const float*,
                                                                               const float*,
                                                                               float*,
                                                                               int64_t );

// Convenience function instantiations
template void sptrsv_lower<int, int, double>( int, const int*, const int*, const double*, const double*, double*, DiagonalType );
template void sptrsv_lower<int, int, float>( int, const int*, const int*, const float*, const float*, float*, DiagonalType );
template void sptrsv_lower<int64_t, int64_t, double>( int64_t,
                                                      const int64_t*,
                                                      const int64_t*,
                                                      const double*,
                                                      const double*,
                                                      double*,
                                                      DiagonalType );
template void sptrsv_lower<int64_t, int64_t, float>( int64_t,
                                                     const int64_t*,
                                                     const int64_t*,
                                                     const float*,
                                                     const float*,
                                                     float*,
                                                     DiagonalType );

template void sptrsv_upper<int, int, double>( int, const int*, const int*, const double*, const double*, double*, DiagonalType );
template void sptrsv_upper<int, int, float>( int, const int*, const int*, const float*, const float*, float*, DiagonalType );
template void sptrsv_upper<int64_t, int64_t, double>( int64_t,
                                                      const int64_t*,
                                                      const int64_t*,
                                                      const double*,
                                                      const double*,
                                                      double*,
                                                      DiagonalType );
template void sptrsv_upper<int64_t, int64_t, float>( int64_t,
                                                     const int64_t*,
                                                     const int64_t*,
                                                     const float*,
                                                     const float*,
                                                     float*,
                                                     DiagonalType );

} // namespace matrix_utils::sparse_cuda
