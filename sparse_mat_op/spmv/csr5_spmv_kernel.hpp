#pragma once

#include "csr5_format.hpp"
#include "utils_core.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <omp.h>

#if defined( __x86_64__ ) || defined( __i386__ ) || defined( _M_X64 ) || defined( _M_IX86 )
#include <immintrin.h>
#endif

#if ( defined( __GNUC__ ) || defined( __clang__ ) ) && \
    ( defined( __x86_64__ ) || defined( __i386__ ) || defined( _M_X64 ) || defined( _M_IX86 ) )
#define MATRIX_UTILS_CSR5_CAN_COMPILE_TARGET_AVX2 1
#define MATRIX_UTILS_CSR5_TARGET_AVX2 __attribute__( ( target( "avx2" ) ) )
#else
#define MATRIX_UTILS_CSR5_CAN_COMPILE_TARGET_AVX2 0
#define MATRIX_UTILS_CSR5_TARGET_AVX2
#endif

namespace matrix_utils
{

inline constexpr std::size_t CSR5_CACHE_LINE_BYTES = 64;

using CSR5SpMVPolicy = CSR5AVX2DoublePolicy;
using CSR5SpMVData = CSR5Data<CSR5SpMVPolicy>;
using CSR5ROWTYPE = typename CSR5SpMVPolicy::ROWTYPE;
using CSR5COLTYPE = typename CSR5SpMVPolicy::COLTYPE;

struct alignas( CSR5_CACHE_LINE_BYTES ) CSR5ThreadBoundaryContribution
{
    double value{};
};

struct CSR5TilePartition
{
    CSR5ROWTYPE begin{};
    CSR5ROWTYPE end{};
};

inline void csr5BuildTilePartitions( const CSR5ROWTYPE num_full_tiles,
                                     const int num_threads,
                                     std::vector<CSR5TilePartition>& tile_partitions )
{
    const int threads = num_threads > 0 ? num_threads : 1;
    tile_partitions.resize( static_cast<std::size_t>( threads ) );
    for ( int tid = 0; tid < threads; ++tid )
    {
        auto [tile_begin, tile_end] =
            utils::LoadBalancedPartitionPos<CSR5ROWTYPE>( num_full_tiles, tid, threads );
        tile_partitions[tid] = CSR5TilePartition{ tile_begin, tile_end };
    }
}

#if MATRIX_UTILS_CSR5_CAN_COMPILE_TARGET_AVX2
MATRIX_UTILS_CSR5_TARGET_AVX2 inline double csr5HorizontalSumAvx2Double( const __m256d values )
{
    double sum = 0.0;
    const __m256d pair_sum = _mm256_add_pd( values, _mm256_permute2f128_pd( values, values, 0x1 ) );
    _mm_store_sd( &sum, _mm_hadd_pd( _mm256_castpd256_pd128( pair_sum ), _mm256_castpd256_pd128( pair_sum ) ) );
    return sum;
}

MATRIX_UTILS_CSR5_TARGET_AVX2 inline __m256d csr5PrefixSumAvx2Double( const __m256d values )
{
    __m256d shifted = _mm256_permute4x64_pd( values, 0x93 );
    __m256d scan = _mm256_add_pd( values, _mm256_blend_pd( shifted, _mm256_setzero_pd(), 0x1 ) );
    shifted = _mm256_permute4x64_pd( values, 0x4E );
    scan = _mm256_add_pd( scan, _mm256_blend_pd( shifted, _mm256_setzero_pd(), 0x3 ) );
    shifted = _mm256_permute4x64_pd( values, 0x39 );
    scan = _mm256_add_pd( scan, _mm256_blend_pd( shifted, _mm256_setzero_pd(), 0x7 ) );
    return scan;
}

MATRIX_UTILS_CSR5_TARGET_AVX2 inline __m256d csr5LoadProductAvx2Double( const double* __restrict values,
                                                                        const CSR5COLTYPE* __restrict col_idx,
                                                                        const double* __restrict x,
                                                                        const __m256d alpha )
{
    static_assert( sizeof( CSR5COLTYPE ) == sizeof( int32_t ),
                   "AVX2 CSR5 kernel expects 32-bit column indices" );

    const double x0 = x[col_idx[0]];
    const double x1 = x[col_idx[1]];
    const double x2 = x[col_idx[2]];
    const double x3 = x[col_idx[3]];
    const __m256d xv = _mm256_set_pd( x3, x2, x1, x0 );
    const __m256d av = _mm256_mul_pd( _mm256_loadu_pd( values ), alpha );
    return _mm256_mul_pd( av, xv );
}

MATRIX_UTILS_CSR5_TARGET_AVX2 inline bool csr5TileStartsNewRowAvx2Double( const CSR5SpMVData& data,
                                                                          const CSR5ROWTYPE tile )
{
    constexpr int bit0_position = 31 - CSR5SpMVPolicy::BIT_Y_OFFSET - CSR5SpMVPolicy::BIT_SEG_OFFSET;
    return ( ( data._tile_desc[tile * CSR5SpMVPolicy::OMEGA] >> bit0_position ) & 1u ) != 0;
}

template <bool BetaIsZero>
MATRIX_UTILS_CSR5_TARGET_AVX2 void csr5WriteOrDeferFirstRowAvx2Double( double* __restrict y,
                                                                       const CSR5COLTYPE row_start,
                                                                       const bool row_started_before_thread,
                                                                       const bool tile_starts_new_row,
                                                                       const double contribution,
                                                                       const double beta,
                                                                       CSR5ThreadBoundaryContribution& thread_boundary_sum )
{
    if ( row_started_before_thread )
    {
        thread_boundary_sum.value += contribution;
    }
    else if ( tile_starts_new_row )
    {
        if constexpr ( BetaIsZero )
        {
            y[row_start] = contribution;
        }
        else
        {
            y[row_start] = beta * y[row_start] + contribution;
        }
    }
    else
    {
        y[row_start] += contribution;
    }
}

template <bool BetaIsZero>
MATRIX_UTILS_CSR5_TARGET_AVX2 inline void csr5StoreStartedRowAvx2Double( double* __restrict y,
                                                                         const int row_offset,
                                                                         const double contribution,
                                                                         const double beta )
{
    if constexpr ( BetaIsZero )
    {
        y[row_offset] = contribution;
    }
    else
    {
        y[row_offset] = beta * y[row_offset] + contribution;
    }
}

template <bool BetaIsZero>
MATRIX_UTILS_CSR5_TARGET_AVX2 void csr5ProcessFastTrackTileAvx2Double( const CSR5SpMVData& data,
                                                                       const CSR5ROWTYPE tile,
                                                                       const double* __restrict x,
                                                                       double* __restrict y,
                                                                       const double alpha,
                                                                       const double beta,
                                                                       const CSR5ROWTYPE thread_start_nnz,
                                                                       CSR5ThreadBoundaryContribution& thread_boundary_sum )
{
    constexpr int OMEGA = CSR5SpMVPolicy::OMEGA;
    constexpr int SIGMA = CSR5SpMVPolicy::SIGMA;
    constexpr int TILE_SIZE = CSR5SpMVPolicy::TILE_SIZE;

    const CSR5ROWTYPE tile_start = tile * TILE_SIZE;
    const CSR5COLTYPE* __restrict col_idx = data._tile_col_idx.data() + tile_start;
    const double* __restrict values = data._tile_val.data() + tile_start;
    const __m256d alpha_vec = _mm256_set1_pd( alpha );

    __m256d sum = _mm256_setzero_pd();
    for ( int i = 0; i < SIGMA; ++i )
    {
        sum = _mm256_add_pd(
            sum, csr5LoadProductAvx2Double( values + i * OMEGA, col_idx + i * OMEGA, x, alpha_vec ) );
    }

    const bool tile_starts_new_row = csr5TileStartsNewRowAvx2Double( data, tile );
    const CSR5COLTYPE row_start = data._tile_ptr[tile];
    const bool row_started_before_thread = data._row_ptr[row_start] < thread_start_nnz;
    csr5WriteOrDeferFirstRowAvx2Double<BetaIsZero>(
        y, row_start, row_started_before_thread, tile_starts_new_row,
        csr5HorizontalSumAvx2Double( sum ), beta, thread_boundary_sum );
}

template <bool BetaIsZero>
MATRIX_UTILS_CSR5_TARGET_AVX2 void csr5ProcessNormalTileAvx2Double( const CSR5SpMVData& data,
                                                                    const CSR5ROWTYPE tile,
                                                                    const double* __restrict x,
                                                                    double* __restrict y,
                                                                    const double alpha,
                                                                    const double beta,
                                                                    const CSR5ROWTYPE thread_start_nnz,
                                                                    CSR5ThreadBoundaryContribution& thread_boundary_sum )
{
    constexpr int OMEGA = CSR5SpMVPolicy::OMEGA;
    constexpr int SIGMA = CSR5SpMVPolicy::SIGMA;
    constexpr int TILE_SIZE = CSR5SpMVPolicy::TILE_SIZE;

    const CSR5ROWTYPE tile_start = tile * TILE_SIZE;
    const CSR5COLTYPE row_start = data._tile_ptr[tile];
    const CSR5COLTYPE* __restrict col_idx = data._tile_col_idx.data() + tile_start;
    const double* __restrict values = data._tile_val.data() + tile_start;
    const __m256d alpha_vec = _mm256_set1_pd( alpha );

    alignas( 32 ) double lane_sum[4]{};
    alignas( 32 ) double lane_first_sum[4]{};
    alignas( 32 ) uint64_t lane_cond[4]{};
    alignas( 16 ) int y_idx[4]{};

    const uint32_t* __restrict desc = data._tile_desc.data() + tile * OMEGA;
    __m128i descriptor = _mm_loadu_si128( reinterpret_cast<const __m128i*>( desc ) );
    __m128i y_offset = _mm_srli_epi32( descriptor, 32 - CSR5SpMVPolicy::BIT_Y_OFFSET );
    __m128i scansum_offset = _mm_slli_epi32( descriptor, CSR5SpMVPolicy::BIT_Y_OFFSET );
    scansum_offset = _mm_srli_epi32( scansum_offset, 32 - CSR5SpMVPolicy::BIT_SEG_OFFSET );
    descriptor = _mm_slli_epi32( descriptor, CSR5SpMVPolicy::BIT_Y_OFFSET + CSR5SpMVPolicy::BIT_SEG_OFFSET );

    descriptor = _mm_or_si128( descriptor, _mm_set_epi32( 0, 0, 0, 0x80000000 ) );
    __m256i local_bit = _mm256_cvtepu32_epi64( _mm_srli_epi32( descriptor, 31 ) );
    __m256i start = _mm256_sub_epi64( _mm256_set1_epi64x( 1 ), local_bit );
    __m256i direct = _mm256_and_si256( local_bit, _mm256_set_epi64x( 1, 1, 1, 0 ) );
    __m256i stop = _mm256_setzero_si256();

    __m256d first_sum = _mm256_setzero_pd();
    __m256d sum = csr5LoadProductAvx2Double( values, col_idx, x, alpha_vec );

    for ( int i = 1; i < SIGMA; ++i )
    {
        local_bit = _mm256_and_si256( _mm256_cvtepu32_epi64( _mm_srli_epi32( descriptor, 31 - i ) ),
                                      _mm256_set1_epi64x( 1 ) );
        const int no_lane_starts = _mm256_testz_si256( local_bit, _mm256_set1_epi64x( -1 ) );
        if ( !no_lane_starts )
        {
            _mm_storeu_si128( reinterpret_cast<__m128i*>( y_idx ), y_offset );
            _mm256_store_pd( lane_sum, sum );
            _mm256_store_si256( reinterpret_cast<__m256i*>( lane_cond ), _mm256_and_si256( direct, local_bit ) );

            int inc0 = 0;
            int inc1 = 0;
            int inc2 = 0;
            int inc3 = 0;
            double* __restrict y_local = y + row_start + 1;
            if ( lane_cond[0] != 0 )
            {
                csr5StoreStartedRowAvx2Double<BetaIsZero>( y_local, y_idx[0], lane_sum[0], beta );
                inc0 = 1;
            }
            if ( lane_cond[1] != 0 )
            {
                csr5StoreStartedRowAvx2Double<BetaIsZero>( y_local, y_idx[1], lane_sum[1], beta );
                inc1 = 1;
            }
            if ( lane_cond[2] != 0 )
            {
                csr5StoreStartedRowAvx2Double<BetaIsZero>( y_local, y_idx[2], lane_sum[2], beta );
                inc2 = 1;
            }
            if ( lane_cond[3] != 0 )
            {
                csr5StoreStartedRowAvx2Double<BetaIsZero>( y_local, y_idx[3], lane_sum[3], beta );
                inc3 = 1;
            }

            y_offset = _mm_add_epi32( y_offset, _mm_set_epi32( inc3, inc2, inc1, inc0 ) );

            __m256i tmp = _mm256_andnot_si256( _mm256_cmpeq_epi64( direct, _mm256_set1_epi64x( 1 ) ),
                                               _mm256_cmpeq_epi64( local_bit, _mm256_set1_epi64x( 1 ) ) );
            first_sum = _mm256_add_pd(
                _mm256_and_pd( _mm256_castsi256_pd( _mm256_cmpeq_epi64( tmp, _mm256_setzero_si256() ) ), first_sum ),
                _mm256_and_pd( _mm256_castsi256_pd( _mm256_cmpeq_epi64( tmp, _mm256_set1_epi64x( -1 ) ) ), sum ) );
            sum = _mm256_and_pd(
                _mm256_castsi256_pd( _mm256_cmpeq_epi64( local_bit, _mm256_setzero_si256() ) ), sum );
            direct = _mm256_or_si256( direct, local_bit );
            stop = _mm256_add_epi64( stop, local_bit );
        }

        sum = _mm256_add_pd(
            sum, csr5LoadProductAvx2Double( values + i * OMEGA, col_idx + i * OMEGA, x, alpha_vec ) );
    }

    __m256i tmp = _mm256_cmpeq_epi64( direct, _mm256_set1_epi64x( 1 ) );
    first_sum = _mm256_and_pd( _mm256_castsi256_pd( tmp ), first_sum );
    first_sum = _mm256_add_pd(
        first_sum,
        _mm256_and_pd( _mm256_castsi256_pd( _mm256_cmpeq_epi64( tmp, _mm256_setzero_si256() ) ), sum ) );

    __m256d last_sum = sum;
    sum = _mm256_and_pd( _mm256_castsi256_pd( _mm256_cmpeq_epi64( start, _mm256_set1_epi64x( 1 ) ) ), first_sum );
    sum = _mm256_permute4x64_pd( sum, 0x39 );
    sum = _mm256_and_pd( _mm256_castsi256_pd( _mm256_setr_epi64x( -1, -1, -1, 0 ) ), sum );

    const __m256d shifted_first_sum = sum;
    sum = csr5PrefixSumAvx2Double( sum );

    scansum_offset = _mm_add_epi32( scansum_offset, _mm_set_epi32( 3, 2, 1, 0 ) );
    __m256i scan_permute = _mm256_castsi128_si256( scansum_offset );
    scan_permute = _mm256_permutevar8x32_epi32( scan_permute, _mm256_set_epi32( 3, 3, 2, 2, 1, 1, 0, 0 ) );
    scan_permute = _mm256_add_epi32( scan_permute, scan_permute );
    scan_permute = _mm256_add_epi32( scan_permute, _mm256_set_epi32( 1, 0, 1, 0, 1, 0, 1, 0 ) );

    sum = _mm256_sub_pd(
        _mm256_castsi256_pd( _mm256_permutevar8x32_epi32( _mm256_castpd_si256( sum ), scan_permute ) ), sum );
    sum = _mm256_add_pd( sum, shifted_first_sum );

    tmp = _mm256_cmpgt_epi64( start, stop );
    tmp = _mm256_cmpeq_epi64( tmp, _mm256_setzero_si256() );
    last_sum = _mm256_add_pd( last_sum, _mm256_and_pd( _mm256_castsi256_pd( tmp ), sum ) );

    _mm_storeu_si128( reinterpret_cast<__m128i*>( y_idx ), y_offset );
    _mm256_store_si256( reinterpret_cast<__m256i*>( lane_cond ), direct );
    _mm256_store_pd( lane_sum, last_sum );

    double first_row_contribution = lane_sum[0];
    double* __restrict y_local = y + row_start + 1;
    if ( lane_cond[0] != 0 )
    {
        _mm256_store_pd( lane_first_sum, first_sum );
        csr5StoreStartedRowAvx2Double<BetaIsZero>( y_local, y_idx[0], lane_sum[0], beta );
        first_row_contribution = lane_first_sum[0];
    }
    if ( lane_cond[1] != 0 )
    {
        csr5StoreStartedRowAvx2Double<BetaIsZero>( y_local, y_idx[1], lane_sum[1], beta );
    }
    if ( lane_cond[2] != 0 )
    {
        csr5StoreStartedRowAvx2Double<BetaIsZero>( y_local, y_idx[2], lane_sum[2], beta );
    }
    if ( lane_cond[3] != 0 )
    {
        csr5StoreStartedRowAvx2Double<BetaIsZero>( y_local, y_idx[3], lane_sum[3], beta );
    }

    const bool tile_starts_new_row = csr5TileStartsNewRowAvx2Double( data, tile );
    const bool row_started_before_thread = data._row_ptr[row_start] < thread_start_nnz;
    csr5WriteOrDeferFirstRowAvx2Double<BetaIsZero>( y, row_start, row_started_before_thread, tile_starts_new_row,
                                                    first_row_contribution, beta, thread_boundary_sum );
}

template <bool BetaIsZero>
MATRIX_UTILS_CSR5_TARGET_AVX2 void csr5SpmvComputeAvx2DoubleTarget( const CSR5SpMVData& data,
                                                                    const double* __restrict x,
                                                                    double* __restrict y,
                                                                    const double alpha,
                                                                    const double beta,
                                                                    const int num_threads,
                                                                    CSR5ThreadBoundaryContribution* __restrict thread_boundary_sums,
                                                                    const CSR5TilePartition* __restrict tile_partitions )
{
    constexpr int TILE_SIZE = CSR5SpMVPolicy::TILE_SIZE;

    for ( int tid = 0; tid < num_threads; ++tid )
    {
        thread_boundary_sums[tid].value = 0.0;
    }

#pragma omp parallel num_threads( num_threads )
    {
        const int tid = omp_get_thread_num();
        const CSR5ROWTYPE tile_begin = tile_partitions[tid].begin;
        const CSR5ROWTYPE tile_end = tile_partitions[tid].end;
        if ( tile_begin < tile_end )
        {
            const CSR5ROWTYPE thread_start_nnz = tile_begin * TILE_SIZE;

            for ( CSR5ROWTYPE tile = tile_begin; tile < tile_end; ++tile )
            {
                if ( data._tile_ptr[tile] == data._tile_ptr[tile + 1] )
                {
                    csr5ProcessFastTrackTileAvx2Double<BetaIsZero>(
                        data, tile, x, y, alpha, beta, thread_start_nnz, thread_boundary_sums[tid] );
                }
                else
                {
                    csr5ProcessNormalTileAvx2Double<BetaIsZero>(
                        data, tile, x, y, alpha, beta, thread_start_nnz, thread_boundary_sums[tid] );
                }
            }
        }
    }

    for ( int tid = 0; tid < num_threads; ++tid )
    {
        const CSR5ROWTYPE tile_begin = tile_partitions[tid].begin;
        const CSR5ROWTYPE tile_end = tile_partitions[tid].end;
        if ( tile_begin < tile_end && thread_boundary_sums[tid].value != 0.0 )
        {
            y[data._tile_ptr[tile_begin]] += thread_boundary_sums[tid].value;
        }
    }

    if ( data._tail_tile_length == 0 )
    {
        return;
    }

    const CSR5ROWTYPE tail_start = data._num_full_tiles * TILE_SIZE;
    const CSR5COLTYPE first_tail_row = data._tile_ptr[data._num_full_tiles];

#pragma omp parallel for num_threads( num_threads ) schedule( static )
    for ( CSR5COLTYPE row = first_tail_row; row < data._num_rows; ++row )
    {
        const CSR5ROWTYPE row_start = std::max( data._row_ptr[row], tail_start );
        const CSR5ROWTYPE row_end = data._row_ptr[row + 1];
        double sum = 0.0;
        for ( CSR5ROWTYPE idx = row_start; idx < row_end; ++idx )
        {
            sum += data._tile_val[idx] * x[data._tile_col_idx[idx]];
        }
        if ( !( row == first_tail_row && data._row_ptr[row] < tail_start ) )
        {
            if constexpr ( BetaIsZero )
            {
                y[row] = alpha * sum;
            }
            else
            {
                y[row] = beta * y[row] + alpha * sum;
            }
        }
        else
        {
            y[row] += alpha * sum;
        }
    }
}
#endif

template <bool BetaIsZero>
bool csr5SpmvComputeAvx2Double( const CSR5SpMVData& data,
                                const double* __restrict x,
                                double* __restrict y,
                                const double alpha,
                                const double beta,
                                const int num_threads,
                                CSR5ThreadBoundaryContribution* __restrict thread_boundary_sums,
                                const int thread_boundary_count,
                                const CSR5TilePartition* __restrict tile_partitions,
                                const int tile_partition_count )
{
#if MATRIX_UTILS_CSR5_CAN_COMPILE_TARGET_AVX2
    static_assert( CSR5SpMVPolicy::OMEGA == 4 && CSR5SpMVPolicy::SIGMA == 16 );
    static_assert( sizeof( CSR5COLTYPE ) == sizeof( int32_t ) );

    if ( thread_boundary_count >= num_threads && tile_partition_count >= num_threads &&
         __builtin_cpu_supports( "avx2" ) )
    {
        csr5SpmvComputeAvx2DoubleTarget<BetaIsZero>( data, x, y, alpha, beta, num_threads,
                                                     thread_boundary_sums, tile_partitions );
        return true;
    }
#endif
    return false;
}

inline void csr5ScaleOutput( const CSR5SpMVData& data, double* __restrict y, const double beta, const int num_threads )
{
    if ( beta == 0.0 )
    {
#pragma omp parallel for num_threads( num_threads )
        for ( CSR5COLTYPE row = 0; row < data._num_rows; ++row )
        {
            y[row] = 0.0;
        }
    }
    else if ( beta != 1.0 )
    {
#pragma omp parallel for num_threads( num_threads )
        for ( CSR5COLTYPE row = 0; row < data._num_rows; ++row )
        {
            y[row] *= beta;
        }
    }
}

template <bool BetaIsZero>
bool csr5SpmvCompute( const CSR5SpMVData& data,
                      const double* __restrict x,
                      double* __restrict y,
                      const double alpha,
                      const double beta,
                      const int num_threads,
                      CSR5ThreadBoundaryContribution* __restrict thread_boundary_sums,
                      const int thread_boundary_count,
                      const CSR5TilePartition* __restrict tile_partitions,
                      const int tile_partition_count )
{
    if ( csr5SpmvComputeAvx2Double<BetaIsZero>( data, x, y, alpha, beta, num_threads, thread_boundary_sums,
                                                thread_boundary_count, tile_partitions, tile_partition_count ) )
    {
        return true;
    }

    return false;
}

inline void csr5Spmv( const CSR5SpMVData& data,
                      const double* __restrict x,
                      double* __restrict y,
                      const double alpha,
                      const double beta,
                      const int num_threads,
                      CSR5ThreadBoundaryContribution* __restrict thread_boundary_sums,
                      const int thread_boundary_count,
                      const CSR5TilePartition* __restrict tile_partitions,
                      const int tile_partition_count )
{
    const int threads = num_threads > 0 ? num_threads : 1;
    if ( data._nnz == 0 || alpha == 0.0 )
    {
        csr5ScaleOutput( data, y, beta, threads );
        return;
    }

    const bool beta_is_zero = beta == 0.0;
    const bool computed =
        beta_is_zero
            ? csr5SpmvCompute<true>( data, x, y, alpha, beta, threads, thread_boundary_sums,
                                     thread_boundary_count, tile_partitions, tile_partition_count )
            : csr5SpmvCompute<false>( data, x, y, alpha, beta, threads, thread_boundary_sums,
                                      thread_boundary_count, tile_partitions, tile_partition_count );
    if ( computed )
    {
        return;
    }

    throw std::runtime_error( "CSR5 SpMV requires AVX2 support" );
}

inline void csr5Spmv( const CSR5SpMVData& data,
                      const double* __restrict x,
                      double* __restrict y,
                      const double alpha,
                      const double beta,
                      const int num_threads )
{
    const int threads = num_threads > 0 ? num_threads : 1;
    std::vector<CSR5ThreadBoundaryContribution> thread_boundary_sums( static_cast<std::size_t>( threads ) );
    std::vector<CSR5TilePartition> tile_partitions;
    csr5BuildTilePartitions( data._num_full_tiles, threads, tile_partitions );
    csr5Spmv( data, x, y, alpha, beta, threads, thread_boundary_sums.data(),
              static_cast<int>( thread_boundary_sums.size() ), tile_partitions.data(),
              static_cast<int>( tile_partitions.size() ) );
}

} // namespace matrix_utils

#undef MATRIX_UTILS_CSR5_CAN_COMPILE_TARGET_AVX2
#undef MATRIX_UTILS_CSR5_TARGET_AVX2
