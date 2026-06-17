#pragma once

#include "csr5_format.hpp"
#include <algorithm>
#include <array>
#include <type_traits>
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

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, typename Policy>
void csr5ComputeTileProductsScalar( const CSR5Data<ROWTYPE, COLTYPE, VALTYPE, Policy>& data,
                                    const ROWTYPE tile,
                                    const VALTYPE* __restrict x,
                                    std::array<VALTYPE, Policy::TILE_SIZE>& products )
{
    constexpr int OMEGA = Policy::OMEGA;
    constexpr int SIGMA = Policy::SIGMA;
    constexpr int TILE_SIZE = Policy::TILE_SIZE;

    const ROWTYPE tile_start = tile * TILE_SIZE;
    const COLTYPE* __restrict col_idx = data._tile_col_idx.data() + tile_start;
    const VALTYPE* __restrict values = data._tile_val.data() + tile_start;

    for ( int lane = 0; lane < OMEGA; ++lane )
    {
        for ( int i = 0; i < SIGMA; ++i )
        {
            const int storage_idx = i * OMEGA + lane;
            products[lane * SIGMA + i] = values[storage_idx] * x[col_idx[storage_idx]];
        }
    }
}

#if MATRIX_UTILS_CSR5_CAN_COMPILE_TARGET_AVX2
template <typename ROWTYPE, typename COLTYPE, typename Policy>
MATRIX_UTILS_CSR5_TARGET_AVX2 void csr5ComputeTileProductsAvx2DoubleTarget(
    const CSR5Data<ROWTYPE, COLTYPE, double, Policy>& data,
    const ROWTYPE tile,
    const double* __restrict x,
    std::array<double, Policy::TILE_SIZE>& products )
{
    constexpr int OMEGA = Policy::OMEGA;
    constexpr int SIGMA = Policy::SIGMA;
    constexpr int TILE_SIZE = Policy::TILE_SIZE;

    static_assert( OMEGA == 4, "The AVX2 double CSR5 product kernel requires omega=4" );
    static_assert( sizeof( COLTYPE ) == sizeof( int32_t ),
                   "AVX2 gather requires 32-bit column indices" );

    const ROWTYPE tile_start = tile * TILE_SIZE;
    const COLTYPE* __restrict col_idx = data._tile_col_idx.data() + tile_start;
    const double* __restrict values = data._tile_val.data() + tile_start;

    for ( int i = 0; i < SIGMA; ++i )
    {
        const int storage_idx = i * OMEGA;
        const __m128i cols = _mm_loadu_si128( reinterpret_cast<const __m128i*>( col_idx + storage_idx ) );
        const __m256d xv = _mm256_i32gather_pd( x, cols, 8 );
        const __m256d vv = _mm256_loadu_pd( values + storage_idx );
        const __m256d prod = _mm256_mul_pd( vv, xv );

        alignas( 32 ) double lanes[4];
        _mm256_store_pd( lanes, prod );
        products[0 * SIGMA + i] = lanes[0];
        products[1 * SIGMA + i] = lanes[1];
        products[2 * SIGMA + i] = lanes[2];
        products[3 * SIGMA + i] = lanes[3];
    }
}
#endif

template <typename ROWTYPE, typename COLTYPE, typename Policy>
void csr5ComputeTileProductsAvx2Double( const CSR5Data<ROWTYPE, COLTYPE, double, Policy>& data,
                                        const ROWTYPE tile,
                                        const double* __restrict x,
                                        std::array<double, Policy::TILE_SIZE>& products )
{
#if MATRIX_UTILS_CSR5_CAN_COMPILE_TARGET_AVX2
    if constexpr ( Policy::OMEGA == 4 && sizeof( COLTYPE ) == sizeof( int32_t ) )
    {
        if ( __builtin_cpu_supports( "avx2" ) )
        {
            csr5ComputeTileProductsAvx2DoubleTarget<ROWTYPE, COLTYPE, Policy>( data, tile, x, products );
            return;
        }
    }
#endif
    {
        csr5ComputeTileProductsScalar<ROWTYPE, COLTYPE, double, Policy>( data, tile, x, products );
    }
}

template <typename COLTYPE, typename VALTYPE>
inline void csr5AddContribution( VALTYPE* __restrict y, const COLTYPE row, const VALTYPE value, const bool use_atomic )
{
    if ( use_atomic )
    {
#pragma omp atomic update
        y[row] += value;
    }
    else
    {
        y[row] += value;
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, typename Policy>
void csr5AccumulateFullTile( const CSR5Data<ROWTYPE, COLTYPE, VALTYPE, Policy>& data,
                             const ROWTYPE tile,
                             const std::array<VALTYPE, Policy::TILE_SIZE>& products,
                             VALTYPE* __restrict y,
                             const VALTYPE alpha,
                             const bool use_atomic )
{
    constexpr int OMEGA = Policy::OMEGA;
    constexpr int SIGMA = Policy::SIGMA;
    constexpr int TILE_SIZE = Policy::TILE_SIZE;

    std::array<uint32_t, OMEGA> bit_flags{};
    for ( int lane = 0; lane < OMEGA; ++lane )
    {
        uint32_t y_offset = 0;
        uint32_t seg_offset = 0;
        data.unpackTileDesc( tile, lane, bit_flags[lane], y_offset, seg_offset );
    }

    COLTYPE row = data._tile_ptr[tile];
    VALTYPE sum = 0;
    for ( int k = 0; k < TILE_SIZE; ++k )
    {
        const int lane = k / SIGMA;
        const int i = k % SIGMA;
        const bool starts_new_row = k != 0 && ( ( bit_flags[lane] & ( uint32_t( 1 ) << i ) ) != 0 );
        if ( starts_new_row )
        {
            csr5AddContribution( y, row, alpha * sum, use_atomic );
            ++row;
            sum = 0;
        }
        sum += products[k];
    }
    csr5AddContribution( y, row, alpha * sum, use_atomic );
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, typename Policy>
void csr5ScaleOutput( const CSR5Data<ROWTYPE, COLTYPE, VALTYPE, Policy>& data,
                      VALTYPE* __restrict y,
                      const VALTYPE beta,
                      const int num_threads )
{
    if ( beta == static_cast<VALTYPE>( 0 ) )
    {
#pragma omp parallel for num_threads( num_threads )
        for ( COLTYPE row = 0; row < data._num_rows; ++row )
        {
            y[row] = static_cast<VALTYPE>( 0 );
        }
    }
    else if ( beta != static_cast<VALTYPE>( 1 ) )
    {
#pragma omp parallel for num_threads( num_threads )
        for ( COLTYPE row = 0; row < data._num_rows; ++row )
        {
            y[row] *= beta;
        }
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, typename Policy>
void csr5SpmvCompute( const CSR5Data<ROWTYPE, COLTYPE, VALTYPE, Policy>& data,
                      const VALTYPE* __restrict x,
                      VALTYPE* __restrict y,
                      const VALTYPE alpha,
                      const int num_threads )
{
    constexpr int TILE_SIZE = Policy::TILE_SIZE;
    const bool use_atomic = num_threads > 1;

#pragma omp parallel for num_threads( num_threads ) schedule( static )
    for ( ROWTYPE tile = 0; tile < data._num_full_tiles; ++tile )
    {
        std::array<VALTYPE, TILE_SIZE> products{};
        if constexpr ( std::is_same_v<VALTYPE, double> )
        {
            csr5ComputeTileProductsAvx2Double<ROWTYPE, COLTYPE, Policy>( data, tile, x, products );
        }
        else
        {
            csr5ComputeTileProductsScalar<ROWTYPE, COLTYPE, VALTYPE, Policy>( data, tile, x, products );
        }
        csr5AccumulateFullTile<ROWTYPE, COLTYPE, VALTYPE, Policy>( data, tile, products, y, alpha, use_atomic );
    }

    if ( data._tail_tile_length == 0 )
    {
        return;
    }

    const ROWTYPE tail_start = data._num_full_tiles * TILE_SIZE;
    const COLTYPE first_tail_row = data._tile_ptr[data._num_full_tiles];

#pragma omp parallel for num_threads( num_threads ) schedule( static )
    for ( COLTYPE row = first_tail_row; row < data._num_rows; ++row )
    {
        const ROWTYPE row_start = std::max( data._row_ptr[row], tail_start );
        const ROWTYPE row_end = data._row_ptr[row + 1];
        VALTYPE sum = 0;
        for ( ROWTYPE idx = row_start; idx < row_end; ++idx )
        {
            sum += data._tile_val[idx] * x[data._tile_col_idx[idx]];
        }
        csr5AddContribution( y, row, alpha * sum, use_atomic );
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, typename Policy>
void csr5Spmv( const CSR5Data<ROWTYPE, COLTYPE, VALTYPE, Policy>& data,
               const VALTYPE* __restrict x,
               VALTYPE* __restrict y,
               const VALTYPE alpha,
               const VALTYPE beta,
               const int num_threads )
{
    csr5ScaleOutput<ROWTYPE, COLTYPE, VALTYPE, Policy>( data, y, beta, num_threads );
    if ( data._nnz == 0 || alpha == static_cast<VALTYPE>( 0 ) )
    {
        return;
    }
    csr5SpmvCompute<ROWTYPE, COLTYPE, VALTYPE, Policy>( data, x, y, alpha, num_threads );
}

} // namespace matrix_utils

#undef MATRIX_UTILS_CSR5_CAN_COMPILE_TARGET_AVX2
#undef MATRIX_UTILS_CSR5_TARGET_AVX2
