#pragma once

#include <cstdint>
#include <type_traits>

namespace matrix_utils
{

constexpr int csr5BitsNeeded( const int value )
{
    int bits = 1;
    int capacity = 2;
    while ( capacity < value )
    {
        capacity <<= 1;
        ++bits;
    }
    return bits;
}

template <int Omega, int Sigma>
struct CSR5StaticPolicy
{
    static_assert( Omega > 0, "CSR5 omega must be positive" );
    static_assert( Sigma > 0, "CSR5 sigma must be positive" );

    static constexpr int OMEGA = Omega;
    static constexpr int SIGMA = Sigma;
    static constexpr int TILE_SIZE = OMEGA * SIGMA;
    static constexpr int BIT_Y_OFFSET = csr5BitsNeeded( TILE_SIZE );
    static constexpr int BIT_SEG_OFFSET = csr5BitsNeeded( OMEGA );
    static constexpr int DESCRIPTOR_BITS = BIT_Y_OFFSET + BIT_SEG_OFFSET + SIGMA;

    static_assert( SIGMA <= 32, "CSR5 sigma bit flags must fit in uint32_t" );
    static_assert( DESCRIPTOR_BITS <= 32,
                   "This CSR5 implementation supports only one 32-bit descriptor packet per lane" );
};

/**
 * @brief Type trait for checking CSR5 policy requirements
 *
 * A valid CSR5 policy must define:
 * - OMEGA: Tile height (number of SIMD lanes, aligned with vector width)
 * - SIGMA: Tile width (tunable parameter for memory access patterns)
 * - TILE_SIZE: Total elements per tile (OMEGA * SIGMA)
 */
template <typename T, typename = void>
struct is_csr5_policy : std::false_type
{
};

template <typename T>
struct is_csr5_policy<T, std::void_t<decltype( T::OMEGA ), decltype( T::SIGMA ), decltype( T::TILE_SIZE ), decltype( T::BIT_Y_OFFSET ), decltype( T::BIT_SEG_OFFSET ), decltype( T::DESCRIPTOR_BITS )>>
    : std::true_type
{
};

template <typename T>
inline constexpr bool is_csr5_policy_v = is_csr5_policy<T>::value;

/**
 * @brief CSR5 policy for AVX2 instruction set
 *
 * AVX2 provides 256-bit SIMD vectors:
 * - For double (64-bit): 256/64 = 4 lanes  → OMEGA = 4
 * - For float (32-bit):  256/32 = 8 lanes  → OMEGA = 8
 *
 * SIGMA is a tunable parameter that controls:
 * - Tile width along the CSR element sequence
 * - Memory access granularity and cache behavior
 * - Typical values: 16-64, default 32 for balance
 *
 * Tile layout (column-major):
 *   For double with OMEGA=4, SIGMA=32, a tile stores 128 elements:
 *
 *   CSR sequence:  [e0, e1, e2, ..., e127]
 *
 *   Tile structure (4 rows × 32 columns, column-major):
 *   Lane 0: e0   e4   e8   e12  ... e124  (elements 0, 4, 8, ..., 124)
 *   Lane 1: e1   e5   e9   e13  ... e125  (elements 1, 5, 9, ..., 125)
 *   Lane 2: e2   e6   e10  e14  ... e126  (elements 2, 6, 10, ..., 126)
 *   Lane 3: e3   e7   e11  e15  ... e127  (elements 3, 7, 11, ..., 127)
 *
 *   Storage order: e0, e1, e2, e3, e4, e5, e6, e7, ...
 *
 * @tparam VALTYPE Value type (double or float)
 */
template <typename VALTYPE>
struct CSR5_AVX2_Policy;

// Specialization for double precision
template <>
struct CSR5_AVX2_Policy<double> : CSR5StaticPolicy<4, 16>
{
};

// Specialization for single precision
template <>
struct CSR5_AVX2_Policy<float> : CSR5StaticPolicy<8, 16>
{
};

// Future policy placeholders:
//
// template<>
// struct CSR5_AVX512_Policy<double> {
//     static constexpr int OMEGA = 8;      // AVX512 double-precision lanes
//     static constexpr int SIGMA = 32;
//     static constexpr int TILE_SIZE = OMEGA * SIGMA;
// };
//
// template<>
// struct CSR5_Scalar_Policy<double> {
//     static constexpr int OMEGA = 1;      // Scalar (no SIMD)
//     static constexpr int SIGMA = 32;
//     static constexpr int TILE_SIZE = OMEGA * SIGMA;
// };

} // namespace matrix_utils
