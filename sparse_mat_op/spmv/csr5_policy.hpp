#pragma once

#include <cstdint>
#include <type_traits>

namespace matrix_utils
{

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
struct is_csr5_policy<T, std::void_t<decltype( T::OMEGA ), decltype( T::SIGMA ), decltype( T::TILE_SIZE )>>
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
struct CSR5_AVX2_Policy<double>
{
    static constexpr int OMEGA = 4;                 // AVX2 double-precision lanes
    static constexpr int SIGMA = 32;                // Tunable tile width
    static constexpr int TILE_SIZE = OMEGA * SIGMA; // 128 elements per tile

    static_assert( OMEGA <= 32, "OMEGA must be <= 32 for bit-flag to fit in uint32_t" );
};

// Specialization for single precision
template <>
struct CSR5_AVX2_Policy<float>
{
    static constexpr int OMEGA = 8;                 // AVX2 single-precision lanes
    static constexpr int SIGMA = 32;                // Tunable tile width
    static constexpr int TILE_SIZE = OMEGA * SIGMA; // 256 elements per tile

    static_assert( OMEGA <= 32, "OMEGA must be <= 32 for bit-flag to fit in uint32_t" );
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
