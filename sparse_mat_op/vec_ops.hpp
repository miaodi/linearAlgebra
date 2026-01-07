#pragma once
#include <algorithm>
#include <numeric>
#include <cmath>

/**
 * @file vec_ops.hpp
 * @brief Optimized vector operations with OpenMP parallelization and SIMD support
 * 
 * This file provides high-performance vector operations optimized for:
 * - Multi-core parallelization using OpenMP
 * - SIMD vectorization for efficient CPU utilization
 * - Static scheduling for balanced workload distribution
 * 
 * @note Requires compilation with OpenMP support (-fopenmp for GCC/Clang)
 */
namespace vec_ops {

/**
 * @brief Copy vector from source to destination
 * @tparam IDX Index type (e.g., int, size_t)
 * @tparam VAL Value type (e.g., float, double)
 * @param size Number of elements to copy
 * @param src Source vector
 * @param dst Destination vector
 * @note Uses std::copy for compiler optimization opportunities
 */
template <typename IDX, typename VAL>
void copy_vec(const IDX size, VAL const *src, VAL *dst) {
    std::copy(src, src + size, dst);
}

/**
 * @brief Scale vector by a scalar: vec = alpha * vec
 * @tparam IDX Index type
 * @tparam VAL Value type
 * @param size Number of elements
 * @param alpha Scaling factor
 * @param vec Vector to scale (modified in-place)
 * @note Parallelized with OpenMP and SIMD instructions
 */
template <typename IDX, typename VAL>
void scale_vec(const IDX size, const VAL alpha, VAL *vec) {
#pragma omp parallel for simd schedule(static)
  for (IDX i = 0; i < size; ++i) {
    vec[i] *= alpha;
  }
}

/**
 * @brief Compute dot product of two vectors: result = vec1 · vec2
 * @tparam IDX Index type
 * @tparam VAL Value type
 * @param size Number of elements
 * @param vec1 First input vector
 * @param vec2 Second input vector
 * @return Dot product result
 * @note Uses parallel reduction for thread-safe accumulation
 */
template <typename IDX, typename VAL>
VAL dot_product(const IDX size, const VAL *vec1, const VAL *vec2) {
  VAL result = static_cast<VAL>(0);
#pragma omp parallel for simd reduction(+:result) schedule(static)
  for (IDX i = 0; i < size; ++i) {
    result += vec1[i] * vec2[i];
  }
  return result;
}

/**
 * @brief Compute L2 norm (Euclidean norm) of a vector: ||vec||₂
 * @tparam IDX Index type
 * @tparam VAL Value type
 * @param size Number of elements
 * @param vec Input vector
 * @return L2 norm of the vector
 * @note Implemented using dot_product for code reuse and optimization
 */
template <typename IDX, typename VAL>
VAL vec_l2_norm(const IDX size, const VAL *vec) {
  return std::sqrt(dot_product(size, vec, vec));
}

/**
 * @brief AXPY operation: y = y + alpha * x
 * @tparam IDX Index type
 * @tparam VAL Value type
 * @param size Number of elements
 * @param alpha Scaling factor for x
 * @param x Input vector x (unchanged)
 * @param y Input/output vector y (modified in-place)
 * @note Classic BLAS Level 1 operation, parallelized with OpenMP
 */
template <typename IDX, typename VAL>
void axpy(const IDX size, const VAL alpha, const VAL *x, VAL *y) {
#pragma omp parallel for simd schedule(static)
  for (IDX i = 0; i < size; ++i) {
    y[i] += alpha * x[i];
  }
}

/**
 * @brief WAXPBY operation: w = alpha * x + beta * y
 * @tparam IDX Index type
 * @tparam VAL Value type
 * @param size Number of elements
 * @param alpha Scaling factor for x
 * @param x First input vector
 * @param beta Scaling factor for y
 * @param y Second input vector
 * @param w Output vector
 * @note Linear combination of two vectors
 */
template <typename IDX, typename VAL>
void waxpby(const IDX size, const VAL alpha, const VAL *x, const VAL beta, const VAL *y, VAL *w) {
#pragma omp parallel for simd schedule(static)
  for (IDX i = 0; i < size; ++i) {
    w[i] = alpha * x[i] + beta * y[i];
  }
}

/**
 * @brief Compute z = x + alpha * (y + beta * w)
 * @tparam IDX Index type
 * @tparam VAL Value type
 * @param size Number of elements
 * @param x First input vector
 * @param alpha Scaling factor for the expression (y + beta * w)
 * @param y Second input vector
 * @param beta Scaling factor for w
 * @param w Third input vector
 * @param z Output vector
 * @note Compound operation used in iterative solvers like BiCGSTAB
 */
template <typename IDX, typename VAL>
void xpay_pbw(const IDX size, const VAL *x, const VAL alpha, const VAL *y, const VAL beta, const VAL *w, VAL *z) {
#pragma omp parallel for simd schedule(static)
  for (IDX i = 0; i < size; ++i) {
    z[i] = x[i] + alpha * (y[i] + beta * w[i]);
  }
}
} // namespace vec_ops