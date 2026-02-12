#pragma once

#include <cstddef>

namespace matrix_utils::sparse_cuda
{

/**
 * @brief Element-wise multiplication: output[i] = a[i] * b[i]
 * 
 * This function launches a CUDA kernel to perform element-wise multiplication
 * of two vectors.
 * 
 * @tparam items_per_thread Number of elements each thread processes (default: 4)
 * @param d_a First device vector
 * @param d_b Second device vector
 * @param d_output Device output vector
 * @param n Vector size
 */
template <int items_per_thread = 4>
void elementwiseMultiply(const double* d_a, const double* d_b,
                         double* d_output, size_t n);

// Additional CUDA kernels for iterative solvers can be added here

} // namespace matrix_utils::sparse_cuda
