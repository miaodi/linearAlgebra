#pragma once

namespace matrix_utils::sparse_cuda
{

/**
 * @brief Enumeration of preconditioner types.
 * left: M^{-1} * A * x = M^{-1} * b
 * right: A * M^{-1} * y = b, x = M^{-1} * y
 * none: A * x = b
 */
enum class PreconditionerType
{
    NONE = 0,
    LEFT = 1,
    RIGHT = 2
};

/**
 * @brief Enumeration of solver convergence states.
 */
enum class State : int
{
    CONVERGED = 0,
    RUNNING = 1,
    MAX_ITER_REACHED = 2,
    FAILED = 3
};

} // namespace matrix_utils::sparse_cuda
