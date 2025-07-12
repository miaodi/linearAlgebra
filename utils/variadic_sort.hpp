#pragma once

#include "traits.hpp"
#include <cstddef>
#include <utility>

namespace utils {

/**
 * @brief Swap the i-th and j-th elements of the arguments
 *
 * @tparam Idx type of the indices
 * @tparam Args types of the arguments
 * @param i index of the first argument
 * @param j index of the second argument
 * @param args arguments to be swapped
 */
template <typename Idx, typename First, typename... Args>
void variadic_swap(const Idx i, const Idx j, First &&first, Args &&...args) {
  static_assert(utils::has_subscript_operator_v<First>,
                "All arguments must have a subscript operator");
  std::swap(first[i], first[j]);
  if constexpr (sizeof...(Args) > 0) {
    variadic_swap(i, j, std::forward<Args>(args)...);
  }
}

/**
 * @brief Assign the i-th element of the first argument to the j-th element of
 * the second argument
 *
 * @tparam Idx type of the indices
 * @tparam First type of the first argument
 * @tparam Second type of the second argument
 * @tparam Args types of the remaining arguments
 * @param i index of the first argument
 * @param j index of the second argument
 * @param first first argument
 * @param second second argument
 * @param args remaining arguments
 */
template <typename Idx, typename First, typename Second, typename... Args>
void variadic_assign(const Idx i, const Idx j, First first, Second second,
                     Args &&...args) {
  static_assert(sizeof...(Args) % 2 == 0,
                "The number of arguments must be even");
  static_assert(std::is_same_v<std::decay_t<First>, std::decay_t<Second>>,
                "The first and second arguments must be the same type");
  static_assert(utils::has_subscript_operator_v<First>,
                "All arguments must have a subscript operator");

  second[j] = first[i];
  if constexpr (sizeof...(Args) >= 2) {
    variadic_assign(i, j, std::forward<Args>(args)...);
  }
}

template <typename Idx, typename T1, typename T2, typename... Args1,
          typename... Args2>
void variadic_assign_uninterleave(const Idx i, const Idx j, T1 first,
                                  Args1... args1, T2 second, Args2... args2) {
  static_assert(sizeof...(Args1) == sizeof...(Args2),
                "The number of arguments must be the same");
  static_assert(utils::has_subscript_operator_v<T1>,
                "All arguments must have a subscript operator");
  static_assert(std::is_same_v<std::decay_t<T1>, std::decay_t<T2>>,
                "The first and second arguments must be the same type");
  second[j] = first[i];
  if constexpr (sizeof...(Args1) > 0) {
    variadic_assign_uninterleave(i, j, std::forward<Args1>(args1)...,
                                 std::forward<Args2>(args2)...);
  }
}

/**
 * @brief Variadic insertion sort
 *
 * @tparam Args types of the arguments
 * @param begin the first index to be sorted
 * @param end after the last index to be sorted
 * @param first first array to be sorted, sort is based on this array
 * @param args remaining arrays to be sorted, must have the same size as first
 */
template <typename Idx, typename First, typename... Args>
void variadic_insertion_sort(const Idx begin, const Idx end, First first,
                             Args... args);

/**
 * @brief Variadic partition
 *
 * @tparam Args types of the arguments
 * @param begin the first index to be partitioned
 * @param end after the last index to be partitioned
 * @param first first array to be partitioned, partition is based on this array
 * @param args remaining arrays to be partitioned, must have the same size as
 * first
 */
template <typename Idx, typename First, typename... Args>
Idx variadic_partition(const Idx begin, const Idx end, First first,
                       Args... args);

/**
 * @brief Variadic quick sort
 *
 * @tparam Args types of the arguments
 * @param begin the first index to be sorted
 * @param end after the last index to be sorted
 * @param first first array to be sorted, sort is based on this array
 * @param args remaining arrays to be sorted, must have the same size as first
 */
template <typename Idx, typename First, typename... Args>
void variadic_quick_sort(const Idx begin, const Idx end, First first,
                         Args... args);

} // namespace utils