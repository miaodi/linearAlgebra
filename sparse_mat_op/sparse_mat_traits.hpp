#pragma once
#include <type_traits>

namespace matrix_utils {

template <typename R, typename C, typename V> struct CSRMatrix;
template <typename R, typename C, typename V> struct CSRMatrixVec;

template <typename T> struct CSRResizable : std::false_type {};

template <typename R, typename C, typename V>
struct CSRResizable<CSRMatrix<R, C, V>> : std::true_type {};

template <typename R, typename C, typename V>
struct CSRResizable<CSRMatrixVec<R, C, V>> : std::true_type {};
} // namespace matrix_utils