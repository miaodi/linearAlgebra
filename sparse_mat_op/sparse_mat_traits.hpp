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

template <typename R, typename C, typename V, typename CSRMatrixType>
struct CSRMatrixFormat {
  static constexpr bool value =
      std::is_same_v<typename CSRMatrixType::ROWTYPE, R> &&
      std::is_same_v<typename CSRMatrixType::COLTYPE, C> &&
      std::is_same_v<typename CSRMatrixType::VALTYPE, V>;
};

template <typename T> struct CSRMatrixRowType {
  using type = typename T::ROWTYPE;
};

template <typename T> struct CSRMatrixIndexType {
  using type = typename T::COLTYPE;
};

template <typename T> struct CSRMatrixValueType {
  using type = typename T::VALTYPE;
};

} // namespace matrix_utils