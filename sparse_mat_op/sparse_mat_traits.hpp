#pragma once
#include <concepts>
#include <type_traits>
#include <cstddef>

namespace matrix_utils {

// ============================================================================
// Type Traits (using alias templates)
// ============================================================================

// Type trait to extract value type from array-like types
template <class Array>
using array_value_type = std::decay_t<decltype(std::declval<Array &>()[0])>;

// Extract row/column/value types from CSR matrices
template <typename T> 
using csr_row_type_t = typename T::ROWTYPE;

template <typename T> 
using csr_col_type_t = typename T::COLTYPE;

template <typename T> 
using csr_value_type_t = typename T::VALTYPE;

// Check if CSR matrix owns its data by detecting static constexpr member
template <typename T>
concept csr_owns_data = requires {
  { T::owns_data } -> std::convertible_to<bool>;
  requires T::owns_data == true;
};

template <typename T>
concept csr_view_data = requires {
  { T::owns_data } -> std::convertible_to<bool>;
  requires T::owns_data == false;
};

// Check if CSR matrix is swappable (uses std::swap or ADL swap)
template <typename T>
concept csr_swappable = std::is_swappable_v<T>;

// ============================================================================
// Concepts (modern C++20 approach)
// ============================================================================

// Core CSR matrix concept
template <typename T>
concept CSR = requires(T obj) {
  typename T::ROWTYPE;
  typename T::COLTYPE;
  typename T::VALTYPE;

  // Check for member functions returning non-const pointers
  { obj.AI() } -> std::same_as<typename T::ROWTYPE *>;
  { obj.AJ() } -> std::same_as<typename T::COLTYPE *>;
  { obj.AV() } -> std::same_as<typename T::VALTYPE *>;

  // Check for member functions returning const pointers
  {
    static_cast<const T &>(obj).AI()
  } -> std::same_as<typename T::ROWTYPE const *>;
  {
    static_cast<const T &>(obj).AJ()
  } -> std::same_as<typename T::COLTYPE const *>;
  {
    static_cast<const T &>(obj).AV()
  } -> std::same_as<typename T::VALTYPE const *>;

  // Check for rows and cols member variables
  { obj.rows } -> std::convertible_to<typename T::COLTYPE>;
  { obj.cols } -> std::convertible_to<typename T::COLTYPE>;
  
  // Check for Base() and NNZ() methods
  { obj.Base() } -> std::same_as<typename T::ROWTYPE>;
  { obj.NNZ() } -> std::same_as<typename T::ROWTYPE>;
};

// Resizable CSR matrix concept
template <typename T>
concept ResizableCSR = CSR<T> && requires(T obj) {
  {
    obj.ResizeAI(std::declval<std::size_t>())
  } -> std::same_as<typename T::ROWTYPE *>;
  {
    obj.ResizeAJ(std::declval<std::size_t>())
  } -> std::same_as<typename T::COLTYPE *>;
  {
    obj.ResizeAV(std::declval<std::size_t>())
  } -> std::same_as<typename T::VALTYPE *>;
};

// Swappable resizable CSR matrix concept
template <typename T>
concept SwappableResizableCSR = ResizableCSR<T> && csr_owns_data<T> && csr_swappable<T>;

// Diagonal matrix support
template <typename T>
concept HasDiagonal = requires(T obj) {
  { obj.Diagonal() } -> std::same_as<typename T::ROWTYPE *>;
  {
    static_cast<const T &>(obj).Diagonal()
  } -> std::same_as<typename T::ROWTYPE const *>;
};

template <typename T>
concept ResizableDiagonal = ResizableCSR<T> && HasDiagonal<T> && 
  requires(T obj) {
    {
      obj.ResizeDiagonal(std::declval<std::size_t>())
    } -> std::same_as<typename T::ROWTYPE *>;
  };

// Operator concepts
template <typename T>
concept SpmvOp = requires(const T op, typename T::VALTYPE const *const b,
                          typename T::VALTYPE *const x,
                          typename T::VALTYPE alpha, typename T::VALTYPE beta) {
  typename T::VALTYPE;
  { op.size() } -> std::convertible_to<std::size_t>;
  { op(b, x, alpha, beta) } -> std::same_as<void>;
} && std::floating_point<typename T::VALTYPE>;

template <typename T>
concept PrecOp = requires(const T prec, typename T::VALTYPE const *const b,
                          typename T::VALTYPE *const x) {
  typename T::VALTYPE;
  { prec.size() } -> std::convertible_to<std::size_t>;
  { prec(b, x) } -> std::same_as<bool>;
};

// Vector-like concept for transformation support
template <typename T, typename VALTYPE>
concept VectorLike = requires(T& v, const T& cv, std::size_t i) {
  typename T::value_type;
  requires std::same_as<typename T::value_type, VALTYPE>;
  { v[i] } -> std::convertible_to<VALTYPE&>;
  { cv.size() } -> std::convertible_to<std::size_t>;
  { std::swap(v, v) } -> std::same_as<void>;
};

// ============================================================================
// Legacy Traits (deprecated - kept for backward compatibility)
// ============================================================================

// Forward declarations for actual CSRMatrix classes (defined in matrix_utils.hpp)
template <typename R, typename C, typename V> struct CSRMatrix;
template <typename R, typename C, typename V> struct CSRMatrixVec;

template <typename T> 
struct CSRResizable : std::false_type {};

template <typename R, typename C, typename V>
struct CSRResizable<CSRMatrix<R, C, V>> : std::true_type {};

template <typename R, typename C, typename V>
struct CSRResizable<CSRMatrixVec<R, C, V>> : std::true_type {};

// NOTE: deprecated, use CSR concept instead
template <typename R, typename C, typename V, typename CSRMatrixType>
struct CSRMatrixFormat {
  static constexpr bool value =
      std::is_same_v<typename CSRMatrixType::ROWTYPE, R> &&
      std::is_same_v<typename CSRMatrixType::COLTYPE, C> &&
      std::is_same_v<typename CSRMatrixType::VALTYPE, V>;
};

// NOTE: deprecated, use csr_row_type_t alias template instead
template <typename T> 
struct CSRMatrixRowType {
  using type = typename T::ROWTYPE;
};

// NOTE: deprecated, use csr_col_type_t alias template instead
template <typename T> 
struct CSRMatrixIndexType {
  using type = typename T::COLTYPE;
};

// NOTE: deprecated, use csr_value_type_t alias template instead
template <typename T> 
struct CSRMatrixValueType {
  using type = typename T::VALTYPE;
};

} // namespace matrix_utils