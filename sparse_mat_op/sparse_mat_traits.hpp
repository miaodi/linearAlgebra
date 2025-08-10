#pragma once
#include <concepts>
#include <type_traits>

namespace matrix_utils {

// NOTE: deprecated, use CSRMatrixType instead
template <typename R, typename C, typename V> struct CSRMatrix;
template <typename R, typename C, typename V> struct CSRMatrixVec;

template <typename T> struct CSRResizable : std::false_type {};

template <typename R, typename C, typename V>
struct CSRResizable<CSRMatrix<R, C, V>> : std::true_type {};

template <typename R, typename C, typename V>
struct CSRResizable<CSRMatrixVec<R, C, V>> : std::true_type {};

// NOTE: deprecated, use CSRMatrixType instead
template <typename R, typename C, typename V, typename CSRMatrixType>
struct CSRMatrixFormat {
  static constexpr bool value =
      std::is_same_v<typename CSRMatrixType::ROWTYPE, R> &&
      std::is_same_v<typename CSRMatrixType::COLTYPE, C> &&
      std::is_same_v<typename CSRMatrixType::VALTYPE, V>;
};

template <typename T>
concept CSRMatrixType = requires(T obj) {
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
};

template <typename T>
concept ResizableCSRMatrixType = CSRMatrixType<T> && requires(T obj) {
  { obj.ResizeAI(std::declval<std::size_t>()) } -> std::same_as<void>;
  { obj.ResizeAJ(std::declval<std::size_t>()) } -> std::same_as<void>;
  { obj.ResizeAV(std::declval<std::size_t>()) } -> std::same_as<void>;
};
// template <typename T>
// concept ResizableCSRMatrixType = true; // Temporarily remove constraints

template <typename T> struct CSRMatrixRowType {
  using type = typename T::ROWTYPE;
};

template <typename T> struct CSRMatrixIndexType {
  using type = typename T::COLTYPE;
};

template <typename T> struct CSRMatrixValueType {
  using type = typename T::VALTYPE;
};

template <typename T>
concept SpmvOpType = requires(const T op, typename T::VALTYPE const *const b,
                              typename T::VALTYPE *const x,
                              const T::VALTYPE alpha, const T::VALTYPE beta) {
  { op.size() } -> std::convertible_to<std::size_t>;
  {op(b, x, alpha, beta)};
  typename T::VALTYPE;
}
&&std::floating_point<typename T::VALTYPE>;

template <typename T>
concept PrecOpType = requires(const T prec, typename T::VALTYPE const *const b,
                              typename T::VALTYPE *const x) {
  { prec.size() } -> std::convertible_to<std::size_t>;
  { prec(b, x) } -> std::same_as<bool>;
  typename T::VALTYPE;
};
} // namespace matrix_utils