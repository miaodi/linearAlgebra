#pragma once
#include <type_traits>
#include <utility>

namespace utils {

template <typename T, typename = std::void_t<>>
struct has_subscript_operator : std::false_type {};

template <typename T>
struct has_subscript_operator<
    T, std::void_t<decltype(std::declval<T>()[std::declval<int>()])>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_subscript_operator_v =
    has_subscript_operator<T>::value;
} // namespace utils
