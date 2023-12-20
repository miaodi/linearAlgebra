#pragma once
#include <chrono>
#include <functional>
#include <iostream>
#include <type_traits>
namespace utils {
int getValue(const int v);
template <typename T = std::ratio<1, 1>> struct Elapse {
  template <typename F>
  static decltype(auto) execute(const std::string &S, F &&func) {
    auto start = std::chrono::steady_clock::now();
    if constexpr (std::is_void_v<decltype(func())>) {
      std::forward<F>(func)();
      std::chrono::duration<float, T> duration =
          (std::chrono::steady_clock::now() - start);
      std::cout << std::string(S) + " uses: Elapsed time: " << duration.count()
                << "(s), VmSize: " << getValue(0)
                << "(kb), VmRSS: " << getValue(1) << "(kb)" << std::endl;
    } else {
      auto &&res = std::forward<F>(func)();
      std::chrono::duration<float, T> duration =
          (std::chrono::steady_clock::now() - start);
      std::cout << std::string(S) + " uses: Elapsed time: " << duration.count()
                << "(s), VmSize: " << getValue(0)
                << "(kb), VmRSS: " << getValue(1) << "(kb)" << std::endl;
      return std::move(res);
    }
  }
};

} // namespace utils