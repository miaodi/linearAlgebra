#pragma once
#include <chrono>
#include <ctime>
#include <functional>
#include <iostream>
#include <type_traits>
namespace utils {
int getValue(const int v);
template <typename T = std::ratio<1, 1>> struct Elapse {
  template <typename F>
  static decltype(auto) execute(const std::string &S, F &&func) {
    auto start = std::chrono::steady_clock::now();
    auto start_cpu_time = std::clock();
    // std::cout << start_cpu_time << std::endl;
    if constexpr (std::is_void_v<decltype(func())>) {
      std::forward<F>(func)();
      std::chrono::duration<float, T> duration =
          (std::chrono::steady_clock::now() - start);
      auto cpu_duration = (std::clock() - start_cpu_time);
      std::cout << std::string(S) + " uses: Elapsed time: " << duration.count()
                << " (s), CPU time: " << cpu_duration / (double)CLOCKS_PER_SEC
#ifdef GET_MEMORY_USAGE
                << "(s), VmSize: " << getValue(0)
                << "(kb), VmRSS: " << getValue(1) << "(kb)" << std::endl;
#else
                << " (s)" << std::endl;
#endif
    } else {
      auto &&res = std::forward<F>(func)();
      std::chrono::duration<float, T> duration =
          (std::chrono::steady_clock::now() - start);
      auto cpu_duration = (std::clock() - start_cpu_time);
      std::cout << std::string(S) + " uses: Elapsed time: " << duration.count()
                << " (s), CPU time: " << cpu_duration / (double)CLOCKS_PER_SEC
#ifdef GET_MEMORY_USAGE
                << "(s), VmSize: " << getValue(0)
                << "(kb), VmRSS: " << getValue(1) << "(kb)" << std::endl;
#else
                << " (s)" << std::endl;
#endif
      return std::move(res);
    }
  }
};

} // namespace utils