#include "variadic_sort.hpp"
namespace utils {

template <typename Idx, typename First, typename... Args>
void variadic_insertion_sort(const Idx begin, const Idx end, First first,
                             Args... args) {
  for (Idx i = begin + 1; i < end; i++) {
    for (Idx j = i; j > begin && first[j] < first[j - 1]; j--) {
      variadic_swap(j, j - 1, first, args...);
    }
  }
}

template <typename Idx, typename First, typename... Args>
Idx variadic_partition(const Idx begin, const Idx end, First first,
                       Args... args) {
  static_assert(utils::has_subscript_operator_v<First>,
                "All arguments must have a subscript operator");
  // Use the last element as the pivot
  auto pivot = first[end - 1];
  Idx i = begin;
  Idx j = end - 1;
  while (i < j) {
    while (i < j && first[i] <= pivot) {
      i++;
    }
    while (i < j && first[j] >= pivot) {
      j--;
    }
    if (i < j) {
      variadic_swap(i, j, first, args...);
    }
  }
  variadic_swap(i, end - 1, first, args...);

  return i;
}

template <typename Idx, typename First, typename... Args>
void variadic_quick_sort(const Idx begin, const Idx end, First first,
                         Args... args) {
  if (begin >= end) {
    return;
  }
  if (end - begin < 16) {
    variadic_insertion_sort(begin, end, first, args...);
    return;
  }

  Idx pivot = variadic_partition(begin, end, first, args...);
  variadic_quick_sort(begin, pivot, first, args...);
  variadic_quick_sort(pivot + 1, end, first, args...);
}

template void variadic_insertion_sort<int, int *>(int, int, int *);
template int variadic_partition<int, int *>(int, int, int *);
template void variadic_quick_sort<int, int *>(int, int, int *);
template void variadic_insertion_sort<int, int *, int *>(int, int, int *,
                                                         int *);
template int variadic_partition<int, int *, int *>(int, int, int *, int *);
template void variadic_quick_sort<int, int *, int *>(int, int, int *, int *);

template void variadic_insertion_sort<int, int *, double *>(int, int, int *,
                                                            double *);
template int variadic_partition<int, int *, double *>(int, int, int *,
                                                      double *);
template void variadic_quick_sort<int, int *, double *>(int, int, int *,
                                                        double *);

template void variadic_insertion_sort<int, int *, float *>(int, int, int *,
                                                           float *);
template int variadic_partition<int, int *, float *>(int, int, int *,
                                                     float *);
template void variadic_quick_sort<int, int *, float *>(int, int, int *,
                                                       float *);

} // namespace utils