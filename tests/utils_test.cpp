#include "../utils/utils.h"
#include <gtest/gtest.h>
#include <iomanip>
#include <memory>

TEST(Utils, knuth_s) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist1(
      0, 100000000); // distribution in range [1, 100000000]
  size_t size = dist1(rng);
  size_t lower_bound = dist1(rng);
  std::uniform_int_distribution<std::mt19937::result_type> dist2(
      100000000, 10000000000); // distribution in range [100000000, 10000000000]

  size_t upper_bound = std::max(size + lower_bound, dist1(rng));

  std::vector<int> randVec(size, 0);

  utils::knuth_s rand;
  for (int i = 0; i < 10; i++) {
    rand(size, lower_bound, upper_bound, randVec.begin());
    for (auto i : randVec) {
      EXPECT_GE(i, lower_bound);
      EXPECT_LT(i, upper_bound);
    }
  }
}

TEST(Utils, MaxHeap) {

  auto compMax = [](const int v1, const int v2) { return v1 > v2; };

  utils::MaxHeap<int, decltype(compMax)> max_heap(compMax);

  for (int i = 10; i >= 0; i--) {
    max_heap.push(i);
    EXPECT_EQ(i, *max_heap.top());
  }
  max_heap.clear();

  for (int i = 10; i >= 0; i--) {
    max_heap.push(i);
    if (max_heap.size() > 2) {
      max_heap.pop();
      EXPECT_EQ(9, *max_heap.top());
    }
  }
}