#include "circularbuffer.hpp"
#include "utils.h"
#include <algorithm>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

// Demonstrate some basic assertions.
TEST(circular_buffer, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
}

// Demonstrate some basic assertions.
TEST(circular_buffer, unshift) {
  utils::CircularBuffer<int> cb(5);
  EXPECT_TRUE(cb.unshift(1));
  EXPECT_EQ(cb.size(), 1);
  EXPECT_EQ(cb.available(), 4);
  EXPECT_TRUE(cb.unshift(2));
  EXPECT_EQ(cb.size(), 2);
  EXPECT_EQ(cb.available(), 3);
  EXPECT_TRUE(cb.unshift(3));
  EXPECT_EQ(cb.size(), 3);
  EXPECT_EQ(cb.available(), 2);
  EXPECT_TRUE(cb.unshift(4));
  EXPECT_EQ(cb.size(), 4);
  EXPECT_EQ(cb.available(), 1);
  EXPECT_TRUE(cb.unshift(5));
  EXPECT_EQ(cb.size(), 5);
  EXPECT_EQ(cb.available(), 0);
  EXPECT_FALSE(cb.unshift(6));
  EXPECT_EQ(cb.size(), 5);
  EXPECT_EQ(cb.available(), 0);
}

// Demonstrate some basic assertions.
TEST(circular_buffer, push) {
  utils::CircularBuffer<int> cb(5);
  EXPECT_TRUE(cb.push(1));
  EXPECT_EQ(cb.size(), 1);
  EXPECT_EQ(cb.available(), 4);
  EXPECT_TRUE(cb.push(2));
  EXPECT_EQ(cb.size(), 2);
  EXPECT_EQ(cb.available(), 3);
  EXPECT_TRUE(cb.push(3));
  EXPECT_EQ(cb.size(), 3);
  EXPECT_EQ(cb.available(), 2);
  EXPECT_TRUE(cb.push(4));
  EXPECT_EQ(cb.size(), 4);
  EXPECT_EQ(cb.available(), 1);
  EXPECT_TRUE(cb.push(5));
  EXPECT_EQ(cb.size(), 5);
  EXPECT_EQ(cb.available(), 0);
  EXPECT_FALSE(cb.push(6));
  EXPECT_EQ(cb.size(), 5);
  EXPECT_EQ(cb.available(), 0);
}

TEST(circular_buffer, random_access) {
  utils::CircularBuffer<int> cb(5);
  cb.unshift(5);
  cb.unshift(4);
  cb.unshift(3);
  EXPECT_EQ(cb[0], 3);
  EXPECT_EQ(cb[1], 4);
  EXPECT_EQ(cb[2], 5);
  EXPECT_EQ(cb.first(), 3);
  EXPECT_EQ(cb.last(), 5);
  EXPECT_EQ(cb[3], 5);

  cb.push(6);
  EXPECT_EQ(cb.first(), 3);
  EXPECT_EQ(cb.last(), 6);
  cb.push(7);
  EXPECT_EQ(cb.first(), 3);
  EXPECT_EQ(cb.last(), 7);
  cb.push(8);
  EXPECT_EQ(cb.first(), 4);
  EXPECT_EQ(cb.last(), 8);
}

TEST(circular_buffer, copy) {
  utils::CircularBuffer<int> cb(5);
  cb.unshift(5);
  cb.unshift(4);
  cb.unshift(3);
  cb.push(6);
  cb.push(7);
  cb.push(8);
  std::vector<int> cp(5);
  cb.copyToVector(cp);
  ASSERT_THAT(cp, testing::ElementsAre(4, 5, 6, 7, 8));
  cb.unshift(2);
  cb.copyToVector(cp);
  ASSERT_THAT(cp, testing::ElementsAre(2, 4, 5, 6, 7));
}

TEST(circular_buffer, resize) {
  utils::CircularBuffer<int> cb(5);
  cb.unshift(5);
  cb.unshift(4);
  cb.unshift(3);
  cb.push(6);
  cb.push(7);
  cb.push(8);
  cb.resizePreserve(6);
  std::vector<int> cp(6, -1);
  cb.copyToVector(cp);
  // std::copy(cp.begin(), cp.end(), std::ostream_iterator<int>(std::cout, "
  // "));
  ASSERT_THAT(cp, testing::ElementsAre(4, 5, 6, 7, 8, -1));
  // cb.unshift(2);
  // cb.copyToVector(cp);
  // ASSERT_THAT(cp, testing::ElementsAre(2, 4, 5, 6, 7));
}