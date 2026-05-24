#include "tree.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <vector>

namespace {

std::vector<int> makeParent(const int base) {
  const std::vector<int> zero_based_parent{0, 3, 3, 3, 3, 0};
  std::vector<int> parent(zero_based_parent.size());
  for (std::size_t i = 0; i < zero_based_parent.size(); i++) {
    parent[i] = zero_based_parent[i] + base;
  }
  return parent;
}

} // namespace

TEST(ParentToChildCSR, GroupsChildrenByParent) {
  constexpr int nnodes = 6;
  const std::vector<int> expected_offsets{0, 1, 1, 1, 4, 4, 4};
  const std::vector<int> expected_children{5, 1, 2, 4};
  const std::vector<int> expected_roots{0, 3};

  for (const int base : {0, 1}) {
    const auto parent = makeParent(base);
    std::vector<int> child_offsets(nnodes + 1);
    std::vector<int> children(nnodes - expected_roots.size());
    std::vector<int> roots(expected_roots.size());

    const auto nroots = graph::parentToChildCSR<true>(
        nnodes, base, parent.data(), child_offsets.data(), children.data(),
        roots.data());

    EXPECT_EQ(nroots, static_cast<int>(expected_roots.size()));
    EXPECT_EQ(child_offsets, expected_offsets);
    EXPECT_EQ(children, expected_children);
    EXPECT_EQ(roots, expected_roots);
  }
}

TEST(ParentToChildCSR, FillRootsFalseDoesNotWriteRoots) {
  constexpr int nnodes = 6;
  constexpr int untouched = -7;
  const auto parent = makeParent(0);
  std::vector<int> child_offsets(nnodes + 1);
  std::vector<int> children(nnodes - 2);
  std::vector<int> roots(2, untouched);

  const auto nroots = graph::parentToChildCSR<false>(
      nnodes, 0, parent.data(), child_offsets.data(), children.data(),
      roots.data());

  EXPECT_EQ(nroots, 2);
  EXPECT_EQ(roots, std::vector<int>({untouched, untouched}));
}

TEST(ParentToChildSibling, TemplateRootFillControlsRootWrites) {
  constexpr int nnodes = 6;
  constexpr int untouched = -7;
  const auto parent = makeParent(0);
  std::vector<int> first_child(nnodes);
  std::vector<int> next_sibling(nnodes);
  std::vector<int> roots(2, untouched);
  std::vector<int> child_count(nnodes);

  const auto nroots_without_fill = graph::parentToChildSibling<false>(
      nnodes, 0, parent.data(), first_child.data(), next_sibling.data(),
      roots.data(), child_count.data());

  EXPECT_EQ(nroots_without_fill, 2);
  EXPECT_EQ(roots, std::vector<int>({untouched, untouched}));
  EXPECT_EQ(child_count, std::vector<int>({1, 0, 0, 3, 0, 0}));

  const auto nroots_with_fill = graph::parentToChildSibling<true>(
      nnodes, 0, parent.data(), first_child.data(), next_sibling.data(),
      roots.data(), child_count.data());

  EXPECT_EQ(nroots_with_fill, 2);
  EXPECT_EQ(roots, std::vector<int>({0, 3}));
}
