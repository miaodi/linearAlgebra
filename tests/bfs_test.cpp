#include "bfs.hpp"
#include "io.hpp"
#include "matrix_utils.hpp"
#include "utils.h"
#include <algorithm>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <omp.h>
#include <random>
#include <unordered_set>

TEST(bfs, serial) {
  // https://dl.acm.org/cms/attachment/039ee79d-efce-4a81-8a76-ed21ffbd1a5b/f1.jpg
  std::vector<int> aiA{0, 3, 5, 8, 12, 16, 20, 24, 27, 28};
  std::vector<int> ajA{1, 2, 3, 0, 2, 0, 1, 3, 0, 2,
                       4, 5, 3, 5, 6, 7, 3, 4, 6,
                       7, 4, 5, 7, 8, 4, 5, 6, 6};

  // Test with LASTLEVEL=false
  graph::BFS<graph::BFSFunc<int, int, false, true>> bfs;
  bfs(9, aiA.data(), ajA.data(), 0);
  EXPECT_EQ(bfs.getHeight(), 5);
  std::vector<int> ref{0, 1, 1, 1, 2, 2, 3, 3, 4};
  for (size_t i = 0; i < ref.size(); i++) {
    EXPECT_EQ(bfs.getLevels()[i], ref[i]);
  }

  // Test with LASTLEVEL=true
  graph::BFS<graph::BFSFunc<int, int, true, true>> bfs2;
  bfs2(9, aiA.data(), ajA.data(), 0);

  EXPECT_EQ(bfs2.getLastLevel().size(), 1);
  EXPECT_EQ(bfs2.getLastLevel()[0], 8);

  // Test with 1-based indexing - use ShiftCSRBase
  matrix_utils::ShiftCSRBase(9, 1, aiA.data(), ajA.data());

  bfs(9, aiA.data(), ajA.data(), 1);
  EXPECT_EQ(bfs.getHeight(), 5);
  for (size_t i = 0; i < ref.size(); i++) {
    EXPECT_EQ(bfs.getLevels()[i], ref[i]);
  }

  bfs2(9, aiA.data(), ajA.data(), 1);

  EXPECT_EQ(bfs2.getLastLevel().size(), 1);
  EXPECT_EQ(bfs2.getLastLevel()[0], 9);
}

TEST(bfs, parallel) {
  // https://dl.acm.org/cms/attachment/039ee79d-efce-4a81-8a76-ed21ffbd1a5b/f1.jpg
  std::vector<int> aiA{0, 3, 5, 8, 12, 16, 20, 24, 27, 28};
  std::vector<int> ajA{1, 2, 3, 0, 2, 0, 1, 3, 0, 2,
                       4, 5, 3, 5, 6, 7, 3, 4, 6,
                       7, 4, 5, 7, 8, 4, 5, 6, 6};

  // Test parallel BFS with LASTLEVEL=false, TRACK=true
  graph::BFS<graph::BFSFunc<int, int, false, true>> bfs;
  bfs(9, aiA.data(), ajA.data(), 0, 5);  // 5 threads
  EXPECT_EQ(bfs.getHeight(), 5);
  std::vector<int> ref{0, 1, 1, 1, 2, 2, 3, 3, 4};
  for (size_t i = 0; i < ref.size(); i++) {
    SCOPED_TRACE("Index i = " + std::to_string(i));
    EXPECT_EQ(bfs.getLevels()[i], ref[i]) << "Failed at index i=" << i;
  }

  // Test parallel BFS with LASTLEVEL=false, TRACK=false
  graph::BFS<graph::BFSFunc<int, int, false, false>> bfs2;
  bfs2(9, aiA.data(), ajA.data(), 0, 5);  // 5 threads
  EXPECT_EQ(bfs2.getHeight(), 5);

  // Test parallel BFS with LASTLEVEL=true, TRACK=true for width and lastLevel
  graph::BFS<graph::BFSFunc<int, int, true, true>> bfs3;
  bfs3(9, aiA.data(), ajA.data(), 0, 5);  // 5 threads
  EXPECT_EQ(bfs3.getHeight(), 5);
  
  // Check levels match
  for (size_t i = 0; i < ref.size(); i++) {
    SCOPED_TRACE("Index i = " + std::to_string(i));
    EXPECT_EQ(bfs3.getLevels()[i], ref[i]) << "Failed at index i=" << i;
  }
  
  // Check width (maximum number of nodes at any level)
  // Level 0: 1 node (0)
  // Level 1: 3 nodes (1,2,3)
  // Level 2: 2 nodes (4,5)
  // Level 3: 2 nodes (6,7)
  // Level 4: 1 node (8)
  // Maximum width should be 3
  EXPECT_EQ(bfs3.getWidth(), 3) << "Width should be 3 (level 1 has nodes 1,2,3)";
  
  // Check lastLevel nodes - should only contain node 8 (the farthest node)
  const auto& lastLevel = bfs3.getLastLevel();
  EXPECT_EQ(lastLevel.size(), 1) << "Should have 1 node in last level";
  EXPECT_EQ(lastLevel[0], 8) << "Last level should contain node 8";
}

TEST(bfs, serial_vs_parallel) {
  const std::vector<std::string> files{"data/ex5.mtx", "data/rdist1.mtx"};
  
  for (const auto &fn : files) {
    std::ifstream f(fn);
    std::vector<int> ai, aj;
    std::vector<double> av;
    matrix_utils::readMatrixMarket(f, ai, aj, av);
    
    const int n = ai.size() - 1;
    auto [ai_1based, aj_1based] = std::make_pair(ai, aj);
    matrix_utils::ShiftCSRBase(n, 1, ai_1based.data(), aj_1based.data());
    
    for (int s = 0; s < n; s++) {
      // Serial BFS reference
      graph::BFS<graph::BFSFunc<int, int, true, true>> bfs;
      bfs(n, ai.data(), aj.data(), s);
      const std::unordered_set<int> bfs_lastLevel_set(bfs.getLastLevel().begin(), bfs.getLastLevel().end());
      
      for (int t : {1, 2, 4, 8}) {
        // Parallel BFS with tracking
        graph::BFS<graph::BFSFunc<int, int, true, true>> pbfs;
        pbfs(n, ai.data(), aj.data(), s, t);
        
        // Verify height, levels, and width match
        EXPECT_EQ(pbfs.getHeight(), bfs.getHeight());
        EXPECT_EQ(pbfs.getLevels(), bfs.getLevels()) 
          << "Mismatch at source=" << s << ", threads=" << t;
        EXPECT_EQ(pbfs.getWidth(), bfs.getWidth())
          << "Width mismatch at source=" << s << ", threads=" << t;
        
        // Verify lastLevel sets match
        const std::unordered_set<int> pbfs_lastLevel_set(pbfs.getLastLevel().begin(), pbfs.getLastLevel().end());
        EXPECT_EQ(pbfs_lastLevel_set, bfs_lastLevel_set);

        // Parallel BFS without tracking (width)
        graph::BFS<graph::BFSFunc<int, int, true, false>> pbfs2;
        pbfs2(n, ai.data(), aj.data(), s, t);
        EXPECT_EQ(pbfs2.getHeight(), bfs.getHeight());
        EXPECT_EQ(pbfs2.getLastLevel().size(), bfs.getLastLevel().size());

        // Test 1-based indexing
        pbfs2(n, ai_1based.data(), aj_1based.data(), s + 1, t);
        std::unordered_set<int> pbfs2_lastLevel_0based;
        std::transform(pbfs2.getLastLevel().begin(), pbfs2.getLastLevel().end(),
                      std::inserter(pbfs2_lastLevel_0based, pbfs2_lastLevel_0based.end()),
                      [](int v) { return v - 1; });
        EXPECT_EQ(pbfs2_lastLevel_0based, bfs_lastLevel_set);
      }
    }
  }
}
