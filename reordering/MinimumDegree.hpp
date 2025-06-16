#pragma once
#include "../config.h"
#include <UnionFind.h>
#include <map>
#include <set>
#include <utility>
#include <vector>
#include <queue>
namespace reordering {

template <typename COLTYPE> class QuotientGraph {
  struct Node {
    COLTYPE degree; // node degree

    std::vector<COLTYPE>
        adjacent_variables;                 // indices of the adjacent variables
    std::vector<COLTYPE> adjacent_elements; // indices of the adjacent elements
    std::vector<COLTYPE> simple_variables;  // indices of the simple variables
  };

public:
  QuotientGraph() = default;
  template <typename ROWTYPE>
  void operator()(const COLTYPE nnodes, ROWTYPE const *ai, COLTYPE const *aj,
                  COLTYPE *iperm, COLTYPE *perm);

protected:
  // check if two nodes are distinguishable
  bool isDistinguishable(const COLTYPE i, const COLTYPE j) const;

  // Adj_G(i)
  std::vector<COLTYPE> &getFillins(const COLTYPE i);

  void merge(const COLTYPE i, const COLTYPE j);

  void massElimination(const COLTYPE i);

  void principleVector(std::vector<COLTYPE> &vec);

  void updateNode(const COLTYPE i);

  template <typename ROWTYPE>
  void initialize(const COLTYPE nnodes, ROWTYPE const *ai, COLTYPE const *aj);

protected:
  std::vector<Node> _nodes;
  UnionFind<COLTYPE, false> _union_find; // union-find structure
  std::map<COLTYPE, std::set<COLTYPE>>
      _degree_to_principle; // map variable to element

  std::vector<COLTYPE> __temp1;
  std::vector<COLTYPE> __temp2; // temporary vectors for sorting
  std::vector<std::vector<COLTYPE> *> __vectors;
};

template <typename COLTYPE>
void mergeKVectors(const std::vector<std::vector<COLTYPE> *> &input_vectors,
                   std::vector<COLTYPE> &output_vector) {
  using Entry = std::tuple<COLTYPE, COLTYPE,
                           COLTYPE>; // (value, list_index, element_index)

  // Min-heap
  std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> minHeap;

  // Initialize heap with the first element of each list
  for (int i = 0; i < input_vectors.size(); ++i) {
    if (!(*input_vectors[i]).empty()) {
      minHeap.emplace((*input_vectors[i])[0], i, 0);
    }
  }
  // Clear the output vector
  output_vector.clear();
  // Merge the vectors
  while (!minHeap.empty()) {
    auto [val, row, col] = minHeap.top();
    minHeap.pop();

    if (output_vector.size() > 0 && output_vector.back() == val) {
      continue; // skip duplicates
    }
    output_vector.push_back(val);

    if (col + 1 < (*input_vectors[row]).size()) {
      minHeap.emplace((*input_vectors[row])[col + 1], row, col + 1);
    }
  }
}

template <typename COLTYPE>
void vectorSubtract(std::vector<COLTYPE> &op1,
                    const std::vector<COLTYPE> &op2) {
  auto it1 = op1.begin();
  auto it1_run = op1.begin();
  auto it2 = op2.begin();

  while (it1_run != op1.end() && it2 != op2.end()) {
    if (*it1_run < *it2) {
      *it1 = *it1_run;
      it1++;
      it1_run++;
    } else if (*it1_run > *it2) {
      it2++;
    } else {
      it1_run++;
    }
  }
  while (it1_run != op1.end()) {
    *it1 = *it1_run;
    it1++;
    it1_run++;
  }
  op1.resize(it1 - op1.begin());
}
} // namespace reordering