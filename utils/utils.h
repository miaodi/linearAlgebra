#pragma once
#include "config.h"
#include "utils_core.hpp"
#include <atomic>
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
#include <queue>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace Eigen {
template <typename T, typename ind> class Triplet;
}
namespace utils {
std::pair<int32_t, int32_t>
ReadFromBinaryEigen(const std::string &filename,
                    std::vector<Eigen::Triplet<double, int32_t>> &coo);

template <typename IVEC, typename VVEC, typename IB>
auto ReadFromBinaryCOO(const std::string &filename, IVEC &rows, IVEC &cols,
                       VVEC &vals, const IB base) {
  using index_type = typename IVEC::value_type;
  std::ifstream file(filename, std::ios::binary);
  int64_t size;
  index_type m = 0, n = 0;
  std::tuple<int32_t, int32_t, double> tmp;
  file.read(reinterpret_cast<char *>(&size), sizeof size);
  rows.resize(size);
  cols.resize(size);
  vals.resize(size);
  for (int64_t i = 0; i < size; i++) {
    file.read(reinterpret_cast<char *>(&tmp), sizeof tmp);
    rows[i] = std::get<0>(tmp);
    cols[i] = std::get<1>(tmp);
    vals[i] = std::get<2>(tmp);
    m = std::max(m, rows[i]);
    n = std::max(n, cols[i]);
  }
  return std::make_pair(m + (1 - base), n + (1 - base));
}

void ReadFromBinaryVec(const std::string &filename, std::vector<double> &vec);

template <typename IVEC, typename VVEC, typename IB>
auto ReadFromBinaryCSR(const std::string &filename, IVEC &ai, IVEC &aj,
                       VVEC &av, const IB base) {
  // using index_type = typename IVEC::value_type;

  IVEC rows;
  auto &cols = aj;
  auto &vals = av;
  auto res = ReadFromBinaryCOO(filename, rows, cols, vals, base);
  ai = IVEC(res.first + 1, 0);

  size_t nnz = cols.size();
  std::vector<size_t> index(nnz);
  for (size_t i = 0; i < index.size(); i++) {
    index[i] = i;
  }
  std::sort(index.begin(), index.end(), [&rows, &cols](size_t a, size_t b) {
    if (rows[a] == rows[b])
      return cols[a] < cols[b];
    return rows[a] < rows[b];
  });
  for (size_t i = 0; i != nnz; i++) {
    size_t current = i;
    while (i != index[current]) {
      size_t next = index[current];
      std::swap(rows[current], rows[next]);
      std::swap(cols[current], cols[next]);
      std::swap(vals[current], vals[next]);
      index[current] = current;
      current = next;
    }
    index[current] = current;
  }

  for (size_t i = 0; i < nnz; i++) {
    ai[rows[i] + (1 - base)]++;
  }
  ai[0] += base;
  for (auto i = 0; i < res.first; i++) {
    ai[i + 1] += ai[i];
  }
  return res;
}

template <typename IVEC, typename VVEC>
void read_matrix_market_csr(
    std::istream &instream, IVEC &rows, IVEC &cols, VVEC &values,
    const fast_matrix_market::read_options &options = {}) {
  fast_matrix_market::matrix_market_header header;
  IVEC coo_rows;
  fast_matrix_market::read_matrix_market_triplet(instream, header, coo_rows,
                                                 cols, values, options);
  rows = IVEC(header.nrows + 1, 0);
  typename IVEC::value_type nnz = cols.size();
  IVEC index(nnz);
  for (typename IVEC::value_type i = 0;
       i < (typename IVEC::value_type)index.size(); i++) {
    index[i] = i;
  }
  std::sort(index.begin(), index.end(),
            [&coo_rows, &cols](typename IVEC::value_type a,
                               typename IVEC::value_type b) {
              if (coo_rows[a] == coo_rows[b])
                return cols[a] < cols[b];
              return coo_rows[a] < coo_rows[b];
            });
  for (typename IVEC::value_type i = 0; i != nnz; i++) {
    typename IVEC::value_type current = i;
    while (i != index[current]) {
      typename IVEC::value_type next = index[current];
      std::swap(coo_rows[current], coo_rows[next]);
      std::swap(cols[current], cols[next]);
      std::swap(values[current], values[next]);
      index[current] = current;
      current = next;
    }
    index[current] = current;
  }
  for (typename IVEC::value_type i = 0; i < nnz; i++) {
    rows[coo_rows[i] + 1]++;
  }
  for (typename IVEC::value_type i = 0; i < header.nrows; i++) {
    rows[i + 1] += rows[i];
  }
}

void printProgress(double percentage);

template <typename COLTYPE>
std::vector<COLTYPE> randomPermute(const COLTYPE n, const COLTYPE base = 0);

template <typename COLTYPE>
void inversePermute(std::vector<COLTYPE> &iperm,
                    const std::vector<COLTYPE> &perm, const COLTYPE base = 0);

template <typename T, typename C> class MaxHeap {
public:
  MaxHeap(C c) : _comp(c) {}

  // return true if the Max Heap is empty, true otherwise.
  bool empty() { return _heap.empty(); }

  // used to insert an item in the priority queue.
  void push(const T &obj) {
    _heap.push_back(obj);
    heapifyUp(_heap.size() - 1);
  }

  // deletes the highest priority item currently in the queue.
  void pop() {
    if (!empty()) {
      std::swap(_heap[0], _heap[static_cast<int>(_heap.size()) - 1]);
      _heap.pop_back();
      if (!empty())
        heapifyDown(0);
    }
  }

  int size() const { return static_cast<int>(_heap.size()); }

  void clear() { _heap.clear(); }

  // return the highest priority item currently in the queue.
  T *top() {
    if (!empty()) {
      return &_heap[0];
    }
    return nullptr;
  }

  std::vector<T> &getHeap() { return _heap; }

  void setComp(C c) { _comp = c; }

protected:
  void heapifyUp(int idx) {
    int parentIdx = parent(idx);
    if (parentIdx < 0)
      return;
    if (_comp(_heap[parentIdx], _heap[idx])) {
      std::swap(_heap[parentIdx], _heap[idx]);
      heapifyUp(parentIdx);
    }
  }

  void heapifyDown(int idx) {
    int largeIdx = idx;
    int leftChildIdx = leftChild(idx), rightChildIdx = rightChild(idx);
    if (leftChildIdx < static_cast<int>(_heap.size())) {
      if (_comp(_heap[largeIdx], _heap[leftChildIdx]))
        largeIdx = leftChildIdx;
    }
    if (rightChildIdx < static_cast<int>(_heap.size())) {
      if (_comp(_heap[largeIdx], _heap[rightChildIdx]))
        largeIdx = rightChildIdx;
    }

    if (largeIdx != idx) {
      std::swap(_heap[largeIdx], _heap[idx]);
      heapifyDown(largeIdx);
    }
  }

  int leftChild(int i) { return 2 * i + 1; }

  int rightChild(int i) { return 2 * i + 2; }

  int parent(int i) { return (i - 1) / 2; }

  std::vector<T> _heap;
  C _comp;
};

/// @brief Compute parallel prefix sum (cumulative sum) with a specified base value
/// @tparam InputIt Input iterator type
/// @tparam OutputIt Output iterator type
/// @param nthreads Number of threads to use for parallel computation
/// @param first Iterator to the beginning of the input range
/// @param last Iterator to the end of the input range
/// @param d_first Iterator to the beginning of the output range (base value stored at *d_first)
/// @return Iterator to one past the last element written (d_first + (last - first) + 1)
///
/// @details This function computes a parallel prefix sum where:
/// - The base value is read from *d_first (typically 0 or 1 for 0-based/1-based indexing)
/// - Output is written starting at d_first[1], with d_first[i+1] = base + sum(input[0..i])
/// - The output array must have size >= (last - first) + 1 to accommodate the base value
///
/// The algorithm uses a three-phase approach:
/// 1. Each thread computes local prefix sums for its partition
/// 2. Thread-local sums are combined using std::partial_sum to compute global offsets
/// 3. Each thread adds its offset to its local results
///
/// @par Example:
/// @code
/// std::vector<int> input = {1, 2, 3, 4, 5};
/// std::vector<int> output(input.size() + 1);
/// output[0] = 0;  // Base value
/// ParallelPrefixSum(4, input.begin(), input.end(), output.begin());
/// // Result: output = {0, 1, 3, 6, 10, 15}
/// @endcode
///
/// @note Thread-safe and uses OpenMP for parallelization
/// @note The base value at d_first[0] is preserved and used as the starting point
template <class InputIt, class OutputIt>
OutputIt ParallelPrefixSum(const int nthreads, InputIt first, InputIt last, OutputIt d_first);

/// @brief In-place inclusive prefix sum (a[i] becomes sum_{k<=i} a[k])
/// @tparam Iter Random access iterator or pointer
/// @param nthreads Number of threads to use
/// @param first Iterator to the beginning of the range (inclusive)
/// @param last Iterator to the end of the range (exclusive)
/// @return Iterator to one past the last element written (same as last)
///
/// @note The operation is in-place; the original values are overwritten by the prefix sums.
template <class Iter>
Iter ParallelPrefixSumInplace(const int nthreads, Iter first, Iter last);

#ifdef USE_BOOST_LIB
template <typename COLTYPE>
void printEliminationTree(const COLTYPE size, const COLTYPE base,
                          COLTYPE *const parent, const std::string &filename);

/// @brief Write adjacency graph to DOT format for GraphViz visualization (original version)
/// @tparam ROWTYPE Row pointer type (typically int or int64_t)
/// @tparam COLTYPE Column index type (typically int or int64_t)
/// @param rows Number of nodes in the graph
/// @param ai Row pointers array (ai[0] contains the base indexing)
/// @param aj Column indices array
/// @param filename Output DOT file path
/// @param title Graph title for the DOT file (default: "Graph")
template <typename ROWTYPE, typename COLTYPE>
void writeAdjacencyGraphDOT(const COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj, 
                           const std::string& filename, const std::string& title = "Graph");

/// @brief Write adjacency graph with node partitioning to DOT format for GraphViz visualization
/// @tparam ROWTYPE Row pointer type (typically int or int64_t)
/// @tparam COLTYPE Column index type (typically int or int64_t)
/// @param rows Number of nodes in the graph
/// @param ai Row pointers array (ai[0] contains the base indexing)
/// @param aj Column indices array
/// @param partition Array where partition[i] gives the partition ID of node i (nullptr for single partition)
/// @param num_partitions Number of partitions
/// @param filename Output DOT file path
/// @param title Graph title for the DOT file (default: "Partitioned Graph")
template <typename ROWTYPE, typename COLTYPE>
void writeAdjacencyGraphDOT(const COLTYPE rows, 
                           ROWTYPE const* ai, 
                           COLTYPE const* aj,
                           COLTYPE const* partition,
                           COLTYPE num_partitions,
                           const std::string& filename, 
                           const std::string& title = "Partitioned Graph");
#endif

// TODO: remove
template <typename T>
class CacheFriendlyVectors : public std::vector<std::vector<T>> {
public:
  CacheFriendlyVectors(const size_t size) : std::vector<std::vector<T>>(size) {}

  void push_back(const size_t to, const T &val) {
    if ((*this)[to].capacity() == 0 && _availableInd < _at) {
      std::swap((*this)[to], (*this)[_availableInd++]);
    }
    (*this)[to].push_back(val);
    _modifiedInd = std::max(to, _modifiedInd);
  }

  void to_next() { (*this)[_at++].clear(); }

  void clear() {
    size_t r = 0;
    for (size_t rr = _availableInd; rr <= _modifiedInd && r < rr; rr++) {
      (*this)[rr].clear();
      if ((*this)[rr].capacity()) {
        std::swap((*this)[rr], (*this)[r++]);
      }
    }
    _availableInd = 0;
    _modifiedInd = 0;
    _at = 0;
  }

protected:
  size_t _availableInd{0};
  size_t _modifiedInd{0};
  size_t _at{0};
};

} // namespace utils
