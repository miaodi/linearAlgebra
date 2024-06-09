#pragma once
#include <atomic>
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <fstream>
#include <iostream>
#include <mkl.h>
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

template <typename Numeric, typename Generator = std::mt19937>
Numeric random(Numeric from, Numeric to) {
  thread_local static Generator gen(std::random_device{}());

  using dist_type =
      typename std::conditional<std::is_integral<Numeric>::value,
                                std::uniform_int_distribution<Numeric>,
                                std::uniform_real_distribution<Numeric>>::type;

  thread_local static dist_type dist;

  return dist(gen, typename dist_type::param_type{from, to});
}

template <typename T> class singleton {
private:
  singleton(); // Disallow instantiation outside of the class.
public:
  singleton(const singleton &) = delete;
  singleton &operator=(const singleton &) = delete;
  singleton(singleton &&) = delete;
  singleton &operator=(singleton &&) = delete;

  static T &instance() {
    static T inst;
    return inst;
  }
};

template <typename Iter>
std::pair<Iter, Iter> LoadBalancedPartition(Iter begin, Iter end, int tid,
                                            int nthreads) {
  const int total_work = std::distance(begin, end);
  const int work_per_thread = total_work / nthreads;
  const int resid = total_work % nthreads;
  return tid >= resid
             ? std::make_pair(begin + tid * work_per_thread + resid,
                              begin + (tid + 1) * work_per_thread + resid)
             : std::make_pair(begin + tid * (work_per_thread + 1),
                              begin + (tid + 1) * (work_per_thread + 1));
}

template <typename Iter>
std::pair<Iter, Iter> LoadPrefixBalancedPartition(Iter begin, Iter end, int tid,
                                                  int nthreads) {
  const int total_work = *end - *begin;
  const int work_per_thread = total_work / nthreads;
  const int resid = total_work % nthreads;
  Iter lb =
      tid == 0
          ? begin
          : std::lower_bound(begin, end,
                             (tid >= resid ? (tid * work_per_thread + resid)
                                           : (tid * (work_per_thread + 1))) +
                                 *begin);
  Iter le = tid == nthreads - 1
                ? end
                : std::lower_bound(begin, end,
                                   (tid >= resid
                                        ? ((tid + 1) * work_per_thread + resid)
                                        : ((tid + 1) * (work_per_thread + 1))) +
                                       *begin);
  return std::make_pair(lb, le);
}

void printProgress(double percentage);

// Programming Pearls column 11.2
// Knuth's algorithm S (3.4.2)
// output M integers (in order) in range [start, end)
// https://stackoverflow.com/questions/33081856/randomly-generated-sorted-arrays-search-performances-comparison
class knuth_s {
public:
  knuth_s() : eng(rd()) {}
  template <typename T, typename Iter>
  void operator()(T M, T start, T end, Iter dest) const {
    double select = M, remaining = end - start;
    for (T i = start; i < end; ++i) {
      if (dist(eng) < select / remaining) {
        *dest++ = i;
        --select;
      }
      --remaining;
    }
  }

protected:
  mutable std::random_device rd;
  mutable std::mt19937 eng;
  mutable std::uniform_real_distribution<> dist; // [0,1)
};

std::vector<MKL_INT> randomPermute(const MKL_INT n, const MKL_INT base = 0);

std::vector<MKL_INT> inversePermute(const std::vector<MKL_INT> &perm,
                                    const MKL_INT base = 0);

bool isPermutation(const std::vector<MKL_INT> &perm, const MKL_INT base = 0);

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
  T top() {
    if (!empty()) {
      return _heap[0];
    }
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