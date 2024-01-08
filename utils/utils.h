#pragma once
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <fstream>
#include <iostream>
#include <mkl.h>
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

template <typename IVEC, typename VVEC, typename IB>
auto ReadFromBinaryCSR(const std::string &filename, IVEC &ai, IVEC &aj,
                       VVEC &av, const IB base) {
  using index_type = typename IVEC::value_type;

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
  for (size_t i = 0; i < res.first; i++) {
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
  for (typename IVEC::value_type i = 0; i < index.size(); i++) {
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
} // namespace utils