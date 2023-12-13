#pragma once
#include <Eigen/Sparse>
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace utils {
std::pair<int32_t, int32_t>
ReadFromBinary(const std::string &filename,
               std::vector<Eigen::Triplet<double, int32_t>> &coo);

template <typename IVEC, typename VVEC>
void read_matrix_market_csr(
    std::istream &instream, IVEC &rows, IVEC &cols, VVEC &values,
    const fast_matrix_market::read_options &options = {}) {
  fast_matrix_market::matrix_market_header header;
  IVEC coo_rows;
  fast_matrix_market::read_matrix_market_triplet(instream, header, coo_rows,
                                                 cols, values, options);
  rows = IVEC(header.nrows + 1, 0);
  IVEC index(header.nnz);
  //   if (header.nnz == 0)
  //     return;
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
  //   typename IVEC::value_type tmp_i, tmp_j, tmp_ind;
  //   typename VVEC::value_type tmp_v;
  //   bool own{false};
  for (typename IVEC::value_type i = 0; i != header.nnz;) {
    typename IVEC::value_type cur = index[i];
    if (cur == i) {
      i++;
    } else {
      std::swap(coo_rows[i], coo_rows[cur]);
      std::swap(cols[i], cols[cur]);
      std::swap(values[i], values[cur]);
      std::swap(index[i], index[cur]);
    }
  }

  for (typename IVEC::value_type i = 0; i < header.nnz; i++) {
    rows[coo_rows[i] + 1]++;
  }
  for (typename IVEC::value_type i = 0; i < header.nrows; i++) {
    rows[i + 1] += rows[i];
  }
}
} // namespace utils