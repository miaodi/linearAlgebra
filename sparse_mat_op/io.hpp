#pragma once
#include "matrix_utils.hpp"

namespace matrix_utils {
    
template <typename CSRMatrixType>
void read_matrix_market(std::istream &instream, 
                        CSRMatrixType &csr_matrix,
                        const fast_matrix_market::read_options &options = {}) {
  using ROWTYPE = typename CSRMatrixType::ROWTYPE;
  using COLTYPE = typename CSRMatrixType::COLTYPE;
  using VALTYPE = typename CSRMatrixType::VALTYPE;

  fast_matrix_market::matrix_market_header header;
  std::vector<ROWTYPE> rows;
  std::vector<COLTYPE> cols;
  std::vector<VALTYPE> values;

  fast_matrix_market::read_matrix_market_triplet(instream, header, rows, cols,
                                                 values, options);

  csr_matrix.rows = header.nrows;
  csr_matrix.cols = header.ncols;
  csr_matrix.Base() = rows[0];
  
  ResizeCSRAI(csr_matrix, csr_matrix.rows + 1);
  ResizeCSRAJ(csr_matrix, values.size());
  ResizeCSRAV(csr_matrix, values.size());

  read_matrix_market_csr(rows, cols, values, csr_matrix.ai, csr_matrix.aj,
                         csr_matrix.av);
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
}