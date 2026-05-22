#pragma once
#include "sparse_mat_traits.hpp"
#include <fast_matrix_market/types.hpp>
#include <istream>
#include <ostream>
#include <vector>
namespace matrix_utils {
template <typename CSRMatrixType>
void readMatrixMarket(std::istream &instream, CSRMatrixType &csr_matrix,
                      const fast_matrix_market::read_options &options = {});

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void readMatrixMarket(std::istream &instream, std::vector<ROWTYPE> &ai,
                      std::vector<COLTYPE> &aj, std::vector<VALTYPE> &av,
                      const fast_matrix_market::read_options &options = {});

template <typename T>
void readMatrixMarketVec(std::istream &instream, std::vector<T> &vec,
                         const fast_matrix_market::read_options &options = {});

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void writeMatrixMarket(const COLTYPE rows, const COLTYPE cols,
                       const ROWTYPE *ai, const COLTYPE *aj, const VALTYPE *av,
                       std::ostream &outstream,
                       const fast_matrix_market::write_options &options = {});

template <typename CSRMatrixType>
void writeMatrixMarket(const CSRMatrixType &csr_matrix, std::ostream &outstream,
                       const fast_matrix_market::write_options &options = {}) {
  writeMatrixMarket(csr_matrix.rows, csr_matrix.cols, csr_matrix.AI(),
                    csr_matrix.AJ(), csr_matrix.AV(), outstream, options);
}

template <typename ROWTYPE, typename COLTYPE>
void writeSVG(const COLTYPE rows, const COLTYPE cols, const ROWTYPE *ai,
              const COLTYPE *aj, std::ostream &outstream,
              const COLTYPE max_display_size = 2000);
} // namespace matrix_utils
