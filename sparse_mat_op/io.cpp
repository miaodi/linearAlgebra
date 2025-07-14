#include "io.hpp"
#include "matrix_utils.hpp"
#include "sparse_mat_traits.hpp"
#include <Eigen/Sparse>
#include <fast_matrix_market/app/Eigen.hpp>
#include <fast_matrix_market/app/triplet.hpp>

namespace matrix_utils {

template <typename CSRMatrixType>
void readMatrixMarket(std::istream &instream, CSRMatrixType &csr_matrix,
                      const fast_matrix_market::read_options &options) {
  using ROWTYPE = typename CSRMatrixType::ROWTYPE;
  using COLTYPE = typename CSRMatrixType::COLTYPE;
  using VALTYPE = typename CSRMatrixType::VALTYPE;

  static_assert(CSRResizable<CSRMatrixType>::value,
                "CSRMatrixType must have a resizable method");

  Eigen::SparseMatrix<VALTYPE, Eigen::RowMajor, COLTYPE> mat;
  fast_matrix_market::read_matrix_market_eigen(instream, mat, options);
  csr_matrix.rows = mat.rows();
  csr_matrix.cols = mat.cols();

  ResizeCSRAI(csr_matrix, mat.rows() + 1);
  ResizeCSRAJ(csr_matrix, mat.nonZeros());
  ResizeCSRAV(csr_matrix, mat.nonZeros());

  std::copy(mat.outerIndexPtr(), mat.outerIndexPtr() + mat.rows() + 1,
            csr_matrix.AI());
  std::copy(mat.innerIndexPtr(), mat.innerIndexPtr() + mat.nonZeros(),
            csr_matrix.AJ());
  std::copy(mat.valuePtr(), mat.valuePtr() + mat.nonZeros(), csr_matrix.AV());
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void writeMatrixMarket(const COLTYPE rows, const COLTYPE cols,
                       const ROWTYPE *ai, const COLTYPE *aj, const VALTYPE *av,
                       std::ostream &outstream,
                       const fast_matrix_market::write_options &options) {
  fast_matrix_market::matrix_market_header header;
  const auto base = ai[0];
  const auto nnz = ai[rows] - base;

  header.nrows = rows;
  header.ncols = cols;
  header.nnz = nnz;
  header.object = fast_matrix_market::matrix;
  header.format = fast_matrix_market::coordinate;

  if (nnz > 0 && (av == nullptr)) {
    header.field = fast_matrix_market::pattern;
  } else if (header.field != fast_matrix_market::pattern &&
             options.fill_header_field_type) {
    header.field = fast_matrix_market::get_field_type((const VALTYPE *)nullptr);
  }
  fast_matrix_market::write_header(outstream, header, options);
  fast_matrix_market::line_formatter<COLTYPE, VALTYPE> lf(header, options);
  auto formatter = fast_matrix_market::csc_formatter(
      lf, ai, ai + rows, aj, aj + nnz, av,
      header.field == fast_matrix_market::pattern ? av : av + nnz, true);
  fast_matrix_market::write_body(outstream, formatter, options);
}

#define INSTANTIATE_READMATRIXMARKET(CSRMatrixType)                            \
  template void readMatrixMarket<CSRMatrixType>(                               \
      std::istream & instream, CSRMatrixType & csr_matrix,                     \
      const fast_matrix_market::read_options &options);
#define INSTANTIATE_WRITEMATRIXMARKET(ROWTYPE, COLTYPE, VALTYPE)                            \
  template void writeMatrixMarket<ROWTYPE, COLTYPE, VALTYPE>(                              \
      const COLTYPE rows, const COLTYPE cols,                                   \
      const ROWTYPE *ai, const COLTYPE *aj, const VALTYPE *av,                 \
      std::ostream &outstream, const fast_matrix_market::write_options &options);

using CSRMatrixTypeDouble = matrix_utils::CSRMatrix<int, int, double>;
INSTANTIATE_READMATRIXMARKET(CSRMatrixTypeDouble);
using CSRMatrixTypeFloat = matrix_utils::CSRMatrix<int, int, float>;
INSTANTIATE_READMATRIXMARKET(CSRMatrixTypeFloat);
using CSRMatrixTypeInt = matrix_utils::CSRMatrix<int, int, int>;
INSTANTIATE_READMATRIXMARKET(CSRMatrixTypeInt);

INSTANTIATE_WRITEMATRIXMARKET(int, int, double);
INSTANTIATE_WRITEMATRIXMARKET(int, int, float);
INSTANTIATE_WRITEMATRIXMARKET(int, int, int);

} // namespace matrix_utils