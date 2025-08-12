#include "io.hpp"
#include "matrix_utils.hpp"
#include "sparse_mat_traits.hpp"
#include <Eigen/Sparse>
#include <algorithm>
#include <fast_matrix_market/app/Eigen.hpp>
#include <fast_matrix_market/app/triplet.hpp>

namespace matrix_utils {

template <ResizableDiagonalType CSRMatrixType>
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

  csr_matrix.ResizeAI(mat.rows() + 1);
  csr_matrix.ResizeAJ(mat.nonZeros() +
                      mat.rows()); // in case diagonal is not provided
  csr_matrix.ResizeAV(mat.nonZeros() +
                      mat.rows()); // in case diagonal is not provided
  csr_matrix.ResizeDiagonal(mat.rows());

  const ROWTYPE base = mat.innerIndexPtr()[0];
  assert(base == 0);

  auto *ai = csr_matrix.AI();
  auto *aj = csr_matrix.AJ();
  auto *av = csr_matrix.AV();
  auto *diagonal = csr_matrix.Diagonal();
  ai[0] = 0;
  ROWTYPE row_nnz;
  for (COLTYPE i = 0; i < mat.rows(); ++i) {
    bool has_diag = false;
    row_nnz = ai[i];
    assert(std::is_sorted(mat.innerIndexPtr() + mat.outerIndexPtr()[i],
                          mat.innerIndexPtr() + mat.outerIndexPtr()[i + 1],
                          std::less<COLTYPE>()));
    for (COLTYPE j = mat.outerIndexPtr()[i]; j < mat.outerIndexPtr()[i + 1];
         ++j) {
      if (mat.innerIndexPtr()[j] == i) {
        diagonal[i] = row_nnz;
        has_diag = true;
      } else if (!has_diag && mat.innerIndexPtr()[j] > i) {
        // insert a zero diagonal entry
        aj[row_nnz] = i;
        av[row_nnz] = VALTYPE(0);
        diagonal[i] = row_nnz++;
        has_diag = true;
      }
      aj[row_nnz] = mat.innerIndexPtr()[j];
      av[row_nnz] = mat.valuePtr()[j];
      row_nnz++;
    }
    if (!has_diag) {
      // insert a zero diagonal entry
      aj[row_nnz] = i;
      av[row_nnz] = VALTYPE(0);
      diagonal[i] = row_nnz++;
    }
    ai[i + 1] = row_nnz;
  }
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

template <typename ROWTYPE, typename COLTYPE>
void writeSVG(const COLTYPE rows, const COLTYPE cols, const ROWTYPE *ai,
              const COLTYPE *aj, std::ostream &outstream) {
  outstream << "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" "
            << "viewBox=\"0 0 " << cols + 2 << " " << rows + 2 << " \">\n"
            << "<style type=\"text/css\" >\n"
            << "     <![CDATA[\n"
            << "      rect.pixel {\n"
            << "          fill:   #ff0000;\n"
            << "      }\n"
            << "    ]]>\n"
            << "  </style>\n\n"
            << "   <rect width=\"" << cols + 2 << "\" height=\"" << rows + 2
            << "\" fill=\"rgb(128, 128, 128)\"/>\n"
            << "   <rect x=\"1\" y=\"1\" width=\"" << cols + 0.1
            << "\" height=\"" << rows + 0.1
            << "\" fill=\"rgb(255, 255, 255)\"/>\n\n";

  const ROWTYPE base = ai[0];
  for (COLTYPE i = 0; i < rows; i++) {
    ROWTYPE previousStart = ai[i] - base;
    ROWTYPE end = ai[i + 1] - base;
    for (ROWTYPE j = previousStart; j < end; j++) {
      outstream << "  <rect class=\"pixel\" x=\"" << aj[j] - base + 1
                << "\" y=\"" << i + 1 << "\" width=\".9\" height=\".9\"/>\n";
    }
  }
  outstream << "</svg>\n";
}

#define INSTANTIATE_READMATRIXMARKET(CSRMatrixType)                            \
  template void readMatrixMarket<CSRMatrixType>(                               \
      std::istream & instream, CSRMatrixType & csr_matrix,                     \
      const fast_matrix_market::read_options &options);
#define INSTANTIATE_WRITEMATRIXMARKET(ROWTYPE, COLTYPE, VALTYPE)               \
  template void writeMatrixMarket<ROWTYPE, COLTYPE, VALTYPE>(                  \
      const COLTYPE rows, const COLTYPE cols, const ROWTYPE *ai,               \
      const COLTYPE *aj, const VALTYPE *av, std::ostream &outstream,           \
      const fast_matrix_market::write_options &options);
#define INSTANTIATE_WRITESVG(ROWTYPE, COLTYPE)                                 \
  template void writeSVG<ROWTYPE, COLTYPE>(                                    \
      const COLTYPE rows, const COLTYPE cols, const ROWTYPE *ai,               \
      const COLTYPE *aj, std::ostream &outstream);

using CSRMatrixTypeDouble = matrix_utils::CSRMatrix<int, int, double>;
INSTANTIATE_READMATRIXMARKET(CSRMatrixTypeDouble);
using CSRMatrixTypeFloat = matrix_utils::CSRMatrix<int, int, float>;
INSTANTIATE_READMATRIXMARKET(CSRMatrixTypeFloat);
using CSRMatrixTypeInt = matrix_utils::CSRMatrix<int, int, int>;
INSTANTIATE_READMATRIXMARKET(CSRMatrixTypeInt);

INSTANTIATE_WRITEMATRIXMARKET(int, int, double);
INSTANTIATE_WRITEMATRIXMARKET(int, int, float);
INSTANTIATE_WRITEMATRIXMARKET(int, int, int);

INSTANTIATE_WRITESVG(int, int);
} // namespace matrix_utils