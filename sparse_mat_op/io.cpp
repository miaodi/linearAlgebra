#include "io.hpp"
#include "matrix_utils.hpp"
#include "sparse_mat_traits.hpp"
#include <Eigen/Sparse>
#include <fast_matrix_market/app/Eigen.hpp>
#include <fast_matrix_market/app/triplet.hpp>

namespace matrix_utils {

template <ResizableCSR CSRMatrixType>
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
  csr_matrix.ResizeAJ(mat.nonZeros());
  csr_matrix.ResizeAV(mat.nonZeros());

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

template <typename ROWTYPE, typename COLTYPE>
void writeSVG(const COLTYPE rows, const COLTYPE cols, const ROWTYPE *ai,
              const COLTYPE *aj, std::ostream &outstream,
              const COLTYPE max_display_size) {
  
  // Determine if downsampling is needed
  const bool needs_sampling = (rows > max_display_size || cols > max_display_size);
  const COLTYPE display_rows = needs_sampling ? max_display_size : rows;
  const COLTYPE display_cols = needs_sampling ? max_display_size : cols;
  
  outstream << "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" "
            << "viewBox=\"0 0 " << display_cols + 2 << " " << display_rows + 2 << " \">\n"
            << "<style type=\"text/css\" >\n"
            << "     <![CDATA[\n"
            << "      rect.pixel {\n"
            << "          fill:   #ff0000;\n"
            << "      }\n"
            << "    ]]>\n"
            << "  </style>\n\n"
            << "   <rect width=\"" << display_cols + 2 << "\" height=\"" << display_rows + 2
            << "\" fill=\"rgb(128, 128, 128)\"/>\n"
            << "   <rect x=\"1\" y=\"1\" width=\"" << display_cols + 0.1
            << "\" height=\"" << display_rows + 0.1
            << "\" fill=\"rgb(255, 255, 255)\"/>\n\n";

  const ROWTYPE base = ai[0];
  
  if (needs_sampling) {
    // Downsampling: use bitmap to track which display pixels have non-zeros
    const double scale_row = static_cast<double>(rows) / display_rows;
    const double scale_col = static_cast<double>(cols) / display_cols;
    
    // Use vector<bool> for space efficiency (1 bit per pixel)
    std::vector<bool> bitmap(display_rows * display_cols, false);
    
    // Map each matrix non-zero to its corresponding display pixel
    for (COLTYPE i = 0; i < rows; i++) {
      const COLTYPE display_i = static_cast<COLTYPE>(i / scale_row);
      const ROWTYPE row_start = ai[i] - base;
      const ROWTYPE row_end = ai[i + 1] - base;
      
      for (ROWTYPE j = row_start; j < row_end; j++) {
        const COLTYPE col = aj[j] - base;
        const COLTYPE display_j = static_cast<COLTYPE>(col / scale_col);
        bitmap[display_i * display_cols + display_j] = true;
      }
    }
    
    // Write only the marked display pixels
    for (COLTYPE i = 0; i < display_rows; i++) {
      for (COLTYPE j = 0; j < display_cols; j++) {
        if (bitmap[i * display_cols + j]) {
          outstream << "  <rect class=\"pixel\" x=\"" << j + 1
                    << "\" y=\"" << i + 1 << "\" width=\".9\" height=\".9\"/>\n";
        }
      }
    }
  } else {
    // No downsampling: write each non-zero directly
    for (COLTYPE i = 0; i < rows; i++) {
      const ROWTYPE row_start = ai[i] - base;
      const ROWTYPE row_end = ai[i + 1] - base;
      for (ROWTYPE j = row_start; j < row_end; j++) {
        outstream << "  <rect class=\"pixel\" x=\"" << aj[j] - base + 1
                  << "\" y=\"" << i + 1 << "\" width=\".9\" height=\".9\"/>\n";
      }
    }
  }
  
  outstream << "</svg>\n";
}

template <typename T>
void readMatrixMarketVec(std::istream &instream, std::vector<T> &vec,
                         const fast_matrix_market::read_options &options) {
  fast_matrix_market::matrix_market_header header;
  fast_matrix_market::read_matrix_market_array(instream, header, vec, 
                                               fast_matrix_market::row_major, options);
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
      const COLTYPE *aj, std::ostream &outstream, const COLTYPE max_display_size);
#define INSTANTIATE_READMATRIXMARKETVEC(T)                                     \
  template void readMatrixMarketVec<T>(                                        \
      std::istream & instream, std::vector<T> & vec,                           \
      const fast_matrix_market::read_options &options);

using CSRMatrixTypeDouble = matrix_utils::CSRMatrix<int, int, double>;
INSTANTIATE_READMATRIXMARKET(CSRMatrixTypeDouble);
using CSRMatrixTypeFloat = matrix_utils::CSRMatrix<int, int, float>;
INSTANTIATE_READMATRIXMARKET(CSRMatrixTypeFloat);
using CSRMatrixTypeInt = matrix_utils::CSRMatrix<int, int, int>;
INSTANTIATE_READMATRIXMARKET(CSRMatrixTypeInt);

// Add instantiations for CSRMatrixVec types
using CSRMatrixVecTypeDouble = matrix_utils::CSRMatrixVec<int, int, double>;
INSTANTIATE_READMATRIXMARKET(CSRMatrixVecTypeDouble);
using CSRMatrixVecTypeFloat = matrix_utils::CSRMatrixVec<int, int, float>;
INSTANTIATE_READMATRIXMARKET(CSRMatrixVecTypeFloat);
using CSRMatrixVecTypeInt = matrix_utils::CSRMatrixVec<int, int, int>;
INSTANTIATE_READMATRIXMARKET(CSRMatrixVecTypeInt);

INSTANTIATE_WRITEMATRIXMARKET(int, int, double);
INSTANTIATE_WRITEMATRIXMARKET(int, int, float);
INSTANTIATE_WRITEMATRIXMARKET(int, int, int);

INSTANTIATE_WRITESVG(int, int);

INSTANTIATE_READMATRIXMARKETVEC(double);
INSTANTIATE_READMATRIXMARKETVEC(float);
INSTANTIATE_READMATRIXMARKETVEC(int);
} // namespace matrix_utils