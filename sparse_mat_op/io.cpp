#include "io.hpp"
#include "matrix_utils.hpp"
#include "sparse_mat_traits.hpp"
#include <Eigen/Sparse>
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <fast_matrix_market/app/Eigen.hpp>
#include <fast_matrix_market/app/triplet.hpp>
#include <stb_image_write.h>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace matrix_utils {

namespace {
template <typename T>
void readBinaryValue(std::istream &stream, T &value, const std::string &filename) {
  if (!stream.read(reinterpret_cast<char *>(&value), sizeof value)) {
    throw std::runtime_error("Failed to read binary matrix file: " + filename);
  }
}

template <typename T>
void writeBinaryValue(std::ostream &stream, const T &value,
                      const std::string &filename) {
  if (!stream.write(reinterpret_cast<const char *>(&value), sizeof value)) {
    throw std::runtime_error("Failed to write binary matrix file: " + filename);
  }
}

template <typename T>
std::int32_t toBinaryIndex(const T value) {
  if (value < static_cast<T>(std::numeric_limits<std::int32_t>::min()) ||
      value > static_cast<T>(std::numeric_limits<std::int32_t>::max())) {
    throw std::overflow_error("Matrix index does not fit binary matrix format");
  }
  return static_cast<std::int32_t>(value);
}

MatrixDataType dataTypeFromFilename(const std::string &filename) {
  if (filename.size() >= 4 &&
      filename.compare(filename.size() - 4, 4, ".mtx") == 0) {
    return MatrixDataType::MatrixMarket;
  }
  if (filename.size() >= 4 &&
      filename.compare(filename.size() - 4, 4, ".bin") == 0) {
    return MatrixDataType::Binary;
  }
  throw std::invalid_argument(
      "Unsupported matrix file extension, expected .mtx or .bin: " + filename);
}

template <ResizableCSR CSRMatrixType>
void readBinaryMatrix(const std::string &filename, CSRMatrixType &csr_matrix) {
  using ROWTYPE = typename CSRMatrixType::ROWTYPE;
  using COLTYPE = typename CSRMatrixType::COLTYPE;
  using VALTYPE = typename CSRMatrixType::VALTYPE;

  std::ifstream file(filename, std::ios::in | std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open binary matrix file: " + filename);
  }

  std::int64_t nnz64 = 0;
  readBinaryValue(file, nnz64, filename);
  if (nnz64 < 0 || static_cast<std::uint64_t>(nnz64) >
                       static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error("Invalid binary matrix nonzero count: " + filename);
  }

  const auto nnz = static_cast<std::size_t>(nnz64);
  std::vector<COLTYPE> rows(nnz);
  std::vector<COLTYPE> cols(nnz);
  std::vector<VALTYPE> vals(nnz);
  COLTYPE nrows = 0;
  COLTYPE ncols = 0;

  for (std::size_t i = 0; i < nnz; ++i) {
    std::tuple<std::int32_t, std::int32_t, double> entry;
    readBinaryValue(file, entry, filename);

    const auto row = std::get<0>(entry);
    const auto col = std::get<1>(entry);
    if (row <= 0 || col <= 0) {
      throw std::runtime_error("Binary matrix uses invalid one-based index: " + filename);
    }

    rows[i] = static_cast<COLTYPE>(row - 1);
    cols[i] = static_cast<COLTYPE>(col - 1);
    vals[i] = static_cast<VALTYPE>(std::get<2>(entry));
    nrows = std::max(nrows, static_cast<COLTYPE>(row));
    ncols = std::max(ncols, static_cast<COLTYPE>(col));
  }

  std::vector<std::size_t> order(nnz);
  for (std::size_t i = 0; i < nnz; ++i) {
    order[i] = i;
  }
  std::sort(order.begin(), order.end(), [&rows, &cols](const auto lhs, const auto rhs) {
    if (rows[lhs] == rows[rhs]) {
      return cols[lhs] < cols[rhs];
    }
    return rows[lhs] < rows[rhs];
  });

  csr_matrix.rows = nrows;
  csr_matrix.cols = ncols;
  auto *ai = csr_matrix.ResizeAI(static_cast<std::size_t>(nrows) + 1);
  auto *aj = csr_matrix.ResizeAJ(nnz);
  auto *av = csr_matrix.ResizeAV(nnz);
  std::fill(ai, ai + static_cast<std::size_t>(nrows) + 1, ROWTYPE{});

  for (const auto idx : order) {
    ++ai[static_cast<std::size_t>(rows[idx]) + 1];
  }
  for (COLTYPE i = 0; i < nrows; ++i) {
    ai[static_cast<std::size_t>(i) + 1] += ai[i];
  }
  for (std::size_t i = 0; i < nnz; ++i) {
    const auto idx = order[i];
    aj[i] = cols[idx];
    av[i] = vals[idx];
  }
}

template <CSR CSRMatrixType>
void writeBinaryMatrix(const CSRMatrixType &csr_matrix, const std::string &filename) {
  std::ofstream file(filename, std::ios::out | std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open binary matrix file for writing: " + filename);
  }

  const auto rows = csr_matrix.rows;
  const auto base = csr_matrix.Base();
  const auto nnz = csr_matrix.NNZ();
  if (nnz < 0) {
    throw std::runtime_error("Cannot write binary matrix with negative nonzero count");
  }
  const auto nnz64 = static_cast<std::int64_t>(nnz);
  writeBinaryValue(file, nnz64, filename);

  const auto *ai = csr_matrix.AI();
  const auto *aj = csr_matrix.AJ();
  const auto *av = csr_matrix.AV();
  if (nnz > 0 && (!ai || !aj || !av)) {
    throw std::runtime_error("Cannot write binary matrix with null CSR storage");
  }

  using COLTYPE = typename CSRMatrixType::COLTYPE;
  for (COLTYPE row = 0; row < rows; ++row) {
    for (auto pos = ai[row] - base; pos < ai[row + 1] - base; ++pos) {
      const std::tuple<std::int32_t, std::int32_t, double> entry{
          toBinaryIndex(row + 1), toBinaryIndex(aj[pos] - base + 1),
          static_cast<double>(av[pos])};
      writeBinaryValue(file, entry, filename);
    }
  }
}
} // namespace

template <ResizableCSR CSRMatrixType>
void readMatrix(const std::string &filename, CSRMatrixType &csr_matrix,
                const fast_matrix_market::read_options &options) {
  readMatrix(filename, csr_matrix, dataTypeFromFilename(filename), options);
}

template <ResizableCSR CSRMatrixType>
void readMatrix(const std::string &filename, CSRMatrixType &csr_matrix,
                const MatrixDataType data_type,
                const fast_matrix_market::read_options &options) {
  switch (data_type) {
  case MatrixDataType::MatrixMarket: {
    std::ifstream stream(filename);
    if (!stream) {
      throw std::runtime_error("Cannot open MatrixMarket file: " + filename);
    }
    readMatrixMarket(stream, csr_matrix, options);
    return;
  }
  case MatrixDataType::Binary:
    readBinaryMatrix(filename, csr_matrix);
    return;
  }

  throw std::invalid_argument("Unsupported matrix data type");
}

template <CSR CSRMatrixType>
void writeMatrix(const CSRMatrixType &csr_matrix, const std::string &filename,
                 const fast_matrix_market::write_options &options) {
  writeMatrix(csr_matrix, filename, dataTypeFromFilename(filename), options);
}

template <CSR CSRMatrixType>
void writeMatrix(const CSRMatrixType &csr_matrix, const std::string &filename,
                 const MatrixDataType data_type,
                 const fast_matrix_market::write_options &options) {
  switch (data_type) {
  case MatrixDataType::MatrixMarket: {
    std::ofstream stream(filename);
    if (!stream) {
      throw std::runtime_error("Cannot open MatrixMarket file for writing: " + filename);
    }
    writeMatrixMarket(csr_matrix, stream, options);
    return;
  }
  case MatrixDataType::Binary:
    writeBinaryMatrix(csr_matrix, filename);
    return;
  }

  throw std::invalid_argument("Unsupported matrix data type");
}

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
  mat.makeCompressed();
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
void readMatrixMarket(std::istream &instream, std::vector<ROWTYPE> &ai,
                      std::vector<COLTYPE> &aj, std::vector<VALTYPE> &av,
                      const fast_matrix_market::read_options &options) {
  CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE> matrix;
  readMatrixMarket(instream, matrix, options);
  ai = std::move(matrix.ai);
  aj = std::move(matrix.aj);
  av = std::move(matrix.av);
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

template <typename ROWTYPE, typename COLTYPE>
void writePNG(const COLTYPE rows, const COLTYPE cols, const ROWTYPE *ai,
              const COLTYPE *aj, const std::string &filename,
              const COLTYPE max_display_size) {
  if (rows <= 0 || cols <= 0) {
    throw std::invalid_argument("Cannot write PNG for an empty matrix");
  }
  if (max_display_size <= 0) {
    throw std::invalid_argument("PNG max_display_size must be positive");
  }

  const COLTYPE image_rows = std::min(rows, max_display_size);
  const COLTYPE image_cols = std::min(cols, max_display_size);
  const std::size_t pixel_count =
      static_cast<std::size_t>(image_rows) * static_cast<std::size_t>(image_cols);
  std::vector<std::uint32_t> counts(pixel_count, 0);

  const ROWTYPE base = ai[0];
  std::uint32_t max_count = 0;
  for (COLTYPE row = 0; row < rows; ++row) {
    const auto y = static_cast<std::size_t>(row) * image_rows / rows;
    for (ROWTYPE pos = ai[row] - base; pos < ai[row + 1] - base; ++pos) {
      const COLTYPE col = aj[pos] - base;
      if (col < 0 || col >= cols) {
        continue;
      }
      const auto x = static_cast<std::size_t>(col) * image_cols / cols;
      auto &count = counts[y * image_cols + x];
      ++count;
      max_count = std::max(max_count, count);
    }
  }

  std::vector<unsigned char> image(pixel_count, 255);
  if (max_count > 0) {
    const double max_log = std::log1p(static_cast<double>(max_count));
    for (std::size_t i = 0; i < pixel_count; ++i) {
      if (counts[i] == 0) {
        continue;
      }
      const double density = std::log1p(static_cast<double>(counts[i])) / max_log;
      image[i] = static_cast<unsigned char>(255.0 * (1.0 - density));
    }
  }

  const int width = static_cast<int>(image_cols);
  const int height = static_cast<int>(image_rows);
  if (!stbi_write_png(filename.c_str(), width, height, 1, image.data(), width)) {
    throw std::runtime_error("Failed to write PNG file: " + filename);
  }
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
#define INSTANTIATE_READMATRIX(CSRMatrixType)                                  \
  template void readMatrix<CSRMatrixType>(                                     \
      const std::string &filename, CSRMatrixType &csr_matrix,                  \
      const MatrixDataType data_type,                                          \
      const fast_matrix_market::read_options &options);
#define INSTANTIATE_READMATRIX_DEDUCED(CSRMatrixType)                          \
  template void readMatrix<CSRMatrixType>(                                     \
      const std::string &filename, CSRMatrixType &csr_matrix,                  \
      const fast_matrix_market::read_options &options);
#define INSTANTIATE_WRITEMATRIX(CSRMatrixType)                                 \
  template void writeMatrix<CSRMatrixType>(                                    \
      const CSRMatrixType &csr_matrix, const std::string &filename,            \
      const MatrixDataType data_type,                                          \
      const fast_matrix_market::write_options &options);
#define INSTANTIATE_WRITEMATRIX_DEDUCED(CSRMatrixType)                         \
  template void writeMatrix<CSRMatrixType>(                                    \
      const CSRMatrixType &csr_matrix, const std::string &filename,            \
      const fast_matrix_market::write_options &options);
#define INSTANTIATE_READMATRIXMARKET_CSR_VECTORS(ROWTYPE, COLTYPE, VALTYPE)    \
  template void readMatrixMarket<ROWTYPE, COLTYPE, VALTYPE>(                   \
      std::istream & instream, std::vector<ROWTYPE> & ai,                      \
      std::vector<COLTYPE> & aj, std::vector<VALTYPE> & av,                    \
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
#define INSTANTIATE_WRITEPNG(ROWTYPE, COLTYPE)                                 \
  template void writePNG<ROWTYPE, COLTYPE>(                                    \
      const COLTYPE rows, const COLTYPE cols, const ROWTYPE *ai,               \
      const COLTYPE *aj, const std::string &filename,                          \
      const COLTYPE max_display_size);
#define INSTANTIATE_READMATRIXMARKETVEC(T)                                     \
  template void readMatrixMarketVec<T>(                                        \
      std::istream & instream, std::vector<T> & vec,                           \
      const fast_matrix_market::read_options &options);

using CSRMatrixTypeDouble = matrix_utils::CSRMatrix<int, int, double>;
INSTANTIATE_READMATRIXMARKET(CSRMatrixTypeDouble);
INSTANTIATE_READMATRIX_DEDUCED(CSRMatrixTypeDouble);
INSTANTIATE_READMATRIX(CSRMatrixTypeDouble);
INSTANTIATE_WRITEMATRIX_DEDUCED(CSRMatrixTypeDouble);
INSTANTIATE_WRITEMATRIX(CSRMatrixTypeDouble);
using CSRMatrixTypeFloat = matrix_utils::CSRMatrix<int, int, float>;
INSTANTIATE_READMATRIXMARKET(CSRMatrixTypeFloat);
INSTANTIATE_READMATRIX_DEDUCED(CSRMatrixTypeFloat);
INSTANTIATE_READMATRIX(CSRMatrixTypeFloat);
INSTANTIATE_WRITEMATRIX_DEDUCED(CSRMatrixTypeFloat);
INSTANTIATE_WRITEMATRIX(CSRMatrixTypeFloat);
using CSRMatrixTypeInt = matrix_utils::CSRMatrix<int, int, int>;
INSTANTIATE_READMATRIXMARKET(CSRMatrixTypeInt);
INSTANTIATE_READMATRIX_DEDUCED(CSRMatrixTypeInt);
INSTANTIATE_READMATRIX(CSRMatrixTypeInt);
INSTANTIATE_WRITEMATRIX_DEDUCED(CSRMatrixTypeInt);
INSTANTIATE_WRITEMATRIX(CSRMatrixTypeInt);

// Add instantiations for CSRMatrixVec types
using CSRMatrixVecTypeDouble = matrix_utils::CSRMatrixVec<int, int, double>;
INSTANTIATE_READMATRIXMARKET(CSRMatrixVecTypeDouble);
INSTANTIATE_READMATRIX_DEDUCED(CSRMatrixVecTypeDouble);
INSTANTIATE_READMATRIX(CSRMatrixVecTypeDouble);
INSTANTIATE_WRITEMATRIX_DEDUCED(CSRMatrixVecTypeDouble);
INSTANTIATE_WRITEMATRIX(CSRMatrixVecTypeDouble);
using CSRMatrixVecTypeFloat = matrix_utils::CSRMatrixVec<int, int, float>;
INSTANTIATE_READMATRIXMARKET(CSRMatrixVecTypeFloat);
INSTANTIATE_READMATRIX_DEDUCED(CSRMatrixVecTypeFloat);
INSTANTIATE_READMATRIX(CSRMatrixVecTypeFloat);
INSTANTIATE_WRITEMATRIX_DEDUCED(CSRMatrixVecTypeFloat);
INSTANTIATE_WRITEMATRIX(CSRMatrixVecTypeFloat);
using CSRMatrixVecTypeInt = matrix_utils::CSRMatrixVec<int, int, int>;
INSTANTIATE_READMATRIXMARKET(CSRMatrixVecTypeInt);
INSTANTIATE_READMATRIX_DEDUCED(CSRMatrixVecTypeInt);
INSTANTIATE_READMATRIX(CSRMatrixVecTypeInt);
INSTANTIATE_WRITEMATRIX_DEDUCED(CSRMatrixVecTypeInt);
INSTANTIATE_WRITEMATRIX(CSRMatrixVecTypeInt);

INSTANTIATE_READMATRIXMARKET_CSR_VECTORS(int, int, double);
INSTANTIATE_READMATRIXMARKET_CSR_VECTORS(int, int, float);
INSTANTIATE_READMATRIXMARKET_CSR_VECTORS(int, int, int);

INSTANTIATE_WRITEMATRIXMARKET(int, int, double);
INSTANTIATE_WRITEMATRIXMARKET(int, int, float);
INSTANTIATE_WRITEMATRIXMARKET(int, int, int);

INSTANTIATE_WRITESVG(int, int);
INSTANTIATE_WRITEPNG(int, int);

INSTANTIATE_READMATRIXMARKETVEC(double);
INSTANTIATE_READMATRIXMARKETVEC(float);
INSTANTIATE_READMATRIXMARKETVEC(int);
} // namespace matrix_utils
