#include "matrix_utils.hpp"

namespace matrix_utils {
template class CSRMatrix<int, int, double>;
template class CSRMatrixVec<int, int, double>;
template class CSRMatrixVec<int, int, int8_t>;
template class CSRMatrixVec<int64_t, int64_t, int8_t>;

template void SerialTranspose<int, int, double>(const int rows, const int cols,
                                                int const *ai,
                                                int const *aj, double const *av,
                                                int *ai_transpose,
                                                int *aj_transpose,
                                                double *av_transpose);

template void ParallelTranspose<int, int, double>(
    const int rows, const int cols, int const *ai,
    int const *aj, double const *av, int *ai_transpose, int *aj_transpose,
    double *av_transpose);

template void ParallelTranspose2<int, int, double>(
    const int rows, const int cols, int const *ai,
    int const *aj, double const *av, int *ai_transpose, int *aj_transpose,
    double *av_transpose);

template bool Diagonal<int, int, double>(const int rows, int const *ai,
                                         int const *aj, double const *av,
                                         int *diagpos, double *diag,
                                         const bool invert);

template bool IsSymmetry<int, int, double>(const int size, int const *ai,
                                            int const *aj, double const *av);

template void SplitLDU(const int rows, const int base, int const *ai,
                       int const *aj, double const *av,
                       CSRMatrix<int, int, double> &L, std::vector<double> &D,
                       CSRMatrix<int, int, double> &U);

template <ResizableCSR CSRMatrixType>
void SplitLU<CSRMatrixType>::operator()(const COLTYPE rows, ROWTYPE const *ai,
                                        ROWTYPE const *diag, COLTYPE const *aj,
                                        VALTYPE const *av, CSRMatrixType &L,
                                        CSRMatrixType &U) {
  const auto base = ai[0];
  L.rows = rows;
  L.cols = rows;
  L.ResizeAI(rows + 1);
  auto L_ai = L.AI();
  L_ai[0] = base;

  U.rows = rows;
  U.cols = rows;
  U.ResizeAI(rows + 1);
  auto U_ai = U.AI();
  U_ai[0] = base;

  prefixL[0] = base;
  prefixU[0] = base;

#pragma omp parallel num_threads(num_threads)
  {
    const int tid = omp_get_thread_num();
    auto [start, end] =
        utils::LoadPrefixBalancedPartitionPos(ai, ai + rows, tid, num_threads);
    prefixL[tid + 1] = 0;
    prefixU[tid + 1] = 0;
    for (auto i = start; i < end; i++) {
      prefixL[tid + 1] += diag[i] - ai[i];
      prefixU[tid + 1] += ai[i + 1] - diag[i];
      L_ai[i + 1] = prefixL[tid + 1];
      U_ai[i + 1] = prefixU[tid + 1];
    }
#pragma omp barrier
#pragma omp single
    {
      for (size_t i = 1; i < prefixL.size(); i++) {
        prefixL[i] += prefixL[i - 1];
        prefixU[i] += prefixU[i - 1];
      }
      const auto L_nnz = prefixL[num_threads] - base;
      const auto U_nnz = prefixU[num_threads] - base;
      L.ResizeAJ(L_nnz);
      L.ResizeAV(L_nnz);
      U.ResizeAJ(U_nnz);
      U.ResizeAV(U_nnz);
    }

    auto L_pos = prefixL[tid] - base;
    auto U_pos = prefixU[tid] - base;
    for (auto i = start; i < end; i++) {
      L_ai[i + 1] += prefixL[tid];
      U_ai[i + 1] += prefixU[tid];

      for (auto j = ai[i]; j < diag[i]; j++) {
        L.AJ()[L_pos] = aj[j];
        L.AV()[L_pos++] = av[j];
      }
      for (auto j = diag[i]; j < ai[i + 1]; j++) {
        U.AJ()[U_pos] = aj[j];
        U.AV()[U_pos++] = av[j];
      }
    }
  }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
ROWTYPE Prune(const COLTYPE rows, ROWTYPE* ai, COLTYPE* aj, VALTYPE* av, const VALTYPE threshold,
              VALTYPE const* row_thresholds)
{
    const ROWTYPE base = ai[0];
    const ROWTYPE old_nnz = ai[rows] - base;

    // Store original ai values before modifying
    std::vector<ROWTYPE> old_ai(rows + 1);
    std::memcpy(old_ai.data(), ai, (rows + 1) * sizeof(ROWTYPE));

    std::vector<ROWTYPE> new_row_sizes(rows);
    std::vector<ROWTYPE> thread_prefix(omp_get_max_threads() + 1, 0);
    std::vector<VALTYPE> av_tmp(old_nnz);
    std::vector<COLTYPE> aj_tmp(old_nnz);

    auto get_threshold = [&](COLTYPE row)
    { return row_thresholds ? row_thresholds[row] : threshold; };

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int num_threads = omp_get_num_threads();
        auto [start, end] =
            utils::LoadPrefixBalancedPartitionPos(old_ai.data(), old_ai.data() + rows, tid, num_threads);

        // Phase 1: Filter and count surviving entries per row
        ROWTYPE local_nnz = 0;
        for (COLTYPE i = start; i < end; i++)
        {
            ROWTYPE row_size = 0;
            const ROWTYPE row_start = old_ai[i] - base;
            const ROWTYPE row_end = old_ai[i + 1] - base;
            const auto row_threshold = get_threshold(i);

            for (ROWTYPE j = row_start; j < row_end; j++)
            {
                if (std::abs(av[j]) > row_threshold)
                {
                    aj_tmp[row_start + row_size] = aj[j];
                    av_tmp[row_start + row_size] = av[j];
                    row_size++;
                }
            }
            new_row_sizes[i] = row_size;
            local_nnz += row_size;
        }
        thread_prefix[tid + 1] = local_nnz;

#pragma omp barrier
#pragma omp single
        {
            // Compute thread prefix sums
            for (size_t i = 1; i < thread_prefix.size(); i++)
            {
                thread_prefix[i] += thread_prefix[i - 1];
            }
        }

        // Phase 2: Compute new ai array (CSR row pointers)
        ROWTYPE row_offset = thread_prefix[tid] + base;
        for (COLTYPE i = start; i < end; i++)
        {
            row_offset += new_row_sizes[i];
            ai[i + 1] = row_offset;
        }

#pragma omp barrier

        // Phase 3: Copy filtered data to final positions
        for (COLTYPE i = start; i < end; i++)
        {
            const ROWTYPE old_row_start = old_ai[i] - base;
            const ROWTYPE new_row_start = ai[i] - base;
            const ROWTYPE row_size = new_row_sizes[i];

            if (row_size > 0)
            {
                std::memcpy(aj + new_row_start, aj_tmp.data() + old_row_start, row_size * sizeof(COLTYPE));
                std::memcpy(av + new_row_start, av_tmp.data() + old_row_start, row_size * sizeof(VALTYPE));
            }
        }
    }
    
    return old_nnz - (ai[rows] - base);
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
ROWTYPE DiagonalScaledPrune(const COLTYPE rows, ROWTYPE* ai, COLTYPE* aj, VALTYPE* av, const VALTYPE threshold)
{
    const ROWTYPE base = ai[0];

    // Step 1: Extract diagonal values
    std::vector<VALTYPE> diag(rows, 0.0);

#pragma omp parallel for
    for (COLTYPE i = 0; i < rows; ++i)
    {
        const ROWTYPE row_start = ai[i] - base;
        const ROWTYPE row_end = ai[i + 1] - base;
        const COLTYPE diag_col = i + base;

        // Binary search for diagonal element
        auto it = std::lower_bound(aj + row_start, aj + row_end, diag_col);
        if (it != aj + row_end && *it == diag_col)
        {
            diag[i] = av[it - aj];
        }
    }

    // Step 2: Zero out entries where |a_ii| * |a_jj| * threshold < |a_ij|
#pragma omp parallel for
    for (COLTYPE i = 0; i < rows; ++i)
    {
        const VALTYPE abs_diag_i = std::abs(diag[i]);
        for (ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; ++j)
        {
            const COLTYPE col = aj[j] - base;
            if (col != i) // Skip diagonal
            {
                const VALTYPE abs_diag_j = std::abs(diag[col]);
                const VALTYPE threshold_ij = abs_diag_i * abs_diag_j * threshold;
                if (av[j] * av[j] < threshold_ij)
                {
                    av[j] = static_cast<VALTYPE>(0);
                }
            }
        }
    }

    // Step 3: Prune zeros
    return Prune(rows, ai, aj, av, static_cast<VALTYPE>(0), (VALTYPE*)nullptr);
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
ROWTYPE RowMaxPrune( const COLTYPE rows,
                     ROWTYPE* ai,
                     COLTYPE* aj,
                     VALTYPE* av,
                     const VALTYPE threshold )
{
    const ROWTYPE base = ai ? ai[0] : ROWTYPE{};
    if (!ai || !aj || !av || rows <= 0)
        return ROWTYPE{};

    std::vector<VALTYPE> row_max(rows, static_cast<VALTYPE>(0));

#pragma omp parallel for
    for (COLTYPE i = 0; i < rows; ++i)
    {
        VALTYPE m = static_cast<VALTYPE>(0);
        for (ROWTYPE p = ai[i] - base; p < ai[i + 1] - base; ++p)
        {
            VALTYPE a = std::abs(av[p]);
            if (a > m)
                m = a;
        }
        row_max[i] = m * threshold;
    }

    return Prune<ROWTYPE, COLTYPE, VALTYPE>(rows, ai, aj, av, static_cast<VALTYPE>(0), row_max.data());
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void EmbedCSR(const COLTYPE rows,
              ROWTYPE const *ai_source, COLTYPE const *aj_source, VALTYPE const *av_source,
              ROWTYPE const *ai_target, COLTYPE const *aj_target, VALTYPE *av_target,
              const int num_threads) {
  
  const ROWTYPE base = ai_source[0];
  const int chunk_size = 32;
  // Process each row in parallel
#pragma omp parallel for schedule(dynamic, chunk_size) num_threads(num_threads)
  for (COLTYPE i = 0; i < rows; ++i) {
    const ROWTYPE source_start = ai_source[i] - base;
    const ROWTYPE source_end = ai_source[i + 1] - base;
    const ROWTYPE target_start = ai_target[i] - base;
    const ROWTYPE target_end = ai_target[i + 1] - base;
    
    ROWTYPE src_idx = source_start;
    
    // Two-pointer merge: for each target position, find matching source or set to zero
    for (ROWTYPE tgt_idx = target_start; tgt_idx < target_end; ++tgt_idx) {
      const COLTYPE tgt_col = aj_target[tgt_idx];
      
      // Since target contains all source elements, check if current source matches
      if (src_idx < source_end && aj_source[src_idx] == tgt_col) {
        // Match found: copy value and advance source pointer
        av_target[tgt_idx] = av_source[src_idx];
        src_idx++;
      } else {
        // No match: target has extra element not in source, set to zero
        av_target[tgt_idx] = static_cast<VALTYPE>(0);
      }
    }
  }
}

template void SplitTriangle<TriangularMatrix::U, int, int, double,
                            CSRMatrix<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    double const *av, CSRMatrix<int, int, double> &U);

template void SplitTriangle<TriangularMatrix::U, int, int, double,
                            CSRMatrixVec<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    double const *av, CSRMatrixVec<int, int, double> &U);

template void SplitTriangle<TriangularMatrix::L, int, int, double,
                            CSRMatrix<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    double const *av, CSRMatrix<int, int, double> &U);

template void SplitTriangle<TriangularMatrix::L, int, int, double,
                            CSRMatrixVec<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    double const *av, CSRMatrixVec<int, int, double> &U);

template void TriangularToFull<TriangularMatrix::U, int, int, double,
                               CSRMatrix<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    double const *av, CSRMatrix<int, int, double> &F);

template void TriangularToFull<TriangularMatrix::U, int, int, double,
                               CSRMatrixVec<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    double const *av, CSRMatrixVec<int, int, double> &F);

#define INSTANTIATE_SPLIT_LU(ROWTYPE, COLTYPE, VALTYPE)                        \
  template struct SplitLU<CSRMatrix<ROWTYPE, COLTYPE, VALTYPE>>;               \
  template struct SplitLU<CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE>>;

INSTANTIATE_SPLIT_LU(std::int32_t, std::int32_t, double)
INSTANTIATE_SPLIT_LU(int, int, float)

template <typename ROWTYPE, typename COLTYPE>
bool CSRToMetisGraph(const COLTYPE nrows,
                     ROWTYPE const *ai, COLTYPE const *aj,
                     ROWTYPE *xadj, COLTYPE *adjncy,
                     const int nthreads) {
  // Detect base indexing from ai[0]
  const ROWTYPE base = ai[0];

  // Compute xadj and fill adjncy with zero-based column indices (excluding diagonal)
  xadj[0] = 0;
  bool all_diag_found = true;
  
#pragma omp parallel for num_threads(nthreads) shared(all_diag_found)
  for (COLTYPE i = 0; i < nrows; ++i) {
    // Compute xadj: xadj[i+1] = ai[i+1] - (i + 1) - base (removes diagonal entries)
    xadj[i + 1] = ai[i + 1] - (i + 1) - base;
    
    // Skip filling adjncy if already found missing diagonal
    if (!all_diag_found) continue;
    
    const ROWTYPE row_start = ai[i] - base;
    const ROWTYPE row_end = ai[i + 1] - base;
    const COLTYPE diag_col = i + base;
    
    // Binary search for diagonal element
    const auto *diag_it = std::lower_bound(aj + row_start, aj + row_end, diag_col);
    
    // Check if diagonal element exists
    if (diag_it == aj + row_end || *diag_it != diag_col) {
      all_diag_found = false;
      continue;
    }
    
    const ROWTYPE diag_pos = diag_it - aj;
    ROWTYPE pos = xadj[i];
    
    // Copy and transform elements before diagonal to zero-based
    std::transform(aj + row_start, aj + diag_pos, adjncy + pos, 
                   [base](COLTYPE col) { return col - base; });
    pos += diag_pos - row_start;
    
    // Copy and transform elements after diagonal
    std::transform(aj + diag_pos + 1, aj + row_end, adjncy + pos,
                   [base](COLTYPE col) { return col - base; });
  }
  
  return all_diag_found;
}

#define INSTANTIATE_MATRIX_OPS(ROWTYPE, COLTYPE, VALTYPE)                     \
  template ROWTYPE Prune<ROWTYPE, COLTYPE, VALTYPE>(                          \
      const COLTYPE rows, ROWTYPE* ai, COLTYPE* aj, VALTYPE* av,              \
      const VALTYPE threshold, VALTYPE const* row_thresholds);                 \
  template ROWTYPE RowMaxPrune<ROWTYPE, COLTYPE, VALTYPE>(                    \
    const COLTYPE rows, ROWTYPE* ai, COLTYPE* aj, VALTYPE* av,              \
    const VALTYPE threshold);                                               \
  template ROWTYPE DiagonalScaledPrune<ROWTYPE, COLTYPE, VALTYPE>(            \
      const COLTYPE rows, ROWTYPE* ai, COLTYPE* aj, VALTYPE* av,              \
      const VALTYPE threshold);                                                \
  template void EmbedCSR<ROWTYPE, COLTYPE, VALTYPE>(                          \
      const COLTYPE rows,                                                      \
      ROWTYPE const *ai_source, COLTYPE const *aj_source, VALTYPE const *av_source, \
      ROWTYPE const *ai_target, COLTYPE const *aj_target, VALTYPE *av_target, \
      const int num_threads);

INSTANTIATE_MATRIX_OPS(int, int, double)
INSTANTIATE_MATRIX_OPS(int, int, float)
INSTANTIATE_MATRIX_OPS(std::int64_t, std::int64_t, double)

// Explicit instantiations for CSRToMetisGraph
template bool CSRToMetisGraph<int, int>(const int nrows,
                                        int const *ai, int const *aj,
                                        int *xadj, int *adjncy,
                                        const int nthreads);
template bool CSRToMetisGraph<int64_t, int64_t>(const int64_t nrows,
                                                 int64_t const *ai, int64_t const *aj,
                                                 int64_t *xadj, int64_t *adjncy,
                                                 const int nthreads);
template bool CSRToMetisGraph<int64_t, int>(const int nrows,
                                             int64_t const *ai, int const *aj,
                                             int64_t *xadj, int *adjncy,
                                             const int nthreads);

} // namespace matrix_utils
