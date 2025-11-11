#include "ilum.hpp"
#include "graph_algs.hpp"
#include "matrix_utils.hpp"
#include "permutation.hpp"
#include "sp_ops.hpp"
#include "spadd.hpp"
#include "spgemm.hpp"
#include "utils.h"
#include <cstdint>

namespace preconditioner
{

// Local helper function to compute E * D^{-1} in-place
// Multiplies each element E[i,j] by 1/D[j]
template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
static void computeEDinv(const VALTYPE* D_diag, const COLTYPE E_rows, const ROWTYPE* E_ai,
                         const COLTYPE* E_aj, VALTYPE* E_av, const int nthreads)
{
    const ROWTYPE E_base = E_ai[0];

// Multiply each element E[i,j] by 1/D[j] with load-balanced partitioning
#pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();
        auto [start, end] = utils::LoadPrefixBalancedPartitionPos(E_ai, E_ai + E_rows, tid, nthreads);

        for (COLTYPE i = start; i < end; ++i)
        {
            for (ROWTYPE j = E_ai[i] - E_base; j < E_ai[i + 1] - E_base; ++j)
            {
                const COLTYPE col = E_aj[j] - E_base;
                E_av[j] *= (1.0 / D_diag[col]);
            }
        }
    }
}

template <matrix_utils::ResizableCSRMatrixType CSRMatrixType>
void ILUMLevel<CSRMatrixType>::operator()(const COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj,
                                          VALTYPE const* av)
{
    // Initialize permutation vectors
    _perm.resize(size);
    _iperm.resize(size);

    // Step 1: Reorder matrix using MIS permutation on symmetric structure
    reordering(size, ai, aj, av);

    // Step 2: Split permuted matrix into blocks D, F, E, C
    split();

    // Step 3: Compute Schur complement: ANext = C - E*D^{-1}*F
    computeSchurComplement();

    // Step 4: Drop small entries from ANext based on tolerance
    if (_tau > 0.0)
    {
        dropSmallEntries();
    }
}

template <matrix_utils::ResizableCSRMatrixType CSRMatrixType>
void ILUMLevel<CSRMatrixType>::reordering(const COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj,
                                          VALTYPE const* av)
{
    // Create symmetric structure A + A^T for MIS reordering
    matrix_utils::APlusATStruct<ROWTYPE, COLTYPE, true> aplusatOp(_nthreads);

    const ROWTYPE base = ai[0];
    const ROWTYPE input_nnz = ai[size] - base;

    // Initialize symmetry struct (APlusATStruct will resize appropriately)
    _APlusAT.ResizeAI(size + 1);

    // Allocate space for symmetric structure (worst case: 2*nnz for fully asymmetric matrix)
    _APlusAT.ResizeAJ(2 * input_nnz);

    // Compute A + A^T structure
    aplusatOp(size, ai, aj, _APlusAT.AI(), _APlusAT.AJ());

    // Perform MIS-based permutation on symmetric structure
    _split_row = matrix_utils::MISPerm(size, _APlusAT.AI(), _APlusAT.AJ(), _perm.data(), _iperm.data());

    // Set dimensions and resize arrays for permuted matrix
    const ROWTYPE nnz = ai[size] - base;
    _PAPT.rows = size;
    _PAPT.cols = size;
    _PAPT.ResizeAI(size + 1);
    _PAPT.ResizeAJ(nnz);
    _PAPT.ResizeAV(nnz);

    matrix_utils::permuteMat(size, size, _perm.data(), _iperm.data(), ai, aj, av, _PAPT.AI(),
                             _PAPT.AJ(), _PAPT.AV());
}

template <matrix_utils::ResizableCSRMatrixType CSRMatrixType>
void ILUMLevel<CSRMatrixType>::split()
{
    // Split the permuted matrix PAPT into 4 blocks:
    // D (top-left), F (top-right), E (bottom-left), C (bottom-right)
    matrix_utils::partitionCSR2x2<CSRMatrixType>(
        _PAPT.rows, _PAPT.cols, _PAPT.AI(), _PAPT.AJ(), _PAPT.AV(),
        _split_row, // row split point
        _split_row, // column split point (same as row for square matrix)
        _D,         // A11 - top-left block
        _F,         // A12 - top-right block
        _EDinv,     // A21 - bottom-left block (will become E * D^-1 later)
        _C          // A22 - bottom-right block
    );

    // Compute E * D^{-1} in-place
    // D is already diagonal, so D.AV() contains the diagonal values
    preconditioner::computeEDinv(_D.AV(), _EDinv.rows, _EDinv.AI(), _EDinv.AJ(), _EDinv.AV(), _nthreads);
}

template <matrix_utils::ResizableCSRMatrixType CSRMatrixType>
void ILUMLevel<CSRMatrixType>::computeSchurComplement()
{
    // Compute ANext = C - EDinv * F
    // where EDinv = E * D^{-1}

    // First compute EDinv * F using SpGEMM
    CSRMatrixType temp;
    matrix_utils::SpGEMM<CSRMatrixType> spgemm(_nthreads);

    // Analysis phase: determine sparsity pattern
    spgemm.analysis(_EDinv.rows, _EDinv.cols, _EDinv.AI(), _EDinv.AJ(), _F.rows, _F.cols, _F.AI(),
                    _F.AJ(), temp);

    // Numerical phase: compute values
    spgemm(_EDinv.rows, _EDinv.cols, _EDinv.AI(), _EDinv.AJ(), _EDinv.AV(), _F.rows, _F.cols,
           _F.AI(), _F.AJ(), _F.AV(), temp);

    // Compute ANext = C - temp using SpADD with alpha=1, beta=-1
    matrix_utils::SpADD<CSRMatrixType> spadd(_nthreads);

    // Analysis phase
    spadd.analysis(_C.rows, _C.cols, _C.AI(), _C.AJ(), temp.rows, temp.cols, temp.AI(), temp.AJ(), _ANext);

    // Numerical phase: ANext = 1.0 * C + (-1.0) * temp
    spadd(_C.rows, _C.cols, _C.AI(), _C.AJ(), _C.AV(),
          1.0, // alpha for C
          temp.rows, temp.cols, temp.AI(), temp.AJ(), temp.AV(),
          -1.0, // beta for temp
          _ANext);
}

template <matrix_utils::ResizableCSRMatrixType CSRMatrixType>
void ILUMLevel<CSRMatrixType>::dropSmallEntries()
{
    // Drop entries in ANext whose absolute value is less than row_max * tau
    // Store result in _ANextDropped
    const ROWTYPE base = _ANext.AI()[0];
    const COLTYPE rows = _ANext.rows;
    const ROWTYPE total_nnz = _ANext.AI()[rows] - base;

    // Set dimensions for dropped matrix
    _ANextDropped.rows = rows;
    _ANextDropped.cols = _ANext.cols;
    _ANextDropped.ResizeAI(rows + 1);

    // Get pointer to row pointers for direct access
    ROWTYPE* dropped_ai = _ANextDropped.AI();

    // Cache row thresholds to avoid recomputing in second pass
    std::vector<VALTYPE> row_thresholds(rows);

    // Thread-local sums for parallel prefix sum
    std::vector<ROWTYPE> thread_sums(_nthreads + 1, 0);

#pragma omp parallel num_threads(_nthreads)
    {
        const int tid = omp_get_thread_num();

        // Phase 1: Compute row thresholds and count non-dropped entries per row (load-balanced)
        auto [count_start, count_end] =
            utils::LoadPrefixBalancedPartitionPos(_ANext.AI(), _ANext.AI() + rows, tid, _nthreads);

        for (COLTYPE i = count_start; i < count_end; ++i)
        {
            // Find row maximum
            VALTYPE row_max = 0.0;
            for (ROWTYPE j = _ANext.AI()[i] - base; j < _ANext.AI()[i + 1] - base; ++j)
            {
                VALTYPE abs_val = std::abs(_ANext.AV()[j]);
                if (abs_val > row_max)
                {
                    row_max = abs_val;
                }
            }

            // Cache threshold for this row
            VALTYPE row_threshold = row_max * _tau;
            row_thresholds[i] = row_threshold;

            // Count entries that exceed row_max * tau
            ROWTYPE count = 0;
            for (ROWTYPE j = _ANext.AI()[i] - base; j < _ANext.AI()[i + 1] - base; ++j)
            {
                if (std::abs(_ANext.AV()[j]) >= row_threshold)
                {
                    count++;
                }
            }
            dropped_ai[i + 1] = count;
        }

#pragma omp barrier

        // Phase 2: Parallel prefix sum - Local scan per thread
        auto [start, end] = utils::LoadBalancedPartitionPos(rows, tid, _nthreads);
        ROWTYPE local_sum = 0;
        for (COLTYPE i = start; i < end; ++i)
        {
            local_sum += dropped_ai[i + 1];
        }
        // Store the last value for global adjustment
        thread_sums[tid + 1] = local_sum;

#pragma omp barrier

#pragma omp single
        {
            thread_sums[0] = base;
            for (int i = 1; i <= _nthreads; ++i)
            {
                thread_sums[i] += thread_sums[i - 1];
            }
            dropped_ai[rows] = thread_sums[_nthreads];
            const ROWTYPE new_nnz = dropped_ai[rows] - base;
            _ANextDropped.ResizeAJ(new_nnz);
            _ANextDropped.ResizeAV(new_nnz);
        }

        const ROWTYPE offset = thread_sums[tid];
        dropped_ai[start] = offset;
        for (COLTYPE i = start; i < end - 1; ++i)
        {
            dropped_ai[i + 1] += dropped_ai[i];
        }
#pragma omp barrier

        // Phase 3: Copy non-dropped entries using cached thresholds (load-balanced)
        for (COLTYPE i = count_start; i < count_end; ++i)
        {
            // Use cached threshold
            VALTYPE row_threshold = row_thresholds[i];

            // Copy entries that exceed threshold
            ROWTYPE new_idx = dropped_ai[i] - base;
            for (ROWTYPE j = _ANext.AI()[i] - base; j < _ANext.AI()[i + 1] - base; ++j)
            {
                if (std::abs(_ANext.AV()[j]) >= row_threshold)
                {
                    _ANextDropped.AJ()[new_idx] = _ANext.AJ()[j];
                    _ANextDropped.AV()[new_idx] = _ANext.AV()[j];
                    new_idx++;
                }
            }
        }
    }
}

// Explicit instantiation for common types
template struct ILUMLevel<matrix_utils::CSRMatrixVec<int, int, double>>;
template struct ILUMLevel<matrix_utils::CSRMatrixVec<int64_t, int64_t, double>>;
template struct ILUMLevel<matrix_utils::CSRMatrixVec<int, int, float>>;
template struct ILUMLevel<matrix_utils::CSRMatrixVec<int64_t, int64_t, float>>;

} // namespace preconditioner
