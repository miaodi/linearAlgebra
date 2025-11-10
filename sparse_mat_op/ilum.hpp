#pragma once
#include "sparse_mat_traits.hpp"
#include <vector>

namespace preconditioner
{
template <matrix_utils::ResizableCSRMatrixType CSRMatrixType>
struct ILUMLevel
{
    using ROWTYPE = typename CSRMatrixType::ROWTYPE;
    using COLTYPE = typename CSRMatrixType::COLTYPE;
    using VALTYPE = typename CSRMatrixType::VALTYPE;

    ILUMLevel() : _nthreads(1), _tau(0.0) {}
    explicit ILUMLevel(int nthreads, VALTYPE tau = 0.0) : _nthreads(nthreads), _tau(tau) {}

    void operator()(const COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj, VALTYPE const* av);
    
    void setNumThreads(int nthreads) { _nthreads = nthreads; }
    void setDropTolerance(VALTYPE tau) { _tau = tau; }

    CSRMatrixType _PAPT;
    CSRMatrixType _D;
    CSRMatrixType _F;
    CSRMatrixType _EDinv;
    CSRMatrixType _C;
    CSRMatrixType _ANext;
    CSRMatrixType _ANextDropped;

    std::vector<COLTYPE> _perm;
    std::vector<COLTYPE> _iperm;
    COLTYPE _split_row; // always zero-based

private:
    void reordering(const COLTYPE size, ROWTYPE const* ai, COLTYPE const* aj, VALTYPE const* av);
    void split();
    void computeEDinv();
    void computeSchurComplement();
    void dropSmallEntries();

    int _nthreads;
    VALTYPE _tau; // drop tolerance
};
} // namespace preconditioner