#include "triangle_solve.hpp"

namespace matrix_utils
{
// template class TriangularSolve<int, int, double>;
template void TriangularSolve<TriangularMatrix::L, int, int, double>(
    const int, int const*, int const*, double const*, double const*, double const*, double* );

template void TriangularSolve<TriangularMatrix::U, int, int, double>(
    const int, int const*, int const*, double const*, double const*, double const*, double* );

template class LevelScheduleTriangularSubstitution<TriangularMatrix::L, int, int, double>;
template class LevelScheduleTriangularSubstitution<TriangularMatrix::U, int, int, double>;

template class P2PTriangularSubstitution<TriangularMatrix::L, int, int, double>;
template class P2PTriangularSubstitution<TriangularMatrix::U, int, int, double>;

template class OptimizedTriangularSolve<FBSubstitutionType::Barrier, TriangularMatrix::L, int, int, double>;

template class OptimizedTriangularSolve<FBSubstitutionType::Barrier, TriangularMatrix::U, int, int, double>;

template class OptimizedTriangularSolve<FBSubstitutionType::NoBarrier, TriangularMatrix::L, int, int, double>;

template class OptimizedTriangularSolve<FBSubstitutionType::NoBarrier, TriangularMatrix::U, int, int, double>;

template class OptimizedTriangularSolve<FBSubstitutionType::NoBarrierSuperNode, TriangularMatrix::L, int, int, double>;

template class OptimizedTriangularSolve<FBSubstitutionType::NoBarrierSuperNode, TriangularMatrix::U, int, int, double>;
} // namespace matrix_utils