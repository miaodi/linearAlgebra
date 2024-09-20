#include "triangle_solve.hpp"

namespace matrix_utils {
template void ForwardSubstitution<int, int, double>(
    const int size, const int base, int const *ai, int const *aj,
    double const *av, double const *const b, double *const x);

template void
BackwardSubstitution<int, int, double>(const int size, const int base,
                                       int const *ai, int const *aj,
                                       double const *av, double const *diag,
                                       double const *const b, double *const x);

template void ForwardSubstitutionT<int, int, double>(
    const int size, const int base, int const *ai, int const *aj,
    double const *av, double const *const b, double *const x);

template void LevelScheduleForwardSubstitution<int, int, double>(
    int const *iperm, int const *prefix, const int lvls, const int rows,
    const int base, int const *ai, int const *aj, double const *av,
    double const *const b, double *const x);

template void LevelScheduleBackwardSubstitution<int, int, double>(
    int const *iperm, int const *prefix, const int lvls, const int rows,
    const int base, int const *ai, int const *aj, double const *av,
    double const *diag, double const *const b, double *const x);

template class OptimizedTriangularSolve<FBSubstitutionType::Barrier,
                                        TriangularMatrix::L, int, int, double>;

template class OptimizedTriangularSolve<FBSubstitutionType::Barrier,
                                        TriangularMatrix::U, int, int, double>;

template class OptimizedTriangularSolve<FBSubstitutionType::NoBarrier,
                                        TriangularMatrix::L, int, int, double>;

template class OptimizedTriangularSolve<FBSubstitutionType::NoBarrier,
                                        TriangularMatrix::U, int, int, double>;

template class OptimizedTriangularSolve<FBSubstitutionType::NoBarrierSuperNode,
                                        TriangularMatrix::L, int, int, double>;

template class OptimizedTriangularSolve<FBSubstitutionType::NoBarrierSuperNode,
                                        TriangularMatrix::U, int, int, double>;

} // namespace matrix_utils