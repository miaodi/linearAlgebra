#include "scaling.hpp"
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace matrix_utils
{
template <typename COLTYPE, typename VALTYPE>
void ScaleVector(const COLTYPE size, VALTYPE* x, VALTYPE const* s, int nthreads)
{
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (COLTYPE i = 0; i < size; ++i)
    {
        x[i] *= s[i];
    }
}

enum class ScalingType
{
    RowOnly,
    ColOnly,
    Both
};

template <ScalingType ST, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
static void ScalingMatInternal(const COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj,
                               VALTYPE* av, VALTYPE const* dr, VALTYPE const* dc, int nthreads)
{
    const auto base = ai[0];

    if constexpr (ST == ScalingType::ColOnly)
    {
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (COLTYPE i = 0; i < rows; ++i)
        {
            for (ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; ++j)
            {
                const COLTYPE col = aj[j] - base;
                av[j] *= dc[col];
            }
        }
    }
    else if constexpr (ST == ScalingType::RowOnly)
    {
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (COLTYPE i = 0; i < rows; ++i)
        {
            const VALTYPE row_scale = dr[i];
            for (ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; ++j)
            {
                av[j] *= row_scale;
            }
        }
    }
    else // ST == ScalingType::Both
    {
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (COLTYPE i = 0; i < rows; ++i)
        {
            const VALTYPE row_scale = dr[i];
            for (ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; ++j)
            {
                const COLTYPE col = aj[j] - base;
                av[j] *= row_scale * dc[col];
            }
        }
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
bool ScaleMat(const COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj, VALTYPE* av,
              VALTYPE const* dr, VALTYPE const* dc, int nthreads)
{
    // Error: both are nullptr
    if (dr == nullptr && dc == nullptr)
    {
        return false;
    }

    // Determine scaling type at runtime and dispatch to appropriate template
    if (dr == nullptr)
    {
        ScalingMatInternal<ScalingType::ColOnly>(rows, ai, aj, av, dr, dc, nthreads);
    }
    else if (dc == nullptr)
    {
        ScalingMatInternal<ScalingType::RowOnly>(rows, ai, aj, av, dr, dc, nthreads);
    }
    else
    {
        ScalingMatInternal<ScalingType::Both>(rows, ai, aj, av, dr, dc, nthreads);
    }

    return true;
}

// Explicit template instantiations for common types
template void ScaleVector<int, double>(const int, double*, double const*, int);
template void ScaleVector<int, float>(const int, float*, float const*, int);

template bool ScaleMat<int, int, double>(const int, int const*, int const*, double*,
                                         double const*, double const*, int);
template bool ScaleMat<int, int, float>(const int, int const*, int const*, float*,
                                        float const*, float const*, int);

} // namespace matrix_utils
