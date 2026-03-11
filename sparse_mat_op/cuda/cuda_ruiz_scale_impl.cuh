#pragma once

namespace matrix_utils::sparse_cuda
{
namespace detail
{
template <typename...>
inline constexpr bool always_false_v = false;

#define DECLARE_RUIZ_SCALE_CUDA_OVERLOADS(ROWTYPE, COLTYPE, VALTYPE)                                            \
    bool RuizScaleCudaCSRImplMaxNorm(const COLTYPE rows, const COLTYPE cols, const ROWTYPE* d_ai,              \
                                     const COLTYPE* d_aj, VALTYPE* d_av, VALTYPE* d_dr, VALTYPE* d_dc,        \
                                     const int max_iters);                                                      \
    bool RuizScaleCudaCSRImplL2Norm(const COLTYPE rows, const COLTYPE cols, const ROWTYPE* d_ai,               \
                                    const COLTYPE* d_aj, VALTYPE* d_av, VALTYPE* d_dr, VALTYPE* d_dc,         \
                                    const int max_iters);                                                       \
    bool RuizScaleCudaTileImplMaxNorm(DeviceTileCOOMatrix<ROWTYPE, COLTYPE, VALTYPE>& tile_mat,               \
                                      VALTYPE* d_dr, VALTYPE* d_dc, const int max_iters);                      \
    bool RuizScaleCudaTileImplL2Norm(DeviceTileCOOMatrix<ROWTYPE, COLTYPE, VALTYPE>& tile_mat,                \
                                     VALTYPE* d_dr, VALTYPE* d_dc, const int max_iters);

DECLARE_RUIZ_SCALE_CUDA_OVERLOADS(int32_t, int32_t, float)
DECLARE_RUIZ_SCALE_CUDA_OVERLOADS(int32_t, int32_t, double)
DECLARE_RUIZ_SCALE_CUDA_OVERLOADS(int64_t, int64_t, float)
DECLARE_RUIZ_SCALE_CUDA_OVERLOADS(int64_t, int64_t, double)

#undef DECLARE_RUIZ_SCALE_CUDA_OVERLOADS
} // namespace detail

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, CudaRuizScalingNormType NORM>
bool RuizScaleCuda(const COLTYPE rows, const COLTYPE cols, const ROWTYPE* d_ai, const COLTYPE* d_aj,
                   VALTYPE* d_av, VALTYPE* d_dr, VALTYPE* d_dc, const int max_iters)
{
    if constexpr (NORM == CudaRuizScalingNormType::MaxNorm)
    {
        return detail::RuizScaleCudaCSRImplMaxNorm(rows, cols, d_ai, d_aj, d_av, d_dr, d_dc, max_iters);
    }
    else if constexpr (NORM == CudaRuizScalingNormType::L2Norm)
    {
        return detail::RuizScaleCudaCSRImplL2Norm(rows, cols, d_ai, d_aj, d_av, d_dr, d_dc, max_iters);
    }
    else
    {
        static_assert(detail::always_false_v<ROWTYPE, COLTYPE, VALTYPE>, "Unsupported RuizScaleCuda norm");
        return false;
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE, CudaRuizScalingNormType NORM>
bool RuizScaleCuda(DeviceTileCOOMatrix<ROWTYPE, COLTYPE, VALTYPE>& tile_mat, VALTYPE* d_dr,
                   VALTYPE* d_dc, const int max_iters)
{
    if constexpr (NORM == CudaRuizScalingNormType::MaxNorm)
    {
        return detail::RuizScaleCudaTileImplMaxNorm(tile_mat, d_dr, d_dc, max_iters);
    }
    else if constexpr (NORM == CudaRuizScalingNormType::L2Norm)
    {
        return detail::RuizScaleCudaTileImplL2Norm(tile_mat, d_dr, d_dc, max_iters);
    }
    else
    {
        static_assert(detail::always_false_v<ROWTYPE, COLTYPE, VALTYPE>, "Unsupported RuizScaleCuda norm");
        return false;
    }
}

} // namespace matrix_utils::sparse_cuda
