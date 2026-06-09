#pragma once

#include "cuda_memory.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace matrix_utils::sparse_cuda
{

template <typename ROWTYPE>
struct ILUUpdateCache
{
    // Strict-lower row offsets. For lower entry k_pos in row i:
    // lower_id = lower_row_ptr[i] + (k_pos - row_begin).
    std::vector<ROWTYPE> lower_row_ptr;
    // Per-lower-entry update offsets into update_jpos/update_pos.
    std::vector<ROWTYPE> update_ptr;
    std::vector<ROWTYPE> update_jpos;
    std::vector<ROWTYPE> update_pos;
    ROWTYPE strict_lower_nnz = 0;
    ROWTYPE total_updates = 0;
    double build_ms = 0.0;

    std::size_t bytes() const
    {
        return ( lower_row_ptr.size() + update_ptr.size() ) * sizeof( ROWTYPE ) +
               ( update_jpos.size() + update_pos.size() ) * sizeof( ROWTYPE );
    }
};

template <typename ROWTYPE>
struct DeviceILUUpdateCache
{
    // Device-resident lower-only cache with the same layout as ILUUpdateCache.
    DeviceArray<ROWTYPE> lower_row_ptr;
    DeviceArray<ROWTYPE> update_ptr;
    DeviceArray<ROWTYPE> update_jpos;
    DeviceArray<ROWTYPE> update_pos;
    ROWTYPE strict_lower_nnz = 0;
    ROWTYPE total_updates = 0;
    double build_ms = 0.0;

    std::size_t bytes() const
    {
        return ( lower_row_ptr.size() + update_ptr.size() + update_jpos.size() + update_pos.size() ) *
               sizeof( ROWTYPE );
    }
};

template <typename ROWTYPE, typename COLTYPE>
ILUUpdateCache<ROWTYPE> BuildILUUpdateCache( COLTYPE n,
                                             const ROWTYPE* lu_ai,
                                             const COLTYPE* lu_aj,
                                             const ROWTYPE* lu_diag,
                                             COLTYPE base,
                                             int threads );

template <typename ROWTYPE, typename COLTYPE>
cudaError_t BuildILUUpdateCacheAsync( COLTYPE n,
                                      const ROWTYPE* d_lu_ai,
                                      const COLTYPE* d_lu_aj,
                                      const ROWTYPE* d_lu_diag,
                                      COLTYPE base,
                                      DeviceILUUpdateCache<ROWTYPE>& cache,
                                      cudaStream_t stream = nullptr );

} // namespace matrix_utils::sparse_cuda
