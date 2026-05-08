#pragma once

#include <cstddef>
#include <vector>

namespace matrix_utils::sparse_cuda
{

template <typename ROWTYPE>
struct ILUUpdateCache
{
    std::vector<ROWTYPE> update_ptr;
    std::vector<ROWTYPE> update_jpos;
    std::vector<ROWTYPE> update_pos;
    double build_ms = 0.0;

    std::size_t bytes() const
    {
        return update_ptr.size() * sizeof( ROWTYPE ) +
               ( update_jpos.size() + update_pos.size() ) * sizeof( ROWTYPE );
    }
};

template <typename ROWTYPE, typename COLTYPE>
ILUUpdateCache<ROWTYPE> BuildILUUpdateCache( COLTYPE n,
                                             const ROWTYPE* lu_ai,
                                             const COLTYPE* lu_aj,
                                             const ROWTYPE* lu_diag,
                                             COLTYPE base,
                                             int threads );

} // namespace matrix_utils::sparse_cuda
