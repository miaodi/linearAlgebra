#include "Reordering.h"
#include "UnionFind.h"
#include "utils.h"
#include <algorithm>
#include <iostream>
#include <span>
#ifdef USE_METIS_LIB
#ifdef USE_MTMETIS
#include <mtmetis.h>
#else
#include <metis.h>
#endif
#endif

namespace reordering
{

// Template implementation
template <typename ROWTYPE, typename COLTYPE>
void NodeDegree( COLTYPE rows, const ROWTYPE* ai, COLTYPE* degrees, int numthreads )
{
    if ( numthreads <= 0 )
    {
        numthreads = omp_get_max_threads();
    }

// Parallel version with SIMD vectorization
#pragma omp parallel for simd num_threads( numthreads )
    for ( COLTYPE i = 0; i < rows; i++ )
    {
        degrees[i] = ai[i + 1] - ai[i];
    }
}

#ifdef USE_METIS_LIB
template <typename ROWTYPE, typename COLTYPE>
int MetisND( const COLTYPE nrows,
             const COLTYPE ncols,
             const ROWTYPE* xadj,
             const COLTYPE* adjncy,
             COLTYPE* iperm,
             COLTYPE* perm,
             const MetisNDOptions& opts )
{
    // METIS requires square matrix
    if ( nrows != ncols )
    {
        return -1;
    }

#ifdef USE_MTMETIS
    using metis_idx_t = mtmetis_vtx_type;
    using metis_adj_t = mtmetis_adj_type;
    using metis_pid_t = mtmetis_pid_type;
    constexpr int SUCCESS_CODE = MTMETIS_SUCCESS;
#else
    using metis_idx_t = idx_t;
    using metis_adj_t = idx_t;
    using metis_pid_t = idx_t;
    constexpr int SUCCESS_CODE = METIS_OK;
#endif

    // Copy xadj and adjncy to METIS types (assuming input is already zero-based with diagonals removed)
    const metis_adj_t nnz = xadj[nrows];
    std::vector<metis_adj_t> xadj_metis( xadj, xadj + nrows + 1 );
    std::vector<metis_idx_t> adjncy_metis( adjncy, adjncy + nnz );

    // Prepare output arrays
    std::vector<metis_pid_t> iperm_metis( nrows );
    std::vector<metis_pid_t> perm_metis( nrows );

    metis_idx_t nvtxs = static_cast<metis_idx_t>( nrows );
    int result;

#ifdef USE_MTMETIS
    double options[MTMETIS_NOPTIONS];
    result = MTMETIS_NodeND( &nvtxs, xadj_metis.data(), adjncy_metis.data(), NULL, options,
                             perm_metis.data(), iperm_metis.data() );
#else
    std::vector<idx_t> options( METIS_NOPTIONS );
    METIS_SetDefaultOptions( options.data() );
    options[METIS_OPTION_NUMBERING] = 0; // Zero-based indexing
    options[METIS_OPTION_NSEPS] = opts.nseps;
    options[METIS_OPTION_NITER] = opts.niter;
    options[METIS_OPTION_SEED] = opts.seed;
    options[METIS_OPTION_COMPRESS] = opts.compress ? 1 : 0;
    options[METIS_OPTION_CCORDER] = opts.ccorder ? 1 : 0;
    options[METIS_OPTION_CTYPE] = opts.ctype;
    options[METIS_OPTION_RTYPE] = opts.rtype;
    options[METIS_OPTION_DBGLVL] = opts.dbglvl;

    result = METIS_NodeND( &nvtxs, xadj_metis.data(), adjncy_metis.data(), NULL, options.data(),
                           perm_metis.data(), iperm_metis.data() );
#endif

    // Convert results back to requested type
    for ( COLTYPE i = 0; i < nrows; ++i )
    {
        iperm[i] = static_cast<COLTYPE>( iperm_metis[i] );
        perm[i] = static_cast<COLTYPE>( perm_metis[i] );
    }

    return ( result == SUCCESS_CODE ) ? 0 : -1;
}
#endif

// Explicit template instantiations for common type combinations
template void NodeDegree<int, int>( int rows, const int* ai, int* degrees, int numthreads );
template void NodeDegree<long, int>( int rows, const long* ai, int* degrees, int numthreads );
template void NodeDegree<long, long>( long rows, const long* ai, long* degrees, int numthreads );

#ifdef USE_METIS_LIB
template int MetisND<int32_t, int32_t>( const int32_t,
                                        const int32_t,
                                        const int32_t*,
                                        const int32_t*,
                                        int32_t*,
                                        int32_t*,
                                        const MetisNDOptions& );
template int MetisND<int64_t, int64_t>( const int64_t,
                                        const int64_t,
                                        const int64_t*,
                                        const int64_t*,
                                        int64_t*,
                                        int64_t*,
                                        const MetisNDOptions& );
template int MetisND<int64_t, int32_t>( const int32_t,
                                        const int32_t,
                                        const int64_t*,
                                        const int32_t*,
                                        int32_t*,
                                        int32_t*,
                                        const MetisNDOptions& );
#endif
} // namespace reordering
