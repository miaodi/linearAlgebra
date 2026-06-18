#pragma once

#include "csr5_convert.hpp"
#include "csr5_spmv_kernel.hpp"
#include <vector>
#include <omp.h>

namespace matrix_utils
{

class CSR5SPMV
{
public:
    using Policy = CSR5AVX2DoublePolicy;
    using ROWTYPE = typename Policy::ROWTYPE;
    using COLTYPE = typename Policy::COLTYPE;
    using VALTYPE = typename Policy::VALTYPE;

    explicit CSR5SPMV( const int num_threads = omp_get_max_threads() )
        : _nthreads( csr5NormalizeThreadCount( num_threads ) ),
          _thread_boundary_sums( static_cast<std::size_t>( _nthreads ) )
    {
        rebuildTilePartitions();
    }

    void setNumThreads( const int num_threads )
    {
        _nthreads = csr5NormalizeThreadCount( num_threads );
        _thread_boundary_sums.resize( static_cast<std::size_t>( _nthreads ) );
        rebuildTilePartitions();
    }

    int numThreads() const { return _nthreads; }

    void preprocess( const COLTYPE size, ROWTYPE const* __restrict ai, COLTYPE const* __restrict aj, VALTYPE const* __restrict av )
    {
        convertCSRtoCSR5<Policy>( size, ai, aj, av, _csr5_data, _nthreads );
        _thread_boundary_sums.resize( static_cast<std::size_t>( _nthreads ) );
        rebuildTilePartitions();
    }

    const CSR5Data<Policy>& data() const { return _csr5_data; }

    void operator()( const VALTYPE* __restrict const b,
                     VALTYPE* __restrict const x,
                     const VALTYPE alpha = static_cast<VALTYPE>( 1 ),
                     const VALTYPE beta = static_cast<VALTYPE>( 0 ) ) const
    {
        ensureThreadBoundaryStorage();
        csr5Spmv( _csr5_data, b, x, alpha, beta, _nthreads, _thread_boundary_sums.data(),
                  static_cast<int>( _thread_boundary_sums.size() ), _tile_partitions.data(),
                  static_cast<int>( _tile_partitions.size() ) );
    }

private:
    void ensureThreadBoundaryStorage() const
    {
        if ( _thread_boundary_sums.size() < static_cast<std::size_t>( _nthreads ) )
        {
            _thread_boundary_sums.resize( static_cast<std::size_t>( _nthreads ) );
        }
        if ( _tile_partitions.size() < static_cast<std::size_t>( _nthreads ) )
        {
            rebuildTilePartitions();
        }
    }

    void rebuildTilePartitions() const
    {
        csr5BuildTilePartitions( _csr5_data._num_full_tiles, _nthreads, _tile_partitions );
    }

    int _nthreads;
    CSR5Data<Policy> _csr5_data;
    mutable std::vector<CSR5ThreadBoundaryContribution> _thread_boundary_sums;
    mutable std::vector<CSR5TilePartition> _tile_partitions;
};

} // namespace matrix_utils
