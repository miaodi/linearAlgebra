#pragma once

#include "csr5_convert.hpp"
#include "csr5_spmv_kernel.hpp"
#include <omp.h>

namespace matrix_utils
{

template <typename ROWTYPE = int, typename COLTYPE = int, typename VALTYPE = double, typename Policy = CSR5_AVX2_Policy<VALTYPE>>
class CSR5SPMV
{
public:
    explicit CSR5SPMV( const int num_threads = omp_get_max_threads() )
        : _nthreads( csr5NormalizeThreadCount( num_threads ) )
    {
    }

    void setNumThreads( const int num_threads )
    {
        _nthreads = csr5NormalizeThreadCount( num_threads );
    }

    int numThreads() const { return _nthreads; }

    void preprocess( const COLTYPE size, ROWTYPE const* __restrict ai, COLTYPE const* __restrict aj, VALTYPE const* __restrict av )
    {
        convertCSRtoCSR5<ROWTYPE, COLTYPE, VALTYPE, Policy>( size, ai, aj, av, _csr5_data, _nthreads );
    }

    const CSR5Data<ROWTYPE, COLTYPE, VALTYPE, Policy>& data() const { return _csr5_data; }

    void operator()( const VALTYPE* __restrict const b,
                     VALTYPE* __restrict const x,
                     const VALTYPE alpha = static_cast<VALTYPE>( 1 ),
                     const VALTYPE beta = static_cast<VALTYPE>( 0 ) ) const
    {
        csr5Spmv<ROWTYPE, COLTYPE, VALTYPE, Policy>( _csr5_data, b, x, alpha, beta, _nthreads );
    }

private:
    int _nthreads;
    CSR5Data<ROWTYPE, COLTYPE, VALTYPE, Policy> _csr5_data;
};

} // namespace matrix_utils
