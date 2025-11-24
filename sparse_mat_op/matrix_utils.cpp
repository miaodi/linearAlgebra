#include "matrix_utils.hpp"

namespace matrix_utils {
template class CSRMatrix<int, int, double>;
template class CSRMatrixVec<int, int, double>;

template void SerialTranspose<int, int, double>(const int rows, const int cols,
                                                const int base, int const *ai,
                                                int const *aj, double const *av,
                                                int *ai_transpose,
                                                int *aj_transpose,
                                                double *av_transpose);

template void ParallelTranspose<int, int, double>(
    const int rows, const int cols, const int base, int const *ai,
    int const *aj, double const *av, int *ai_transpose, int *aj_transpose,
    double *av_transpose);

template void ParallelTranspose2<int, int, double>(
    const int rows, const int cols, const int base, int const *ai,
    int const *aj, double const *av, int *ai_transpose, int *aj_transpose,
    double *av_transpose);

template void permutedAI<int, int>(const int rows, const int base,
                                   int const *ai, int const *iperm,
                                   int *permed_ai);

template void permute<int, int, double>(const int rows, const int base,
                                        int const *ai, int const *aj,
                                        double const *av, int const *iperm,
                                        int const *perm, int *permed_ai,
                                        int *permed_aj, double *permed_av);

template void permuteRow<int, int, double>(const int rows, const int base,
                                           int const *ai, int const *aj,
                                           double const *av, int const *iperm,
                                           int *permed_ai, int *permed_aj,
                                           double *permed_av);

template void symPermute<int, int, double>(const int rows, const int base,
                                           int const *ai, int const *aj,
                                           double const *av, int const *iperm,
                                           int *permed_ai, int *permed_aj,
                                           double *permed_av);

template <typename ROWTYPE, typename COLTYPE>
COLTYPE KahnSerial<ROWTYPE, COLTYPE>::operator()( const COLTYPE nodes,
                                                      ROWTYPE const* ai,
                                                      COLTYPE const* aj,
                                                      COLTYPE* perm,
                                                      COLTYPE* prefix,
                                                      bool has_diagonal )
{
    _degrees.resize( nodes );
    const auto base = ai[0];
    const auto nnz = ai[nodes] - base;

    _t_ai.resize( nodes + 1 );
    _t_aj.resize( nnz );
    COLTYPE processed = 0;
    COLTYPE level = 0;

    // reverse graph to get out edges
    ParallelTranspose2( nodes, nodes, base, ai, aj, (double*)nullptr,
                        _t_ai.data(), _t_aj.data(), (double*)nullptr );

    prefix[0] = base;

    auto degree_for = [&]( const COLTYPE i )
    {
        COLTYPE degree = ai[i + 1] - ai[i];
        if ( has_diagonal && degree > 0 )
        {
            --degree;
        }
        return degree;
    };

    for ( COLTYPE i = 0; i < nodes; ++i )
    {
        _degrees[i] = degree_for( i );
        if ( _degrees[i] == 0 )
        {
            perm[processed++] = i + base;
        }
    }

    prefix[1] = processed + base;

    auto process_row = [&]( const COLTYPE idx, const auto& handle_neighbor )
    {
        const auto row_start = _t_ai[idx] - base;
        const auto row_end = _t_ai[idx + 1] - base;
        for ( auto pos = row_start; pos < row_end; ++pos )
        {
            if ( has_diagonal )
            {
                if ( _t_aj[pos] - base == idx )
                {
                    continue;
                }
            }
            handle_neighbor( _t_aj[pos] );
        }
    };

    // process levels
    while ( processed != nodes )
    {
        for ( COLTYPE i = prefix[level] - base; i < prefix[level + 1] - base; ++i )
        {
            const auto idx = perm[i] - base;
            process_row( idx,
                         [&]( const COLTYPE neighbor )
                         {
                             if ( --_degrees[neighbor - base] == 0 )
                             {
                                 perm[processed++] = neighbor;
                             }
                         } );
        }
        ++level;
        prefix[level + 1] = processed + base;
    }

    return level + 1;
}

template <typename ROWTYPE, typename COLTYPE>
COLTYPE KahnParallel<ROWTYPE, COLTYPE>::operator()( const COLTYPE nodes,
                                                        ROWTYPE const* ai,
                                                        COLTYPE const* aj,
                                                        COLTYPE* perm,
                                                        COLTYPE* prefix,
                                                        bool has_diagonal )
{
    if ( _degrees_size < nodes )
    {
        _degrees.reset( new std::atomic<COLTYPE>[nodes] );
        _degrees_size = nodes;
    }
    const auto base = ai[0];
    const auto nnz = ai[nodes] - base;

    _t_ai.resize( nodes + 1 );
    _t_aj.resize( nnz );
    COLTYPE processed = 0;
    COLTYPE level = 0;

    // reverse graph
    ParallelTranspose2( nodes, nodes, base, ai, aj, (double*)nullptr,
                        _t_ai.data(), _t_aj.data(), (double*)nullptr );

    prefix[0] = base;
    
    auto degree_for = [&]( const COLTYPE i )
    {
        COLTYPE degree = ai[i + 1] - ai[i];
        if ( has_diagonal && degree > 0 )
        {
            --degree;
        }
        return degree;
    };

    auto process_row = [&]( const COLTYPE idx, const auto& handle_neighbor )
    {
        const auto row_start = _t_ai[idx] - base;
        const auto row_end = _t_ai[idx + 1] - base;
        for ( auto pos = row_start; pos < row_end; ++pos )
        {
            if( has_diagonal )
            {
                if ( _t_aj[pos] - base == idx )
                {
                    continue;
                }
            }
            handle_neighbor( _t_aj[pos] );
        }
    };

#pragma omp parallel num_threads( _nthreads )
    {
        const int thread_id = omp_get_thread_num();
        _threads_nodes[thread_id].clear();
        _threads_prefix[thread_id + 1] = 0;

        auto chunk_begin = thread_id * nodes / _nthreads;
        auto chunk_end = ( thread_id + 1 ) * nodes / _nthreads;

        for ( COLTYPE i = chunk_begin; i < chunk_end; ++i )
        {
            _degrees[i] =  degree_for( i );
            if ( _degrees[i] == 0 )
            {
                _threads_nodes[thread_id].push_back( i + base );
            }
        }
        _threads_prefix[thread_id + 1] = _threads_nodes[thread_id].size();
#pragma omp barrier
#pragma omp single
        {
            _threads_prefix[0] = base;
            for ( size_t i = 1; i < _threads_prefix.size(); ++i )
            {
                _threads_prefix[i] += _threads_prefix[i - 1];
            }
            prefix[1] = _threads_prefix[_nthreads];
            processed = _threads_prefix[_nthreads] - base;
        }
        auto thread_start = _threads_prefix[thread_id] - base;
        for ( const auto node : _threads_nodes[thread_id] )
        {
            perm[thread_start++] = node;
        }
#pragma omp barrier
        while ( processed != nodes )
        {
            _threads_prefix[thread_id + 1] = 0;
            _threads_nodes[thread_id].clear();
            auto start_range = prefix[level] - base +
                               ( prefix[level + 1] - prefix[level] ) * thread_id / _nthreads;
            auto end_range =
                prefix[level] - base +
                ( prefix[level + 1] - prefix[level] ) * ( thread_id + 1 ) / _nthreads;

            for ( COLTYPE idx_pos = start_range; idx_pos < end_range; ++idx_pos )
            {
                const auto idx = perm[idx_pos] - base;
                process_row( idx,
                             [&]( const COLTYPE neighbor )
                             {
                                 if ( _degrees[neighbor - base].fetch_sub(
                                          1, std::memory_order_relaxed ) == 1 )
                                 {
                                     _threads_nodes[thread_id].push_back( neighbor );
                                 }
                             } );
            }
            _threads_prefix[thread_id + 1] = _threads_nodes[thread_id].size();
#pragma omp barrier
#pragma omp single
            {
                _threads_prefix[0] = processed + base;
                for ( size_t i = 1; i < _threads_prefix.size(); ++i )
                {
                    _threads_prefix[i] += _threads_prefix[i - 1];
                }
                ++level;
                prefix[level + 1] = _threads_prefix[_nthreads];
                processed = _threads_prefix[_nthreads] - base;
            }
            auto local_start = _threads_prefix[thread_id] - base;
            for ( const auto node : _threads_nodes[thread_id] )
            {
                perm[local_start++] = node;
            }
#pragma omp barrier
        }
    }

    return level + 1;
}

template <typename ROWTYPE, typename COLTYPE, TriangularMatrix TS>
COLTYPE TopologicalSort2<ROWTYPE, COLTYPE, TS>::operator()( const COLTYPE nodes,
                                                            ROWTYPE const* ai,
                                                            COLTYPE const* aj,
                                                            COLTYPE* perm,
                                                            COLTYPE* prefix,
                                                            bool has_diagonal )
{
    _degrees.resize( nodes );
    std::fill( _degrees.begin(), _degrees.end(), 0 );

    const auto base = ai[0];

    COLTYPE level = 0;
    for ( COLTYPE offset = 0; offset < nodes; ++offset )
    {
        COLTYPE i = offset;
        if constexpr ( TS == TriangularMatrix::U )
        {
            i = nodes - 1 - offset;
        }
        auto row_begin = ai[i] - base;
        auto row_end = ai[i + 1] - base;
        if ( has_diagonal && row_end > row_begin )
        {
            if constexpr ( TS == TriangularMatrix::U )
            {
                ++row_begin;
            }
            else
            {
                --row_end;
            }
        }
        for ( auto j = row_begin; j < row_end; ++j )
        {
            _degrees[i] = std::max(
                _degrees[i], static_cast<COLTYPE>( _degrees[aj[j] - base] + 1 ) );
        }
        level = std::max( level, _degrees[i] + 1 );
    }
    std::fill( prefix, prefix + level + 1, 0 );
    prefix[0] = base;
    for ( COLTYPE i = 0; i < nodes; i++ )
    {
        prefix[_degrees[i] + 1]++;
    }
    std::inclusive_scan( prefix, prefix + level + 1, prefix );

    for ( COLTYPE i = 0; i < nodes; i++ )
    {
        perm[prefix[_degrees[i]]++ - base] = i + base;
    }
    for ( COLTYPE i = level; i > 0; i-- )
    {
        prefix[i] = prefix[i - 1];
    }
    prefix[0] = base;
    return level;
}

template bool Diagonal<int, int, double>(const int rows, int const *ai,
                                         int const *aj, double const *av,
                                         int *diagpos, double *diag,
                                         const bool invert);

template void SplitLDU(const int rows, const int base, int const *ai,
                       int const *aj, double const *av,
                       CSRMatrix<int, int, double> &L, std::vector<double> &D,
                       CSRMatrix<int, int, double> &U);

template <ResizableCSRMatrixType CSRMatrixType>
void SplitLU<CSRMatrixType>::operator()(const COLTYPE rows, ROWTYPE const *ai,
                                        ROWTYPE const *diag, COLTYPE const *aj,
                                        VALTYPE const *av, CSRMatrixType &L,
                                        CSRMatrixType &U) {
  const auto base = ai[0];
  L.rows = rows;
  L.cols = rows;
  L.ResizeAI(rows + 1);
  auto L_ai = L.AI();
  L_ai[0] = base;

  U.rows = rows;
  U.cols = rows;
  U.ResizeAI(rows + 1);
  auto U_ai = U.AI();
  U_ai[0] = base;

  prefixL[0] = base;
  prefixU[0] = base;

#pragma omp parallel num_threads(num_threads)
  {
    const int tid = omp_get_thread_num();
    auto [start, end] =
        utils::LoadPrefixBalancedPartitionPos(ai, ai + rows, tid, num_threads);
    prefixL[tid + 1] = 0;
    prefixU[tid + 1] = 0;
    for (auto i = start; i < end; i++) {
      prefixL[tid + 1] += diag[i] - ai[i];
      prefixU[tid + 1] += ai[i + 1] - diag[i];
      L_ai[i + 1] = prefixL[tid + 1];
      U_ai[i + 1] = prefixU[tid + 1];
    }
#pragma omp barrier
#pragma omp single
    {
      for (size_t i = 1; i < prefixL.size(); i++) {
        prefixL[i] += prefixL[i - 1];
        prefixU[i] += prefixU[i - 1];
      }
      const auto L_nnz = prefixL[num_threads] - base;
      const auto U_nnz = prefixU[num_threads] - base;
      L.ResizeAJ(L_nnz);
      L.ResizeAV(L_nnz);
      U.ResizeAJ(U_nnz);
      U.ResizeAV(U_nnz);
    }

    auto L_pos = prefixL[tid] - base;
    auto U_pos = prefixU[tid] - base;
    for (auto i = start; i < end; i++) {
      L_ai[i + 1] += prefixL[tid];
      U_ai[i + 1] += prefixU[tid];

      for (auto j = ai[i]; j < diag[i]; j++) {
        L.AJ()[L_pos] = aj[j];
        L.AV()[L_pos++] = av[j];
      }
      for (auto j = diag[i]; j < ai[i + 1]; j++) {
        U.AJ()[U_pos] = aj[j];
        U.AV()[U_pos++] = av[j];
      }
    }
  }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void Prune(const COLTYPE rows, ROWTYPE* ai, COLTYPE* aj, VALTYPE* av, const VALTYPE threshold,
           VALTYPE const* row_thresholds)
{
    const ROWTYPE base = ai[0];
    const ROWTYPE old_nnz = ai[rows] - base;

    // Store original ai values before modifying
    std::vector<ROWTYPE> old_ai(rows + 1);
    std::memcpy(old_ai.data(), ai, (rows + 1) * sizeof(ROWTYPE));

    std::vector<ROWTYPE> new_row_sizes(rows);
    std::vector<ROWTYPE> thread_prefix(omp_get_max_threads() + 1, 0);
    std::vector<VALTYPE> av_tmp(old_nnz);
    std::vector<COLTYPE> aj_tmp(old_nnz);

    auto get_threshold = [&](COLTYPE row)
    { return row_thresholds ? row_thresholds[row] : threshold; };

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int num_threads = omp_get_num_threads();
        auto [start, end] =
            utils::LoadPrefixBalancedPartitionPos(old_ai.data(), old_ai.data() + rows, tid, num_threads);

        // Phase 1: Filter and count surviving entries per row
        ROWTYPE local_nnz = 0;
        for (COLTYPE i = start; i < end; i++)
        {
            ROWTYPE row_size = 0;
            const ROWTYPE row_start = old_ai[i] - base;
            const ROWTYPE row_end = old_ai[i + 1] - base;
            const auto row_threshold = get_threshold(i);

            for (ROWTYPE j = row_start; j < row_end; j++)
            {
                if (std::abs(av[j]) > row_threshold)
                {
                    aj_tmp[row_start + row_size] = aj[j];
                    av_tmp[row_start + row_size] = av[j];
                    row_size++;
                }
            }
            new_row_sizes[i] = row_size;
            local_nnz += row_size;
        }
        thread_prefix[tid + 1] = local_nnz;

#pragma omp barrier
#pragma omp single
        {
            // Compute thread prefix sums
            for (size_t i = 1; i < thread_prefix.size(); i++)
            {
                thread_prefix[i] += thread_prefix[i - 1];
            }
        }

        // Phase 2: Compute new ai array (CSR row pointers)
        ROWTYPE row_offset = thread_prefix[tid] + base;
        for (COLTYPE i = start; i < end; i++)
        {
            row_offset += new_row_sizes[i];
            ai[i + 1] = row_offset;
        }

#pragma omp barrier

        // Phase 3: Copy filtered data to final positions
        for (COLTYPE i = start; i < end; i++)
        {
            const ROWTYPE old_row_start = old_ai[i] - base;
            const ROWTYPE new_row_start = ai[i] - base;
            const ROWTYPE row_size = new_row_sizes[i];

            if (row_size > 0)
            {
                std::memcpy(aj + new_row_start, aj_tmp.data() + old_row_start, row_size * sizeof(COLTYPE));
                std::memcpy(av + new_row_start, av_tmp.data() + old_row_start, row_size * sizeof(VALTYPE));
            }
        }
    }
}

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void DiagonalScaledPrune(const COLTYPE rows, ROWTYPE* ai, COLTYPE* aj, VALTYPE* av, const VALTYPE threshold)
{
    const ROWTYPE base = ai[0];

    // Step 1: Extract diagonal values
    std::vector<VALTYPE> diag(rows, 0.0);

#pragma omp parallel for
    for (COLTYPE i = 0; i < rows; ++i)
    {
        const ROWTYPE row_start = ai[i] - base;
        const ROWTYPE row_end = ai[i + 1] - base;
        const COLTYPE diag_col = i + base;

        // Binary search for diagonal element
        auto it = std::lower_bound(aj + row_start, aj + row_end, diag_col);
        if (it != aj + row_end && *it == diag_col)
        {
            diag[i] = av[it - aj];
        }
    }

    // Step 2: Zero out entries where |a_ii| * |a_jj| * threshold < |a_ij|
#pragma omp parallel for
    for (COLTYPE i = 0; i < rows; ++i)
    {
        const VALTYPE abs_diag_i = std::abs(diag[i]);
        for (ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; ++j)
        {
            const COLTYPE col = aj[j] - base;
            if (col != i) // Skip diagonal
            {
                const VALTYPE abs_diag_j = std::abs(diag[col]);
                const VALTYPE threshold_ij = abs_diag_i * abs_diag_j * threshold;
                if (av[j] * av[j] < threshold_ij)
                {
                    av[j] = 0.0;
                }
            }
        }
    }

    // Step 3: Prune zeros
    Prune(rows, ai, aj, av, static_cast<VALTYPE>(0.0), (VALTYPE*)nullptr);
}

template void SplitTriangle<TriangularMatrix::U, int, int, double,
                            CSRMatrix<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    double const *av, CSRMatrix<int, int, double> &U);

template void SplitTriangle<TriangularMatrix::U, int, int, double,
                            CSRMatrixVec<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    double const *av, CSRMatrixVec<int, int, double> &U);

template void SplitTriangle<TriangularMatrix::L, int, int, double,
                            CSRMatrix<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    double const *av, CSRMatrix<int, int, double> &U);

template void SplitTriangle<TriangularMatrix::L, int, int, double,
                            CSRMatrixVec<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    double const *av, CSRMatrixVec<int, int, double> &U);

template void TriangularToFull<TriangularMatrix::U, int, int, double,
                               CSRMatrix<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    double const *av, CSRMatrix<int, int, double> &F);

template void TriangularToFull<TriangularMatrix::U, int, int, double,
                               CSRMatrixVec<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    double const *av, CSRMatrixVec<int, int, double> &F);


#define INSTANTIATE_TOPOLOGICAL_SORT(ROWTYPE, COLTYPE)                       \
    template struct KahnSerial<ROWTYPE, COLTYPE>;                            \
    template struct KahnParallel<ROWTYPE, COLTYPE>;                          \
    template struct TopologicalSort2<ROWTYPE, COLTYPE, TriangularMatrix::L>; \
    template struct TopologicalSort2<ROWTYPE, COLTYPE, TriangularMatrix::U>;

INSTANTIATE_TOPOLOGICAL_SORT(std::int32_t, std::int32_t)
INSTANTIATE_TOPOLOGICAL_SORT(std::int64_t, std::int64_t)

#define INSTANTIATE_SPLIT_LU(ROWTYPE, COLTYPE, VALTYPE)                        \
  template struct SplitLU<CSRMatrix<ROWTYPE, COLTYPE, VALTYPE>>;               \
  template struct SplitLU<CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE>>;

INSTANTIATE_SPLIT_LU(std::int32_t, std::int32_t, double)
INSTANTIATE_SPLIT_LU(int, int, float)

#define INSTANTIATE_MATRIX_OPS(ROWTYPE, COLTYPE, VALTYPE)                     \
  template void Prune<ROWTYPE, COLTYPE, VALTYPE>(                             \
      const COLTYPE rows, ROWTYPE* ai, COLTYPE* aj, VALTYPE* av,              \
      const VALTYPE threshold, VALTYPE const* row_thresholds);                 \
  template void DiagonalScaledPrune<ROWTYPE, COLTYPE, VALTYPE>(               \
      const COLTYPE rows, ROWTYPE* ai, COLTYPE* aj, VALTYPE* av,              \
      const VALTYPE threshold);

INSTANTIATE_MATRIX_OPS(int, int, double)
INSTANTIATE_MATRIX_OPS(int, int, float)
INSTANTIATE_MATRIX_OPS(std::int64_t, std::int64_t, double)

} // namespace matrix_utils
