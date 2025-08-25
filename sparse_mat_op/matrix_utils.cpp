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
COLTYPE KahnSerial<ROWTYPE, COLTYPE>::operator()(
    const TriangularMatrix TS, const COLTYPE nodes, ROWTYPE const *ai,
    COLTYPE const *aj, COLTYPE *perm, COLTYPE *prefix) {
  _degrees.resize(nodes);
  const auto base = ai[0];
  const auto nnz = ai[nodes] - base;

  _t_ai.resize(nodes + 1);
  _t_aj.resize(nnz);
  COLTYPE processed = 0;
  COLTYPE level = 0;

  //   reverse graph to get out edges
  ParallelTranspose2(nodes, nodes, base, ai, aj, (double *)nullptr,
                     _t_ai.data(), _t_aj.data(), (double *)nullptr);

  if (TS == TriangularMatrix::L) {
    _start = 0;
    _end = nodes;
    _inc = 1;
  } else {
    _start = nodes - 1;
    _end = -1;
    _inc = -1;
  }
  prefix[0] = base;

  //   node in-degrees and prepare level 0 nodes
  for (COLTYPE i = _start; i != _end; i += _inc) {
    _degrees[i] = ai[i + 1] - ai[i] - (TS == TriangularMatrix::U);
    if (_degrees[i] == 0) {
      perm[processed++] = i + base;
    }
  }

  prefix[1] = processed + base;

  //   process levels
  while (processed != nodes) {
    for (COLTYPE i = prefix[level] - base; i < prefix[level + 1] - base; i++) {
      const auto idx = perm[i] - base;
      for (auto j = _t_ai[idx] - base + (TS == TriangularMatrix::U);
           j < _t_ai[idx + 1] - base; j++) {
        if (--_degrees[_t_aj[j] - base] == 0) {
          perm[processed++] = _t_aj[j];
        }
      }
    }
    prefix[++level + 1] = processed + base;
  }

  return level + 1;
}

template <typename ROWTYPE, typename COLTYPE>
COLTYPE KahnParallel<ROWTYPE, COLTYPE>::operator()(
    const TriangularMatrix TS, const COLTYPE nodes, ROWTYPE const *ai,
    COLTYPE const *aj, COLTYPE *perm, COLTYPE *prefix) {
  if (_degrees_size < nodes) {
    _degrees.reset(new std::atomic<COLTYPE>[nodes]);
    _degrees_size = nodes;
  }
  const auto base = ai[0];
  const auto nnz = ai[nodes] - base;

  _t_ai.resize(nodes + 1);
  _t_aj.resize(nnz);
  COLTYPE processed = 0;
  COLTYPE level = 0;

  //   reverse graph
  ParallelTranspose2(nodes, nodes, base, ai, aj, (double *)nullptr,
                     _t_ai.data(), _t_aj.data(), (double *)nullptr);

  if (TS == TriangularMatrix::L) {
    _start = 0;
    _end = nodes;
    _inc = 1;
  } else {
    _start = nodes - 1;
    _end = -1;
    _inc = -1;
  }
  prefix[0] = base;
  _threads_prefix[0] = base;
//   node degrees and prepare level 0 nodes
#pragma omp parallel num_threads(_nthreads)
  {

    const int thread_id = omp_get_thread_num();
    _threads_nodes[thread_id].clear();
    _threads_prefix[thread_id + 1] = 0;
    auto _start_thread = _start + thread_id * (_end - _start) / _nthreads;
    auto _end_thread = (_start + (thread_id + 1) * (_end - _start) / _nthreads);
    for (COLTYPE i = _start_thread; i != _end_thread; i += _inc) {
      _degrees[i].store(ai[i + 1] - ai[i] - (TS == TriangularMatrix::U),
                        std::memory_order_relaxed);
      if (_degrees[i].load(std::memory_order_relaxed) == 0) {
        _threads_nodes[thread_id].push_back(i + base);
      }
    }
    _threads_prefix[thread_id + 1] = _threads_nodes[thread_id].size();
#pragma omp barrier
#pragma omp single
    {
      for (size_t i = 1; i < _threads_prefix.size(); i++) {
        _threads_prefix[i] += _threads_prefix[i - 1];
      }
      prefix[1] = _threads_prefix[_nthreads];
      processed = _threads_prefix[_nthreads] - base;
    }
    auto thread_start = _threads_prefix[thread_id] - base;
    for (const auto i : _threads_nodes[thread_id]) {
      perm[thread_start++] = i;
    }

#pragma omp barrier
#pragma omp single
    { _threads_prefix[0] = _threads_prefix[_nthreads]; }
    while (processed != nodes) {
      _threads_prefix[thread_id + 1] = 0;
      _threads_nodes[thread_id].clear();
      auto _start_thread =
          prefix[level] - base +
          (prefix[level + 1] - prefix[level]) * thread_id / _nthreads;
      auto _end_thread =
          prefix[level] - base +
          (prefix[level + 1] - prefix[level]) * (thread_id + 1) / _nthreads;

      for (COLTYPE i = _start_thread; i < _end_thread; i++) {
        const auto idx = perm[i] - base;
        for (auto j = _t_ai[idx] - base + (TS == TriangularMatrix::U);
             j < _t_ai[idx + 1] - base; j++) {
          if (_degrees[_t_aj[j] - base].fetch_sub(
                  1, std::memory_order_relaxed) == 1) {
            _threads_nodes[thread_id].push_back(_t_aj[j]);
          }
        }
      }
      _threads_prefix[thread_id + 1] = _threads_nodes[thread_id].size();
#pragma omp barrier
#pragma omp single
      {
        for (size_t i = 1; i < _threads_prefix.size(); i++) {
          _threads_prefix[i] += _threads_prefix[i - 1];
        }
        prefix[++level + 1] = _threads_prefix[_nthreads];
        processed = _threads_prefix[_nthreads] - base;
      }
      auto thread_start = _threads_prefix[thread_id] - base;
      for (const auto i : _threads_nodes[thread_id]) {
        perm[thread_start++] = i;
      }
#pragma omp barrier
#pragma omp single
      { _threads_prefix[0] = _threads_prefix[_nthreads]; }
    }
  }

  return level + 1;
}

template <typename ROWTYPE, typename COLTYPE>
COLTYPE TopologicalSort2<ROWTYPE, COLTYPE>::operator()(
    const TriangularMatrix TS, const COLTYPE nodes, ROWTYPE const *ai,
    COLTYPE const *aj, COLTYPE *perm, COLTYPE *prefix) {
  _degrees.resize(nodes);
  std::fill(_degrees.begin(), _degrees.end(), 0);

  const auto base = ai[0];

  COLTYPE start, end, inc;
  if (TS == L) {
    start = 0;
    end = nodes;
    inc = 1;
  } else {
    start = nodes - 1;
    end = -1;
    inc = -1;
  }

  COLTYPE level = 0;
  for (COLTYPE i = start; i != end; i += inc) {
    for (auto j = ai[i] - base + (TS == TriangularMatrix::U);
         j < ai[i + 1] - base; j++) {
      _degrees[i] = std::max(_degrees[i], _degrees[aj[j] - base] + 1);
    }
    level = std::max(level, _degrees[i] + 1);
  }
  std::fill(prefix, prefix + level + 1, 0);
  prefix[0] = base;
  for (COLTYPE i = 0; i < nodes; i++) {
    prefix[_degrees[i] + 1]++;
  }
  std::inclusive_scan(prefix, prefix + level + 1, prefix);

  for (COLTYPE i = 0; i < nodes; i++) {
    perm[prefix[_degrees[i]]++ - base] = i + base;
  }
  for (COLTYPE i = level; i > 0; i--) {
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

template void Block<int, int, double, CSRMatrixVec<int, int, double>>(
    const int rows, const int base, int const *ai, int const *aj,
    double const *av, const int i, const int j, const int p, const int q,
    CSRMatrixVec<int, int, double> &);

#define INSTANTIATE_TOPOLOGICAL_SORT(ROWTYPE, COLTYPE)                         \
  template struct KahnSerial<ROWTYPE, COLTYPE>;                                \
  template struct KahnParallel<ROWTYPE, COLTYPE>;                              \
  template struct TopologicalSort2<ROWTYPE, COLTYPE>;

INSTANTIATE_TOPOLOGICAL_SORT(std::int32_t, std::int32_t)
INSTANTIATE_TOPOLOGICAL_SORT(std::int64_t, std::int64_t)

#define INSTANTIATE_SPLIT_LU(ROWTYPE, COLTYPE, VALTYPE)                        \
  template struct SplitLU<CSRMatrix<ROWTYPE, COLTYPE, VALTYPE>>;               \
  template struct SplitLU<CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE>>;

INSTANTIATE_SPLIT_LU(std::int32_t, std::int32_t, double)
INSTANTIATE_SPLIT_LU(int, int, float)

} // namespace matrix_utils