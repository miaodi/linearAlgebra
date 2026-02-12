#include "cuda_csr_utils.cuh"
#include "cuda_ilu_symbolic.cuh"
#include "cuda_spmm.cuh"
#include <cstdint>
#include <limits>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <type_traits>

namespace matrix_utils::sparse_cuda
{
namespace
{
/// Predicate: true if row > col (key = (row<<32)|col)
struct PackedCooRowGtCol
{
    __host__ __device__ bool operator()(uint64_t key) const
    {
        return (key >> 32) > (key & 0xFFFFFFFFu);
    }
};

/// Predicate: true if row <= col (upper triangle incl. diagonal)
struct PackedCooRowLeCol
{
    __host__ __device__ bool operator()(uint64_t key) const
    {
        return (key >> 32) <= (key & 0xFFFFFFFFu);
    }
};

/// Functor: swap (i,j) to (j,i) (key = (row<<32)|col)
struct PackedCooSwap
{
    __host__ __device__ uint64_t operator()(uint64_t key) const
    {
        uint32_t row = static_cast<uint32_t>(key >> 32);
        uint32_t col = static_cast<uint32_t>(key & 0xFFFFFFFFu);
        return (static_cast<uint64_t>(col) << 32) | row;
    }
};

// 1. diff = packed_coo \ packed_coo_prev; for (i,j) with i>j swap to (j,i); sort+unique; PackedCOOtoCSR -> frontier_matrix
// 2. packed_coo_prev_out = packed_coo ∪ packed_coo_prev
template <typename ROWTYPE, typename COLTYPE>
bool frontier_advance(const DeviceArray<uint64_t>& packed_coo, const DeviceArray<uint64_t>& packed_coo_prev,
                      COLTYPE n, ROWTYPE base, DeviceCSRMatrix<ROWTYPE, COLTYPE>& frontier_matrix,
                      DeviceArray<uint64_t>& packed_coo_prev_out)
{
    const size_t coo_sz = packed_coo.size();
    const size_t prev_sz = packed_coo_prev.size();

    if (coo_sz == 0)
    {
        frontier_matrix.n_rows = n;
        frontier_matrix.base = base;
        frontier_matrix.ai.resize(static_cast<size_t>(n) + 1);
        frontier_matrix.aj.resize(0);
        thrust::fill(thrust::device, frontier_matrix.ai.data(), frontier_matrix.ai.data() + (n + 1),
                     static_cast<ROWTYPE>(base));
        packed_coo_prev_out.resize(0);
        return true;
    }

    DeviceArray<uint64_t> d_diff;
    d_diff.resize(coo_sz);
    uint64_t* diff_end;
    if (prev_sz == 0)
    {
        thrust::copy(thrust::device, packed_coo.data(), packed_coo.data() + coo_sz, d_diff.data());
        diff_end = d_diff.data() + coo_sz;
    }
    else
    {
        diff_end = thrust::set_difference(thrust::device, packed_coo.data(),
                                          packed_coo.data() + coo_sz, packed_coo_prev.data(),
                                          packed_coo_prev.data() + prev_sz, d_diff.data());
    }
    const size_t diff_sz = static_cast<size_t>(diff_end - d_diff.data());
    d_diff.resize(diff_sz);

    // Filter: keep only (i,j) where i > j, then swap to (j,i)
    DeviceArray<uint64_t> d_lower;
    d_lower.resize(diff_sz);
    uint64_t* lower_end = thrust::copy_if(thrust::device, d_diff.data(), d_diff.data() + diff_sz,
                                          d_lower.data(), PackedCooRowGtCol{});
    const size_t lower_sz = static_cast<size_t>(lower_end - d_lower.data());
    d_lower.resize(lower_sz);

    if (lower_sz == 0)
    {
        frontier_matrix.n_rows = n;
        frontier_matrix.base = base;
        frontier_matrix.ai.resize(static_cast<size_t>(n) + 1);
        frontier_matrix.aj.resize(0);
        thrust::fill(thrust::device, frontier_matrix.ai.data(), frontier_matrix.ai.data() + (n + 1),
                     static_cast<ROWTYPE>(base));
    }
    else
    {
        thrust::transform(thrust::device, d_lower.data(), d_lower.data() + lower_sz, d_lower.data(),
                          PackedCooSwap{});
        thrust::sort(thrust::device, d_lower.data(), d_lower.data() + lower_sz);
        if (!PackedCOOtoCSR<ROWTYPE, COLTYPE>(d_lower.data(), static_cast<ROWTYPE>(lower_sz), n, base, frontier_matrix))
            return false;
    }

    DeviceArray<uint64_t> d_union;
    d_union.resize(coo_sz + prev_sz);
    uint64_t* union_end;
    if (prev_sz == 0)
    {
        thrust::copy(thrust::device, packed_coo.data(), packed_coo.data() + coo_sz, d_union.data());
        union_end = d_union.data() + coo_sz;
    }
    else
    {
        union_end = thrust::set_union(thrust::device, packed_coo.data(), packed_coo.data() + coo_sz,
                                      packed_coo_prev.data(), packed_coo_prev.data() + prev_sz,
                                      d_union.data());
    }
    const size_t union_sz = static_cast<size_t>(union_end - d_union.data());
    packed_coo_prev_out.resize(union_sz);
    thrust::copy(thrust::device, d_union.data(), d_union.data() + union_sz, packed_coo_prev_out.data());
    return true;
}
} // namespace

// ILUSymbolicU_SpMM_CUDA: U-row symbolic factorization using SpMM approach
template <typename ROWTYPE, typename COLTYPE>
bool ILUSymbolicU_SpMM_CUDA(COLTYPE n, const ROWTYPE* d_ai, const COLTYPE* d_aj, int lvl,
                            COLTYPE base, DeviceCSRMatrix<ROWTYPE, COLTYPE>& U_matrix)
{
    static_assert(std::is_same<ROWTYPE, int>::value || std::is_same<ROWTYPE, std::int64_t>::value,
                  "ROWTYPE must be int or int64_t");
    static_assert(std::is_same<COLTYPE, int>::value, "COLTYPE must be int");
    DeviceCSRMatrix<ROWTYPE, COLTYPE> frontier_matrix;
    DeviceArray<uint64_t> packed_coo;
    DeviceArray<uint64_t> packed_coo_prev;

    CSRDiagDevice<ROWTYPE, COLTYPE>(n, base, frontier_matrix);
    for (int k = 0; k <= lvl; ++k)
    {
        SpMMStruct<ROWTYPE, COLTYPE>(n, frontier_matrix.ai.data(), frontier_matrix.aj.data(), d_ai,
                                     d_aj, base, packed_coo);
        if (!frontier_advance<ROWTYPE, COLTYPE>(packed_coo, packed_coo_prev, n, base, frontier_matrix, packed_coo_prev))
        {
            return false;
        }
    }

    // Extract upper triangular part of packed_coo_prev and convert to CSR
    const size_t prev_sz = packed_coo_prev.size();
    if (prev_sz == 0)
    {
        U_matrix.n_rows = n;
        U_matrix.base = base;
        U_matrix.ai.resize(static_cast<size_t>(n) + 1);
        U_matrix.aj.resize(0);
        thrust::fill(thrust::device, U_matrix.ai.data(), U_matrix.ai.data() + (n + 1),
                     static_cast<ROWTYPE>(base));
        return true;
    }
    DeviceArray<uint64_t> d_upper;
    d_upper.resize(prev_sz);
    uint64_t* upper_end =
        thrust::copy_if(thrust::device, packed_coo_prev.data(), packed_coo_prev.data() + prev_sz,
                        d_upper.data(), PackedCooRowLeCol{});
    const size_t upper_sz = static_cast<size_t>(upper_end - d_upper.data());
    d_upper.resize(upper_sz);
    if (upper_sz == 0)
    {
        U_matrix.n_rows = n;
        U_matrix.base = base;
        U_matrix.ai.resize(static_cast<size_t>(n) + 1);
        U_matrix.aj.resize(0);
        thrust::fill(thrust::device, U_matrix.ai.data(), U_matrix.ai.data() + (n + 1),
                     static_cast<ROWTYPE>(base));
        return true;
    }
    return PackedCOOtoCSR<ROWTYPE, COLTYPE>(d_upper.data(), static_cast<ROWTYPE>(upper_sz), n, base, U_matrix);
}

template bool ILUSymbolicU_SpMM_CUDA<int, int>(int n, const int* d_ai, const int* d_aj, int lvl,
                                               int base, DeviceCSRMatrix<int, int>& U_matrix);
template bool ILUSymbolicU_SpMM_CUDA<std::int64_t, int>(int n, const std::int64_t* d_ai,
                                                        const int* d_aj, int lvl, int base,
                                                        DeviceCSRMatrix<std::int64_t, int>& U_matrix);
} // namespace matrix_utils::sparse_cuda
