#pragma once

#include "BitVector.hpp"
#include "utils.h"
#include <chrono>
#include <execution>
#include <fstream>
#include <memory>
#include <numeric>
#include <omp.h>
#include <span>
#include <tuple>
#include <type_traits>

#include "matrix_utils.hpp"

namespace matrix_utils
{

/// @brief Combined triangular solve function using TriangularMatrix enum with standard CSR format
/// @tparam TM TriangularMatrix::L for forward substitution, TriangularMatrix::U for backward substitution
/// @tparam ROWTYPE Row pointer type
/// @tparam COLTYPE Column index type
/// @tparam VALTYPE Value type
/// @param size Matrix dimension
/// @param ai Row pointers (ai[i] to ai[i+1]-1 are indices for row i, ai[0] is base)
/// @param aj Column indices
/// @param av Matrix values
/// @param diag Diagonal values (nullptr for unit diagonal)
/// @param b Right-hand side vector
/// @param x Solution vector
template <TriangularMatrix TM, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
void TriangularSolve( const COLTYPE size,
                      ROWTYPE const* ai,
                      COLTYPE const* aj,
                      VALTYPE const* av,
                      VALTYPE const* diag,
                      VALTYPE const* const b,
                      VALTYPE* const x );

/// @brief Level-scheduled triangular substitution with OpenMP parallelization
/// @tparam TM TriangularMatrix::L for forward substitution, TriangularMatrix::U for backward substitution
/// @tparam ROWTYPE Row pointer type
/// @tparam COLTYPE Column index type
/// @tparam VALTYPE Value type
/// @param iperm Permutation array for level scheduling
/// @param prefix Level prefix array (prefix[l] to prefix[l+1]-1 are nodes in level l)
/// @param lvls Number of levels
/// @param rows Number of rows
/// @param ai Row pointers
/// @param aj Column indices
/// @param av Matrix values
/// @param diag Diagonal values (required for backward substitution, ignored for forward)
/// @param b Right-hand side vector
/// @param x Solution vector
template <TriangularMatrix TM, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
struct LevelScheduleTriangularSubstitution
{
    LevelScheduleTriangularSubstitution( const int num_threads = omp_get_max_threads() )
        : _nthreads{ num_threads }
    {
    }
    void analysis( const COLTYPE size,
                   ROWTYPE const* ai,
                   COLTYPE const* aj,
                   VALTYPE const* av,
                   VALTYPE const* diag = nullptr );

    void operator()( VALTYPE const* const b, VALTYPE* const x ) const;
    void set_num_threads( const int num_threads )
    {
        _nthreads = num_threads;
    }

    int _nthreads;
    std::vector<COLTYPE> _iperm;
    std::vector<COLTYPE> _levelPrefix;
    COLTYPE _levels;
    TopologicalSort2<ROWTYPE, COLTYPE, TM> _topSort;

    ROWTYPE const* _ai;
    COLTYPE const* _aj;
    VALTYPE const* _av;
    VALTYPE const* _diag{ nullptr };
    COLTYPE _size{ 0 };
};

template <TriangularMatrix TM, typename ROWTYPE, typename COLTYPE, typename VALTYPE>
struct P2PTriangularSubstitution
{
    P2PTriangularSubstitution( const int num_threads = omp_get_max_threads() )
        : _nthreads{ num_threads }
    {
    }
    
    void analysis( const COLTYPE size,
                   ROWTYPE const* ai,
                   COLTYPE const* aj,
                   VALTYPE const* av,
                   VALTYPE const* diag = nullptr );

    // void operator()( VALTYPE const* const b, VALTYPE* const x ) const;
    
    int _nthreads;
    std::vector<COLTYPE> _iperm;
    std::vector<COLTYPE> _levelPrefix;
    COLTYPE _levels;
    TopologicalSort2<ROWTYPE, COLTYPE, TM> _topSort;
};

enum class FBSubstitutionType
{
    Barrier,
    NoBarrier,
    NoBarrierSuperNode
};

template <FBSubstitutionType FBST = FBSubstitutionType::Barrier,
          TriangularMatrix TS = TriangularMatrix::L,
          typename ROWTYPE = int,
          typename COLTYPE = int,
          typename VALTYPE = double>
class OptimizedTriangularSolve
{
public:
    OptimizedTriangularSolve( const int num_threads = omp_get_num_threads() )
        : _nthreads{ num_threads }
    {
    }

    void analysis( const COLTYPE rows,
                   const int base,
                   ROWTYPE const* ai,
                   COLTYPE const* aj,
                   VALTYPE const* av,
                   VALTYPE const* diag = nullptr );

    void operator()( const VALTYPE* const b, VALTYPE* const x ) const;

    void BarrierOp( const VALTYPE* const b, VALTYPE* const x ) const;

    void NoBarrierOp( const VALTYPE* const b, VALTYPE* const x ) const;

    void NoBarrierSuperNodeOp( const VALTYPE* const b, VALTYPE* const x ) const;

    void build_task_graph();

    int get_num_threads() const
    {
        return _nthreads;
    }

protected:
    int _nthreads;
    COLTYPE _size;
    std::vector<COLTYPE> _iperm;
    std::vector<COLTYPE> _levelPrefix;
    mutable std::vector<double> _vec;

    COLTYPE _levels;
    std::vector<std::vector<COLTYPE>> _threadlevels; // level prefix for each thread, zero based
    std::vector<COLTYPE> _threadiperm;
    std::vector<COLTYPE> _threadperm;
    CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE> _reorderedMat;
    VALTYPE const* _diag{ nullptr };

    // super node level scheduling data
    // always zero based
    COLTYPE _tasks;
    CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE> _taskAdjGraph;    // children
    CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE> _taskInvAdjGraph; // parents
    CSRMatrixVec<ROWTYPE, COLTYPE, VALTYPE> _taskInvAdjGraph2; // parents after transisive edge removal
    std::vector<COLTYPE> _threadTaskPrefix; // tasks on each thread
    std::vector<COLTYPE> _taskBoundaryPrefix; // num of rows in each task size task + 1
    std::vector<ROWTYPE> _threadPrefixSum; //
    // std::vector<ROWTYPE> _threadPrefixSum2; //
    std::vector<COLTYPE> _reorderedRowIdToTaskId;
    std::vector<std::vector<COLTYPE>> _taskInvAdj; // thread local
    std::vector<COLTYPE> _transitiveEdgeRemoveAj;

    mutable utils::BitVector<COLTYPE> _bv;

    // debugging
    // std::vector<COLTYPE> taskSizes;
};
} // namespace matrix_utils

#include "triangle_solve.tpp"