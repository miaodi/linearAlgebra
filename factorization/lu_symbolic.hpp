#pragma once

#include "matrix_utils.hpp"

#include <vector>

namespace factorization
{

/// @brief EDAGS symbolic LU factorization pattern builder.
///
/// Implements the elimination DAG algorithm of Gilbert and Liu,
/// "Elimination Structures for Unsymmetric Sparse LU Factors", SIAM J. Matrix
/// Anal. Appl. 14(2), 1993, DOI: 10.1137/0614024. The algorithm models
/// symbolic LU without pivoting under the generic structural-pattern assumption:
/// coincidental numerical cancellation is ignored.
///
/// The input is a square CSR structure whose row column indices are sorted. Each
/// row must contain an explicit diagonal entry; initialization fails otherwise.
/// The CSR base is taken from `ai[0]` and is preserved in all input/output CSR
/// structures and EDAGs.
///
/// `apply()` writes the combined row-oriented L/U pattern to one appendable CSR
/// output. Output values are resized only after the symbolic structure is built
/// and are not meaningful numeric factors. `lowerEdag()` returns G(L), while
/// `upperEdag()` returns G(U); EDAG rows are intentionally unsorted and should be
/// treated as unordered adjacency lists.
template <matrix_utils::AppendableCSR CSRMatrixType>
class SymbolicLUEdags
{
public:
    using ROWTYPE = typename CSRMatrixType::ROWTYPE;
    using COLTYPE = typename CSRMatrixType::COLTYPE;
    using GraphType = matrix_utils::CSRStructVec<ROWTYPE, COLTYPE>;

    bool apply( COLTYPE nnodes, ROWTYPE const* ai, COLTYPE const* aj, CSRMatrixType& lu );

    /// Edag adjacency rows are intentionally not sorted; callers should treat
    /// them as unordered adjacency lists.
    const GraphType& lowerEdag() const { return _lEdag; }
    const GraphType& upperEdag() const { return _uEdag; }

private:
    enum class ReduceOrder
    {
        Forward,
        Reverse
    };

    bool initialize( COLTYPE nnodes, ROWTYPE const* ai, COLTYPE const* aj );
    bool setUpperEdag( const GraphType& upper_edag );

    /// @brief Step 1 of EDAGS: compute Struct(L_i,) by traversing the upper edag.
    ///
    /// `row` is zero-based regardless of the matrix base. The pattern is appended
    /// directly to `lu.aj`; row pointers are not updated here.
    void appendLowerRowPattern( COLTYPE row, ROWTYPE const* ai, COLTYPE const* aj, CSRMatrixType& lu );

    /// @brief Step 2 of EDAGS: reduce the current lower row into the lower edag.
    ///
    /// This expects `[candidates_begin, candidates_end)` to contain the sorted
    /// lower row pattern produced by step 1, before diagonal/upper entries are
    /// appended.
    void reduceLowerEdagRow( COLTYPE row, COLTYPE const* candidates_begin, COLTYPE const* candidates_end );

    /// @brief Step 3 of EDAGS: compute Struct(U_i,) by a Gustavson row union.
    ///
    /// This appends the diagonal/upper part of row `row` directly to `lu`, using
    /// the current lower-edag row as the list of previous U rows to merge. Row
    /// pointers are not updated here.
    void appendUpperRowPattern( COLTYPE row, ROWTYPE const* ai, COLTYPE const* aj, CSRMatrixType& lu );

    void extendUpperReachabilityColumn( COLTYPE col );
    void rebuildUpperEdag( const CSRMatrixType& lu );

    bool validGraph( const GraphType& graph ) const;
    void reduceEdagRow( GraphType& edag,
                        COLTYPE row,
                        COLTYPE const* candidates_begin,
                        COLTYPE const* candidates_end,
                        ReduceOrder order,
                        bool skip_self );
    COLTYPE nextEpoch( std::vector<COLTYPE>& visited, COLTYPE& epoch );

private:
    COLTYPE _nnodes{};
    ROWTYPE _base{};
    std::vector<ROWTYPE> _diag;
    GraphType _lEdag;
    GraphType _uEdag;
    std::vector<COLTYPE> _reachVisited;
    std::vector<COLTYPE> _reduceVisited;
    std::vector<COLTYPE> _unionVisited;
    std::vector<COLTYPE> _stack;
    std::vector<ROWTYPE> _uCursor;
    std::vector<std::vector<COLTYPE>> _uReachRows;
    std::vector<std::vector<COLTYPE>> _uColumnRows;
    COLTYPE _reachEpoch{};
    COLTYPE _reduceEpoch{};
    COLTYPE _unionEpoch{};
};

} // namespace factorization
