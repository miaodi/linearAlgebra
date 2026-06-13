#pragma once
#include <deque>
#include <limits>
#include <vector>
namespace reordering
{

template <typename ROWTYPE, typename COLTYPE, typename VALTYPE>
class HungarianAlgorithm
{
public:
    HungarianAlgorithm() = default;
    void operator()( const COLTYPE n,
                     ROWTYPE const* ai,
                     COLTYPE const* aj,
                     VALTYPE const* av,
                     COLTYPE* matching_row,
                     COLTYPE* matching_col,
                     VALTYPE* potential_row,
                     VALTYPE* potential_col );

private:
    void initialize();
    void initialize_row( const COLTYPE row );
    void augment( COLTYPE t );
    bool update_potentials();
    void prep_row( const COLTYPE row );
    bool match_row( const COLTYPE row );

private:
    std::vector<COLTYPE> parent;
    std::vector<char> S; // rows in the alternating tree Z
    std::vector<char> T; // columns in the alternating tree Z
    std::vector<VALTYPE> min_slack;
    // std::vector<VALTYPE> min_slack_cpy;
    std::deque<COLTYPE> Q;
    static constexpr COLTYPE INVALID = std::numeric_limits<COLTYPE>::max();
    static constexpr VALTYPE MAX_VAL = std::numeric_limits<VALTYPE>::max();
    static constexpr VALTYPE TOL_VAL = 10 * std::numeric_limits<VALTYPE>::epsilon();

    // Data that does not owned by this class
    VALTYPE* potential_row;
    VALTYPE* potential_col;
    COLTYPE* matching_row;
    COLTYPE* matching_col;

    COLTYPE n;
    ROWTYPE const* ai;
    COLTYPE const* aj;
    VALTYPE const* av;
    ROWTYPE base;
};
} // namespace reordering
