#include <gtest/gtest.h>

#include "csr5_convert.hpp"
#include "csr5_format.hpp"
#include "csr5_policy.hpp"
#include "io.hpp"
#include "matrix_utils.hpp"

#include "format_avx2.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using Policy = matrix_utils::CSR5_AVX2_Policy<double>;
using Matrix = matrix_utils::CSRMatrixVec<int, int, double>;

static_assert( Policy::OMEGA == ANONYMOUSLIB_CSR5_OMEGA );
static_assert( Policy::SIGMA == ANONYMOUSLIB_CSR5_SIGMA );
static_assert( Policy::DESCRIPTOR_BITS <= 32 );

struct ReferenceCSR5
{
    int num_tiles = 0;
    std::vector<uint32_t> tile_ptr;
    std::vector<uint32_t> tile_desc;
    std::vector<int> col_idx;
    std::vector<double> values;
};

struct ComparisonStats
{
    int compared = 0;
    int skipped_empty = 0;
    int skipped_no_shared_tiles = 0;
    int skipped_fast_track_data_tiles = 0;
};

std::vector<std::filesystem::path> testMatrixPaths()
{
    std::vector<std::filesystem::path> paths;
    const std::filesystem::path root{ "data" };
    if ( !std::filesystem::exists( root ) )
    {
        return paths;
    }

    for ( const auto& entry : std::filesystem::recursive_directory_iterator( root ) )
    {
        if ( entry.is_regular_file() && entry.path().extension() == ".mtx" )
        {
            paths.push_back( entry.path() );
        }
    }
    std::sort( paths.begin(), paths.end() );
    return paths;
}

Matrix readNormalizedMatrix( const std::filesystem::path& path )
{
    std::ifstream input( path );
    if ( !input )
    {
        throw std::runtime_error( "Cannot open MatrixMarket file: " + path.string() );
    }

    Matrix matrix;
    matrix_utils::readMatrixMarket( input, matrix );

    if ( matrix.ai.empty() )
    {
        return matrix;
    }

    const int base = matrix.ai.front();
    if ( base != 0 )
    {
        for ( int& row_ptr : matrix.ai )
        {
            row_ptr -= base;
        }
        for ( int& col : matrix.aj )
        {
            col -= base;
        }
    }
    return matrix;
}

bool hasEmptyRows( const Matrix& matrix )
{
    for ( int row = 0; row < matrix.rows; ++row )
    {
        if ( matrix.ai[row] == matrix.ai[row + 1] )
        {
            return true;
        }
    }
    return false;
}

ReferenceCSR5 buildReferenceCSR5( const Matrix& matrix )
{
    constexpr int num_packet = 1;

    ReferenceCSR5 ref;
    const int nnz = matrix.NNZ();
    ref.num_tiles = ( nnz + Policy::TILE_SIZE - 1 ) / Policy::TILE_SIZE;
    ref.tile_ptr.resize( static_cast<std::size_t>( ref.num_tiles ) + 1 );
    ref.tile_desc.assign( static_cast<std::size_t>( ref.num_tiles ) * Policy::OMEGA, 0 );
    ref.col_idx = matrix.aj;
    ref.values = matrix.av;

    std::vector<int> descriptor_offset_pointer( static_cast<std::size_t>( ref.num_tiles ) + 1, 0 );
    int num_offsets = 0;

    generate_partition_pointer<int, uint32_t>( Policy::SIGMA, ref.num_tiles, matrix.rows, nnz,
                                               ref.tile_ptr.data(), matrix.ai.data() );

    generate_partition_descriptor<int, uint32_t>(
        Policy::SIGMA, ref.num_tiles, matrix.rows, Policy::BIT_Y_OFFSET, Policy::BIT_SEG_OFFSET,
        num_packet, matrix.ai.data(), ref.tile_ptr.data(), ref.tile_desc.data(),
        descriptor_offset_pointer.data(), &num_offsets );

    aosoa_transpose<int, uint32_t, double>( Policy::SIGMA, nnz, ref.tile_ptr.data(),
                                            ref.col_idx.data(), ref.values.data(), true );

    return ref;
}

void compareMatrixWithReference( const std::filesystem::path& path, ComparisonStats& stats )
{
    SCOPED_TRACE( path.string() );

    const Matrix matrix = readNormalizedMatrix( path );
    if ( hasEmptyRows( matrix ) )
    {
        ++stats.skipped_empty;
        return;
    }

    matrix_utils::CSR5Data<int, int, double, Policy> actual;
    matrix_utils::convertCSRtoCSR5<int, int, double, Policy>(
        matrix.rows, matrix.ai.data(), matrix.aj.data(), matrix.av.data(), actual, 4 );

    const ReferenceCSR5 ref = buildReferenceCSR5( matrix );

    // The reference kernels reserve their final partition for CSR tail handling.
    const int reference_full_prefix = std::max( 0, ref.num_tiles - 1 );
    const int shared_full_tiles = std::min<int>( actual._num_full_tiles, reference_full_prefix );
    if ( shared_full_tiles == 0 )
    {
        ++stats.skipped_no_shared_tiles;
        return;
    }

    ASSERT_GE( actual._tile_ptr.size(), static_cast<std::size_t>( shared_full_tiles ) + 1 );
    ASSERT_GE( ref.tile_ptr.size(), static_cast<std::size_t>( shared_full_tiles ) + 1 );

    for ( int tile = 0; tile <= shared_full_tiles; ++tile )
    {
        EXPECT_EQ( static_cast<uint32_t>( actual._tile_ptr[tile] ), ref.tile_ptr[tile] & 0x7FFFFFFFu )
            << "tile_ptr[" << tile << "]";
    }

    for ( int tile = 0; tile < shared_full_tiles; ++tile )
    {
        for ( int lane = 0; lane < Policy::OMEGA; ++lane )
        {
            const int desc_idx = tile * Policy::OMEGA + lane;
            EXPECT_EQ( actual._tile_desc[desc_idx], ref.tile_desc[desc_idx] )
                << "tile_desc[tile=" << tile << ", lane=" << lane << "]";
        }

        const bool fast_track = actual._tile_ptr[tile] == actual._tile_ptr[tile + 1];
        if ( fast_track )
        {
            ++stats.skipped_fast_track_data_tiles;
            continue;
        }

        const int tile_start = tile * Policy::TILE_SIZE;
        for ( int local = 0; local < Policy::TILE_SIZE; ++local )
        {
            const int idx = tile_start + local;
            EXPECT_EQ( actual._tile_col_idx[idx], ref.col_idx[idx] )
                << "tile_col_idx[tile=" << tile << ", local=" << local << "]";
            EXPECT_EQ( actual._tile_val[idx], ref.values[idx] )
                << "tile_val[tile=" << tile << ", local=" << local << "]";
        }
    }

    ++stats.compared;
}

} // namespace

TEST( CSR5ReferencePreprocessTest, MatchesReferenceForTestDataMatrices )
{
    const std::vector<std::filesystem::path> paths = testMatrixPaths();
    if ( paths.empty() )
    {
        GTEST_SKIP() << "No MatrixMarket test data found under the test data directory";
    }

    ComparisonStats stats;
    for ( const std::filesystem::path& path : paths )
    {
        compareMatrixWithReference( path, stats );
    }

    EXPECT_GT( stats.compared, 0 )
        << "No CSR5 v1-compatible matrix with a shared full-tile prefix was found. "
        << "Skipped empty-row matrices: " << stats.skipped_empty
        << ", skipped small/tail-only matrices: " << stats.skipped_no_shared_tiles;
}
