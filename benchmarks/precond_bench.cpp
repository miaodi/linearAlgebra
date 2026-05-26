#include "io.hpp"
#include "matrix_utils.hpp"
#include "precond.hpp"
#include "utils.h"
#include <algorithm>
#include <benchmark/benchmark.h>
#include <memory>
#include <numeric>
#include <omp.h>
#include <vector>

static std::unique_ptr<matrix_utils::CSRMatrixVec<int, int, double>> mat;
class MyFixture : public benchmark::Fixture
{
public:
    MyFixture()
    {
        // load matrix
        if ( mat == nullptr )
        {
            mat.reset( new matrix_utils::CSRMatrixVec<int, int, double>() );
            std::ifstream f( "data/thermal2.mtx" );
            f.clear();
            f.seekg( 0, std::ios::beg );
            matrix_utils::readMatrixMarket( f, mat->ai, mat->aj, mat->av );
            mat->rows = mat->ai.size() - 1;
            std::cout << "matrix size: " << mat->rows << "\n";
        }
    }
};

BENCHMARK_DEFINE_F( MyFixture, llist_small )( benchmark::State& state )
{
    std::vector<double> x( mat->rows, 0.0 );
    std::vector<double> b( mat->rows, 1.0 );

    matrix_utils::CSRMatrix<int, int, double> U, ICC;
    matrix_utils::SplitTriangle<matrix_utils::TriangularMatrix::U>(
        mat->rows, mat->ai[0], mat->AI(), mat->AJ(), mat->AV(), U );

    for ( auto _ : state )
    {
        matrix_utils::ICCLevelSymbolic0( mat->rows, U.ai.get(), U.aj.get(), U.ai.get(), state.range( 0 ), ICC );
    }
}

BENCHMARK_REGISTER_F( MyFixture, llist_small )->Arg( 1 )->Arg( 3 )->Arg( 5 )->Arg( 7 );

BENCHMARK_DEFINE_F( MyFixture, vector_merge_small )( benchmark::State& state )
{
    std::vector<double> x( mat->rows, 0.0 );
    std::vector<double> b( mat->rows, 1.0 );

    matrix_utils::CSRMatrix<int, int, double> U, ICC;
    matrix_utils::SplitTriangle<matrix_utils::TriangularMatrix::U>(
        mat->rows, mat->ai[0], mat->AI(), mat->AJ(), mat->AV(), U );

    for ( auto _ : state )
    {
        matrix_utils::ICCLevelSymbolic1( mat->rows, U.ai.get(), U.aj.get(), U.ai.get(), state.range( 0 ), ICC );
    }
}

BENCHMARK_REGISTER_F( MyFixture, vector_merge_small )->Arg( 1 )->Arg( 3 )->Arg( 5 )->Arg( 7 );

BENCHMARK_DEFINE_F( MyFixture, vector_balanced_merge_small )
( benchmark::State& state )
{
    std::vector<double> x( mat->rows, 0.0 );
    std::vector<double> b( mat->rows, 1.0 );

    matrix_utils::CSRMatrix<int, int, double> U, ICC;
    matrix_utils::SplitTriangle<matrix_utils::TriangularMatrix::U>(
        mat->rows, mat->ai[0], mat->AI(), mat->AJ(), mat->AV(), U );

    for ( auto _ : state )
    {
        matrix_utils::ICCLevelSymbolic2( mat->rows, U.ai.get(), U.aj.get(), U.ai.get(), state.range( 0 ), ICC );
    }
}

BENCHMARK_REGISTER_F( MyFixture, vector_balanced_merge_small )->Arg( 1 )->Arg( 3 )->Arg( 5 )->Arg( 7 );

BENCHMARK_DEFINE_F( MyFixture, vector_merge_1st_balanced_merge_small )
( benchmark::State& state )
{
    std::vector<double> x( mat->rows, 0.0 );
    std::vector<double> b( mat->rows, 1.0 );

    matrix_utils::CSRMatrix<int, int, double> U, ICC;
    matrix_utils::SplitTriangle<matrix_utils::TriangularMatrix::U>(
        mat->rows, mat->ai[0], mat->AI(), mat->AJ(), mat->AV(), U );

    for ( auto _ : state )
    {
        matrix_utils::ICCLevelSymbolic3( mat->rows, U.ai.get(), U.aj.get(), U.ai.get(), state.range( 0 ), ICC );
    }
}

BENCHMARK_REGISTER_F( MyFixture, vector_merge_1st_balanced_merge_small )
    ->Arg( 1 )
    ->Arg( 3 )
    ->Arg( 5 )
    ->Arg( 7 );

BENCHMARK_MAIN();
