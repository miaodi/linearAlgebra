#include "cholesky_symbolic.hpp"
#include "io.hpp"
#include "matrix_utils.hpp"
#include "permutation.hpp"
#include "tree.hpp"
#include <benchmark/benchmark.h>
#include <cxxopts.hpp>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using CSRMatrixType = matrix_utils::CSRMatrix<int, int, double>;

struct BenchInput
{
    CSRMatrixType matrix;
    std::vector<int> parent;
};

BenchInput loadInput( const std::string& filename, const int nthreads )
{
    CSRMatrixType matrix;
    matrix_utils::readMatrix( filename, matrix );
    if ( matrix.rows != matrix.cols )
    {
        throw std::runtime_error( "Cholesky symbolic benchmark requires a square matrix" );
    }

    std::vector<int> parent( matrix.rows );
    std::vector<int> ancestor( matrix.rows );
    graph::eliminationTree( matrix.rows, matrix.AI(), matrix.AJ(), parent.data(), ancestor.data() );

    std::vector<int> perm( matrix.rows );
    std::vector<int> iperm( matrix.rows );
    std::vector<int> permed_parent( matrix.rows );
    graph::PostOrderNoRecur<int> postorder;
    postorder.apply( matrix.rows, matrix.Base(), parent.data(), permed_parent.data(), perm.data(),
                     iperm.data() );

    CSRMatrixType permed_matrix;
    permed_matrix.rows = matrix.rows;
    permed_matrix.cols = matrix.cols;
    permed_matrix.ResizeAI( matrix.rows + 1 );
    permed_matrix.ResizeAJ( matrix.NNZ() );
    permed_matrix.ResizeAV( matrix.NNZ() );
    matrix_utils::permuteMat( matrix.rows, matrix.cols, perm.data(), iperm.data(), matrix.AI(),
                              matrix.AJ(), matrix.AV(), permed_matrix.AI(), permed_matrix.AJ(),
                              permed_matrix.AV(), nthreads );

    return { std::move( permed_matrix ), std::move( permed_parent ) };
}

static void setCounters( benchmark::State& state, const CSRMatrixType& matrix, const CSRMatrixType* factor = nullptr )
{
    state.counters["rows"] = matrix.rows;
    state.counters["nnz"] = matrix.NNZ();
    if ( factor != nullptr )
    {
        state.counters["L_nnz"] = factor->NNZ();
    }
}

static void BM_SymbolicCholeskyCol( benchmark::State& state, const BenchInput* input )
{
    const int nthreads = static_cast<int>( state.range( 0 ) );
    factorization::SymbolicCholeskyCol<CSRMatrixType> symbolic_cholesky( nthreads );
    CSRMatrixType factor;

    for ( auto _ : state )
    {
        if ( !symbolic_cholesky.apply( input->matrix.rows, input->matrix.AI(), input->matrix.AJ(),
                                       input->parent.data(), factor ) )
        {
            state.SkipWithError( "matrix is missing explicit diagonal entries" );
            break;
        }
        benchmark::DoNotOptimize( factor.AI() );
        benchmark::DoNotOptimize( factor.AJ() );
    }
    setCounters( state, input->matrix, &factor );
}

static void BM_SymbolicCholeskyColV2( benchmark::State& state, const BenchInput* input )
{
    const int nthreads = static_cast<int>( state.range( 0 ) );
    factorization::SymbolicCholeskyColV2<CSRMatrixType> symbolic_cholesky( nthreads );
    CSRMatrixType factor;

    for ( auto _ : state )
    {
        if ( !symbolic_cholesky.apply( input->matrix.rows, input->matrix.AI(), input->matrix.AJ(),
                                       input->parent.data(), factor ) )
        {
            state.SkipWithError( "matrix is missing explicit diagonal entries" );
            break;
        }
        benchmark::DoNotOptimize( factor.AI() );
        benchmark::DoNotOptimize( factor.AJ() );
    }
    setCounters( state, input->matrix, &factor );
}

int main( int argc, char** argv )
{
    cxxopts::Options options( "cholesky_symbolic_bench",
                              "Benchmark Cholesky symbolic factorization" );
    options.allow_unrecognised_options().add_options()(
        "m,matrix", "Matrix Market file path",
        cxxopts::value<std::string>()->default_value( "data/thermal2.mtx" ) )(
        "n,nt", "Number of threads for parallel benchmarks",
        cxxopts::value<int>()->default_value( "1" ) )( "h,help", "Print usage" );

    auto result = options.parse( argc, argv );
    if ( result.count( "help" ) )
    {
        std::cout << options.help() << std::endl;
        benchmark::PrintDefaultHelp();
        return 0;
    }

    const auto matrix_file = result["matrix"].as<std::string>();
    const int nthreads = std::max( 1, result["nt"].as<int>() );
    BenchInput input = loadInput( matrix_file, nthreads );

    std::cout << "Matrix loaded: rows=" << input.matrix.rows << " nnz=" << input.matrix.NNZ()
              << " base=" << input.matrix.Base() << " threads=" << nthreads << std::endl;

    benchmark::RegisterBenchmark( "SymbolicCholeskyCol", BM_SymbolicCholeskyCol, &input )->Arg( nthreads );
    benchmark::RegisterBenchmark( "SymbolicCholeskyColV2", BM_SymbolicCholeskyColV2, &input )->Arg( nthreads );

    benchmark::Initialize( &argc, argv );
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
