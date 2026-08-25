#include "cuda_bicgstab.cuh"
#include "cuda_spmv.cuh"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

namespace cuda_utils = matrix_utils::sparse_cuda;

namespace
{

struct SolveOptions
{
    size_t max_iterations = 100;
    size_t max_restarts = 2;
    size_t residual_replacement_frequency = 50;
    double relative_tolerance = 1e-12;
    double absolute_tolerance = 0.0;
    cuda_utils::PreconditionerType preconditioner_type = cuda_utils::PreconditionerType::NONE;
};

struct SolveResult
{
    cuda_utils::State state;
    int iterations;
    int restarts;
};

std::vector<double> csr_matvec( const int rows,
                                const std::vector<int>& ia,
                                const std::vector<int>& ja,
                                const std::vector<double>& values,
                                const std::vector<double>& x )
{
    std::vector<double> result( static_cast<size_t>( rows ), 0.0 );
    for ( int row = 0; row < rows; ++row )
    {
        for ( int idx = ia[static_cast<size_t>( row )]; idx < ia[static_cast<size_t>( row + 1 )]; ++idx )
        {
            result[static_cast<size_t>( row )] += values[static_cast<size_t>( idx )] *
                                                  x[static_cast<size_t>( ja[static_cast<size_t>( idx )] )];
        }
    }
    return result;
}

SolveResult solve_csr( const int rows,
                       const std::vector<int>& ia,
                       const std::vector<int>& ja,
                       const std::vector<double>& values,
                       const std::vector<double>& rhs,
                       std::vector<double>& solution,
                       const SolveOptions& options = {} )
{
    EXPECT_EQ( ia.size(), static_cast<size_t>( rows + 1 ) );
    EXPECT_EQ( ja.size(), values.size() );
    EXPECT_EQ( rhs.size(), static_cast<size_t>( rows ) );
    EXPECT_EQ( solution.size(), static_cast<size_t>( rows ) );

    cuda_utils::DeviceArray<int> d_ia;
    cuda_utils::DeviceArray<int> d_ja;
    cuda_utils::DeviceArray<double> d_values;
    d_ia.copy<cuda_utils::MemoryLocation::Host>( ia.data(), ia.size() );
    d_ja.copy<cuda_utils::MemoryLocation::Host>( ja.data(), ja.size() );
    d_values.copy<cuda_utils::MemoryLocation::Host>( values.data(), values.size() );

    cuda_utils::CSRScalarSPMV<int, int, double> spmv;
    spmv.preprocess( rows, d_ia.data(), d_ja.data(), d_values.data(), 0, static_cast<int>( values.size() ) );

    cuda_utils::CudaBiCGSTAB solver;
    solver.setMaxIter( options.max_iterations );
    solver.setMaxRestarts( options.max_restarts );
    solver.setResidualReplacementFrequency( options.residual_replacement_frequency );
    solver.setRelTol( options.relative_tolerance );
    solver.setAbsTol( options.absolute_tolerance );
    solver.setPreconditionerType( options.preconditioner_type );
    solver.setupOperator( &spmv );

    cuda_utils::JacobiPreconditioner jacobi;
    if ( options.preconditioner_type != cuda_utils::PreconditionerType::NONE )
    {
        std::vector<double> diagonal( static_cast<size_t>( rows ), 1.0 );
        for ( int row = 0; row < rows; ++row )
        {
            for ( int idx = ia[static_cast<size_t>( row )]; idx < ia[static_cast<size_t>( row + 1 )]; ++idx )
            {
                if ( ja[static_cast<size_t>( idx )] == row )
                {
                    diagonal[static_cast<size_t>( row )] = values[static_cast<size_t>( idx )];
                    break;
                }
            }
        }
        jacobi.setup( static_cast<size_t>( rows ), diagonal.data() );
        solver.setPreconditioner( &jacobi );
    }

    const cuda_utils::State state = solver.solve<false>( rhs.data(), solution.data() );
    const cudaError_t sync_status = cudaDeviceSynchronize();
    EXPECT_EQ( sync_status, cudaSuccess ) << cudaGetErrorString( sync_status );

    return { state, solver.getLastIterations(), solver.getLastRestarts() };
}

void expect_solution_near( const std::vector<double>& actual,
                           const std::vector<double>& expected,
                           const double tolerance = 1e-10 )
{
    ASSERT_EQ( actual.size(), expected.size() );
    for ( size_t idx = 0; idx < actual.size(); ++idx )
    {
        EXPECT_NEAR( actual[idx], expected[idx], tolerance ) << "index " << idx;
    }
}

} // namespace

TEST( CudaBiCGSTAB, ConvergesOnExactAlphaStep )
{
    const std::vector<int> ia = { 0, 1, 2, 3 };
    const std::vector<int> ja = { 0, 1, 2 };
    const std::vector<double> values = { 1.0, 1.0, 1.0 };
    const std::vector<double> rhs = { 1.0, -2.0, 3.0 };
    std::vector<double> solution( rhs.size(), 0.0 );

    const SolveResult result = solve_csr( 3, ia, ja, values, rhs, solution );

    EXPECT_EQ( result.state, cuda_utils::State::CONVERGED );
    EXPECT_EQ( result.iterations, 1 );
    EXPECT_EQ( result.restarts, 0 );
    expect_solution_near( solution, rhs, 0.0 );
}

TEST( CudaBiCGSTAB, AcceptsExactInitialGuessAndZeroRightHandSide )
{
    const std::vector<int> ia = { 0, 1, 2 };
    const std::vector<int> ja = { 0, 1 };
    const std::vector<double> values = { 2.0, 3.0 };

    const std::vector<double> exact_solution = { 2.0, -3.0 };
    const std::vector<double> rhs = csr_matvec( 2, ia, ja, values, exact_solution );
    std::vector<double> solution = exact_solution;
    SolveResult result = solve_csr( 2, ia, ja, values, rhs, solution );
    EXPECT_EQ( result.state, cuda_utils::State::CONVERGED );
    EXPECT_EQ( result.iterations, 0 );
    expect_solution_near( solution, exact_solution, 0.0 );

    const std::vector<double> zero_rhs( 2, 0.0 );
    solution.assign( 2, 0.0 );
    result = solve_csr( 2, ia, ja, values, zero_rhs, solution );
    EXPECT_EQ( result.state, cuda_utils::State::CONVERGED );
    EXPECT_EQ( result.iterations, 0 );
    expect_solution_near( solution, zero_rhs, 0.0 );
}

TEST( CudaBiCGSTAB, RepairsOrthogonalShadowResidual )
{
    // For r=(1,0), r.A.r is exactly zero although A is nonsingular. The
    // conventional shadow residual therefore breaks down before alpha.
    const std::vector<int> ia = { 0, 1, 3 };
    const std::vector<int> ja = { 1, 0, 1 };
    const std::vector<double> values = { 1.0, 1.0, 1.0 };
    const std::vector<double> rhs = { 1.0, 0.0 };
    const std::vector<double> expected = { -1.0, 1.0 };
    std::vector<double> solution( 2, 0.0 );

    const SolveResult result = solve_csr( 2, ia, ja, values, rhs, solution );

    EXPECT_EQ( result.state, cuda_utils::State::CONVERGED );
    EXPECT_EQ( result.restarts, 0 );
    expect_solution_near( solution, expected );
}

TEST( CudaBiCGSTAB, HandlesVerySmallButWellScaledDenominator )
{
    const std::vector<int> ia = { 0, 1 };
    const std::vector<int> ja = { 0 };
    const std::vector<double> values = { 1e-30 };
    const std::vector<double> rhs = { 2e-30 };
    const std::vector<double> expected = { 2.0 };
    std::vector<double> solution( 1, 0.0 );

    const SolveResult result = solve_csr( 1, ia, ja, values, rhs, solution );

    EXPECT_EQ( result.state, cuda_utils::State::CONVERGED );
    expect_solution_near( solution, expected );
}

TEST( CudaBiCGSTAB, ConvergesWithReliableResidualReplacement )
{
    const std::vector<int> ia = { 0, 2, 5, 8, 11 };
    const std::vector<int> ja = { 0, 1, 0, 1, 2, 1, 2, 3, 0, 2, 3 };
    const std::vector<double> values = { 4.0, 1.0, -2.0, 5.0, 1.0, -1.0, 3.0, 1.0, 1.0, -1.0, 4.0 };
    const std::vector<double> expected = { 1.0, -2.0, 0.5, 3.0 };
    const std::vector<double> rhs = csr_matvec( 4, ia, ja, values, expected );
    std::vector<double> solution( 4, 0.0 );

    SolveOptions options;
    options.residual_replacement_frequency = 2;
    const SolveResult result = solve_csr( 4, ia, ja, values, rhs, solution, options );

    EXPECT_EQ( result.state, cuda_utils::State::CONVERGED );
    expect_solution_near( solution, expected );
}

TEST( CudaBiCGSTAB, SupportsLeftAndRightJacobiPreconditioning )
{
    const std::vector<int> ia = { 0, 2, 5, 8, 10 };
    const std::vector<int> ja = { 0, 1, 0, 1, 2, 1, 2, 3, 2, 3 };
    const std::vector<double> values = { 10.0, 1.0, -2.0, 20.0, 2.0, -1.0, 30.0, 3.0, -3.0, 40.0 };
    const std::vector<double> expected = { 1.0, -2.0, 0.5, 3.0 };
    const std::vector<double> rhs = csr_matvec( 4, ia, ja, values, expected );

    for ( const cuda_utils::PreconditionerType type :
          { cuda_utils::PreconditionerType::LEFT, cuda_utils::PreconditionerType::RIGHT } )
    {
        std::vector<double> solution( 4, 0.0 );
        SolveOptions options;
        options.preconditioner_type = type;
        const SolveResult result = solve_csr( 4, ia, ja, values, rhs, solution, options );

        EXPECT_EQ( result.state, cuda_utils::State::CONVERGED );
        expect_solution_near( solution, expected );
    }
}

TEST( CudaBiCGSTAB, ReturnsFiniteFailureAfterRepeatedOmegaBreakdown )
{
    // For a real skew-symmetric matrix, s.A.s is always zero. BiCGSTAB's
    // minimal-residual omega step is therefore undefined and restarts cannot
    // repair the method; the solver must stop without propagating NaNs.
    const std::vector<int> ia = { 0, 1, 2 };
    const std::vector<int> ja = { 1, 0 };
    const std::vector<double> values = { 1.0, -1.0 };
    const std::vector<double> rhs = { 1.0, 0.0 };
    std::vector<double> solution( 2, 0.0 );

    SolveOptions options;
    options.max_iterations = 10;
    options.max_restarts = 2;
    const SolveResult result = solve_csr( 2, ia, ja, values, rhs, solution, options );

    EXPECT_EQ( result.state, cuda_utils::State::FAILED );
    EXPECT_EQ( result.restarts, 2 );
    EXPECT_TRUE( std::isfinite( solution[0] ) );
    EXPECT_TRUE( std::isfinite( solution[1] ) );
}

TEST( CudaBiCGSTAB, ReturnsFiniteFailureAfterRepeatedAlphaBreakdown )
{
    // The first residual lies in the null space, so A*p is zero and alpha is
    // undefined. Bounded restarts must terminate without dividing by zero.
    const std::vector<int> ia = { 0, 0, 1 };
    const std::vector<int> ja = { 1 };
    const std::vector<double> values = { 1.0 };
    const std::vector<double> rhs = { 1.0, 0.0 };
    std::vector<double> solution( 2, 0.0 );

    SolveOptions options;
    options.max_iterations = 10;
    options.max_restarts = 2;
    const SolveResult result = solve_csr( 2, ia, ja, values, rhs, solution, options );

    EXPECT_EQ( result.state, cuda_utils::State::FAILED );
    EXPECT_EQ( result.restarts, 2 );
    EXPECT_TRUE( std::isfinite( solution[0] ) );
    EXPECT_TRUE( std::isfinite( solution[1] ) );
}
