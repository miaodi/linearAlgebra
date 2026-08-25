#include "cuda_bicgstab.cuh"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace matrix_utils::sparse_cuda
{

namespace
{

bool near_inner_product_breakdown( const double inner_product, const double lhs_norm, const double rhs_norm, const double tolerance )
{
    // Extension beyond the original recurrence [1]: use an angle/scale-aware
    // test instead of an absolute threshold. The near-breakdown analysis in
    // [2] is formulated in terms of angles between Krylov and shadow spaces.
    // [2]: https://doi.org/10.1007/BF02140769
    if ( !std::isfinite( inner_product ) || !std::isfinite( lhs_norm ) || !std::isfinite( rhs_norm ) )
    {
        return true;
    }

    if ( lhs_norm == 0.0 || rhs_norm == 0.0 )
    {
        return true;
    }

    const long double scale = static_cast<long double>( lhs_norm ) * static_cast<long double>( rhs_norm );
    return std::abs( static_cast<long double>( inner_product ) ) <= static_cast<long double>( tolerance ) * scale;
}

} // namespace

CudaBiCGSTAB::CudaBiCGSTAB()
    : _cublas_handle( nullptr ),
      _cusparse_handle( nullptr ),
      _spmv_operator( nullptr ),
      _preconditioner( &_default_preconditioner ),
      _default_preconditioner(),
      _max_iter( 100 ),
      _abs_tol( 0.0 ),
      _rel_tol( 1e-8 ),
      _breakdown_tol( std::sqrt( std::numeric_limits<double>::epsilon() ) ),
      _prec_type( PreconditionerType::LEFT ),
      _max_restarts( 2 ),
      _residual_replacement_frequency( 50 ),
      _verbose( false ),
      _last_iterations( 0 ),
      _last_restarts( 0 ),
      _is_operator_setup( false ),
      _n( 0 )
{
    initialize_cuda();
}

CudaBiCGSTAB::~CudaBiCGSTAB()
{
    cleanup_cuda();
}

void CudaBiCGSTAB::initialize_cuda()
{
    try
    {
        // Create cuBLAS handle
        check_cublas_error( cublasCreate( &_cublas_handle ), "Failed to create cuBLAS handle" );

        // Create cuSPARSE handle
        check_cusparse_error( cusparseCreate( &_cusparse_handle ),
                              "Failed to create cuSPARSE handle" );
    }
    catch ( ... )
    {
        cleanup_cuda();
        throw;
    }
}

void CudaBiCGSTAB::cleanup_cuda()
{
    // Free workspace memory
    _d_r0.release();
    _d_r.release();
    _d_p.release();
    _d_v.release();
    _d_s.release();
    _d_t.release();
    _d_x_hat.release();
    _d_tmp.release();

    // Destroy handles
    if ( _cusparse_handle )
        cusparseDestroy( _cusparse_handle );
    if ( _cublas_handle )
        cublasDestroy( _cublas_handle );
}

void CudaBiCGSTAB::initialize_workspace( size_t n )
{
    if ( _n == n )
    {
        return; // Already initialized for this size
    }

    _n = n;

    // Allocate device memory for BiCGSTAB vectors
    _d_r0.resize( n );    // Reference residual vector
    _d_r.resize( n );     // Current residual
    _d_p.resize( n );     // Search direction
    _d_v.resize( n );     // A * p (with preconditioning)
    _d_s.resize( n );     // Intermediate residual
    _d_t.resize( n );     // A * s (with preconditioning)
    _d_x_hat.resize( n ); // Accumulated solution updates
    _d_tmp.resize( n );   // Temporary storage
}

void CudaBiCGSTAB::setupOperator( SpMVOperator<double>* spmv_operator )
{
    if ( !spmv_operator )
    {
        throw std::runtime_error( "setupOperator: spmv_operator cannot be nullptr" );
    }

    // Store operator pointer
    _spmv_operator = spmv_operator;

    // Get matrix size from operator
    const size_t n = _spmv_operator->size();
    if ( n > static_cast<size_t>( std::numeric_limits<int>::max() ) )
    {
        throw std::runtime_error( "CudaBiCGSTAB matrix size exceeds the cuBLAS integer limit" );
    }

    // Initialize workspace for this problem size
    initialize_workspace( n );

    // Create vector descriptors using DeviceVectorView
    _view_prec_x.create( static_cast<size_t>( _n ), nullptr );
    _view_prec_y.create( static_cast<size_t>( _n ), nullptr );
    _view_prec_tmp.create( static_cast<size_t>( _n ), nullptr );

    // Create DeviceVectorView wrappers for RHS and solution vectors
    _view_d_b.create( static_cast<size_t>( _n ), nullptr );
    _view_d_x.create( static_cast<size_t>( _n ), nullptr );

    // Set up DeviceVectorView wrappers to point to allocated memory
    _view_prec_tmp.setData( _d_tmp.data() );

    _is_operator_setup = true;
}

void CudaBiCGSTAB::setPreconditioner( Preconditioner* preconditioner )
{
    _preconditioner = preconditioner;
}

template <bool ZeroInitialGuess>
State CudaBiCGSTAB::solve( const double* h_b, double* h_x )
{
    // Check if setup has been called
    if ( !_is_operator_setup )
    {
        throw std::runtime_error( "setupOperator must be called before solve" );
    }

    // Check if preconditioner is required but not setup
    if ( _prec_type != PreconditionerType::NONE && ( !_preconditioner || !_preconditioner->isSetup() ) )
    {
        throw std::runtime_error(
            "A setup preconditioner is required before solve when preconditioning is enabled" );
    }
    if ( _n > 0 && ( !h_b || !h_x ) )
    {
        throw std::runtime_error( "CudaBiCGSTAB solve requires non-null b and x pointers" );
    }

    // Copy host data to device
    _d_b.copy<MemoryLocation::Host>( h_b, static_cast<size_t>( _n ) );

    if constexpr ( ZeroInitialGuess )
    {
        // Initialize device memory to zero instead of copying
        _d_x.resize( static_cast<size_t>( _n ) ); // Ensure size
        check_cuda_error( cudaMemset( _d_x.data(), 0, static_cast<size_t>( _n ) * sizeof( double ) ),
                          "Failed to zero the initial guess" );
    }
    else
    {
        // Copy initial guess from host
        _d_x.copy<MemoryLocation::Host>( h_x, static_cast<size_t>( _n ) );
    }

    // Set up DeviceVectorView wrappers
    _view_d_b.setData( _d_b.data() );
    _view_d_x.setData( _d_x.data() );

    // Solve on device
    State result = deviceSolve( _view_d_b, _view_d_x );

    // Copy solution back to host
    check_cuda_error( cudaMemcpy( h_x, _d_x.data(), static_cast<size_t>( _n ) * sizeof( double ), cudaMemcpyDeviceToHost ),
                      "Failed to copy the BiCGSTAB solution to host" );

    return result;
}

// Explicit template instantiations
template State CudaBiCGSTAB::solve<false>( const double* h_b, double* h_x );
template State CudaBiCGSTAB::solve<true>( const double* h_b, double* h_x );

State CudaBiCGSTAB::deviceSolve( const DeviceVectorView& d_b, DeviceVectorView& d_x )
{
    // Check if setup has been called
    if ( !_is_operator_setup )
    {
        throw std::runtime_error( "setupOperator must be called before solve" );
    }

    // Check if preconditioner is required but not setup
    if ( _prec_type != PreconditionerType::NONE && ( !_preconditioner || !_preconditioner->isSetup() ) )
    {
        throw std::runtime_error(
            "A setup preconditioner is required before solve when preconditioning is enabled" );
    }

    _last_iterations = 0;
    _last_restarts = 0;
    if ( _n == 0 )
    {
        return State::CONVERGED;
    }

    check_cuda_error( cudaMemset( _d_x_hat.data(), 0, static_cast<size_t>( _n ) * sizeof( double ) ),
                      "Failed to initialize the BiCGSTAB solution update" );

    double init_true_resid = 0.0;
    const double init_resid = compute_residual( d_b, d_x, &init_true_resid );
    if ( !std::isfinite( init_resid ) || !std::isfinite( init_true_resid ) )
    {
        return State::FAILED;
    }
    if ( check_convergence( init_true_resid, init_true_resid ) )
    {
        return State::CONVERGED;
    }

    double resid = init_resid;
    double rho = 0.0;
    double r0_norm = 0.0;
    bool at_recurrence_start = false;
    bool shadow_repaired = false;

    auto initialize_recurrence = [&]()
    {
        _d_r0.copy<MemoryLocation::Device>( _d_r.data(), static_cast<size_t>( _n ) );
        _d_p.copy<MemoryLocation::Device>( _d_r.data(), static_cast<size_t>( _n ) );
        r0_norm = resid;
        check_cublas_error( cublasDdot( _cublas_handle, _n, _d_r0.data(), 1, _d_r.data(), 1, &rho ),
                            "Failed to initialize rho" );
        at_recurrence_start = true;
        shadow_repaired = false;
    };

    enum class RefreshOutcome
    {
        RESTARTED,
        CONVERGED,
        FAILED
    };

    auto refresh_solution_and_residual = [&]( const bool breakdown_restart )
    {
        // Extension beyond the original recurrence [1]: commit the grouped
        // update and recompute b-A*x before accepting convergence or restarting.
        // [3] analyzes reliable residual updates and restart trade-offs. Its
        // "flying restart" preserves more recurrence state; this implementation
        // deliberately performs a full recurrence restart to remain simple and
        // to discard state after an invalid denominator.
        // [3]: https://doi.org/10.1007/BF02309342
        update_solution( d_x );
        check_cuda_error( cudaMemset( _d_x_hat.data(), 0, static_cast<size_t>( _n ) * sizeof( double ) ),
                          "Failed to clear the committed BiCGSTAB solution update" );

        double true_resid = 0.0;
        resid = compute_residual( d_b, d_x, &true_resid );
        if ( !std::isfinite( resid ) || !std::isfinite( true_resid ) )
        {
            return RefreshOutcome::FAILED;
        }
        if ( check_convergence( true_resid, init_true_resid ) )
        {
            return RefreshOutcome::CONVERGED;
        }

        if ( breakdown_restart )
        {
            if ( static_cast<size_t>( _last_restarts ) >= _max_restarts )
            {
                return RefreshOutcome::FAILED;
            }
            ++_last_restarts;
        }

        initialize_recurrence();
        return RefreshOutcome::RESTARTED;
    };

    initialize_recurrence();

    // BiCGSTAB main iteration loop
    for ( size_t iter = 0; iter < _max_iter; ++iter )
    {
        // step 1: Compute alpha
        // step 1.1: Compute v = A * p (with preconditioning)
        _view_prec_x.setData( _d_p.data() );
        _view_prec_y.setData( _d_v.data() );
        apply_operator_with_preconditioning( _view_prec_x, _view_prec_y );

        // step 1.2: alpha = rho / (r_tilde, v)
        double rtilde_v;
        check_cublas_error( cublasDdot( _cublas_handle, _n, _d_r0.data(), 1, _d_v.data(), 1, &rtilde_v ),
                            "Failed to compute <r_tilde, v>" );
        double v_norm = 0.0;
        check_cublas_error( cublasDnrm2( _cublas_handle, _n, _d_v.data(), 1, &v_norm ),
                            "Failed to compute ||v||" );

        bool alpha_breakdown = near_inner_product_breakdown( rtilde_v, r0_norm, v_norm, _breakdown_tol ) ||
                               near_inner_product_breakdown( rho, r0_norm, resid, _breakdown_tol );
        if ( alpha_breakdown && at_recurrence_start && !shadow_repaired && resid > 0.0 && v_norm > 0.0 )
        {
            // Extension beyond [1]: r_tilde is arbitrary, so repair a nearly
            // orthogonal initial shadow vector deterministically. This exact
            // repair formula is an implementation policy; [2] provides the
            // finite-precision shadow-space/near-breakdown motivation.
            // [2]: https://doi.org/10.1007/BF02140769
            _d_r0.copy<MemoryLocation::Device>( _d_r.data(), static_cast<size_t>( _n ) );
            const double shadow_scale = resid / v_norm;
            check_cublas_error(
                cublasDaxpy( _cublas_handle, _n, &shadow_scale, _d_v.data(), 1, _d_r0.data(), 1 ),
                "Failed to repair the BiCGSTAB shadow residual" );
            check_cublas_error( cublasDnrm2( _cublas_handle, _n, _d_r0.data(), 1, &r0_norm ),
                                "Failed to compute the repaired shadow residual norm" );
            check_cublas_error( cublasDdot( _cublas_handle, _n, _d_r0.data(), 1, _d_r.data(), 1, &rho ),
                                "Failed to recompute rho after shadow repair" );
            check_cublas_error( cublasDdot( _cublas_handle, _n, _d_r0.data(), 1, _d_v.data(), 1, &rtilde_v ),
                                "Failed to recompute <r_tilde, v> after shadow repair" );
            shadow_repaired = true;
            alpha_breakdown = near_inner_product_breakdown( rtilde_v, r0_norm, v_norm, _breakdown_tol ) ||
                              near_inner_product_breakdown( rho, r0_norm, resid, _breakdown_tol );
        }

        if ( alpha_breakdown )
        {
            _last_iterations = static_cast<int>( iter + 1 );
            const RefreshOutcome outcome = refresh_solution_and_residual( true );
            if ( outcome == RefreshOutcome::CONVERGED )
            {
                return State::CONVERGED;
            }
            if ( outcome == RefreshOutcome::FAILED )
            {
                return State::FAILED;
            }
            continue;
        }

        const double alpha = rho / rtilde_v;
        if ( !std::isfinite( alpha ) )
        {
            _last_iterations = static_cast<int>( iter + 1 );
            const RefreshOutcome outcome = refresh_solution_and_residual( true );
            if ( outcome == RefreshOutcome::CONVERGED )
            {
                return State::CONVERGED;
            }
            if ( outcome == RefreshOutcome::FAILED )
            {
                return State::FAILED;
            }
            continue;
        }

        // step 2: s = r - alpha * v.
        _d_s.copy<MemoryLocation::Device>( _d_r.data(), static_cast<size_t>( _n ) );
        const double neg_alpha = -alpha;
        check_cublas_error( cublasDaxpy( _cublas_handle, _n, &neg_alpha, _d_v.data(), 1, _d_s.data(), 1 ),
                            "Failed to compute s" );
        double s_norm = 0.0;
        check_cublas_error( cublasDnrm2( _cublas_handle, _n, _d_s.data(), 1, &s_norm ),
                            "Failed to compute ||s||" );

        // Practical guard around [1]'s recurrence: the alpha step may already
        // be the exact solution. Algorithm 1 in [3], which reproduces standard
        // Bi-CGSTAB [1], proceeds directly to omega; checking here prevents the
        // resulting exact 0/0 when s=0.
        // [3]: https://doi.org/10.1007/BF02309342
        if ( check_convergence( s_norm, init_resid ) )
        {
            check_cublas_error( cublasDaxpy( _cublas_handle, _n, &alpha, _d_p.data(), 1, _d_x_hat.data(), 1 ),
                                "Failed to commit the converged alpha update" );
            _last_iterations = static_cast<int>( iter + 1 );
            const RefreshOutcome outcome = refresh_solution_and_residual( false );
            if ( outcome == RefreshOutcome::CONVERGED )
            {
                return State::CONVERGED;
            }
            if ( outcome == RefreshOutcome::FAILED )
            {
                return State::FAILED;
            }
            continue;
        }

        // step 3: Compute omega
        // step 3.1: Compute t = A * s (with preconditioning)
        _view_prec_x.setData( _d_s.data() );
        _view_prec_y.setData( _d_t.data() );
        apply_operator_with_preconditioning( _view_prec_x, _view_prec_y );

        // step 3.2: omega = (t, s) / (t, t)
        double t_s = 0.0;
        double t_t = 0.0;
        check_cublas_error( cublasDdot( _cublas_handle, _n, _d_t.data(), 1, _d_s.data(), 1, &t_s ),
                            "Failed to compute <t, s>" );
        check_cublas_error( cublasDdot( _cublas_handle, _n, _d_t.data(), 1, _d_t.data(), 1, &t_t ),
                            "Failed to compute <t, t>" );
        const double t_norm = t_t > 0.0 ? std::sqrt( t_t ) : 0.0;
        const bool omega_breakdown =
            t_t <= 0.0 || near_inner_product_breakdown( t_s, t_norm, s_norm, _breakdown_tol );
        if ( omega_breakdown )
        {
            // The alpha update is valid even if the minimal-residual omega
            // step breaks down, so preserve it before the full restart.
            check_cublas_error( cublasDaxpy( _cublas_handle, _n, &alpha, _d_p.data(), 1, _d_x_hat.data(), 1 ),
                                "Failed to preserve the alpha update before restart" );
            _last_iterations = static_cast<int>( iter + 1 );
            const RefreshOutcome outcome = refresh_solution_and_residual( true );
            if ( outcome == RefreshOutcome::CONVERGED )
            {
                return State::CONVERGED;
            }
            if ( outcome == RefreshOutcome::FAILED )
            {
                return State::FAILED;
            }
            continue;
        }

        const double omega = t_s / t_t;
        if ( !std::isfinite( omega ) )
        {
            check_cublas_error( cublasDaxpy( _cublas_handle, _n, &alpha, _d_p.data(), 1, _d_x_hat.data(), 1 ),
                                "Failed to preserve the alpha update before restart" );
            _last_iterations = static_cast<int>( iter + 1 );
            const RefreshOutcome outcome = refresh_solution_and_residual( true );
            if ( outcome == RefreshOutcome::CONVERGED )
            {
                return State::CONVERGED;
            }
            if ( outcome == RefreshOutcome::FAILED )
            {
                return State::FAILED;
            }
            continue;
        }

        // step 4: x_hat = x_hat + alpha * p + omega * s.
        check_cublas_error( cublasDaxpy( _cublas_handle, _n, &alpha, _d_p.data(), 1, _d_x_hat.data(), 1 ),
                            "Failed to apply the alpha solution update" );
        check_cublas_error( cublasDaxpy( _cublas_handle, _n, &omega, _d_s.data(), 1, _d_x_hat.data(), 1 ),
                            "Failed to apply the omega solution update" );

        // step 5: r = s - omega * t.
        _d_r.copy<MemoryLocation::Device>( _d_s.data(), static_cast<size_t>( _n ) );
        const double neg_omega = -omega;
        check_cublas_error( cublasDaxpy( _cublas_handle, _n, &neg_omega, _d_t.data(), 1, _d_r.data(), 1 ),
                            "Failed to update the residual" );

        // step 6: test the recursively updated residual.
        check_cublas_error( cublasDnrm2( _cublas_handle, _n, _d_r.data(), 1, &resid ),
                            "Failed to compute the residual norm" );

        if ( _verbose )
        {
            print_iteration_info( static_cast<int>( iter ), resid, init_resid );
        }

        if ( !std::isfinite( resid ) )
        {
            _last_iterations = static_cast<int>( iter + 1 );
            return State::FAILED;
        }

        if ( check_convergence( resid, init_resid ) )
        {
            _last_iterations = static_cast<int>( iter + 1 );
            const RefreshOutcome outcome = refresh_solution_and_residual( false );
            if ( outcome == RefreshOutcome::CONVERGED )
            {
                return State::CONVERGED;
            }
            if ( outcome == RefreshOutcome::FAILED )
            {
                return State::FAILED;
            }
            continue;
        }

        if ( _residual_replacement_frequency > 0 && ( iter + 1 ) % _residual_replacement_frequency == 0 )
        {
            // Extension beyond [1]: fixed-frequency residual replacement is a
            // simple version of the reliable-update framework in [3]. The
            // interval of 50 is an implementation default, not prescribed by
            // that paper; set the frequency to zero to disable it.
            // [3]: https://doi.org/10.1007/BF02309342
            _last_iterations = static_cast<int>( iter + 1 );
            const RefreshOutcome outcome = refresh_solution_and_residual( false );
            if ( outcome == RefreshOutcome::CONVERGED )
            {
                return State::CONVERGED;
            }
            if ( outcome == RefreshOutcome::FAILED )
            {
                return State::FAILED;
            }
            continue;
        }

        // step 7: rho_new = (r_tilde, r).
        double rho_new = 0.0;
        check_cublas_error( cublasDdot( _cublas_handle, _n, _d_r0.data(), 1, _d_r.data(), 1, &rho_new ),
                            "Failed to compute the next rho" );
        if ( near_inner_product_breakdown( rho_new, r0_norm, resid, _breakdown_tol ) )
        {
            _last_iterations = static_cast<int>( iter + 1 );
            const RefreshOutcome outcome = refresh_solution_and_residual( true );
            if ( outcome == RefreshOutcome::CONVERGED )
            {
                return State::CONVERGED;
            }
            if ( outcome == RefreshOutcome::FAILED )
            {
                return State::FAILED;
            }
            continue;
        }

        const double beta = ( rho_new / rho ) * ( alpha / omega );
        if ( !std::isfinite( beta ) )
        {
            _last_iterations = static_cast<int>( iter + 1 );
            const RefreshOutcome outcome = refresh_solution_and_residual( true );
            if ( outcome == RefreshOutcome::CONVERGED )
            {
                return State::CONVERGED;
            }
            if ( outcome == RefreshOutcome::FAILED )
            {
                return State::FAILED;
            }
            continue;
        }
        rho = rho_new;

        // step 8: p = r + beta * (p - omega * v).
        check_cublas_error( cublasDaxpy( _cublas_handle, _n, &neg_omega, _d_v.data(), 1, _d_p.data(), 1 ),
                            "Failed to compute p - omega*v" );
        check_cublas_error( cublasDscal( _cublas_handle, _n, &beta, _d_p.data(), 1 ),
                            "Failed to scale the search direction" );
        const double one = 1.0;
        check_cublas_error( cublasDaxpy( _cublas_handle, _n, &one, _d_r.data(), 1, _d_p.data(), 1 ),
                            "Failed to add the residual to the search direction" );
        at_recurrence_start = false;
    }

    _last_iterations = static_cast<int>( _max_iter );
    update_solution( d_x );
    return State::MAX_ITER_REACHED;
}

double CudaBiCGSTAB::compute_residual( const DeviceVectorView& d_b, const DeviceVectorView& d_x, double* true_residual )
{
    // Compute r = b - Ax
    // First compute Ax into a temporary vector
    _spmv_operator->operator()( d_x.data(), _d_tmp.data(), 1.0, 0.0 );

    // Then compute r = b - Ax using axpy: r = b + (-1.0) * Ax
    _d_r.copy<MemoryLocation::Device>( d_b.data(), _n );
    const double neg_one = -1.0;
    check_cublas_error( cublasDaxpy( _cublas_handle, _n, &neg_one, _d_tmp.data(), 1, _d_r.data(), 1 ),
                        "Failed to compute residual" );

    double norm = 0.0;
    check_cublas_error( cublasDnrm2( _cublas_handle, _n, _d_r.data(), 1, &norm ),
                        "Failed to compute the true residual norm" );
    if ( true_residual )
    {
        *true_residual = norm;
    }

    // Apply left preconditioning after recording the true residual norm.
    if ( _prec_type == PreconditionerType::LEFT )
    {
        _view_prec_x.setData( _d_r.data() );
        _view_prec_y.setData( _d_r.data() );
        _preconditioner->operator()( _view_prec_x, _view_prec_y );
        check_cublas_error( cublasDnrm2( _cublas_handle, _n, _d_r.data(), 1, &norm ),
                            "Failed to compute the preconditioned residual norm" );
    }

    return norm;
}

void CudaBiCGSTAB::apply_operator_with_preconditioning( const DeviceVectorView& d_input, DeviceVectorView& d_output )
{
    switch ( _prec_type )
    {
    case PreconditionerType::RIGHT:
        // Right preconditioning: A * M^{-1} * input
        _preconditioner->operator()( d_input, _view_prec_tmp );
        _spmv_operator->operator()( _view_prec_tmp.data(), d_output.data(), 1.0, 0.0 );
        break;

    case PreconditionerType::LEFT:
        // Left preconditioning: M^{-1} * A * input
        _spmv_operator->operator()( d_input.data(), d_output.data(), 1.0, 0.0 );
        _preconditioner->operator()( d_output, d_output );
        break;

    case PreconditionerType::NONE:
        // No preconditioning: A * input
        _spmv_operator->operator()( d_input.data(), d_output.data(), 1.0, 0.0 );
        break;
    }
}

void CudaBiCGSTAB::update_solution( DeviceVectorView& d_x )
{
    if ( _prec_type == PreconditionerType::RIGHT )
    {
        // Right preconditioning: x = x + M^{-1} * x_hat
        _view_prec_x.setData( _d_x_hat.data() );
        _view_prec_y.setData( _d_tmp.data() );
        _preconditioner->operator()( _view_prec_x, _view_prec_y );
        const double one = 1.0;
        check_cublas_error( cublasDaxpy( _cublas_handle, _n, &one, _d_tmp.data(), 1, d_x.data(), 1 ),
                            "Failed to update solution with preconditioned x_hat" );
    }
    else
    {
        // Left or no preconditioning: x = x + x_hat
        const double one = 1.0;
        check_cublas_error( cublasDaxpy( _cublas_handle, _n, &one, _d_x_hat.data(), 1, d_x.data(), 1 ),
                            "Failed to update solution with x_hat" );
    }
}

bool CudaBiCGSTAB::check_convergence( double resid, double init_resid ) const
{
    if ( !std::isfinite( resid ) || !std::isfinite( init_resid ) )
    {
        return false;
    }

    const double tolerance = std::max( _abs_tol, _rel_tol * std::abs( init_resid ) );
    return std::abs( resid ) <= tolerance;
}

void CudaBiCGSTAB::print_iteration_info( int iter, double resid, double init_resid ) const
{
    std::cout << "iter: " << std::setw( 4 ) << iter << " resid: " << std::scientific
              << std::setprecision( 4 ) << std::abs( resid )
              << " relative resid: " << std::scientific << std::setprecision( 4 )
              << std::abs( resid ) / std::max( std::abs( init_resid ), std::numeric_limits<double>::min() )
              << std::endl;
}

void CudaBiCGSTAB::check_cuda_error( cudaError_t error, const char* message )
{
    if ( error != cudaSuccess )
    {
        std::cerr << "CUDA Error: " << message << " - " << cudaGetErrorString( error ) << std::endl;
        throw std::runtime_error( message );
    }
}

void CudaBiCGSTAB::check_cublas_error( cublasStatus_t status, const char* message )
{
    if ( status != CUBLAS_STATUS_SUCCESS )
    {
        std::cerr << "cuBLAS Error: " << message << " - Status: " << status << std::endl;
        throw std::runtime_error( message );
    }
}

void CudaBiCGSTAB::check_cusparse_error( cusparseStatus_t status, const char* message )
{
    if ( status != CUSPARSE_STATUS_SUCCESS )
    {
        std::cerr << "cuSPARSE Error: " << message << " - Status: " << status << std::endl;
        throw std::runtime_error( message );
    }
}

} // namespace matrix_utils::sparse_cuda
