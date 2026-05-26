#include "io.hpp"
#include "matrix_utils.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cxxopts.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#define CHECK_CUDA( call )                                                        \
    do                                                                            \
    {                                                                             \
        cudaError_t err = call;                                                   \
        if ( err != cudaSuccess )                                                 \
        {                                                                         \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString( err ) << std::endl;                  \
            exit( 1 );                                                            \
        }                                                                         \
    } while ( 0 )

#define CHECK_CUSPARSE( call )                                                                             \
    do                                                                                                     \
    {                                                                                                      \
        cusparseStatus_t err = call;                                                                       \
        if ( err != CUSPARSE_STATUS_SUCCESS )                                                              \
        {                                                                                                  \
            std::cerr << "cuSPARSE error at " << __FILE__ << ":" << __LINE__ << " - " << err << std::endl; \
            exit( 1 );                                                                                     \
        }                                                                                                  \
    } while ( 0 )

// Use CSRMatrix from matrix_utils.hpp
using CSRMatrix = matrix_utils::CSRMatrixVec<int, int, double>;

class CudaLUBenchmark
{
private:
    int n;

    // Host data for L and U matrices
    CSRMatrix h_L, h_U;
    std::vector<double> h_x, h_b, h_y; // y is intermediate vector for L*y = b
    std::vector<double> h_x_ref;

    // Device data for L matrix
    int *d_L_rowPtr, *d_L_colInd;
    double* d_L_val;

    // Device data for U matrix
    int *d_U_rowPtr, *d_U_colInd;
    double* d_U_val;

    // Device vectors
    double *d_x, *d_b, *d_y;

    // cuSPARSE handle and descriptors
    cusparseHandle_t cusparse_handle;
    cusparseSpMatDescr_t matL, matU;
    cusparseDnVecDescr_t vecB, vecX, vecY;
    cusparseDnMatDescr_t matB, matX, matY; // Dense matrix descriptors for SpSM
    cusparseSpSVDescr_t spsvDescrL, spsvDescrU;
    cusparseSpSMDescr_t spsmDescrL, spsmDescrU;

    // Buffers for SpSV
    size_t bufferSizeL, bufferSizeU;
    void* externalBufferL;
    void* externalBufferU;

    // Buffers for SpSM
    size_t bufferSizeSML, bufferSizeSMU;
    void* externalBufferSML;
    void* externalBufferSMU;

public:
    CudaLUBenchmark( int matrix_size,
                     double sparsity = 0.1,
                     const std::string& l_file = "",
                     const std::string& u_file = "" )
        : n( matrix_size )
    {
        // Initialize cuSPARSE
        CHECK_CUSPARSE( cusparseCreate( &cusparse_handle ) );

        if ( !l_file.empty() && !u_file.empty() )
        {
            readSeparateLUFiles( l_file, u_file );
        }
        else
        {
            generateLUMatrices( sparsity );
        }

        allocateDeviceMemory();
        copyToDevice();
        createSparseDescriptors();
        analyzeSpSV();
        analyzeSpSM();
    }

    ~CudaLUBenchmark()
    {
        // Clean up device memory
        cudaFree( d_L_rowPtr );
        cudaFree( d_L_colInd );
        cudaFree( d_L_val );
        cudaFree( d_U_rowPtr );
        cudaFree( d_U_colInd );
        cudaFree( d_U_val );
        cudaFree( d_x );
        cudaFree( d_b );
        cudaFree( d_y );
        cudaFree( externalBufferL );
        cudaFree( externalBufferU );
        cudaFree( externalBufferSML );
        cudaFree( externalBufferSMU );

        // Destroy cuSPARSE objects
        cusparseDestroySpMat( matL );
        cusparseDestroySpMat( matU );
        cusparseDestroyDnVec( vecB );
        cusparseDestroyDnVec( vecX );
        cusparseDestroyDnVec( vecY );
        cusparseDestroyDnMat( matB );
        cusparseDestroyDnMat( matX );
        cusparseDestroyDnMat( matY );
        cusparseSpSV_destroyDescr( spsvDescrL );
        cusparseSpSV_destroyDescr( spsvDescrU );
        cusparseSpSM_destroyDescr( spsmDescrL );
        cusparseSpSM_destroyDescr( spsmDescrU );
        cusparseDestroy( cusparse_handle );
    }

    void readSeparateLUFiles( const std::string& l_filename, const std::string& u_filename )
    {
        // Read L matrix using matrix_utils::readMatrixMarket
        std::ifstream l_file( l_filename );
        if ( !l_file.is_open() )
        {
            throw std::runtime_error( "Cannot open L matrix file: " + l_filename );
        }
        matrix_utils::readMatrixMarket( l_file, h_L );
        l_file.close();

        // Read U matrix using matrix_utils::readMatrixMarket
        std::ifstream u_file( u_filename );
        if ( !u_file.is_open() )
        {
            throw std::runtime_error( "Cannot open U matrix file: " + u_filename );
        }
        matrix_utils::readMatrixMarket( u_file, h_U );
        u_file.close();

        std::vector<int> ipermL( h_L.rows );
        std::vector<int> prefixL( h_L.rows + 1 );

        graph::TopologicalSort2<int, int, matrix_utils::TriangularMatrix::L> kahnL;
        auto levelL = kahnL( h_L.rows, h_L.AI(), h_L.AJ(), ipermL.data(), prefixL.data() );
        int maxL = 0;
        for ( int i = 0; i < levelL; ++i )
            if ( prefixL[i + 1] - prefixL[i] > maxL )
                maxL = prefixL[i + 1] - prefixL[i];

        std::vector<int> ipermU( h_U.rows );
        std::vector<int> prefixU( h_U.rows + 1 );
        graph::TopologicalSort2<int, int, matrix_utils::TriangularMatrix::U> kahnU;
        auto levelU = kahnU( h_U.rows, h_U.AI(), h_U.AJ(), ipermU.data(), prefixU.data() );
        int maxU = 0;
        for ( int i = 0; i < levelU; ++i )
            if ( prefixU[i + 1] - prefixU[i] > maxU )
                maxU = prefixU[i + 1] - prefixU[i];

        // Validate dimensions
        if ( h_L.rows != h_U.rows )
        {
            throw std::runtime_error( "L and U matrices must have the same dimensions" );
        }

        n = h_L.rows;
        generateRandomRHS();

        std::cout << "Read L matrix (" << h_L.rows << "x" << h_L.rows << ") from " << l_filename << std::endl;
        std::cout << "L matrix: " << h_L.NNZ() << " non-zeros" << std::endl;
        std::cout << "Read U matrix (" << h_U.rows << "x" << h_U.rows << ") from " << u_filename << std::endl;
        std::cout << "U matrix: " << h_U.NNZ() << " non-zeros" << std::endl;
        std::cout << "L matrix levels: " << levelL << std::endl;
        std::cout << "U matrix levels: " << levelU << std::endl;
        std::cout << "L matrix maximum width: " << maxL << std::endl;
        std::cout << "U matrix maximum width: " << maxU << std::endl;
    }

    void generateLUMatrices( double sparsity )
    {
        std::random_device rd;
        std::mt19937 gen( 42 ); // Fixed seed for reproducibility
        std::uniform_real_distribution<double> val_dist( -1.0, 1.0 );
        std::uniform_real_distribution<double> sparse_dist( 0.0, 1.0 );

        h_L = CSRMatrix();
        h_L.rows = n;
        h_L.cols = n;
        h_U = CSRMatrix();
        h_U.rows = n;
        h_U.cols = n;

        std::vector<std::vector<std::pair<int, double>>> L_rows( n ), U_rows( n );

        // Generate L matrix (lower triangular with unit diagonal)
        for ( int i = 0; i < n; ++i )
        {
            // Add unit diagonal
            L_rows[i].emplace_back( i, 1.0 );

            // Add off-diagonal elements
            for ( int j = 0; j < i; ++j )
            {
                if ( sparse_dist( gen ) < sparsity )
                {
                    L_rows[i].emplace_back( j, val_dist( gen ) );
                }
            }
            std::sort( L_rows[i].begin(), L_rows[i].end() );
        }

        // Generate U matrix (upper triangular)
        for ( int i = 0; i < n; ++i )
        {
            // Add diagonal element
            double diag_val = val_dist( gen );
            if ( std::abs( diag_val ) < 0.1 )
                diag_val = ( diag_val >= 0 ) ? 0.1 : -0.1;
            U_rows[i].emplace_back( i, diag_val );

            // Add off-diagonal elements
            for ( int j = i + 1; j < n; ++j )
            {
                if ( sparse_dist( gen ) < sparsity )
                {
                    U_rows[i].emplace_back( j, val_dist( gen ) );
                }
            }
            std::sort( U_rows[i].begin(), U_rows[i].end() );
        }

        // Convert to CSR format
        convertToCSR( L_rows, h_L );
        convertToCSR( U_rows, h_U );

        generateRandomRHS();

        std::cout << "Generated " << n << "x" << n << " LU matrices" << std::endl;
        std::cout << "L matrix: " << h_L.NNZ() << " non-zeros (density: " << std::fixed
                  << std::setprecision( 4 ) << 100.0 * h_L.NNZ() / ( double( n ) * n ) << "%)" << std::endl;
        std::cout << "U matrix: " << h_U.NNZ() << " non-zeros (density: " << std::fixed
                  << std::setprecision( 4 ) << 100.0 * h_U.NNZ() / ( double( n ) * n ) << "%)" << std::endl;
    }

    void convertToCSR( const std::vector<std::vector<std::pair<int, double>>>& rows, CSRMatrix& matrix )
    {
        matrix.ai.resize( n + 1 );
        matrix.ai[0] = 0;
        for ( int i = 0; i < n; ++i )
        {
            for ( const auto& [col, val] : rows[i] )
            {
                matrix.aj.push_back( col );
                matrix.av.push_back( val );
            }
            matrix.ai[i + 1] = matrix.aj.size();
        }
    }

    void generateRandomRHS()
    {
        std::random_device rd;
        std::mt19937 gen( 42 );
        std::uniform_real_distribution<double> val_dist( -1.0, 1.0 );

        h_b.resize( n );
        for ( int i = 0; i < n; ++i )
        {
            h_b[i] = val_dist( gen );
        }

        h_x.resize( n, 0.0 );
        h_y.resize( n, 0.0 );
        h_x_ref.resize( n, 0.0 );
    }

    void allocateDeviceMemory()
    {
        // Allocate L matrix
        CHECK_CUDA( cudaMalloc( &d_L_rowPtr, ( n + 1 ) * sizeof( int ) ) );
        CHECK_CUDA( cudaMalloc( &d_L_colInd, h_L.NNZ() * sizeof( int ) ) );
        CHECK_CUDA( cudaMalloc( &d_L_val, h_L.NNZ() * sizeof( double ) ) );

        // Allocate U matrix
        CHECK_CUDA( cudaMalloc( &d_U_rowPtr, ( n + 1 ) * sizeof( int ) ) );
        CHECK_CUDA( cudaMalloc( &d_U_colInd, h_U.NNZ() * sizeof( int ) ) );
        CHECK_CUDA( cudaMalloc( &d_U_val, h_U.NNZ() * sizeof( double ) ) );

        // Allocate vectors
        CHECK_CUDA( cudaMalloc( &d_x, n * sizeof( double ) ) );
        CHECK_CUDA( cudaMalloc( &d_b, n * sizeof( double ) ) );
        CHECK_CUDA( cudaMalloc( &d_y, n * sizeof( double ) ) );
    }

    void copyToDevice()
    {
        // Copy L matrix
        CHECK_CUDA( cudaMemcpy( d_L_rowPtr, h_L.ai.data(), ( n + 1 ) * sizeof( int ), cudaMemcpyHostToDevice ) );
        CHECK_CUDA( cudaMemcpy( d_L_colInd, h_L.aj.data(), h_L.NNZ() * sizeof( int ), cudaMemcpyHostToDevice ) );
        CHECK_CUDA( cudaMemcpy( d_L_val, h_L.av.data(), h_L.NNZ() * sizeof( double ), cudaMemcpyHostToDevice ) );

        // Copy U matrix
        CHECK_CUDA( cudaMemcpy( d_U_rowPtr, h_U.ai.data(), ( n + 1 ) * sizeof( int ), cudaMemcpyHostToDevice ) );
        CHECK_CUDA( cudaMemcpy( d_U_colInd, h_U.aj.data(), h_U.NNZ() * sizeof( int ), cudaMemcpyHostToDevice ) );
        CHECK_CUDA( cudaMemcpy( d_U_val, h_U.av.data(), h_U.NNZ() * sizeof( double ), cudaMemcpyHostToDevice ) );

        // Copy vectors
        CHECK_CUDA( cudaMemcpy( d_b, h_b.data(), n * sizeof( double ), cudaMemcpyHostToDevice ) );
    }

    void createSparseDescriptors()
    {
        // Create sparse matrix descriptors for L and U
        CHECK_CUSPARSE( cusparseCreateCsr( &matL, n, n, h_L.NNZ(), d_L_rowPtr, d_L_colInd, d_L_val, CUSPARSE_INDEX_32I,
                                           CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F ) );

        CHECK_CUSPARSE( cusparseCreateCsr( &matU, n, n, h_U.NNZ(), d_U_rowPtr, d_U_colInd, d_U_val, CUSPARSE_INDEX_32I,
                                           CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F ) );

        // Create dense vector descriptors
        CHECK_CUSPARSE( cusparseCreateDnVec( &vecB, n, d_b, CUDA_R_64F ) );
        CHECK_CUSPARSE( cusparseCreateDnVec( &vecX, n, d_x, CUDA_R_64F ) );
        CHECK_CUSPARSE( cusparseCreateDnVec( &vecY, n, d_y, CUDA_R_64F ) );

        // Create dense matrix descriptors (treating vectors as n x 1 matrices for SpSM)
        CHECK_CUSPARSE( cusparseCreateDnMat( &matB, n, 1, n, d_b, CUDA_R_64F, CUSPARSE_ORDER_COL ) );
        CHECK_CUSPARSE( cusparseCreateDnMat( &matX, n, 1, n, d_x, CUDA_R_64F, CUSPARSE_ORDER_COL ) );
        CHECK_CUSPARSE( cusparseCreateDnMat( &matY, n, 1, n, d_y, CUDA_R_64F, CUSPARSE_ORDER_COL ) );

        // Set matrix properties
        cusparseFillMode_t fillModeL = CUSPARSE_FILL_MODE_LOWER;
        cusparseDiagType_t diagTypeL = CUSPARSE_DIAG_TYPE_UNIT;
        cusparseSpMatSetAttribute( matL, CUSPARSE_SPMAT_FILL_MODE, &fillModeL, sizeof( cusparseFillMode_t ) );
        cusparseSpMatSetAttribute( matL, CUSPARSE_SPMAT_DIAG_TYPE, &diagTypeL, sizeof( cusparseDiagType_t ) );

        cusparseFillMode_t fillModeU = CUSPARSE_FILL_MODE_UPPER;
        cusparseDiagType_t diagTypeU = CUSPARSE_DIAG_TYPE_NON_UNIT;
        cusparseSpMatSetAttribute( matU, CUSPARSE_SPMAT_FILL_MODE, &fillModeU, sizeof( cusparseFillMode_t ) );
        cusparseSpMatSetAttribute( matU, CUSPARSE_SPMAT_DIAG_TYPE, &diagTypeU, sizeof( cusparseDiagType_t ) );

        // Create SpSV descriptors
        CHECK_CUSPARSE( cusparseSpSV_createDescr( &spsvDescrL ) );
        CHECK_CUSPARSE( cusparseSpSV_createDescr( &spsvDescrU ) );
    }

    void analyzeSpSV()
    {
        const double alpha = 1.0;

        // Analyze L matrix for forward substitution
        CHECK_CUSPARSE( cusparseSpSV_bufferSize( cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 &alpha, matL, vecB, vecY, CUDA_R_64F,
                                                 CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &bufferSizeL ) );

        CHECK_CUDA( cudaMalloc( &externalBufferL, bufferSizeL ) );

        CHECK_CUSPARSE( cusparseSpSV_analysis( cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                               matL, vecB, vecY, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
                                               spsvDescrL, externalBufferL ) );

        // Analyze U matrix for backward substitution
        CHECK_CUSPARSE( cusparseSpSV_bufferSize( cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 &alpha, matU, vecY, vecX, CUDA_R_64F,
                                                 CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, &bufferSizeU ) );

        CHECK_CUDA( cudaMalloc( &externalBufferU, bufferSizeU ) );

        CHECK_CUSPARSE( cusparseSpSV_analysis( cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                               matU, vecY, vecX, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
                                               spsvDescrU, externalBufferU ) );
    }

    void analyzeSpSM()
    {
        const double alpha = 1.0;

        // Create SpSM descriptors
        CHECK_CUSPARSE( cusparseSpSM_createDescr( &spsmDescrL ) );
        CHECK_CUSPARSE( cusparseSpSM_createDescr( &spsmDescrU ) );

        // Analyze L matrix for forward substitution (SpSM)
        CHECK_CUSPARSE( cusparseSpSM_bufferSize(
            cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            matL, matB, matY, CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT, spsmDescrL, &bufferSizeSML ) );

        CHECK_CUDA( cudaMalloc( &externalBufferSML, bufferSizeSML ) );

        CHECK_CUSPARSE( cusparseSpSM_analysis(
            cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            matL, matB, matY, CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT, spsmDescrL, externalBufferSML ) );

        // Analyze U matrix for backward substitution (SpSM)
        CHECK_CUSPARSE( cusparseSpSM_bufferSize(
            cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            matU, matY, matX, CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT, spsmDescrU, &bufferSizeSMU ) );

        CHECK_CUDA( cudaMalloc( &externalBufferSMU, bufferSizeSMU ) );

        CHECK_CUSPARSE( cusparseSpSM_analysis(
            cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            matU, matY, matX, CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT, spsmDescrU, externalBufferSMU ) );
    }

    void cudaForwardBackwardSpSV()
    {
        const double alpha = 1.0;

        // Reset solution vectors
        CHECK_CUDA( cudaMemset( d_y, 0, n * sizeof( double ) ) );
        CHECK_CUDA( cudaMemset( d_x, 0, n * sizeof( double ) ) );

        // Forward substitution: L * y = b using cuSPARSE SpSV
        CHECK_CUSPARSE( cusparseSpSV_solve( cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matL,
                                            vecB, vecY, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL ) );

        // Backward substitution: U * x = y using cuSPARSE SpSV
        CHECK_CUSPARSE( cusparseSpSV_solve( cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matU,
                                            vecY, vecX, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU ) );
    }

    void cudaForwardBackwardSpSM()
    {
        const double alpha = 1.0;

        // Reset solution vectors
        CHECK_CUDA( cudaMemset( d_y, 0, n * sizeof( double ) ) );
        CHECK_CUDA( cudaMemset( d_x, 0, n * sizeof( double ) ) );

        // Forward substitution: L * y = b using cuSPARSE SpSM
        CHECK_CUSPARSE( cusparseSpSM_solve( cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matL, matB, matY,
                                            CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT, spsmDescrL ) );

        // Backward substitution: U * x = y using cuSPARSE SpSM
        CHECK_CUSPARSE( cusparseSpSM_solve( cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matU, matY, matX,
                                            CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT, spsmDescrU ) );
    }

    double benchmarkForwardBackward( int num_iterations = 1000 )
    {
        // Warm up GPU
        cudaForwardBackwardSpSV();
        CHECK_CUDA( cudaDeviceSynchronize() );

        // Benchmark GPU-based forward-backward substitution
        auto start = std::chrono::high_resolution_clock::now();

        for ( int i = 0; i < num_iterations; ++i )
        {
            cudaForwardBackwardSpSV();
        }

        CHECK_CUDA( cudaDeviceSynchronize() );
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end - start );
        double avg_time_ms = duration.count() / ( 1000.0 * num_iterations );

        // Copy result back from GPU
        CHECK_CUDA( cudaMemcpy( h_x.data(), d_x, n * sizeof( double ), cudaMemcpyDeviceToHost ) );

        return avg_time_ms;
    }

    double benchmarkForwardBackwardSpSM( int num_iterations = 1000 )
    {
        // Warm up GPU
        cudaForwardBackwardSpSM();
        CHECK_CUDA( cudaDeviceSynchronize() );

        // Benchmark GPU-based forward-backward substitution using SpSM
        auto start = std::chrono::high_resolution_clock::now();

        for ( int i = 0; i < num_iterations; ++i )
        {
            cudaForwardBackwardSpSM();
        }

        CHECK_CUDA( cudaDeviceSynchronize() );
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end - start );
        double avg_time_ms = duration.count() / ( 1000.0 * num_iterations );

        // Copy result back from GPU
        CHECK_CUDA( cudaMemcpy( h_x.data(), d_x, n * sizeof( double ), cudaMemcpyDeviceToHost ) );

        return avg_time_ms;
    }

    void cpuReference()
    {
        std::fill( h_x_ref.begin(), h_x_ref.end(), 0.0 );
        std::fill( h_y.begin(), h_y.end(), 0.0 );

        // Forward substitution: L * y = b
        for ( int i = 0; i < n; ++i )
        {
            double sum = h_b[i];
            for ( int j = h_L.ai[i]; j < h_L.ai[i + 1]; ++j )
            {
                int col = h_L.aj[j];
                if ( col < i )
                {
                    sum -= h_L.av[j] * h_y[col];
                }
            }
            h_y[i] = sum; // L has unit diagonal
        }

        // Backward substitution: U * x = y
        for ( int i = n - 1; i >= 0; --i )
        {
            double sum = h_y[i];
            double diag_val = 1.0;

            for ( int j = h_U.ai[i]; j < h_U.ai[i + 1]; ++j )
            {
                int col = h_U.aj[j];
                if ( col == i )
                {
                    diag_val = h_U.av[j];
                }
                else if ( col > i )
                {
                    sum -= h_U.av[j] * h_x_ref[col];
                }
            }
            h_x_ref[i] = sum / diag_val;
        }
    }

    double computeError()
    {
        double sum_squared_error = 0.0;
        double sum_squared_ref = 0.0;

        for ( int i = 0; i < n; ++i )
        {
            double diff = h_x[i] - h_x_ref[i];
            sum_squared_error += diff * diff;
            sum_squared_ref += h_x_ref[i] * h_x_ref[i];
        }

        // Compute relative L2 norm: ||x - x_ref||_2 / ||x_ref||_2
        if ( sum_squared_ref > 1e-30 ) // Avoid division by zero
        {
            return std::sqrt( sum_squared_error / sum_squared_ref );
        }
        else
        {
            // If reference is very small, return absolute L2 norm
            return std::sqrt( sum_squared_error );
        }
    }

    void printResults( double gpu_time_ms, const std::string& method )
    {
        // Compute CPU reference for verification
        cpuReference();
        double error = computeError();

        double flops = 2.0 * ( h_L.NNZ() + h_U.NNZ() ); // Approximate FLOPs for forward + backward substitution
        double gflops = ( flops / 1e9 ) / ( gpu_time_ms / 1000.0 );

        std::cout << std::fixed << std::setprecision( 6 );
        std::cout << "cuSPARSE " << method << " time: " << gpu_time_ms << " ms" << std::endl;
        std::cout << "Performance: " << std::setprecision( 3 ) << gflops << " GFLOP/s" << std::endl;
        std::cout << "Relative L2 error vs CPU reference: " << std::scientific
                  << std::setprecision( 2 ) << error << std::endl;
        std::cout << "Implementation: cuSPARSE " << method;
        if ( method == "SpSV" )
        {
            std::cout << " (Sparse triangular solve)";
        }
        else
        {
            std::cout << " (Sparse matrix-dense matrix multiply)";
        }
        std::cout << std::endl;
        std::cout << "L matrix nnz: " << h_L.NNZ() << ", U matrix nnz: " << h_U.NNZ() << std::endl;
    }

    void printResultsSpSM( double gpu_time_ms )
    {
        // Compute CPU reference for verification
        cpuReference();
        double error = computeError();

        double flops = 2.0 * ( h_L.NNZ() + h_U.NNZ() ); // Approximate FLOPs for forward + backward substitution
        double gflops = ( flops / 1e9 ) / ( gpu_time_ms / 1000.0 );

        std::cout << std::fixed << std::setprecision( 6 );
        std::cout << "cuSPARSE SpSM time: " << gpu_time_ms << " ms" << std::endl;
        std::cout << "Performance: " << std::setprecision( 3 ) << gflops << " GFLOP/s" << std::endl;
        std::cout << "Relative L2 error vs CPU reference: " << std::scientific
                  << std::setprecision( 2 ) << error << std::endl;
        std::cout << "Implementation: cuSPARSE SpSM (Sparse matrix-dense "
                     "matrix multiply)"
                  << std::endl;
        std::cout << "L matrix nnz: " << h_L.NNZ() << ", U matrix nnz: " << h_U.NNZ() << std::endl;
    }
};

void printDeviceInfo()
{
    int device;
    CHECK_CUDA( cudaGetDevice( &device ) );

    cudaDeviceProp prop;
    CHECK_CUDA( cudaGetDeviceProperties( &prop, device ) );

    std::cout << "=== CUDA Device Information ===" << std::endl;
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Memory: " << prop.totalGlobalMem / ( 1024 * 1024 ) << " MB" << std::endl;
    std::cout << "SMs: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "================================" << std::endl << std::endl;
}

int main( int argc, char** argv )
{
    // Parse command line arguments with cxxopts
    cxxopts::Options options( "cuSPARSE SpSV vs SpSM Performance Test",
                              "Benchmark cuSPARSE SpSV and SpSM performance for triangular solve" );

    options.add_options()( "n,size", "Matrix size (ignored if file is provided)",
                           cxxopts::value<int>()->default_value( "10000" ) )(
        "s,sparsity", "Matrix sparsity for generated matrices (0.0-1.0)",
        cxxopts::value<double>()->default_value( "0.05" ) )(
        "l,lfile", "L matrix Market file path (optional)", cxxopts::value<std::string>()->default_value( "" ) )(
        "u,ufile", "U matrix Market file path (optional)", cxxopts::value<std::string>()->default_value( "" ) )(
        "i,iterations", "Number of benchmark iterations", cxxopts::value<int>()->default_value( "1000" ) )(
        "m,method", "Benchmark method: spsv, spsm, or both",
        cxxopts::value<std::string>()->default_value( "both" ) )( "h,help", "Show help message" );

    auto result = options.parse( argc, argv );

    if ( result.count( "help" ) )
    {
        std::cout << options.help() << std::endl;
        return 0;
    }

    // Extract parameters
    int matrix_size = result["size"].as<int>();
    double sparsity = result["sparsity"].as<double>();
    std::string l_file = result["lfile"].as<std::string>();
    std::string u_file = result["ufile"].as<std::string>();
    int num_iterations = result["iterations"].as<int>();
    std::string method = result["method"].as<std::string>();

    // Validate file options
    bool has_separate_files = !l_file.empty() && !u_file.empty();
    bool has_partial_files = !l_file.empty() || !u_file.empty();

    if ( has_partial_files && !has_separate_files )
    {
        std::cerr << "Error: Both L file (-l) and U file (-u) must be "
                     "specified together"
                  << std::endl;
        return 1;
    }

    // Validate parameters
    if ( matrix_size <= 0 )
    {
        std::cerr << "Error: Matrix size must be positive" << std::endl;
        return 1;
    }
    if ( sparsity <= 0.0 || sparsity > 1.0 )
    {
        std::cerr << "Error: Sparsity must be between 0.0 and 1.0" << std::endl;
        return 1;
    }
    if ( num_iterations <= 0 )
    {
        std::cerr << "Error: Number of iterations must be positive" << std::endl;
        return 1;
    }
    if ( method != "spsv" && method != "spsm" && method != "both" )
    {
        std::cerr << "Error: Method must be 'spsv', 'spsm', or 'both'" << std::endl;
        return 1;
    }

    printDeviceInfo();

    std::cout << "=== cuSPARSE SpSV vs SpSM Performance Test ===" << std::endl;
    if ( !l_file.empty() && !u_file.empty() )
    {
        std::cout << "L matrix file: " << l_file << std::endl;
        std::cout << "U matrix file: " << u_file << std::endl;
    }
    else
    {
        std::cout << "Matrix size: " << matrix_size << "x" << matrix_size << std::endl;
        std::cout << "Target sparsity: " << std::fixed << std::setprecision( 4 ) << sparsity << std::endl;
    }

    std::cout << "Benchmark iterations: " << num_iterations << std::endl;
    std::cout << "=========================================" << std::endl << std::endl;

    try
    {
        CudaLUBenchmark benchmark( matrix_size, sparsity, l_file, u_file );

        bool run_spsv = ( method == "spsv" || method == "both" );
        bool run_spsm = ( method == "spsm" || method == "both" );

        double spsv_time = 0.0, spsm_time = 0.0;

        // Benchmark SpSV
        if ( run_spsv )
        {
            std::cout << "=== Testing cuSPARSE SpSV ===" << std::endl;
            spsv_time = benchmark.benchmarkForwardBackward( num_iterations );
            benchmark.printResults( spsv_time, "SpSV" );
            std::cout << std::endl;
        }

        // Benchmark SpSM
        if ( run_spsm )
        {
            std::cout << "=== Testing cuSPARSE SpSM ===" << std::endl;
            spsm_time = benchmark.benchmarkForwardBackwardSpSM( num_iterations );
            benchmark.printResults( spsm_time, "SpSM" );
            std::cout << std::endl;
        }

        // Performance comparison
        if ( run_spsv && run_spsm )
        {
            std::cout << "=== Performance Comparison ===" << std::endl;
            std::cout << std::fixed << std::setprecision( 2 );
            double speedup = spsm_time / spsv_time;
            if ( speedup > 1.0 )
            {
                std::cout << "SpSV is " << speedup << "x faster than SpSM" << std::endl;
            }
            else
            {
                std::cout << "SpSM is " << ( 1.0 / speedup ) << "x faster than SpSV" << std::endl;
            }
            std::cout << "SpSV/SpSM time ratio: " << speedup << std::endl;
        }
    }
    catch ( const std::exception& e )
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
