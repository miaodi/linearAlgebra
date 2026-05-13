#include "matrix_utils.hpp"
#include "io.hpp"
#include "spgemm/spgemm_contract.cuh"
#include "spgemm/spgemm_expand.cuh"
#include "spgemm/spgemm_sort.cuh"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cxxopts.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using matrix_utils::sparse_cuda::DeviceArray;

void checkCuda( cudaError_t status, const char* message )
{
    if ( status != cudaSuccess )
    {
        throw std::runtime_error( std::string( message ) + ": " + cudaGetErrorString( status ) );
    }
}

void checkCuSparse( cusparseStatus_t status, const char* message )
{
    if ( status != CUSPARSE_STATUS_SUCCESS )
    {
        throw std::runtime_error( std::string( message ) + ": " + cusparseGetErrorString( status ) );
    }
}

void requireCudaDevice()
{
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount( &device_count );
    if ( status != cudaSuccess || device_count == 0 )
    {
        throw std::runtime_error( std::string( "CUDA device unavailable: " ) + cudaGetErrorString( status ) );
    }
}

template <typename Func>
double timeMs( Func&& func )
{
    checkCuda( cudaDeviceSynchronize(), "pre-timing synchronize" );
    const auto begin = std::chrono::high_resolution_clock::now();
    func();
    checkCuda( cudaDeviceSynchronize(), "post-timing synchronize" );
    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>( end - begin ).count();
}

void printTimingRow( const char* phase, double milliseconds, double total_ms )
{
    const double percent = total_ms > 0.0 ? 100.0 * milliseconds / total_ms : 0.0;
    std::cout << "  " << std::left << std::setw( 12 ) << phase << std::right << std::setw( 10 )
              << milliseconds << std::setw( 9 ) << percent << '\n';
}

struct CusparseHandleGuard
{
    cusparseHandle_t handle = nullptr;

    CusparseHandleGuard() { checkCuSparse( cusparseCreate( &handle ), "create cuSPARSE handle" ); }

    ~CusparseHandleGuard()
    {
        if ( handle != nullptr )
        {
            cusparseDestroy( handle );
        }
    }
};

struct CusparseSpMatGuard
{
    cusparseSpMatDescr_t descr = nullptr;

    ~CusparseSpMatGuard()
    {
        if ( descr != nullptr )
        {
            cusparseDestroySpMat( descr );
        }
    }
};

struct CusparseSpGEMMDescrGuard
{
    cusparseSpGEMMDescr_t descr = nullptr;

    CusparseSpGEMMDescrGuard()
    {
        checkCuSparse( cusparseSpGEMM_createDescr( &descr ), "create cuSPARSE SpGEMM descriptor" );
    }

    ~CusparseSpGEMMDescrGuard()
    {
        if ( descr != nullptr )
        {
            cusparseSpGEMM_destroyDescr( descr );
        }
    }
};

struct DeviceCsrProduct
{
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    DeviceArray<int> row_ptr;
    DeviceArray<int> col_ind;
    DeviceArray<double> values;
};

void runCuSparseSpGEMMAA( int rows, int nnz, int* d_row_ptr, int* d_col_ind, double* d_values, DeviceCsrProduct& product )
{
    constexpr cusparseIndexType_t index_type = CUSPARSE_INDEX_32I;
    constexpr cusparseIndexBase_t index_base = CUSPARSE_INDEX_BASE_ZERO;
    constexpr cudaDataType value_type = CUDA_R_64F;
    constexpr cusparseSpGEMMAlg_t algorithm = CUSPARSE_SPGEMM_DEFAULT;
    const double alpha = 1.0;
    const double beta = 0.0;

    CusparseHandleGuard handle;
    CusparseSpMatGuard mat_a;
    CusparseSpMatGuard mat_b;
    DeviceArray<int> c_col_placeholder;
    DeviceArray<double> c_value_placeholder;
    CusparseSpMatGuard mat_c;
    CusparseSpGEMMDescrGuard spgemm_descr;
    product.rows = rows;
    product.cols = rows;
    product.nnz = 0;
    product.row_ptr.resize( static_cast<size_t>( rows + 1 ) );
    c_col_placeholder.resize( 1 );
    c_value_placeholder.resize( 1 );

    checkCuSparse( cusparseCreateCsr( &mat_a.descr, rows, rows, nnz, d_row_ptr, d_col_ind, d_values,
                                      index_type, index_type, index_base, value_type ),
                   "create cuSPARSE A CSR descriptor" );
    checkCuSparse( cusparseCreateCsr( &mat_b.descr, rows, rows, nnz, d_row_ptr, d_col_ind, d_values,
                                      index_type, index_type, index_base, value_type ),
                   "create cuSPARSE B CSR descriptor" );
    checkCuSparse( cusparseCreateCsr( &mat_c.descr, rows, rows, 0, product.row_ptr.data(),
                                      c_col_placeholder.data(), c_value_placeholder.data(),
                                      index_type, index_type, index_base, value_type ),
                   "create cuSPARSE C CSR descriptor" );

    size_t buffer1_bytes = 0;
    checkCuSparse( cusparseSpGEMM_workEstimation( handle.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a.descr,
                                                  mat_b.descr, &beta, mat_c.descr, value_type, algorithm,
                                                  spgemm_descr.descr, &buffer1_bytes, nullptr ),
                   "query cuSPARSE SpGEMM work-estimation buffer" );
    DeviceArray<std::uint8_t> buffer1;
    buffer1.resize( buffer1_bytes );
    checkCuSparse( cusparseSpGEMM_workEstimation( handle.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a.descr,
                                                  mat_b.descr, &beta, mat_c.descr, value_type, algorithm,
                                                  spgemm_descr.descr, &buffer1_bytes, buffer1.data() ),
                   "run cuSPARSE SpGEMM work estimation" );

    size_t buffer2_bytes = 0;
    checkCuSparse( cusparseSpGEMM_compute( handle.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a.descr,
                                           mat_b.descr, &beta, mat_c.descr, value_type, algorithm,
                                           spgemm_descr.descr, &buffer2_bytes, nullptr ),
                   "query cuSPARSE SpGEMM compute buffer" );
    DeviceArray<std::uint8_t> buffer2;
    buffer2.resize( buffer2_bytes );
    checkCuSparse( cusparseSpGEMM_compute( handle.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a.descr,
                                           mat_b.descr, &beta, mat_c.descr, value_type, algorithm,
                                           spgemm_descr.descr, &buffer2_bytes, buffer2.data() ),
                   "run cuSPARSE SpGEMM compute" );

    int64_t c_rows = 0;
    int64_t c_cols = 0;
    int64_t c_nnz = 0;
    checkCuSparse( cusparseSpMatGetSize( mat_c.descr, &c_rows, &c_cols, &c_nnz ),
                   "query cuSPARSE SpGEMM output size" );
    if ( c_rows > std::numeric_limits<int>::max() || c_cols > std::numeric_limits<int>::max() ||
         c_nnz > std::numeric_limits<int>::max() )
    {
        throw std::runtime_error( "cuSPARSE SpGEMM output exceeds int CSR limits." );
    }

    product.rows = static_cast<int>( c_rows );
    product.cols = static_cast<int>( c_cols );
    product.nnz = static_cast<int>( c_nnz );
    product.col_ind.resize( static_cast<size_t>( product.nnz ) );
    product.values.resize( static_cast<size_t>( product.nnz ) );

    checkCuSparse( cusparseCsrSetPointers( mat_c.descr, product.row_ptr.data(),
                                           product.col_ind.data(), product.values.data() ),
                   "set cuSPARSE SpGEMM output CSR pointers" );
    checkCuSparse( cusparseSpGEMM_copy( handle.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a.descr, mat_b.descr,
                                        &beta, mat_c.descr, value_type, algorithm, spgemm_descr.descr ),
                   "copy cuSPARSE SpGEMM output" );
    checkCuda( cudaDeviceSynchronize(),
               "synchronize cuSPARSE SpGEMM before descriptor destruction" );
}

template <typename T>
struct ExactComparison
{
    bool match = true;
    size_t lhs_size = 0;
    size_t rhs_size = 0;
    size_t first_mismatch = 0;
    T lhs_value{};
    T rhs_value{};
};

template <typename T>
ExactComparison<T> compareExact( const std::vector<T>& lhs, const std::vector<T>& rhs )
{
    ExactComparison<T> result;
    result.lhs_size = lhs.size();
    result.rhs_size = rhs.size();
    if ( lhs.size() != rhs.size() )
    {
        result.match = false;
        return result;
    }

    const auto mismatch = std::mismatch( lhs.begin(), lhs.end(), rhs.begin() );
    if ( mismatch.first != lhs.end() )
    {
        result.match = false;
        result.first_mismatch = static_cast<size_t>( std::distance( lhs.begin(), mismatch.first ) );
        result.lhs_value = *mismatch.first;
        result.rhs_value = *mismatch.second;
    }
    return result;
}

struct ValueComparison
{
    bool match = true;
    size_t lhs_size = 0;
    size_t rhs_size = 0;
    size_t first_mismatch = 0;
    double lhs_value = 0.0;
    double rhs_value = 0.0;
    double max_abs_diff = 0.0;
};

ValueComparison compareValues( const std::vector<double>& lhs,
                               const std::vector<double>& rhs,
                               double abs_tol = 1.0e-8,
                               double rel_tol = 1.0e-8 )
{
    ValueComparison result;
    result.lhs_size = lhs.size();
    result.rhs_size = rhs.size();
    if ( lhs.size() != rhs.size() )
    {
        result.match = false;
        return result;
    }

    for ( size_t i = 0; i < lhs.size(); ++i )
    {
        const double diff = std::abs( lhs[i] - rhs[i] );
        result.max_abs_diff = std::max( result.max_abs_diff, diff );
        const double tolerance = abs_tol + rel_tol * std::max( std::abs( lhs[i] ), std::abs( rhs[i] ) );
        if ( result.match && diff > tolerance )
        {
            result.match = false;
            result.first_mismatch = i;
            result.lhs_value = lhs[i];
            result.rhs_value = rhs[i];
        }
    }
    return result;
}

template <typename T>
void printExactComparison( const char* name, const ExactComparison<T>& comparison )
{
    std::cout << "  " << name << ": ";
    if ( comparison.match )
    {
        std::cout << "match\n";
        return;
    }
    if ( comparison.lhs_size != comparison.rhs_size )
    {
        std::cout << "mismatch (size custom=" << comparison.lhs_size
                  << ", cusparse=" << comparison.rhs_size << ")\n";
        return;
    }
    std::cout << "mismatch at " << comparison.first_mismatch << " (custom=" << comparison.lhs_value
              << ", cusparse=" << comparison.rhs_value << ")\n";
}

void printValueComparison( const char* name, const ValueComparison& comparison )
{
    std::cout << "  " << name << ": ";
    if ( comparison.match )
    {
        std::cout << "match (max abs diff=" << std::scientific << std::setprecision( 3 )
                  << comparison.max_abs_diff << std::defaultfloat << ")\n";
        return;
    }
    if ( comparison.lhs_size != comparison.rhs_size )
    {
        std::cout << "mismatch (size custom=" << comparison.lhs_size
                  << ", cusparse=" << comparison.rhs_size << ")\n";
        return;
    }
    std::cout << "mismatch at " << comparison.first_mismatch << " (custom=" << comparison.lhs_value
              << ", cusparse=" << comparison.rhs_value << ", max abs diff=" << std::scientific
              << std::setprecision( 3 ) << comparison.max_abs_diff << std::defaultfloat << ")\n";
}

} // namespace

int main( int argc, char** argv )
{
    try
    {
        cxxopts::Options options(
            "cuda_spgemm_phase_demo",
            "Read a MatrixMarket CSR matrix, run A*A SpGEMM phases, and print timings." );
        options.add_options()( "f,file", "Input MatrixMarket file path",
                               cxxopts::value<std::string>() )( "h,help", "Show help" );

        const auto parsed = options.parse( argc, argv );
        if ( parsed.count( "help" ) )
        {
            std::cout << options.help() << '\n';
            return 0;
        }
        if ( !parsed.count( "file" ) )
        {
            std::cout << options.help() << '\n';
            return 1;
        }

        requireCudaDevice();

        using HostCSR = matrix_utils::CSRMatrixVec<int, int, double>;
        HostCSR matrix;

        const std::string file_path = parsed["file"].as<std::string>();
        {
            std::ifstream input( file_path );
            if ( !input.is_open() )
            {
                throw std::runtime_error( "Cannot open MatrixMarket file: " + file_path );
            }
            matrix_utils::readMatrixMarket( input, matrix );
        }

        if ( matrix.rows != matrix.cols )
        {
            throw std::runtime_error( "A*A requires a square matrix." );
        }
        if ( matrix.Base() != 0 )
        {
            throw std::runtime_error( "This scratch demo expects a base-0 CSR matrix." );
        }

        const int rows = matrix.rows;
        const int nnz = static_cast<int>( matrix.NNZ() );
        std::cout << "Input\n"
                  << "  file: " << file_path << '\n'
                  << "  shape: " << rows << " x " << matrix.cols << '\n'
                  << "  nnz(A): " << nnz << '\n'
                  << "  base: " << matrix.Base() << '\n';

        DeviceArray<int> d_row_ptr;
        DeviceArray<int> d_col_ind;
        DeviceArray<double> d_values;
        d_row_ptr.copyFromHost( matrix.AI(), static_cast<size_t>( rows + 1 ) );
        d_col_ind.copyFromHost( matrix.AJ(), static_cast<size_t>( nnz ) );
        d_values.copyFromHost( matrix.AV(), static_cast<size_t>( nnz ) );

        matrix_utils::sparse_cuda::SpGEMMSymbolicResult<int, int> symbolic;
        double symbolic_ms = timeMs(
            [&]
            {
                if ( !matrix_utils::sparse_cuda::SpGEMMSymbolicAnalyzeCSR<int, int>(
                         rows, matrix.cols, d_row_ptr.data(), d_col_ind.data(), rows, d_row_ptr.data(), 0, symbolic ) )
                {
                    throw std::runtime_error( "SpGEMMSymbolicAnalyzeCSR failed." );
                }
            } );

        matrix_utils::sparse_cuda::SpGEMMExpandedProducts<int, double> expanded;
        double expansion_ms = timeMs(
            [&]
            {
                if ( !matrix_utils::sparse_cuda::SpGEMMExpandCSR<int, int, double>(
                         rows, matrix.cols, d_row_ptr.data(), d_col_ind.data(), d_values.data(), rows,
                         d_row_ptr.data(), d_col_ind.data(), d_values.data(), 0, symbolic, expanded ) )
                {
                    throw std::runtime_error( "SpGEMMExpandCSR failed." );
                }
            } );

        matrix_utils::sparse_cuda::SpGEMMExpandedProducts<int, double> sorted;
        double sorting_ms = timeMs(
            [&]
            {
                if ( !matrix_utils::sparse_cuda::SpGEMMSortExpandedProductsByColumn<int, int, double>(
                         symbolic, expanded, sorted ) )
                {
                    throw std::runtime_error( "SpGEMMSortExpandedProductsByColumn failed." );
                }
            } );

        matrix_utils::sparse_cuda::SpGEMMReducedProducts<int, double> reduced;
        double contraction_ms = timeMs(
            [&]
            {
                if ( !matrix_utils::sparse_cuda::SpGEMMContractSortedProducts<int, int, double>(
                         symbolic, sorted, reduced ) )
                {
                    throw std::runtime_error( "SpGEMMContractSortedProducts failed." );
                }
            } );

        matrix_utils::sparse_cuda::DeviceCSRMatrix<int, int> contracted;
        DeviceArray<double> contracted_values;
        double construct_ms = timeMs(
            [&]
            {
                if ( !matrix_utils::sparse_cuda::SpGEMMConstructCSR<int, int, double>(
                         symbolic, reduced, contracted, contracted_values ) )
                {
                    throw std::runtime_error( "SpGEMMConstructCSR failed." );
                }
            } );

        std::vector<int> contracted_row_ptr( static_cast<size_t>( rows + 1 ) );
        contracted.ai.copyToHost( contracted_row_ptr.data() );
        const int contracted_nnz = contracted_row_ptr.back() - contracted.base;
        std::vector<int> contracted_col_ind( static_cast<size_t>( contracted_nnz ) );
        std::vector<double> contracted_av( static_cast<size_t>( contracted_nnz ) );
        contracted.aj.copyToHost( contracted_col_ind.data() );
        contracted_values.copyToHost( contracted_av.data() );

        DeviceCsrProduct cusparse_product;
        double cusparse_ms = timeMs(
            [&]
            {
                runCuSparseSpGEMMAA( rows, nnz, d_row_ptr.data(), d_col_ind.data(), d_values.data(), cusparse_product );
            } );

        std::vector<int> cusparse_row_ptr( static_cast<size_t>( rows + 1 ) );
        std::vector<int> cusparse_col_ind( static_cast<size_t>( cusparse_product.nnz ) );
        std::vector<double> cusparse_av( static_cast<size_t>( cusparse_product.nnz ) );
        cusparse_product.row_ptr.copyToHost( cusparse_row_ptr.data() );
        cusparse_product.col_ind.copyToHost( cusparse_col_ind.data() );
        cusparse_product.values.copyToHost( cusparse_av.data() );

        const auto ai_comparison = compareExact( contracted_row_ptr, cusparse_row_ptr );
        const auto aj_comparison = compareExact( contracted_col_ind, cusparse_col_ind );
        const auto av_comparison = compareValues( contracted_av, cusparse_av );

        const int thread_rows = symbolic.classEnd( matrix_utils::sparse_cuda::SpGEMMRowClass::Thread ) -
                                symbolic.classBegin( matrix_utils::sparse_cuda::SpGEMMRowClass::Thread );
        const int warp_rows = symbolic.classEnd( matrix_utils::sparse_cuda::SpGEMMRowClass::Warp ) -
                              symbolic.classBegin( matrix_utils::sparse_cuda::SpGEMMRowClass::Warp );
        const int cta_rows = symbolic.classEnd( matrix_utils::sparse_cuda::SpGEMMRowClass::CTA ) -
                             symbolic.classBegin( matrix_utils::sparse_cuda::SpGEMMRowClass::CTA );
        const int global_rows = symbolic.classEnd( matrix_utils::sparse_cuda::SpGEMMRowClass::Global ) -
                                symbolic.classBegin( matrix_utils::sparse_cuda::SpGEMMRowClass::Global );

        const double total_ms = symbolic_ms + expansion_ms + sorting_ms + contraction_ms + construct_ms;

        std::cout << "\nA*A Result\n"
                  << "  nnz(C_hat): " << symbolic.total_expanded_nnz << '\n'
                  << "  nnz(C): " << contracted_nnz << '\n'
                  << "  nnz(C cuSPARSE): " << cusparse_product.nnz << '\n'
                  << "  row classes: thread=" << thread_rows << ", warp=" << warp_rows
                  << ", cta=" << cta_rows << ", global=" << global_rows << '\n';

        std::cout << "\nComparison vs cuSPARSE\n";
        printExactComparison( "ai", ai_comparison );
        printExactComparison( "aj", aj_comparison );
        printValueComparison( "av", av_comparison );

        std::cout << "\nTiming\n"
                  << std::fixed << std::setprecision( 3 ) << "  " << std::left << std::setw( 12 ) << "phase"
                  << std::right << std::setw( 10 ) << "ms" << std::setw( 9 ) << "% custom" << '\n';
        printTimingRow( "symbolic", symbolic_ms, total_ms );
        printTimingRow( "expansion", expansion_ms, total_ms );
        printTimingRow( "sorting", sorting_ms, total_ms );
        printTimingRow( "contraction", contraction_ms, total_ms );
        printTimingRow( "construct", construct_ms, total_ms );
        printTimingRow( "custom total", total_ms, total_ms );
        printTimingRow( "cusparse", cusparse_ms, total_ms );

        return 0;
    }
    catch ( const std::exception& err )
    {
        std::cerr << "error: " << err.what() << '\n';
        return 1;
    }
}
