#include <iostream>
#include <vector>
#include "../../sparse_mat_op/cuda/cuda_memory.cuh"
#include "../../sparse_mat_op/cuda/cuda_spmm.cuh"

using namespace matrix_utils::sparse_cuda;

int main( int argc, char** argv )
{
    // Test parameters
    const int n_rows = 5; // Number of rows
    const int base = 0;   // 0-based indexing

    // Host arrays for row pointers (ai_A and ai_B)
    // User will fill these - representing CSR row pointer arrays
    // For example, if matrix A has row pointer [0, 2, 5, 7, 10, 12]
    std::vector<int> ai_A( n_rows + 1 );
    std::vector<int> ai_B( n_rows + 1 );

    // TODO: Fill ai_A and ai_B here
    // Example initialization (user should replace this):
    ai_A = { 0, 3, 6, 9, 12, 15 }; // Each row has 3 non-zeros
    ai_B = { 0, 2, 5, 6, 8, 9 };   // Each row has 2 non-zeros

    std::cout << "Testing SpMMAnalyze with n_rows=" << n_rows << std::endl;
    std::cout << "ai_A: ";
    for ( int val : ai_A )
        std::cout << val << " ";
    std::cout << "\nai_B: ";
    for ( int val : ai_B )
        std::cout << val << " ";
    std::cout << "\n\n";

    // Step 1: Copy ai_A and ai_B to device using cuda_memory
    DeviceArray<int> d_ai_A;
    DeviceArray<int> d_ai_B;

    d_ai_A.copyFromHost( ai_A.data(), ai_A.size() );
    d_ai_B.copyFromHost( ai_B.data(), ai_B.size() );

    std::cout << "Copied ai_A and ai_B to device" << std::endl;

    // Step 2: Allocate workload_prefix on device
    DeviceArray<int> d_workload_prefix;
    d_workload_prefix.resize( n_rows + 1 );

    std::cout << "Allocated workload_prefix array on device" << std::endl;

    // Step 3: Run SpMMAnalyze
    int required_array_size = 0;
    bool success = SpMMAnalyze<int, int>( n_rows, d_ai_A.data(), d_ai_B.data(), base,
                                          d_workload_prefix.data(), &required_array_size );

    if ( !success )
    {
        std::cerr << "SpMMAnalyze failed!" << std::endl;
        return 1;
    }

    std::cout << "SpMMAnalyze completed successfully" << std::endl;
    std::cout << "Required array size: " << required_array_size << std::endl;

    // Step 4: Copy workload_prefix back to host using cuda_memory
    std::vector<int> h_workload_prefix( n_rows + 1 );
    d_workload_prefix.copyToHost( h_workload_prefix.data() );

    // Step 5: Print results
    std::cout << "\nWorkload prefix array:" << std::endl;
    for ( int i = 0; i <= n_rows; i++ )
    {
        std::cout << "  workload_prefix[" << i << "] = " << h_workload_prefix[i];
        if ( i > 0 )
        {
            int workload = h_workload_prefix[i] - h_workload_prefix[i - 1];
            int nnz_A = ai_A[i] - ai_A[i - 1];
            int nnz_B = ai_B[i] - ai_B[i - 1];
            std::cout << "  (row " << ( i - 1 ) << " workload: " << nnz_A << " x " << nnz_B << " = "
                      << workload << ")";
        }
        std::cout << std::endl;
    }

    std::cout << "\nTest completed successfully!" << std::endl;

    return 0;
}
