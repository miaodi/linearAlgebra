#include "io.hpp"
#include "spmv.hpp"
#include "utils.h"
#include <fstream>
#include <iostream>
#include <vector>

using namespace matrix_utils;

int main()
{
    std::vector<int> csr_rows;
    std::vector<int> csr_cols;
    std::vector<double> csr_vals;

    std::ifstream f;
    f.open( "data/ex5.mtx" );
    matrix_utils::readMatrixMarket( f, csr_rows, csr_cols, csr_vals );
    f.close();

    int n = csr_rows.size() - 1;
    int base = 0;
    std::vector<double> b( n, 1.0 );
    std::vector<double> x_serial( n, 0.0 );
    std::vector<double> x_albus( n, 0.0 );

    SerialSPMV spmv;
    spmv( n, base, csr_rows.data(), csr_cols.data(), csr_vals.data(), b.data(), x_serial.data(), 1.0, 0.0 );

    ALBUSSPMV<int, int, double> albus_spmv( 2 );
    albus_spmv.preprocess( n, csr_rows.data(), csr_cols.data(), csr_vals.data() );
    albus_spmv( n, csr_rows.data(), csr_cols.data(), csr_vals.data(), b.data(), x_albus.data(), 1.0, 0.0 );

    // Find rows with largest errors
    double max_rel_error = 0.0;
    int max_error_row = -1;
    for ( int i = 0; i < n; i++ )
    {
        double abs_err = std::abs( x_albus[i] - x_serial[i] );
        double rel_err = std::abs( x_serial[i] ) > 1e-15 ? abs_err / std::abs( x_serial[i] ) : abs_err;
        if ( rel_err > max_rel_error )
        {
            max_rel_error = rel_err;
            max_error_row = i;
        }
    }

    std::cout << "Row with max error: " << max_error_row << std::endl;
    std::cout << "Serial result: " << x_serial[max_error_row] << std::endl;
    std::cout << "ALBUS result:  " << x_albus[max_error_row] << std::endl;
    std::cout << "Abs error: " << std::abs( x_albus[max_error_row] - x_serial[max_error_row] ) << std::endl;
    std::cout << "Rel error: " << max_rel_error << std::endl;

    // Check if this row spans multiple threads
    std::cout << "\nRow " << max_error_row << " info:" << std::endl;
    std::cout << "Row start: " << csr_rows[max_error_row]
              << ", end: " << csr_rows[max_error_row + 1] << std::endl;
    std::cout << "NNZ in row: " << ( csr_rows[max_error_row + 1] - csr_rows[max_error_row] ) << std::endl;

    return 0;
}
