#include "io.hpp"
#include "sp_ops.hpp"
#include "utils.h"
#include <cxxopts.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include "graph_algs.hpp"
#include "permutation.hpp"

using namespace matrix_utils;

int main(int argc, char** argv) {
    cxxopts::Options options("test_ilum", "Generate sparsity pattern visualizations for matrix and A+A^T");
    
    options.add_options()
        ("f,file", "Input matrix file (Matrix Market format)", 
         cxxopts::value<std::string>()->default_value("../data/ex5.mtx"))
        ("o,output", "Output file prefix (default: matrix)", 
         cxxopts::value<std::string>()->default_value("matrix"))
        ("s,size", "Maximum display size for SVG (default: 2000)", 
         cxxopts::value<int>()->default_value("2000"))
        ("t,threads", "Number of threads for A+A^T computation (default: 4)", 
         cxxopts::value<int>()->default_value("4"))
        ("h,help", "Print help");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        std::cout << "\nExample usage:\n";
        std::cout << "  " << argv[0] << " -f ../data/ex5.mtx\n";
        std::cout << "  " << argv[0] << " -f ../data/ex5.mtx -o output -s 3000 -t 8\n";
        std::cout << "  " << argv[0] << " (uses default: ../data/ex5.mtx)\n";
        return 0;
    }

    std::string matrix_file = result["file"].as<std::string>();
    std::string output_prefix = result["output"].as<std::string>();
    int max_display_size = result["size"].as<int>();
    int num_threads = result["threads"].as<int>();

    std::cout << "Reading matrix from: " << matrix_file << std::endl;

    // Read the matrix
    std::ifstream f(matrix_file);
    if (!f.good()) {
        std::cerr << "Error: Cannot open file " << matrix_file << std::endl;
        return 1;
    }

    std::vector<int> ai, aj;
    std::vector<double> av;
    utils::read_matrix_market_csr(f, ai, aj, av);
    f.close();

    if (ai.size() == 0) {
        std::cerr << "Error: Could not read matrix" << std::endl;
        return 1;
    }

    const int size = ai.size() - 1;
    const int base = ai[0];
    const int nnz = ai[size] - base;

    std::cout << "Matrix properties:" << std::endl;
    std::cout << "  Size: " << size << " x " << size << std::endl;
    std::cout << "  NNZ: " << nnz << std::endl;
    std::cout << "  Base: " << base << std::endl;
    std::cout << "  Density: " << (100.0 * nnz / (size * size)) << "%" << std::endl;
    std::cout << "  Threads: " << num_threads << std::endl;
    std::cout << "  Max display size: " << max_display_size << std::endl;

    // Generate SVG for original matrix
    std::string original_svg = output_prefix + "_original.svg";
    std::cout << "\nGenerating original sparsity pattern: " << original_svg << std::endl;
    {
        std::ofstream svg_out(original_svg);
        if (!svg_out.good()) {
            std::cerr << "Error: Cannot create file " << original_svg << std::endl;
            return 1;
        }
        writeSVG(size, size, ai.data(), aj.data(), svg_out, max_display_size);
        svg_out.close();
    }
    std::cout << "  Written to: " << original_svg << std::endl;

    // Compute A+A^T structure
    std::cout << "\nComputing A+A^T structure..." << std::endl;
    
    // Test with KEEPDIAG=true
    {
        APlusATStruct<int, int, true> aplusat_op(num_threads);
        
        std::vector<int> result_ai(size + 1);
        std::vector<int> result_aj(2 * nnz); // Upper bound

        aplusat_op(size, ai.data(), aj.data(), result_ai.data(), result_aj.data());

        const int result_nnz = result_ai[size] - result_ai[0];
        std::cout << "A+A^T properties (with diagonal):" << std::endl;
        std::cout << "  Size: " << size << " x " << size << std::endl;
        std::cout << "  NNZ: " << result_nnz << std::endl;
        std::cout << "  Original NNZ: " << nnz << std::endl;
        std::cout << "  Ratio: " << (double)result_nnz / nnz << "x" << std::endl;
        std::cout << "  Density: " << (100.0 * result_nnz / (size * size)) << "%" << std::endl;

        // Generate SVG for A+A^T
        std::string aplusat_svg = output_prefix + "_aplusat_keepdiag.svg";
        std::cout << "\nGenerating A+A^T sparsity pattern: " << aplusat_svg << std::endl;
        {
            std::ofstream svg_out(aplusat_svg);
            if (!svg_out.good()) {
                std::cerr << "Error: Cannot create file " << aplusat_svg << std::endl;
                return 1;
            }
            writeSVG(size, size, result_ai.data(), result_aj.data(), svg_out, max_display_size);
            svg_out.close();
        }
        std::cout << "  Written to: " << aplusat_svg << std::endl;
    }
    
    // Compute MIS permutation
    std::cout << "\nComputing MIS permutation..." << std::endl;
    std::vector<int> perm(size);
    std::vector<int> iperm(size);
    matrix_utils::MISPerm(size, ai.data(), aj.data(), perm.data(), iperm.data());
    std::cout << "  MIS permutation computed" << std::endl;
    
    // Verify permutation
    if (!matrix_utils::isPermutation(size, base, perm.data())) {
        std::cerr << "Error: Invalid permutation generated!" << std::endl;
        return 1;
    }
    std::cout << "  Permutation verified: valid" << std::endl;
    
    // Permute the matrix: perm_A = P * A * P^T
    std::cout << "\nPermuting matrix..." << std::endl;
    std::vector<int> perm_ai(size + 1);
    std::vector<int> perm_aj(nnz);
    std::vector<double> perm_av(nnz);
    
    matrix_utils::permuteMat(size, size, perm.data(), iperm.data(),
                            ai.data(), aj.data(), av.data(),
                            perm_ai.data(), perm_aj.data(), perm_av.data());
    
    std::cout << "  Matrix permuted" << std::endl;
    std::cout << "  Permuted matrix NNZ: " << (perm_ai[size] - perm_ai[0]) << std::endl;
    
    // Generate SVG for permuted matrix
    std::string permuted_svg = output_prefix + "_mis_permuted.svg";
    std::cout << "\nGenerating MIS permuted sparsity pattern: " << permuted_svg << std::endl;
    {
        std::ofstream svg_out(permuted_svg);
        if (!svg_out.good()) {
            std::cerr << "Error: Cannot create file " << permuted_svg << std::endl;
            return 1;
        }
        writeSVG(size, size, perm_ai.data(), perm_aj.data(), svg_out, max_display_size);
        svg_out.close();
    }
    std::cout << "  Written to: " << permuted_svg << std::endl;

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Generated SVG files:" << std::endl;
    std::cout << "  1. " << output_prefix << "_original.svg (original matrix)" << std::endl;
    std::cout << "  2. " << output_prefix << "_aplusat_keepdiag.svg (A+A^T with diagonal)" << std::endl;
    std::cout << "  3. " << output_prefix << "_mis_permuted.svg (MIS permuted matrix)" << std::endl;
    std::cout << "\nDisplay resolution: " << max_display_size << "x" << max_display_size << std::endl;
    std::cout << "Open the SVG files in a web browser to view the sparsity patterns." << std::endl;

    return 0;
}
