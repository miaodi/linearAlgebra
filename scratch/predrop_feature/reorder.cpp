#include "io.hpp"
#include "matrix_utils.hpp"
#include "permutation.hpp"
#include "sp_ops.hpp"
#include "Reordering.h"
#include <chrono>
#include <cxxopts.hpp>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>
#include "utils.h"
struct Options
{
    std::string filename;
    std::string output_file;
    int max_display_size;
    int threads;
};

void printOptions(const Options& opts)
{
    std::cout << "Options:" << std::endl;
    std::cout << "  filename: " << opts.filename << std::endl;
    std::cout << "  output: " << opts.output_file << std::endl;
    std::cout << "  max_display_size: " << opts.max_display_size << std::endl;
    std::cout << "  threads: " << opts.threads << std::endl;
}

int main(int argc, char* argv[])
{
    cxxopts::Options options("rcm_reorder", "RCM reordering of matrix adjacency graph");

    // clang-format off
    options.add_options()
        ("f,filename", "Matrix Market file to read", cxxopts::value<std::string>()->default_value("../../tests/data/ex5.mtx"))
        ("o,output", "Output SVG file path", cxxopts::value<std::string>()->default_value("rcm_reordered.svg"))
        ("s,size", "Maximum display size (pixels)", cxxopts::value<int>()->default_value("200"))
        ("n,threads", "Number of threads", cxxopts::value<int>()->default_value("1"))
        ("h,help", "Print usage");
    // clang-format on

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        return 0;
    }

    Options opts;
    opts.filename = result["filename"].as<std::string>();
    opts.output_file = result["output"].as<std::string>();
    opts.max_display_size = result["size"].as<int>();
    opts.threads = result["threads"].as<int>();

    printOptions(opts);

    omp_set_num_threads(opts.threads);

    // Read matrix
    std::ifstream f(opts.filename);
    if (!f.is_open())
    {
        std::cerr << "Failed to open file: " << opts.filename << std::endl;
        return -1;
    }

    matrix_utils::CSRMatrix<int, int, double> matrix;
    matrix_utils::readMatrixMarket(f, matrix);
    f.close();

    std::cout << "\nOriginal matrix: " << matrix.rows << " x " << matrix.cols
              << ", NNZ: " << matrix.NNZ() << std::endl;

    // Write original matrix to SVG
    std::ofstream out_orig("original_matrix.svg");
    if (out_orig.is_open())
    {
        std::cout << "\nWriting original matrix to SVG..." << std::endl;
        matrix_utils::writeSVG(matrix.rows, matrix.cols, matrix.AI(), matrix.AJ(),
                              out_orig, opts.max_display_size);
        out_orig.close();
        std::cout << "Original matrix written to original_matrix.svg" << std::endl;
    }

    // Compute A+A^T (symmetric adjacency graph without diagonal)
    std::cout << "\n=== Computing A+A^T (adjacency graph) ===" << std::endl;
    std::vector<int> xadj(matrix.rows + 1);
    
    auto aplusat_start = std::chrono::high_resolution_clock::now();
    matrix_utils::APlusATPrefix<int, int, false>(
        matrix.rows, matrix.AI(), matrix.AJ(), xadj.data());
    
    // Allocate and fill adjacency list
    int actual_edges = xadj[matrix.rows] - xadj[0];
    std::vector<int> adjncy(actual_edges);
    matrix_utils::APlusATFill<int, int, false>(
        matrix.rows, matrix.AI(), matrix.AJ(), xadj.data(), adjncy.data());
    
    auto aplusat_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> aplusat_time = aplusat_end - aplusat_start;
    
    std::cout << "Adjacency graph created (A+A^T without diagonal):" << std::endl;
    std::cout << "  Vertices: " << matrix.rows << std::endl;
    std::cout << "  Edges: " << actual_edges << std::endl;
    std::cout << "  Time: " << aplusat_time.count() << " s" << std::endl;

    // Compute RCM ordering
    std::cout << "\n=== Computing RCM ordering ===" << std::endl;
    std::vector<int> perm(matrix.rows);
    
    auto rcm_start = std::chrono::high_resolution_clock::now();
    reordering::RCM_MultiComponent(matrix.rows, xadj.data(), adjncy.data(),
                                   perm.data(), opts.threads);
    auto rcm_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> rcm_time = rcm_end - rcm_start;
    
    std::cout << "RCM ordering computed successfully" << std::endl;

    // Compute inverse permutation
    std::vector<int> iperm(matrix.rows);
    matrix_utils::invPerm(matrix.rows, matrix.Base(), perm.data(), iperm.data(), opts.threads);

    // Allocate permuted matrix
    matrix_utils::CSRMatrix<int, int, double> permuted;
    permuted.rows = matrix.rows;
    permuted.cols = matrix.cols;
    permuted.ResizeAI(matrix.rows + 1);
    permuted.ResizeAJ(matrix.NNZ());
    permuted.ResizeAV(matrix.NNZ());
    
    // Permute the matrix: permuted = P * matrix * P^T (symmetric permutation)
    std::cout << "\n=== Permuting matrix ===" << std::endl;
    auto perm_start = std::chrono::high_resolution_clock::now();
    matrix_utils::permuteMat(matrix.rows, matrix.cols,
                            perm.data(), iperm.data(),
                            matrix.AI(), matrix.AJ(), matrix.AV(),
                            permuted.AI(), permuted.AJ(), permuted.AV(), opts.threads);
    auto perm_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> perm_time = perm_end - perm_start;
    
    std::cout << "Matrix permuted successfully" << std::endl;
    std::cout << "  Time: " << perm_time.count() << " s" << std::endl;
    std::cout << "  Permuted matrix: " << permuted.rows << " x " << permuted.cols 
              << ", NNZ: " << permuted.NNZ() << std::endl;

    // Write permuted matrix to SVG
    std::cout << "\nWriting RCM-reordered matrix to SVG..." << std::endl;
    std::ofstream out(opts.output_file);
    if (!out.is_open())
    {
        std::cerr << "Failed to create output file: " << opts.output_file << std::endl;
        return -1;
    }

    matrix_utils::writeSVG(permuted.rows, permuted.cols, permuted.AI(), permuted.AJ(),
                          out, opts.max_display_size);
    out.close();
    std::cout << "RCM-reordered matrix written to: " << opts.output_file << std::endl;

    // Summary
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Total time: " << (aplusat_time.count() + rcm_time.count() + perm_time.count()) << " s" << std::endl;
    std::cout << "  A+A^T construction: " << aplusat_time.count() << " s" << std::endl;
    std::cout << "  RCM ordering:       " << rcm_time.count() << " s" << std::endl;
    std::cout << "  Matrix permutation: " << perm_time.count() << " s" << std::endl;

    return 0;
}
